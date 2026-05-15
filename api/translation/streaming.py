"""Map OpenAI/xAI chat-completions SSE chunks into Anthropic Messages SSE events.

The Anthropic event sequence we MUST emit, in order:

    event: message_start
      data: {"type":"message_start","message":{...empty content, usage{input,output:0}}}
    (then per content block, by index:)
      event: content_block_start
        data: {"type":"content_block_start","index":N,"content_block":{...}}
      event: content_block_delta  (many)
        data: {"type":"content_block_delta","index":N,"delta":{...text_delta | input_json_delta}}
      event: content_block_stop
        data: {"type":"content_block_stop","index":N}
    event: message_delta
      data: {"type":"message_delta","delta":{"stop_reason":...,"stop_sequence":null},"usage":{"output_tokens":...}}
    event: message_stop
      data: {"type":"message_stop"}

OpenAI-side SSE chunks look like:

    data: {"id":"...","choices":[{"index":0,"delta":{"role":"assistant"}}], ...}\n\n
    data: {"id":"...","choices":[{"index":0,"delta":{"content":"Hello"}}], ...}\n\n
    data: {"id":"...","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_x","type":"function","function":{"name":"get","arguments":""}}]}}], ...}\n\n
    data: {"id":"...","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"q\""}}]}}], ...}\n\n
    data: {"id":"...","choices":[{"index":0,"delta":{},"finish_reason":"stop"}], ...}\n\n
    data: {"usage":{"prompt_tokens":N,"completion_tokens":M, "total_tokens":N+M}, "choices":[]}\n\n
    data: [DONE]\n\n

Tool-call deltas: index-keyed; the model emits id+name once (on the first chunk for
that tool) and streams `arguments` incrementally. We assemble those into a single
Anthropic ``tool_use`` content block per ``index``, with ``input_json_delta`` deltas
streaming the partial JSON of ``arguments``.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator

from api.translation.openai_to_anthropic import map_finish_reason


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


@dataclass
class _ToolCallState:
    anthropic_index: int  # index in the Anthropic content array
    id: str
    name: str
    started: bool = False


@dataclass
class _StreamState:
    message_id: str
    anthropic_model: str
    text_block_index: int | None = None  # which Anthropic content index holds the running text block
    text_block_started: bool = False
    text_block_closed: bool = False
    next_index: int = 0
    tool_calls: dict[int, _ToolCallState] = field(default_factory=dict)  # keyed by OpenAI tool_call index
    stop_reason: str | None = None
    output_tokens: int = 0
    input_tokens: int = 0
    message_start_sent: bool = False


def _emit_message_start(state: _StreamState) -> bytes:
    return _sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": state.message_id,
                "type": "message",
                "role": "assistant",
                "model": state.anthropic_model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": state.input_tokens,
                    "output_tokens": 0,
                },
            },
        },
    )


def _close_text_block(state: _StreamState) -> bytes | None:
    if state.text_block_started and not state.text_block_closed and state.text_block_index is not None:
        state.text_block_closed = True
        return _sse("content_block_stop", {"type": "content_block_stop", "index": state.text_block_index})
    return None


def _handle_content_delta(state: _StreamState, text: str) -> list[bytes]:
    out: list[bytes] = []
    if not text:
        return out
    if not state.text_block_started:
        state.text_block_index = state.next_index
        state.next_index += 1
        state.text_block_started = True
        out.append(
            _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": state.text_block_index,
                    "content_block": {"type": "text", "text": ""},
                },
            )
        )
    out.append(
        _sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": state.text_block_index,
                "delta": {"type": "text_delta", "text": text},
            },
        )
    )
    return out


def _handle_tool_call_delta(state: _StreamState, tc_chunk: dict) -> list[bytes]:
    out: list[bytes] = []
    # OpenAI streams tool-call deltas keyed by `index`. The id+name appear on first
    # appearance; arguments stream incrementally as a JSON-fragment string.
    oai_idx = tc_chunk.get("index")
    if oai_idx is None:
        # Some providers omit `index` when there's only one tool call. Fall back to 0.
        oai_idx = 0
    fn = tc_chunk.get("function") or {}
    name = fn.get("name")
    args_fragment = fn.get("arguments")
    call_id = tc_chunk.get("id")

    st = state.tool_calls.get(oai_idx)
    if st is None:
        # First chunk for this tool call. We may not yet have `name` (rare), in which
        # case we hold off on emitting content_block_start until we do.
        if not name and not call_id and not args_fragment:
            return out
        st = _ToolCallState(
            anthropic_index=state.next_index,
            id=call_id or f"toolu_{uuid.uuid4().hex[:24]}",
            name=name or "",
        )
        state.tool_calls[oai_idx] = st
        state.next_index += 1
    else:
        # Subsequent chunks may carry updates to id/name (uncommon but legal).
        if call_id and not st.id.startswith("toolu_"):
            st.id = call_id
        if name and not st.name:
            st.name = name

    if not st.started:
        # Emit the content_block_start once we know at least the name (Anthropic SDK
        # requires `name` to be non-empty). If we still don't have a name, hold.
        if not st.name:
            # Buffer the args fragment for later — accumulate into a list on the state.
            if args_fragment:
                st_args = getattr(st, "_buffered_args", "")
                st._buffered_args = st_args + args_fragment  # type: ignore[attr-defined]
            return out
        # Close any open text block before opening a tool-use block, so the indices stay
        # ordered correctly.
        closing = _close_text_block(state)
        if closing:
            out.append(closing)
        st.started = True
        out.append(
            _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": st.anthropic_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": st.id,
                        "name": st.name,
                        "input": {},
                    },
                },
            )
        )
        # Flush any buffered args we held while waiting for the name.
        buffered = getattr(st, "_buffered_args", "")
        if buffered:
            out.append(
                _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": st.anthropic_index,
                        "delta": {"type": "input_json_delta", "partial_json": buffered},
                    },
                )
            )
            st._buffered_args = ""  # type: ignore[attr-defined]

    if args_fragment:
        out.append(
            _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": st.anthropic_index,
                    "delta": {"type": "input_json_delta", "partial_json": args_fragment},
                },
            )
        )
    return out


def _emit_final(state: _StreamState) -> list[bytes]:
    out: list[bytes] = []
    closing = _close_text_block(state)
    if closing:
        out.append(closing)
    for st in state.tool_calls.values():
        if st.started:
            out.append(_sse("content_block_stop", {"type": "content_block_stop", "index": st.anthropic_index}))
    out.append(
        _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": state.stop_reason or "end_turn",
                    "stop_sequence": None,
                },
                "usage": {"output_tokens": state.output_tokens},
            },
        )
    )
    out.append(_sse("message_stop", {"type": "message_stop"}))
    return out


async def _iter_sse_data(byte_iter: AsyncIterator[bytes]) -> AsyncIterator[str]:
    """Parse an SSE byte stream into ``data:`` payload strings (one per event).

    Joins multi-line ``data:`` fields per the SSE spec. Ignores any ``event:`` /
    ``id:`` / comment lines — xAI doesn't use them.
    """

    buf = b""
    async for chunk in byte_iter:
        buf += chunk
        while True:
            sep_idx = buf.find(b"\n\n")
            if sep_idx == -1:
                # Some servers emit \r\n\r\n; check that too.
                sep_idx = buf.find(b"\r\n\r\n")
                if sep_idx == -1:
                    break
                event_bytes = buf[:sep_idx]
                buf = buf[sep_idx + 4 :]
            else:
                event_bytes = buf[:sep_idx]
                buf = buf[sep_idx + 2 :]
            lines = event_bytes.split(b"\n")
            data_parts: list[str] = []
            for raw in lines:
                line = raw.rstrip(b"\r")
                if line.startswith(b"data:"):
                    data_parts.append(line[5:].lstrip(b" ").decode("utf-8", errors="replace"))
            if data_parts:
                yield "\n".join(data_parts)


async def openai_stream_to_anthropic_events(
    upstream: AsyncIterator[bytes],
    *,
    anthropic_model: str,
    input_tokens_estimate: int = 0,
) -> AsyncIterator[bytes]:
    """Async generator yielding fully formatted Anthropic SSE event bytes."""

    state = _StreamState(
        message_id=f"msg_{uuid.uuid4().hex[:24]}",
        anthropic_model=anthropic_model,
        input_tokens=input_tokens_estimate,
    )

    finalized = False
    try:
        async for data_str in _iter_sse_data(upstream):
            if data_str == "[DONE]":
                break
            try:
                payload = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if not state.message_start_sent:
                # Pull a better message id off the upstream if it gave us one.
                upstream_id = payload.get("id")
                if upstream_id:
                    state.message_id = (
                        upstream_id if upstream_id.startswith("msg_") else f"msg_{upstream_id}"
                    )
                state.message_start_sent = True
                yield _emit_message_start(state)

            # Some OpenAI-compatible providers emit a final standalone `usage` chunk
            # with `choices: []` when `stream_options.include_usage: true` is set.
            usage = payload.get("usage")
            if usage:
                state.input_tokens = int(usage.get("prompt_tokens") or state.input_tokens or 0)
                state.output_tokens = int(usage.get("completion_tokens") or state.output_tokens or 0)

            for choice in payload.get("choices") or []:
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if isinstance(content, str) and content:
                    for ev in _handle_content_delta(state, content):
                        yield ev
                elif isinstance(content, list):
                    # Some providers stream multimodal content; flatten text parts.
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            for ev in _handle_content_delta(state, part.get("text") or ""):
                                yield ev

                for tc_chunk in delta.get("tool_calls") or []:
                    if isinstance(tc_chunk, dict):
                        for ev in _handle_tool_call_delta(state, tc_chunk):
                            yield ev

                fr = choice.get("finish_reason")
                if fr:
                    state.stop_reason = map_finish_reason(fr) or state.stop_reason
    finally:
        if not state.message_start_sent:
            # Upstream died before any chunk — still emit a valid empty Anthropic stream.
            state.message_start_sent = True
            yield _emit_message_start(state)
        if not finalized:
            for ev in _emit_final(state):
                yield ev
            finalized = True
