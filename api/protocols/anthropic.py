"""AnthropicInbound — translates Anthropic Messages API bodies <-> Canonical.

This is the only place in the codebase that knows the Anthropic wire format. The
canonical layer downstream never sees an ``Anthropic*`` type; provider adapters
never see one either.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator

from api.canonical.events import (
    ContentBlockDone,
    ContentBlockStart,
    ContentTextDelta,
    ContentToolCallDelta,
    MessageDelta,
    StreamDone,
    StreamError,
    StreamStart,
)
from api.canonical.types import (
    CanonicalContent,
    CanonicalImage,
    CanonicalMessage,
    CanonicalRequest,
    CanonicalResponse,
    CanonicalText,
    CanonicalTool,
    CanonicalToolCall,
    CanonicalToolChoice,
    CanonicalToolResult,
)
from api.v1.schemas import MessagesRequest

ANTHROPIC_PROTOCOL = "anthropic-messages-v1"


# ---------------------------------------------------------------------------
# decode: Anthropic body -> CanonicalRequest
# ---------------------------------------------------------------------------


def _decode_system(system: str | list[dict[str, Any]] | None) -> list[CanonicalContent] | None:
    if system is None:
        return None
    if isinstance(system, str):
        return [CanonicalText(text=system)] if system else None
    out: list[CanonicalContent] = []
    for b in system:
        if isinstance(b, dict) and b.get("type") == "text":
            out.append(CanonicalText(text=b.get("text") or ""))
    return out or None


def _decode_image(block: dict[str, Any]) -> CanonicalImage | None:
    source = block.get("source") or {}
    src_type = source.get("type")
    if src_type == "url":
        url = source.get("url")
        if not url:
            return None
        return CanonicalImage(media_type=source.get("media_type") or "image/png", url=url)
    if src_type == "base64":
        data = source.get("data")
        if not data:
            return None
        return CanonicalImage(media_type=source.get("media_type") or "image/png", data=data)
    return None


def _flatten_tool_result_content(content: str | list[dict[str, Any]]) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for b in content or []:
        if isinstance(b, dict) and b.get("type") == "text":
            parts.append(b.get("text") or "")
        elif isinstance(b, dict):
            parts.append(json.dumps(b, ensure_ascii=False))
    return "\n".join(p for p in parts if p)


def _decode_message_content(raw: str | list[dict[str, Any]]) -> list[CanonicalContent]:
    if isinstance(raw, str):
        return [CanonicalText(text=raw)] if raw else []
    out: list[CanonicalContent] = []
    for b in raw or []:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "text":
            out.append(CanonicalText(text=b.get("text") or ""))
        elif t == "image":
            img = _decode_image(b)
            if img:
                out.append(img)
        elif t == "tool_use":
            out.append(
                CanonicalToolCall(
                    id=b.get("id") or "",
                    name=b.get("name") or "",
                    input=b.get("input") or {},
                )
            )
        elif t == "tool_result":
            out.append(
                CanonicalToolResult(
                    tool_call_id=b.get("tool_use_id") or "",
                    content=_flatten_tool_result_content(b.get("content") or ""),
                    is_error=bool(b.get("is_error") or False),
                )
            )
        # thinking + unknown -> dropped, retained as metadata key on the canonical request
    return out


def _decode_tools(tools: list[Any] | None) -> list[CanonicalTool]:
    if not tools:
        return []
    out: list[CanonicalTool] = []
    for t in tools:
        td = t.model_dump(exclude_none=True) if hasattr(t, "model_dump") else dict(t)
        out.append(
            CanonicalTool(
                name=td.get("name") or "",
                description=td.get("description") or "",
                json_schema=td.get("input_schema") or {},
            )
        )
    return out


def _decode_tool_choice(tc: dict[str, Any] | None) -> CanonicalToolChoice | None:
    if not tc:
        return None
    t = tc.get("type")
    if t == "auto":
        return CanonicalToolChoice(mode="auto")
    if t == "any":
        return CanonicalToolChoice(mode="required")
    if t == "none":
        return CanonicalToolChoice(mode="none")
    if t == "tool":
        return CanonicalToolChoice(mode="tool", name=tc.get("name") or "")
    return None


class AnthropicInbound:
    """Stateless. All methods are pure."""

    protocol_name = ANTHROPIC_PROTOCOL

    @staticmethod
    def decode_request(req: MessagesRequest) -> CanonicalRequest:
        canonical_messages: list[CanonicalMessage] = []
        for m in req.messages:
            blocks = _decode_message_content(m.content)
            canonical_messages.append(CanonicalMessage(role=m.role, content_blocks=blocks))

        metadata: dict[str, Any] = {}
        if req.metadata:
            metadata["anthropic_metadata"] = req.metadata
        if req.thinking is not None:
            # Preserved so downstream observability can see it; provider adapters that
            # don't support thinking ignore it.
            metadata["anthropic_thinking"] = req.thinking
        if req.top_k is not None:
            metadata["top_k"] = req.top_k

        return CanonicalRequest(
            model=req.model,
            messages=canonical_messages,
            system=_decode_system(req.system),
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stream=req.stream,
            tools=_decode_tools(req.tools),
            tool_choice=_decode_tool_choice(req.tool_choice),
            stop_sequences=req.stop_sequences or [],
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # encode: CanonicalResponse -> Anthropic body
    # ------------------------------------------------------------------

    @staticmethod
    def encode_response(canonical: CanonicalResponse) -> dict[str, Any]:
        anth_blocks: list[dict[str, Any]] = []
        for b in canonical.content_blocks:
            if isinstance(b, CanonicalText):
                anth_blocks.append({"type": "text", "text": b.text})
            elif isinstance(b, CanonicalToolCall):
                anth_blocks.append(
                    {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
                )
            elif isinstance(b, CanonicalImage):
                # Assistants don't typically emit images today; encode for completeness.
                src: dict[str, Any]
                if b.url:
                    src = {"type": "url", "media_type": b.media_type, "url": b.url}
                else:
                    src = {"type": "base64", "media_type": b.media_type, "data": b.data or ""}
                anth_blocks.append({"type": "image", "source": src})
            elif isinstance(b, CanonicalToolResult):
                # Tool results from an assistant are non-sensical; drop.
                continue

        # Anthropic clients tolerate empty content list, but the SDK's `content[0]` access
        # crashes — emit an empty text block for safety.
        if not anth_blocks:
            anth_blocks.append({"type": "text", "text": ""})

        response_id = canonical.id
        if not response_id.startswith("msg_"):
            response_id = f"msg_{response_id}"

        return {
            "id": response_id,
            "type": "message",
            "role": "assistant",
            "model": canonical.model,
            "content": anth_blocks,
            "stop_reason": canonical.stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": canonical.usage.input_tokens,
                "output_tokens": canonical.usage.output_tokens,
            },
        }

    # ------------------------------------------------------------------
    # encode: canonical event stream -> Anthropic SSE byte stream
    # ------------------------------------------------------------------

    @staticmethod
    async def encode_stream(
        events: AsyncIterator,  # AsyncIterator[CanonicalStreamEvent]
    ) -> AsyncIterator[bytes]:
        """Map canonical events to Anthropic SSE.

        Anthropic event sequence we must emit:
            message_start
              (per content block:)
                content_block_start
                content_block_delta (text_delta or input_json_delta) *
                content_block_stop
            message_delta (stop_reason + final usage)
            message_stop
        """

        started = False
        message_id = ""
        model = ""
        async for ev in events:
            if isinstance(ev, StreamStart):
                message_id = ev.message_id if ev.message_id.startswith("msg_") else f"msg_{ev.message_id}"
                model = ev.model
                started = True
                yield _sse(
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": message_id,
                            "type": "message",
                            "role": "assistant",
                            "model": model,
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {
                                "input_tokens": ev.usage_estimate.input_tokens,
                                "output_tokens": ev.usage_estimate.output_tokens,
                            },
                        },
                    },
                )
            elif isinstance(ev, ContentBlockStart):
                yield _sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": ev.index,
                        "content_block": _encode_block(ev.block),
                    },
                )
            elif isinstance(ev, ContentTextDelta):
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": ev.index,
                        "delta": {"type": "text_delta", "text": ev.text},
                    },
                )
            elif isinstance(ev, ContentToolCallDelta):
                if ev.partial_input_json:
                    yield _sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": ev.index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": ev.partial_input_json,
                            },
                        },
                    )
            elif isinstance(ev, ContentBlockDone):
                yield _sse(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": ev.index},
                )
            elif isinstance(ev, MessageDelta):
                yield _sse(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": ev.stop_reason or "end_turn",
                            "stop_sequence": None,
                        },
                        "usage": {"output_tokens": ev.usage.output_tokens},
                    },
                )
            elif isinstance(ev, StreamError):
                yield _sse(
                    "error",
                    {"type": "error", "error": {"type": "provider_error", "message": ev.message}},
                )
            elif isinstance(ev, StreamDone):
                yield _sse("message_stop", {"type": "message_stop"})

        if not started:
            # Upstream produced nothing — synthesise a minimum valid stream so the
            # consumer's SSE parser terminates cleanly.
            message_id = f"msg_{uuid.uuid4().hex[:24]}"
            yield _sse(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "model": model or "unknown",
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                },
            )
            yield _sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 0},
                },
            )
            yield _sse("message_stop", {"type": "message_stop"})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


def _encode_block(block: CanonicalContent) -> dict[str, Any]:
    if isinstance(block, CanonicalText):
        return {"type": "text", "text": block.text}
    if isinstance(block, CanonicalToolCall):
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    if isinstance(block, CanonicalImage):
        if block.url:
            return {
                "type": "image",
                "source": {"type": "url", "media_type": block.media_type, "url": block.url},
            }
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": block.media_type, "data": block.data or ""},
        }
    if isinstance(block, CanonicalToolResult):
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_call_id,
            "content": block.content,
            "is_error": block.is_error,
        }
    return {"type": "text", "text": ""}
