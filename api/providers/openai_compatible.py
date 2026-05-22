"""OpenAICompatibleProvider — base for any provider that speaks the OpenAI
``/v1/chat/completions`` wire format.

This module owns ALL of the Canonical <-> OpenAI translation: request
encoding, non-streaming response decoding, SSE stream decoding, tool-use
round-tripping, and usage extraction. Subclasses specialize ONLY:

  - ``name``                       (e.g. ``"xai"``, ``"openai"``)
  - ``model_name_prefixes``        (model ids this provider claims, e.g. ``("grok",)``)
  - ``twin_override_env_var``      (e.g. ``"GROK_BASE_URL"``, ``"OPENAI_BASE_URL"``)
  - ``__init__`` defaults from the right settings

The wire format these providers share — xAI Grok, OpenAI's own GPT / o-series,
plus any compatible third-party (Groq, Fireworks, Together, OpenRouter, ...) —
is OpenAI's chat-completions schema. Reasoning models (OpenAI o-series, GPT-5)
add a ``completion_tokens_details.reasoning_tokens`` field inside ``usage``;
those tokens are ALREADY counted in ``completion_tokens``, so canonical
``output_tokens`` (which we set from ``completion_tokens``) correctly includes
them. We do not currently surface reasoning_tokens separately in
``CanonicalUsage``; if a future caller needs it, it lives in the recorded
``provider_response`` JSON blob.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, ClassVar

import httpx

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
    CanonicalImage,
    CanonicalRequest,
    CanonicalResponse,
    CanonicalText,
    CanonicalToolCall,
    CanonicalToolResult,
    CanonicalUsage,
)
from api.providers.base import ProviderError
from api.twin import get_override

# ---------------------------------------------------------------------------
# OpenAI finish_reason <-> canonical stop_reason
# ---------------------------------------------------------------------------

_FINISH_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "stop_sequence",
}


def _map_finish_reason(fr: str | None) -> str | None:
    if fr is None:
        return None
    return _FINISH_REASON_MAP.get(fr, "end_turn")


# ---------------------------------------------------------------------------
# Canonical -> OpenAI body
# ---------------------------------------------------------------------------


def _encode_user_blocks(blocks) -> list[dict[str, Any]]:
    """A user-role canonical message may interleave Text/Image and ToolResult blocks.
    ToolResults become standalone OpenAI ``role:"tool"`` messages; the rest become a
    single ``role:"user"`` message, preserving original order."""

    out: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    def flush() -> None:
        if not pending:
            return
        if len(pending) == 1 and pending[0].get("type") == "text":
            out.append({"role": "user", "content": pending[0]["text"]})
        else:
            out.append({"role": "user", "content": list(pending)})
        pending.clear()

    for b in blocks:
        if isinstance(b, CanonicalText):
            pending.append({"type": "text", "text": b.text})
        elif isinstance(b, CanonicalImage):
            url = b.url or (f"data:{b.media_type};base64,{b.data}" if b.data else None)
            if url:
                pending.append({"type": "image_url", "image_url": {"url": url}})
        elif isinstance(b, CanonicalToolResult):
            flush()
            out.append({"role": "tool", "tool_call_id": b.tool_call_id, "content": b.content})
        elif isinstance(b, CanonicalToolCall):
            # A tool_call in a user message is non-sensical; drop quietly.
            continue
    flush()
    return out


def _encode_assistant_message(blocks) -> dict[str, Any]:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for b in blocks:
        if isinstance(b, CanonicalText):
            text_parts.append(b.text)
        elif isinstance(b, CanonicalToolCall):
            tool_calls.append(
                {
                    "id": b.id,
                    "type": "function",
                    "function": {
                        "name": b.name,
                        "arguments": json.dumps(b.input or {}, ensure_ascii=False),
                    },
                }
            )
    msg: dict[str, Any] = {"role": "assistant"}
    text = "".join(text_parts)
    msg["content"] = text if text else None
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _encode_tool_choice(tc) -> str | dict[str, Any] | None:
    if tc is None:
        return None
    if tc.mode == "auto":
        return "auto"
    if tc.mode == "required":
        return "required"
    if tc.mode == "none":
        return "none"
    if tc.mode == "tool":
        return {"type": "function", "function": {"name": tc.name or ""}}
    return None


def compute_prompt_cache_key(canonical: CanonicalRequest) -> str:
    """Stable per-(system+tools) hash, suitable as OpenAI's ``prompt_cache_key``.

    OpenAI auto-caches request prefixes >=1024 tokens, but routing to a
    consistent cache shard is best-effort without an explicit hint. Setting
    ``prompt_cache_key`` to a stable value tied to the static portion of the
    request (system messages + tool definitions) tells OpenAI to route same-
    key requests to the same shard, dramatically improving hit rate across a
    multi-turn tool-use loop where the system + tools never change but the
    message history does.

    The key MUST be deterministic across calls that share the same prefix.
    We hash the JSON-serialized (system, tools) tuple — message history is
    intentionally excluded so an Nth-turn continuation hits the cache built
    by turn 1. Returned value is a 32-char hex digest (well under OpenAI's
    64-char cap). The hash carries no semantic — it's a routing token only;
    nothing inside the gateway parses it back.
    """
    payload = {
        "system": [b.model_dump(exclude_none=True) for b in (canonical.system or [])],
        "tools": [t.model_dump(exclude_none=True) for t in canonical.tools],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:32]


def encode_request(
    canonical: CanonicalRequest,
    *,
    target_model: str,
    max_tokens_param_name: str = "max_tokens",
    cache_key: str | None = None,
) -> dict[str, Any]:
    """Canonical -> OpenAI chat-completions request body.

    ``max_tokens_param_name`` lets a subclass pick the correct field name for
    its provider. xAI (and older OpenAI models) accept ``max_tokens``; OpenAI's
    GPT-5 / o-series / o4 deprecate it in favor of ``max_completion_tokens``
    and 400 the request if the old name is used. Subclasses pass the right
    value via their ``OpenAICompatibleProvider.max_tokens_param_name`` ClassVar.

    ``cache_key``, when non-None, is emitted as the OpenAI-only
    ``prompt_cache_key`` parameter. Subclasses that don't speak real OpenAI
    (i.e. xAI/Grok) should pass ``None`` — the parameter is silently
    accepted by some compatible-but-not-actual-OpenAI backends but does
    nothing there. See :func:`compute_prompt_cache_key` for key construction.
    """
    messages: list[dict[str, Any]] = []
    if canonical.system:
        system_text = "\n".join(
            b.text for b in canonical.system if isinstance(b, CanonicalText) and b.text
        )
        if system_text:
            messages.append({"role": "system", "content": system_text})

    for m in canonical.messages:
        if m.role == "user":
            messages.extend(_encode_user_blocks(m.content_blocks))
        elif m.role == "assistant":
            messages.append(_encode_assistant_message(m.content_blocks))
        else:  # "tool" — canonical doesn't normally emit this, but pass through
            text = "\n".join(
                b.content for b in m.content_blocks if isinstance(b, CanonicalToolResult)
            )
            if text:
                messages.append({"role": "tool", "content": text})

    body: dict[str, Any] = {
        "model": target_model,
        "messages": messages,
        "stream": canonical.stream,
        max_tokens_param_name: canonical.max_tokens,
    }
    if cache_key:
        body["prompt_cache_key"] = cache_key
    if canonical.temperature is not None:
        body["temperature"] = canonical.temperature
    if canonical.top_p is not None:
        body["top_p"] = canonical.top_p
    if canonical.stop_sequences:
        body["stop"] = list(canonical.stop_sequences)

    if canonical.tools:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.json_schema or {"type": "object", "properties": {}},
                },
            }
            for t in canonical.tools
        ]
    tc = _encode_tool_choice(canonical.tool_choice)
    if tc is not None:
        body["tool_choice"] = tc

    if canonical.stream:
        body["stream_options"] = {"include_usage": True}

    return body


# ---------------------------------------------------------------------------
# OpenAI body -> CanonicalResponse
# ---------------------------------------------------------------------------


def _safe_parse_arguments(args: Any) -> dict[str, Any]:
    if args is None:
        return {}
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        if not args.strip():
            return {}
        try:
            parsed = json.loads(args)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            return {"_unparsed_arguments": args}
    return {"value": args}


def decode_response(provider_body: dict[str, Any], *, canonical_model: str) -> CanonicalResponse:
    """OpenAI chat-completions response body -> CanonicalResponse."""
    choices = provider_body.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    finish_reason = choice.get("finish_reason")

    blocks: list = []
    text = message.get("content")
    if isinstance(text, str) and text:
        blocks.append(CanonicalText(text=text))
    elif isinstance(text, list):
        for part in text:
            if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                blocks.append(CanonicalText(text=part["text"]))

    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        blocks.append(
            CanonicalToolCall(
                id=call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                name=fn.get("name") or "",
                input=_safe_parse_arguments(fn.get("arguments")),
            )
        )

    usage_in = provider_body.get("usage") or {}
    # OpenAI's o-series / GPT-5 report reasoning_tokens inside
    # completion_tokens_details — those tokens are ALREADY counted in
    # completion_tokens, so output_tokens below correctly includes them.
    response_id = provider_body.get("id") or f"msg_{uuid.uuid4().hex[:24]}"

    return CanonicalResponse(
        id=response_id,
        model=canonical_model,
        content_blocks=blocks,
        stop_reason=_map_finish_reason(finish_reason),
        usage=_canonical_usage_from_openai(usage_in),
    )


def _canonical_usage_from_openai(usage_in: dict[str, Any]) -> CanonicalUsage:
    """Map OpenAI's ``usage`` block to ``CanonicalUsage``.

    OpenAI's ``prompt_tokens`` is the TOTAL input including any cached portion;
    ``prompt_tokens_details.cached_tokens`` is the cached subset (priced 10x
    cheaper). Canonical convention is "input_tokens excludes cached" (matches
    Anthropic, matches what downstream cost computations expect). So we split:

        canonical.input_tokens = prompt_tokens - cached_tokens   (uncached only)
        canonical.cache_tokens = cached_tokens                   (cached subset)

    cache_tokens stays ``None`` when no cached_tokens field is present — that
    way xAI / older OpenAI responses round-trip unchanged.
    """
    total_input = int(usage_in.get("prompt_tokens") or 0)
    output = int(usage_in.get("completion_tokens") or 0)
    details = usage_in.get("prompt_tokens_details") or {}
    cached_raw = details.get("cached_tokens") if isinstance(details, dict) else None
    if cached_raw in (None, 0):
        return CanonicalUsage(input_tokens=total_input, output_tokens=output)
    cached = int(cached_raw)
    uncached = max(total_input - cached, 0)
    return CanonicalUsage(input_tokens=uncached, output_tokens=output, cache_tokens=cached)


# ---------------------------------------------------------------------------
# Streaming: OpenAI SSE bytes -> Canonical events
# ---------------------------------------------------------------------------


@dataclass
class _ToolCallState:
    canonical_index: int
    id: str
    name: str
    started: bool = False
    buffered_args: str = ""


@dataclass
class _StreamState:
    message_id: str
    model: str
    text_index: int | None = None
    text_started: bool = False
    text_closed: bool = False
    next_index: int = 0
    tool_calls: dict[int, _ToolCallState] = field(default_factory=dict)
    stop_reason: str | None = None
    output_tokens: int = 0
    input_tokens: int = 0
    # OpenAI's terminal usage frame carries ``prompt_tokens_details.cached_tokens``
    # when the request hit the prompt cache. Tracked here so MessageDelta carries
    # the split-out cache_tokens through to the protocol encoders.
    cache_tokens: int = 0
    started_emitted: bool = False


async def _iter_sse_data(byte_iter: AsyncIterator[bytes]) -> AsyncIterator[str]:
    buf = b""
    async for chunk in byte_iter:
        buf += chunk
        while True:
            sep = buf.find(b"\n\n")
            sep_len = 2
            if sep == -1:
                sep = buf.find(b"\r\n\r\n")
                sep_len = 4
                if sep == -1:
                    break
            event_bytes = buf[:sep]
            buf = buf[sep + sep_len :]
            data_parts: list[str] = []
            for raw in event_bytes.split(b"\n"):
                line = raw.rstrip(b"\r")
                if line.startswith(b"data:"):
                    data_parts.append(line[5:].lstrip(b" ").decode("utf-8", errors="replace"))
            if data_parts:
                yield "\n".join(data_parts)


async def decode_stream(
    raw_chunks: AsyncIterator[bytes], *, canonical_model: str, input_tokens_estimate: int = 0
):
    """Yield ``CanonicalStreamEvent``s from an OpenAI-compatible SSE byte stream."""

    state = _StreamState(
        message_id=f"msg_{uuid.uuid4().hex[:24]}",
        model=canonical_model,
        input_tokens=input_tokens_estimate,
    )

    async for data_str in _iter_sse_data(raw_chunks):
        if data_str == "[DONE]":
            break
        try:
            payload = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        if not state.started_emitted:
            upstream_id = payload.get("id")
            if upstream_id:
                state.message_id = upstream_id
            state.started_emitted = True
            yield StreamStart(
                message_id=state.message_id,
                model=state.model,
                usage_estimate=CanonicalUsage(
                    input_tokens=state.input_tokens, output_tokens=0
                ),
            )

        usage = payload.get("usage")
        if usage:
            total_input = int(usage.get("prompt_tokens") or 0) or state.input_tokens
            state.output_tokens = int(usage.get("completion_tokens") or state.output_tokens)
            details = usage.get("prompt_tokens_details") or {}
            cached_raw = details.get("cached_tokens") if isinstance(details, dict) else None
            cached = int(cached_raw) if cached_raw else 0
            # Same split as the non-streaming path: ``input_tokens`` becomes the
            # uncached portion only; cached tokens move into their own bucket.
            state.cache_tokens = cached
            state.input_tokens = max(total_input - cached, 0)

        for choice in payload.get("choices") or []:
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if isinstance(content, str) and content:
                for ev in _handle_text_delta(state, content):
                    yield ev
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        for ev in _handle_text_delta(state, part.get("text") or ""):
                            yield ev
            for tc_chunk in delta.get("tool_calls") or []:
                if isinstance(tc_chunk, dict):
                    for ev in _handle_tool_call_delta(state, tc_chunk):
                        yield ev
            fr = choice.get("finish_reason")
            if fr:
                state.stop_reason = _map_finish_reason(fr) or state.stop_reason

    # finalize
    if not state.started_emitted:
        state.started_emitted = True
        yield StreamStart(
            message_id=state.message_id,
            model=state.model,
            usage_estimate=CanonicalUsage(input_tokens=state.input_tokens, output_tokens=0),
        )
    if state.text_started and not state.text_closed and state.text_index is not None:
        state.text_closed = True
        yield ContentBlockDone(index=state.text_index)
    for st in state.tool_calls.values():
        if st.started:
            yield ContentBlockDone(index=st.canonical_index)
    yield MessageDelta(
        stop_reason=state.stop_reason or "end_turn",
        usage=CanonicalUsage(
            input_tokens=state.input_tokens,
            output_tokens=state.output_tokens,
            cache_tokens=state.cache_tokens or None,
        ),
    )
    yield StreamDone()


def _handle_text_delta(state: _StreamState, text: str):
    if not text:
        return
    if not state.text_started:
        state.text_index = state.next_index
        state.next_index += 1
        state.text_started = True
        yield ContentBlockStart(index=state.text_index, block=CanonicalText(text=""))
    yield ContentTextDelta(index=state.text_index, text=text)


def _handle_tool_call_delta(state: _StreamState, tc_chunk: dict):
    oai_idx = tc_chunk.get("index")
    if oai_idx is None:
        oai_idx = 0
    fn = tc_chunk.get("function") or {}
    name = fn.get("name")
    args_fragment = fn.get("arguments")
    call_id = tc_chunk.get("id")

    st = state.tool_calls.get(oai_idx)
    if st is None:
        if not name and not call_id and not args_fragment:
            return
        st = _ToolCallState(
            canonical_index=state.next_index,
            id=call_id or f"toolu_{uuid.uuid4().hex[:24]}",
            name=name or "",
        )
        state.tool_calls[oai_idx] = st
        state.next_index += 1
    else:
        if call_id:
            st.id = call_id
        if name and not st.name:
            st.name = name

    if not st.started:
        if not st.name:
            if args_fragment:
                st.buffered_args += args_fragment
            return
        if state.text_started and not state.text_closed and state.text_index is not None:
            state.text_closed = True
            yield ContentBlockDone(index=state.text_index)
        st.started = True
        yield ContentBlockStart(
            index=st.canonical_index,
            block=CanonicalToolCall(id=st.id, name=st.name, input={}),
        )
        if st.buffered_args:
            yield ContentToolCallDelta(
                index=st.canonical_index,
                id=st.id,
                name=st.name,
                partial_input_json=st.buffered_args,
            )
            st.buffered_args = ""

    if args_fragment:
        yield ContentToolCallDelta(
            index=st.canonical_index,
            id=st.id,
            name=st.name,
            partial_input_json=args_fragment,
        )


# ---------------------------------------------------------------------------
# Base adapter class — encode/decode + HTTP transport
# ---------------------------------------------------------------------------


class OpenAICompatibleProvider:
    """Base class for providers that speak ``POST /v1/chat/completions`` with
    ``Authorization: Bearer <api_key>``.

    Subclasses override the four ClassVars (``name``, ``model_name_prefixes``,
    ``twin_override_env_var``) and the ``__init__`` defaults. Everything else —
    canonical encode/decode, streaming, HTTP transport — is reused.
    """

    name: ClassVar[str] = ""
    # canonical-side model ids this provider claims (lowercase prefix match)
    model_name_prefixes: ClassVar[tuple[str, ...]] = ()
    # env-var name the X-Twin-* middleware overrides for this provider's base URL
    twin_override_env_var: ClassVar[str] = ""
    # Some providers renamed ``max_tokens`` for newer / reasoning models.
    # Default is the historical OpenAI spelling; the OpenAI subclass overrides
    # to ``max_completion_tokens`` because GPT-5 / o-series 400 on the old name.
    max_tokens_param_name: ClassVar[str] = "max_tokens"
    # Real OpenAI exposes a ``prompt_cache_key`` request parameter that hints
    # the cache layer to route same-key requests to the same shard (10x cheaper
    # input on hits). xAI/Grok don't have it. Default off; the OpenAI subclass
    # flips this on. When True, ``encode_request`` computes the key from the
    # canonical request's (system, tools) static portion — same shape, same
    # key, cache hit on every continuation.
    supports_prompt_cache_key: ClassVar[bool] = False

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        default_model: str,
        timeout_seconds: float,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._configured_base_url = base_url
        self._default_model = default_model
        self._timeout = timeout_seconds
        self._client = client  # injectable for tests

    # ------- routing helpers

    def claims_model(self, canonical_model: str) -> bool:
        """True iff the given canonical model id is one this provider handles."""
        if not canonical_model:
            return False
        lo = canonical_model.lower()
        return any(lo.startswith(p) for p in self.model_name_prefixes)

    @property
    def default_model(self) -> str:
        return self._default_model

    # ------- canonical boundary

    def encode_request(self, canonical: CanonicalRequest) -> dict[str, Any]:
        cache_key = (
            compute_prompt_cache_key(canonical) if self.supports_prompt_cache_key else None
        )
        return encode_request(
            canonical,
            target_model=self.map_model(canonical.model),
            max_tokens_param_name=self.max_tokens_param_name,
            cache_key=cache_key,
        )

    def decode_response(
        self, provider_body: dict[str, Any], *, canonical_request_model: str
    ) -> CanonicalResponse:
        return decode_response(provider_body, canonical_model=canonical_request_model)

    def decode_stream(
        self, raw_chunks: AsyncIterator[bytes], *, canonical_request_model: str
    ):
        return decode_stream(raw_chunks, canonical_model=canonical_request_model)

    def map_model(self, canonical_model: str) -> str:
        """If this provider claims the model, pass it through; otherwise fall back
        to ``default_model``. Routing should have selected the right provider
        already, so the fallback is mainly a safety net for unusual inbound ids.
        """
        if not canonical_model:
            return self._default_model
        return canonical_model if self.claims_model(canonical_model) else self._default_model

    # ------- HTTP transport

    def _resolved_base_url(self) -> str:
        override = (
            get_override(self.twin_override_env_var) if self.twin_override_env_var else None
        )
        return (override or self._configured_base_url).rstrip("/")

    def _headers(self) -> dict[str, str]:
        if not self._api_key:
            raise ProviderError(
                status_code=500,
                message=f"{self.name.upper()}_API_KEY not configured on the gateway.",
            )
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _stream_headers(self) -> dict[str, str]:
        h = self._headers()
        h["Accept"] = "text/event-stream"
        return h

    def _new_client(self) -> httpx.AsyncClient:
        return self._client or httpx.AsyncClient(timeout=self._timeout)

    async def send(self, provider_body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._resolved_base_url()}/chat/completions"
        client = self._new_client()
        owns_client = self._client is None
        try:
            try:
                resp = await client.post(url, json=provider_body, headers=self._headers())
            except httpx.RequestError as exc:
                raise ProviderError(
                    status_code=502, message=f"Upstream provider request failed: {exc}"
                ) from exc
            if resp.status_code >= 400:
                raise ProviderError(
                    status_code=resp.status_code,
                    message=f"Provider returned {resp.status_code}",
                    provider_body=resp.text,
                )
            return resp.json()
        finally:
            if owns_client:
                await client.aclose()

    async def send_stream(self, provider_body: dict[str, Any]) -> AsyncIterator[bytes]:
        url = f"{self._resolved_base_url()}/chat/completions"
        client = self._new_client()
        owns_client = self._client is None
        body = {**provider_body, "stream": True}
        try:
            try:
                async with client.stream(
                    "POST", url, json=body, headers=self._stream_headers()
                ) as resp:
                    if resp.status_code >= 400:
                        text = (await resp.aread()).decode("utf-8", errors="replace")
                        raise ProviderError(
                            status_code=resp.status_code,
                            message=f"Provider returned {resp.status_code} on stream open",
                            provider_body=text,
                        )
                    async for chunk in resp.aiter_bytes():
                        if chunk:
                            yield chunk
            except httpx.RequestError as exc:
                raise ProviderError(
                    status_code=502, message=f"Upstream provider request failed: {exc}"
                ) from exc
        finally:
            if owns_client:
                await client.aclose()


def _wrap_stream_error(exc: ProviderError):
    return StreamError(message=exc.message)
