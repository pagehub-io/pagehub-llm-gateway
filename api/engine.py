"""Engine — orchestrates one request through the canonical pipeline + recording.

Pipeline:

    inbound HTTP (Anthropic-shaped) ─decode──► CanonicalRequest
                                                    │
                                                    ▼
                       ProviderAdapter.encode_request  ──► provider body
                                                    │
                                                    ▼
                              HTTPS to provider (xAI / ...)
                                                    │
                                                    ▼
                       ProviderAdapter.decode_response ──► CanonicalResponse
                                                    │
                                                    ▼
                      AnthropicInbound.encode_response ──► Anthropic-shaped body
                                                    │
                                                    ▼
                                          outbound HTTP

The same diagram for streaming substitutes ``decode_stream`` (provider chunks
-> Canonical events) and ``encode_stream`` (Canonical events -> Anthropic SSE)
for the response halves. The recorder receives an event at every step so the
DB has a replay-exact view of what happened.
"""

from __future__ import annotations

import json
import logging
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
from api.canonical.types import CanonicalRequest, CanonicalResponse
from api.protocols.anthropic import ANTHROPIC_PROTOCOL, AnthropicInbound
from api.providers.base import ProviderAdapter, ProviderError
from api.providers.registry import ProviderRegistry
from api.shared.recorder import Recorder

logger = logging.getLogger(__name__)


class _SingleProviderRegistry:
    """Test-shim: behaves like ``ProviderRegistry`` but always returns the same
    provider regardless of model id. Used when the engine is constructed with
    the legacy ``provider=`` kwarg (most existing tests). The shim's ``pick``
    contract matches :class:`ProviderRegistry.pick`; production wiring uses the
    real registry so unknown models fall through to a 400.
    """

    def __init__(self, provider: ProviderAdapter) -> None:
        self._provider = provider

    @property
    def providers(self) -> list[ProviderAdapter]:
        return [self._provider]

    def pick(self, canonical_model: str) -> ProviderAdapter:  # noqa: ARG002
        return self._provider


class Engine:
    def __init__(
        self,
        *,
        provider: ProviderAdapter | None = None,
        registry: ProviderRegistry | _SingleProviderRegistry | None = None,
        recorder: Recorder,
    ) -> None:
        """One of ``provider=`` (single backend) or ``registry=`` (model-routed)
        is required.

        ``provider=`` is the original v1 API and stays supported for tests and
        anyone embedding the engine with a fixed backend. ``registry=`` is
        used in production wiring so the engine picks the right adapter per
        request (xAI for ``grok-*``, OpenAI for ``gpt-*``/``o*``, ...).
        """
        if registry is None and provider is None:
            raise ValueError("Engine requires either provider= or registry=")
        if registry is not None and provider is not None:
            raise ValueError("Engine: pass provider= OR registry=, not both")
        self._registry: ProviderRegistry | _SingleProviderRegistry = (
            registry if registry is not None else _SingleProviderRegistry(provider)
        )
        # Back-compat alias: legacy tests/code may inspect ``.provider``. When
        # constructed with a registry, no single provider is meaningful, so the
        # alias is ``None`` and callers must use ``self.provider_for(model)``.
        self.provider: ProviderAdapter | None = provider
        self.recorder = recorder

    def provider_for(self, canonical_model: str) -> ProviderAdapter:
        """Routes a canonical model id to its provider. Raises
        :class:`ProviderError(status_code=400)` if no provider claims it."""
        return self._registry.pick(canonical_model)

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    async def run_anthropic(
        self,
        *,
        client_request_id: str | None,
        raw_inbound_body: dict[str, Any],
        canonical_request: CanonicalRequest,
    ) -> tuple[dict[str, Any], uuid.UUID]:
        """Run a non-streaming Anthropic-protocol request end-to-end.

        Returns ``(anthropic_response_dict, request_id)``. ``request_id`` is useful
        for caller-side correlation with the recorded events.
        """

        provider = self.provider_for(canonical_request.model)
        provider_model = provider.map_model(canonical_request.model)
        rid = await self.recorder.start_request(
            client_request_id=client_request_id,
            inbound_protocol=ANTHROPIC_PROTOCOL,
            inbound_body=raw_inbound_body,
            canonical_request=canonical_request.model_dump(exclude_none=True),
            provider=provider.name,
            provider_model_requested=provider_model,
        )
        await self.recorder.add_event(rid, "inbound_received", raw_inbound_body)
        await self.recorder.add_event(rid, "decoded_canonical", canonical_request.model_dump(exclude_none=True))

        provider_body = provider.encode_request(canonical_request)
        await self.recorder.add_event(rid, "provider_request", provider_body)

        try:
            provider_response = await provider.send(provider_body)
        except ProviderError as exc:
            await self.recorder.add_event(
                rid,
                "error",
                {"status": exc.status_code, "message": exc.message, "body": exc.provider_body},
            )
            await self.recorder.finalize_request(
                rid,
                outbound_body=provider_body,
                error=exc.message,
            )
            raise

        await self.recorder.add_event(rid, "provider_response", provider_response)

        canonical_response = provider.decode_response(
            provider_response, canonical_request_model=canonical_request.model
        )
        await self.recorder.add_event(
            rid, "decoded_canonical_response", canonical_response.model_dump(exclude_none=True)
        )

        anth_body = AnthropicInbound.encode_response(canonical_response)
        await self.recorder.add_event(rid, "encoded_outbound", anth_body)

        await self.recorder.finalize_request(
            rid,
            outbound_body=provider_body,
            provider_response=provider_response,
            canonical_response=canonical_response.model_dump(exclude_none=True),
            outbound_body_returned=anth_body,
            input_tokens=canonical_response.usage.input_tokens,
            output_tokens=canonical_response.usage.output_tokens,
            stop_reason=canonical_response.stop_reason,
        )
        return anth_body, rid

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def stream_anthropic(
        self,
        *,
        client_request_id: str | None,
        raw_inbound_body: dict[str, Any],
        canonical_request: CanonicalRequest,
        input_tokens_estimate: int = 0,
    ) -> AsyncIterator[bytes]:
        """Run a streaming Anthropic-protocol request end-to-end, yielding Anthropic SSE bytes."""

        provider = self.provider_for(canonical_request.model)
        provider_model = provider.map_model(canonical_request.model)
        rid = await self.recorder.start_request(
            client_request_id=client_request_id,
            inbound_protocol=ANTHROPIC_PROTOCOL,
            inbound_body=raw_inbound_body,
            canonical_request=canonical_request.model_dump(exclude_none=True),
            provider=provider.name,
            provider_model_requested=provider_model,
        )
        await self.recorder.add_event(rid, "inbound_received", raw_inbound_body)
        await self.recorder.add_event(rid, "decoded_canonical", canonical_request.model_dump(exclude_none=True))

        provider_body = provider.encode_request(canonical_request)
        await self.recorder.add_event(rid, "provider_request", provider_body)

        recorder = self.recorder
        canonical_model = canonical_request.model

        # We need to (a) yield the Anthropic SSE bytes downstream, (b) record each
        # canonical event in order, (c) assemble a "final" canonical response object
        # for the requests row. Use a single generator that does all three.
        final_response = {
            "id": "",
            "model": canonical_model,
            "content_blocks": [],
            "stop_reason": None,
            "input_tokens": input_tokens_estimate,
            "output_tokens": 0,
        }

        async def canonical_events():
            """Decode provider stream -> canonical events, recording each. Errors map
            to a StreamError event so AnthropicInbound.encode_stream produces a clean
            terminator."""
            try:
                provider_stream = provider.send_stream(provider_body)
                async for ev in provider.decode_stream(
                    provider_stream, canonical_request_model=canonical_model
                ):
                    await recorder.add_event(rid, "canonical_event", ev.model_dump(exclude_none=True))
                    _accumulate_final(final_response, ev)
                    yield ev
            except ProviderError as exc:
                await recorder.add_event(
                    rid,
                    "error",
                    {"status": exc.status_code, "message": exc.message, "body": exc.provider_body},
                )
                yield StreamError(message=exc.message)

        # We need to write out the bytes AND keep track of them for the final
        # outbound_body_returned. Accumulate as we go.
        emitted_bytes: list[bytes] = []
        try:
            async for raw in AnthropicInbound.encode_stream(canonical_events()):
                emitted_bytes.append(raw)
                yield raw
        finally:
            final_canonical = CanonicalResponse(
                id=final_response["id"] or f"msg_{uuid.uuid4().hex[:24]}",
                model=final_response["model"],
                content_blocks=final_response["content_blocks"],
                stop_reason=final_response["stop_reason"],
                usage=_build_usage(final_response),
            )
            await self.recorder.add_event(
                rid, "decoded_canonical_response", final_canonical.model_dump(exclude_none=True)
            )
            await self.recorder.add_event(
                rid,
                "outbound_sent",
                {"sse_bytes": sum(len(b) for b in emitted_bytes)},
            )
            await self.recorder.finalize_request(
                rid,
                outbound_body=provider_body,
                provider_response={"streamed": True},
                canonical_response=final_canonical.model_dump(exclude_none=True),
                outbound_body_returned={"sse_event_count": len(emitted_bytes)},
                input_tokens=final_canonical.usage.input_tokens,
                output_tokens=final_canonical.usage.output_tokens,
                stop_reason=final_canonical.stop_reason,
            )


# ---------------------------------------------------------------------------
# Helpers for streaming accumulation
# ---------------------------------------------------------------------------


def _build_usage(final_response: dict[str, Any]):
    from api.canonical.types import CanonicalUsage

    return CanonicalUsage(
        input_tokens=int(final_response.get("input_tokens") or 0),
        output_tokens=int(final_response.get("output_tokens") or 0),
    )


def _accumulate_final(final: dict[str, Any], ev) -> None:
    """Assemble the canonical "final" content as we observe each event."""
    if isinstance(ev, StreamStart):
        final["id"] = ev.message_id
        final["model"] = ev.model
        final["input_tokens"] = max(final.get("input_tokens") or 0, ev.usage_estimate.input_tokens)
        return
    if isinstance(ev, ContentBlockStart):
        # Append a placeholder mirroring the start block.
        block_dict = ev.block.model_dump()
        final["content_blocks"].append(block_dict)
        return
    if isinstance(ev, ContentTextDelta):
        if 0 <= ev.index < len(final["content_blocks"]):
            block = final["content_blocks"][ev.index]
            if block.get("type") == "text":
                block["text"] = (block.get("text") or "") + ev.text
        return
    if isinstance(ev, ContentToolCallDelta):
        if 0 <= ev.index < len(final["content_blocks"]):
            block = final["content_blocks"][ev.index]
            if block.get("type") == "tool_call":
                if ev.id and not block.get("id"):
                    block["id"] = ev.id
                if ev.name and not block.get("name"):
                    block["name"] = ev.name
                buf = block.setdefault("_partial_json", "")
                if ev.partial_input_json:
                    block["_partial_json"] = buf + ev.partial_input_json
        return
    if isinstance(ev, ContentBlockDone):
        # If a tool_call block accumulated partial_json, parse it into ``input``.
        if 0 <= ev.index < len(final["content_blocks"]):
            block = final["content_blocks"][ev.index]
            if block.get("type") == "tool_call" and "_partial_json" in block:
                raw = block.pop("_partial_json")
                try:
                    parsed = json.loads(raw) if raw else {}
                    block["input"] = parsed if isinstance(parsed, dict) else {"value": parsed}
                except json.JSONDecodeError:
                    block["input"] = {"_unparsed_arguments": raw}
        return
    if isinstance(ev, MessageDelta):
        final["stop_reason"] = ev.stop_reason or final["stop_reason"]
        if ev.usage.output_tokens:
            final["output_tokens"] = ev.usage.output_tokens
        if ev.usage.input_tokens:
            final["input_tokens"] = ev.usage.input_tokens
        return
    if isinstance(ev, StreamDone):
        return
    if isinstance(ev, StreamError):
        final["stop_reason"] = "error"
        return
