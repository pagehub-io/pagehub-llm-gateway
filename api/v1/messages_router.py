"""POST /v1/messages and GET /v1/models — Anthropic-compatible surface.

The router is intentionally thin: validate the body, decode it to Canonical via
``AnthropicInbound``, hand to the engine, return the result. The engine owns
the canonical pipeline + recording.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.auth import require_auth
from api.config import settings
from api.engine import Engine
from api.metrics import Timer, get_metrics
from api.protocols.anthropic import AnthropicInbound
from api.providers.base import ProviderError
from api.v1.schemas import (
    MessagesRequest,
    MessagesResponse,
    ModelInfo,
    ModelListResponse,
)

logger = logging.getLogger("pagehub_llm_gateway")

router = APIRouter(tags=["anthropic"])


def get_engine(request: Request) -> Engine:
    return request.app.state.engine


@router.get("/v1/models", response_model=ModelListResponse, dependencies=[Depends(require_auth)])
async def list_models(request: Request) -> ModelListResponse:
    """Advertise the default model id of every registered provider. Clients
    can address other models on those providers by name — these are just the
    fallback ids the gateway will use when a request omits ``model``."""
    registry = getattr(request.app.state, "registry", None)
    legacy = getattr(request.app.state, "provider", None)
    ids: list[str] = []
    if registry is not None:
        for p in registry.providers:
            m = getattr(p, "default_model", "")
            if m:
                ids.append(m)
    elif legacy is not None:
        m = getattr(legacy, "default_model", settings.grok_default_model)
        if m:
            ids.append(m)
    if not ids:
        ids.append(settings.grok_default_model)

    data = [
        ModelInfo(id=mid, display_name=mid, created_at="2024-01-01T00:00:00Z") for mid in ids
    ]
    return ModelListResponse(
        data=data,
        first_id=ids[0],
        last_id=ids[-1],
        has_more=False,
    )


def _estimate_input_tokens(canonical) -> int:
    """Cheap byte-based estimate. Placeholder until the provider reports a real count."""
    total = 0
    if canonical.system:
        for b in canonical.system:
            if getattr(b, "type", None) == "text":
                total += len(b.text or "")
    for m in canonical.messages:
        for b in m.content_blocks:
            t = getattr(b, "type", None)
            if t == "text":
                total += len(b.text or "")
            elif t == "tool_result":
                total += len(b.content or "")
    return max(1, total // 4)


@router.post("/v1/messages", dependencies=[Depends(require_auth)], response_model=None)
async def messages(
    request: Request,
    body: MessagesRequest,
    engine: Engine = Depends(get_engine),
):
    request_id = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex[:16]}"
    timer = Timer()

    raw_body = body.model_dump(exclude_none=True)
    canonical = AnthropicInbound.decode_request(body)

    if body.stream:

        async def event_source():
            first = True
            try:
                async for chunk in engine.stream_anthropic(
                    client_request_id=request_id,
                    raw_inbound_body=raw_body,
                    canonical_request=canonical,
                    input_tokens_estimate=_estimate_input_tokens(canonical),
                ):
                    if first:
                        timer.mark_first_byte()
                        first = False
                    yield chunk
            finally:
                get_metrics().record_request(
                    route="/v1/messages",
                    status_code=200,
                    ttfb_ms=timer.ttfb_ms(),
                    total_ms=timer.total_ms(),
                    input_tokens=0,
                    output_tokens=0,
                )
                logger.info(
                    "messages-stream rid=%s model=%s ttfb_ms=%.1f total_ms=%.1f",
                    request_id,
                    canonical.model,
                    timer.ttfb_ms() or 0.0,
                    timer.total_ms(),
                )

        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Request-Id": request_id,
            },
        )

    try:
        result, _ = await engine.run_anthropic(
            client_request_id=request_id,
            raw_inbound_body=raw_body,
            canonical_request=canonical,
        )
    except ProviderError as exc:
        timer.mark_first_byte()
        get_metrics().record_request(
            route="/v1/messages",
            status_code=exc.status_code,
            ttfb_ms=timer.ttfb_ms(),
            total_ms=timer.total_ms(),
            input_tokens=0,
            output_tokens=0,
        )
        logger.warning(
            "provider error rid=%s status=%s msg=%s", request_id, exc.status_code, exc.message
        )
        raise HTTPException(
            status_code=exc.status_code,
            detail={"error": {"type": "provider_error", "message": exc.message}},
        ) from exc

    timer.mark_first_byte()
    MessagesResponse.model_validate(result)
    usage = result.get("usage") or {}
    get_metrics().record_request(
        route="/v1/messages",
        status_code=200,
        ttfb_ms=timer.ttfb_ms(),
        total_ms=timer.total_ms(),
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
    )
    logger.info(
        "messages ok rid=%s model=%s in=%s out=%s total_ms=%.1f finish=%s",
        request_id,
        canonical.model,
        usage.get("input_tokens"),
        usage.get("output_tokens"),
        timer.total_ms(),
        result.get("stop_reason"),
    )
    return result
