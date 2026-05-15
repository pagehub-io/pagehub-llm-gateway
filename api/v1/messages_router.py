"""POST /v1/messages and GET /v1/models — the Anthropic-compatible surface."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.auth import require_auth
from api.config import settings
from api.metrics import Timer, get_metrics
from api.providers.base import Provider, ProviderError
from api.translation.anthropic_to_openai import translate_request
from api.translation.openai_to_anthropic import translate_response
from api.translation.streaming import openai_stream_to_anthropic_events
from api.v1.schemas import (
    MessagesRequest,
    MessagesResponse,
    ModelInfo,
    ModelListResponse,
)

logger = logging.getLogger("pagehub_llm_gateway")

router = APIRouter(tags=["anthropic"])


def get_provider(request: Request) -> Provider:
    return request.app.state.provider


@router.get("/v1/models", response_model=ModelListResponse, dependencies=[Depends(require_auth)])
async def list_models() -> ModelListResponse:
    default = settings.grok_default_model
    info = ModelInfo(id=default, display_name=default, created_at="2024-01-01T00:00:00Z")
    return ModelListResponse(data=[info], first_id=default, last_id=default, has_more=False)


def _estimate_input_tokens(req: MessagesRequest) -> int:
    """Cheap byte-based estimate. Not for billing — just a placeholder for
    ``message_start.usage.input_tokens`` before the provider's real count arrives."""
    total = 0
    if req.system:
        if isinstance(req.system, str):
            total += len(req.system)
        else:
            for b in req.system:
                if isinstance(b, dict) and b.get("type") == "text":
                    total += len(b.get("text") or "")
    for m in req.messages:
        if isinstance(m.content, str):
            total += len(m.content)
        else:
            for b in m.content or []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text":
                    total += len(b.get("text") or "")
                elif b.get("type") == "tool_result":
                    c = b.get("content")
                    if isinstance(c, str):
                        total += len(c)
                    elif isinstance(c, list):
                        for sub in c:
                            if isinstance(sub, dict) and sub.get("type") == "text":
                                total += len(sub.get("text") or "")
    return max(1, total // 4)


def _provider_error_payload(exc: ProviderError) -> dict[str, Any]:
    return {"error": {"type": "provider_error", "message": exc.message}}


async def _handle_non_streaming(
    req: MessagesRequest,
    provider: Provider,
    request_id: str,
    timer: Timer,
) -> dict[str, Any]:
    target_model = provider.map_model(req.model)
    body = translate_request(req, target_model)
    body["stream"] = False
    body.pop("stream_options", None)
    try:
        upstream = await provider.chat_completion(body)
    except ProviderError as exc:
        get_metrics().record_request(
            route="/v1/messages",
            status_code=exc.status_code,
            ttfb_ms=None,
            total_ms=timer.total_ms(),
            input_tokens=0,
            output_tokens=0,
        )
        logger.warning(
            "provider error rid=%s status=%s msg=%s", request_id, exc.status_code, exc.message
        )
        raise HTTPException(
            status_code=exc.status_code, detail=_provider_error_payload(exc)
        ) from exc

    timer.mark_first_byte()
    anth = translate_response(upstream, anthropic_model=req.model)
    MessagesResponse.model_validate(anth)
    usage = anth.get("usage") or {}
    get_metrics().record_request(
        route="/v1/messages",
        status_code=200,
        ttfb_ms=timer.ttfb_ms(),
        total_ms=timer.total_ms(),
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
    )
    logger.info(
        "messages ok rid=%s model=%s target=%s in=%s out=%s ttfb_ms=%.1f total_ms=%.1f finish=%s",
        request_id,
        req.model,
        target_model,
        usage.get("input_tokens"),
        usage.get("output_tokens"),
        timer.ttfb_ms() or 0.0,
        timer.total_ms(),
        anth.get("stop_reason"),
    )
    return anth


async def _handle_streaming(
    req: MessagesRequest,
    provider: Provider,
    request_id: str,
    timer: Timer,
) -> StreamingResponse:
    target_model = provider.map_model(req.model)
    body = translate_request(req, target_model)
    body["stream"] = True
    input_estimate = _estimate_input_tokens(req)

    async def event_source():
        first = True
        output_tokens = 0
        input_tokens = input_estimate
        try:
            provider_stream = provider.chat_completion_stream(body)
            async for event_bytes in openai_stream_to_anthropic_events(
                provider_stream,
                anthropic_model=req.model,
                input_tokens_estimate=input_estimate,
            ):
                if first:
                    timer.mark_first_byte()
                    first = False
                if b"message_delta" in event_bytes:
                    tok = _extract_output_tokens(event_bytes)
                    if tok is not None:
                        output_tokens = tok
                if b"message_start" in event_bytes:
                    tok = _extract_input_tokens(event_bytes)
                    if tok:
                        input_tokens = tok
                yield event_bytes
        except ProviderError as exc:
            err = {"type": "error", "error": {"type": "provider_error", "message": exc.message}}
            yield f"event: error\ndata: {json.dumps(err)}\n\n".encode("utf-8")
        finally:
            get_metrics().record_request(
                route="/v1/messages",
                status_code=200,
                ttfb_ms=timer.ttfb_ms(),
                total_ms=timer.total_ms(),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            logger.info(
                "messages-stream done rid=%s model=%s target=%s in=%s out=%s ttfb_ms=%.1f total_ms=%.1f",
                request_id,
                req.model,
                target_model,
                input_tokens,
                output_tokens,
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


def _extract_output_tokens(event_bytes: bytes) -> int | None:
    try:
        text = event_bytes.decode("utf-8", errors="replace")
        data_line = next((ln for ln in text.split("\n") if ln.startswith("data:")), None)
        if not data_line:
            return None
        payload = json.loads(data_line[5:].strip())
        usage = payload.get("usage") or {}
        if "output_tokens" in usage:
            return int(usage["output_tokens"])
    except (ValueError, KeyError, AttributeError):
        return None
    return None


def _extract_input_tokens(event_bytes: bytes) -> int | None:
    try:
        text = event_bytes.decode("utf-8", errors="replace")
        data_line = next((ln for ln in text.split("\n") if ln.startswith("data:")), None)
        if not data_line:
            return None
        payload = json.loads(data_line[5:].strip())
        usage = (payload.get("message") or {}).get("usage") or {}
        if "input_tokens" in usage:
            return int(usage["input_tokens"])
    except (ValueError, KeyError, AttributeError):
        return None
    return None


@router.post(
    "/v1/messages",
    dependencies=[Depends(require_auth)],
    response_model=None,
)
async def messages(
    request: Request,
    body: MessagesRequest,
    provider: Provider = Depends(get_provider),
):
    request_id = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex[:16]}"
    timer = Timer()

    if settings.log_content:
        logger.debug("inbound rid=%s body=%s", request_id, body.model_dump(exclude_none=True))

    if body.stream:
        return await _handle_streaming(body, provider, request_id, timer)
    return await _handle_non_streaming(body, provider, request_id, timer)
