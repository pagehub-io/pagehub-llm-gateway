"""GET /v1/admin/requests + GET /v1/admin/requests/{id} — inspect logged traffic."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from api.auth import require_admin_auth
from api.shared.recorder import Recorder

router = APIRouter(tags=["admin"])


def get_recorder(request: Request) -> Recorder:
    return request.app.state.recorder


class RequestSummary(BaseModel):
    id: str
    ts: str | None = None
    client_request_id: str | None = None
    inbound_protocol: str
    provider: str
    provider_model_requested: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    stop_reason: str | None = None
    error: str | None = None
    latency_ms: int | None = None


class RequestListResponse(BaseModel):
    items: list[RequestSummary]
    limit: int
    offset: int


class RequestEvent(BaseModel):
    seq: int
    ts: str | None = None
    kind: str
    payload: Any | None = None


class RequestDetail(BaseModel):
    id: str
    ts: str | None = None
    client_request_id: str | None = None
    inbound_protocol: str
    provider: str
    provider_model_requested: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    stop_reason: str | None = None
    error: str | None = None
    latency_ms: int | None = None
    inbound_body: Any | None = None
    canonical_request: Any | None = None
    outbound_body: Any | None = None
    provider_response: Any | None = None
    canonical_response: Any | None = None
    outbound_body_returned: Any | None = None
    started_at: str | None = None
    finished_at: str | None = None
    events: list[RequestEvent] = Field(default_factory=list)


@router.get(
    "/v1/admin/requests",
    response_model=RequestListResponse,
    dependencies=[Depends(require_admin_auth)],
)
async def list_requests(
    recorder: Recorder = Depends(get_recorder),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> RequestListResponse:
    rows = await recorder.list_requests(limit=limit, offset=offset)
    return RequestListResponse(
        items=[RequestSummary(**r) for r in rows], limit=limit, offset=offset
    )


@router.get(
    "/v1/admin/requests/{request_id}",
    response_model=RequestDetail,
    dependencies=[Depends(require_admin_auth)],
)
async def get_request_detail(
    request_id: str,
    recorder: Recorder = Depends(get_recorder),
) -> RequestDetail:
    try:
        rid = uuid.UUID(request_id)
    except ValueError:
        raise HTTPException(
            status_code=400, detail={"error": {"type": "invalid_id", "message": "Bad UUID"}}
        )
    row = await recorder.get_request(rid)
    if row is None:
        raise HTTPException(
            status_code=404, detail={"error": {"type": "not_found", "message": "Unknown request id"}}
        )
    return RequestDetail(**row)
