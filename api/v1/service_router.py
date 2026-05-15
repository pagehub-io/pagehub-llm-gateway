from __future__ import annotations

from fastapi import APIRouter

from api.config import settings
from api.metrics import get_metrics
from api.v1.service_schemas import HealthResponse, MetricsResponse

router = APIRouter(tags=["service"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        commit=settings.git_commit,
        env=settings.env,
        service=settings.service_name,
        default_model=settings.grok_default_model,
    )


@router.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    return MetricsResponse(**get_metrics().snapshot())
