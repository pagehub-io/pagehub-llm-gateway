from __future__ import annotations

from fastapi import APIRouter, Request

from api.config import settings
from api.metrics import get_metrics
from api.v1.service_schemas import HealthResponse, MetricsResponse, ProviderInfo

router = APIRouter(tags=["service"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    # Reflect whatever was wired into the engine — production registry, single
    # test provider, etc. Falls back to settings on cold-start edge cases.
    registry = getattr(request.app.state, "registry", None)
    legacy = getattr(request.app.state, "provider", None)
    providers: list[ProviderInfo] = []
    if registry is not None:
        for p in registry.providers:
            providers.append(
                ProviderInfo(
                    name=getattr(p, "name", type(p).__name__),
                    default_model=getattr(p, "default_model", ""),
                    model_name_prefixes=list(getattr(p, "model_name_prefixes", ()) or ()),
                )
            )
    elif legacy is not None:
        providers.append(
            ProviderInfo(
                name=getattr(legacy, "name", type(legacy).__name__),
                default_model=getattr(legacy, "default_model", settings.grok_default_model),
                model_name_prefixes=list(getattr(legacy, "model_name_prefixes", ()) or ()),
            )
        )

    default = providers[0].default_model if providers else settings.grok_default_model
    return HealthResponse(
        status="ok",
        commit=settings.git_commit,
        env=settings.env,
        service=settings.service_name,
        default_model=default,
        providers=providers,
    )


@router.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    return MetricsResponse(**get_metrics().snapshot())
