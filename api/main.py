"""FastAPI app factory + lifespan for pagehub-llm-gateway."""

from __future__ import annotations

import contextlib
import logging

from fastapi import FastAPI

from api.config import settings
from api.engine import Engine
from api.middleware import BodySizeLimitMiddleware, TwinHeaderMiddleware
from api.providers.base import ProviderAdapter
from api.providers.openai import OpenAIProvider
from api.providers.registry import ProviderRegistry
from api.providers.xai import XAIProvider
from api.shared.db import apply_schema, close_pool, init_pool
from api.shared.recorder import MemoryRecorder, PostgresRecorder, Recorder
from api.v1.admin_router import router as admin_router
from api.v1.messages_router import router as messages_router
from api.v1.service_router import router as service_router

logger = logging.getLogger(__name__)


def _build_registry() -> ProviderRegistry:
    """Order matters: ``pick()`` walks the list and returns the first claim.
    Adding a new provider is one line here."""
    return ProviderRegistry(
        [
            XAIProvider(),
            OpenAIProvider(),
        ]
    )


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Provider wiring. Tests may have set either:
    #   - app.state.registry  (multi-provider, the new way)
    #   - app.state.provider  (single-provider, the legacy / fake-injection way
    #     used by tests/conftest.py and tests/test_e2e_app.py)
    # We respect either, falling back to building the real production
    # registry from settings.
    registry: ProviderRegistry | None = getattr(app.state, "registry", None)
    legacy_provider: ProviderAdapter | None = getattr(app.state, "provider", None)
    if registry is None and legacy_provider is None:
        registry = _build_registry()

    # Recorder: PostgresRecorder when DATABASE_URL is set, else MemoryRecorder. This
    # keeps the local-dev / no-DB path working out of the box.
    if not hasattr(app.state, "recorder"):
        if settings.database_url:
            try:
                pool = await init_pool(settings.database_url)
                await apply_schema()
                recorder: Recorder = PostgresRecorder(pool, log_content=settings.log_content)
                logger.info("recorder: postgres (log_content=%s)", settings.log_content)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Postgres init failed — falling back to MemoryRecorder. Set DATABASE_URL to a reachable DSN."
                )
                recorder = MemoryRecorder(log_content=settings.log_content)
        else:
            recorder = MemoryRecorder(log_content=settings.log_content)
            logger.info("recorder: memory (DATABASE_URL not set — request log is not persisted)")
        app.state.recorder = recorder

    if registry is not None:
        app.state.registry = registry
        app.state.engine = Engine(registry=registry, recorder=app.state.recorder)
    else:
        # Legacy single-provider path (tests). The engine still routes via its
        # internal single-provider shim so the request semantics are identical.
        app.state.provider = legacy_provider
        app.state.engine = Engine(provider=legacy_provider, recorder=app.state.recorder)

    try:
        yield
    finally:
        if settings.database_url:
            with contextlib.suppress(Exception):
                await close_pool()


def create_app() -> FastAPI:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    app = FastAPI(
        title="pagehub-llm-gateway",
        description=(
            "Anthropic-compatible gateway that routes Claude-style /v1/messages calls to "
            "non-Anthropic backends through a provider-neutral canonical layer. v1 "
            "backend: xAI Grok. Adding new providers / new inbound protocols is one "
            "adapter each — not pairwise translators."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.request_body_max_bytes)
    app.add_middleware(TwinHeaderMiddleware)

    app.include_router(service_router)
    app.include_router(messages_router)
    app.include_router(admin_router)
    return app


app = create_app()
