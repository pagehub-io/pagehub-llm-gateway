"""FastAPI app factory + lifespan for pagehub-llm-gateway."""

from __future__ import annotations

import contextlib
import logging

from fastapi import FastAPI

from api.config import settings
from api.engine import Engine
from api.middleware import BodySizeLimitMiddleware, TwinHeaderMiddleware
from api.providers.base import ProviderAdapter
from api.providers.xai import XAIProvider
from api.shared.db import apply_schema, close_pool, init_pool
from api.shared.recorder import MemoryRecorder, PostgresRecorder, Recorder
from api.v1.admin_router import router as admin_router
from api.v1.messages_router import router as messages_router
from api.v1.service_router import router as service_router

logger = logging.getLogger(__name__)


def _build_provider() -> ProviderAdapter:
    return XAIProvider()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # If the test/build wired a provider in already, respect it.
    provider: ProviderAdapter = getattr(app.state, "provider", None) or _build_provider()

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

    app.state.provider = provider
    app.state.engine = Engine(provider=provider, recorder=app.state.recorder)

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
