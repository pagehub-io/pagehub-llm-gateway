from __future__ import annotations

import os

os.environ.setdefault("GATEWAY_AUTH_TOKEN", "test-token")
os.environ.setdefault("ADMIN_AUTH_TOKEN", "test-admin-token")
os.environ.setdefault("ENV", "development")
os.environ.setdefault("XAI_API_KEY", "test-xai-key-not-used")
os.environ.setdefault("GROK_DEFAULT_MODEL", "grok-4")
# Default tests to MemoryRecorder (no postgres). The `db` marker selects tests that
# need a real DATABASE_URL.
os.environ.setdefault("DATABASE_URL", "")
os.environ.setdefault("LOG_CONTENT", "true")

import pytest
from fastapi.testclient import TestClient

from api.engine import Engine
from api.main import create_app
from api.shared.recorder import MemoryRecorder
from tests.fakes import EchoProvider, ScriptedXAIProvider


def _wire(app, provider, recorder):
    """Eagerly wire provider/recorder/engine onto the app.

    FastAPI's TestClient does NOT run the lifespan unless used as a context
    manager, so engine/recorder must be set explicitly for sync test usage.
    """
    app.state.provider = provider
    app.state.recorder = recorder
    app.state.engine = Engine(provider=provider, recorder=recorder)
    return app


@pytest.fixture
def echo_provider() -> EchoProvider:
    return EchoProvider()


@pytest.fixture
def scripted_xai() -> ScriptedXAIProvider:
    return ScriptedXAIProvider()


@pytest.fixture
def memory_recorder() -> MemoryRecorder:
    return MemoryRecorder(log_content=True)


@pytest.fixture
def app_with_echo(echo_provider, memory_recorder):
    return _wire(create_app(), echo_provider, memory_recorder)


@pytest.fixture
def app_with_xai(scripted_xai, memory_recorder):
    return _wire(create_app(), scripted_xai, memory_recorder)


@pytest.fixture
def app(app_with_xai):
    """Default: full e2e app with the scripted xAI provider."""
    return app_with_xai


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"x-api-key": "test-token"}


@pytest.fixture
def admin_headers() -> dict[str, str]:
    return {"x-api-key": "test-admin-token"}
