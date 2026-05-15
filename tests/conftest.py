from __future__ import annotations

import os

# Ensure tests run with predictable settings — auth token + dev env.
os.environ.setdefault("GATEWAY_AUTH_TOKEN", "test-token")
os.environ.setdefault("ENV", "development")
os.environ.setdefault("XAI_API_KEY", "test-xai-key-not-used")
os.environ.setdefault("GROK_DEFAULT_MODEL", "grok-4")

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from tests.fake_provider import FakeProvider


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def app(fake_provider: FakeProvider):
    a = create_app()
    a.state.provider = fake_provider
    return a


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"x-api-key": "test-token"}
