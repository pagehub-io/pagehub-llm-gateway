"""X-Twin-* header behavior: honored in development, stripped in production/staging.

We can't directly observe contextvar state from a TestClient request, so we mount a
debug route that reports what the middleware extracted. The real grok provider has
its own dependency on :func:`api.twin.get_override` — covered indirectly in the
``test_grok_provider`` tests.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from api import middleware, twin
from api.middleware import BodySizeLimitMiddleware, TwinHeaderMiddleware


def _build_app(*, env: str) -> FastAPI:
    # Force the env to a specific value so the middleware re-reads it.
    import api.config as config_mod

    config_mod.settings.env = env
    middleware._DEV_ENV = "development"  # ensure default

    app = FastAPI()
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=1024 * 1024)
    app.add_middleware(TwinHeaderMiddleware)

    @app.get("/_twin")
    async def show(request: Request):
        return {
            "override": twin.get_override("GROK_BASE_URL"),
            "raw_header": request.headers.get("x-twin-grok-base-url"),
        }

    return app


@pytest.fixture(autouse=True)
def _reset_twin_state():
    twin.reset()
    yield
    twin.reset()
    # Restore env to development for subsequent tests.
    import api.config as config_mod

    config_mod.settings.env = "development"
    importlib.reload(middleware)


def test_twin_override_active_in_development():
    app = _build_app(env="development")
    client = TestClient(app)
    resp = client.get(
        "/_twin",
        headers={"X-Twin-Grok-Base-Url": "http://twin.example/v1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    # In dev: the header reaches the handler AND the contextvar carries the override.
    assert data["override"] == "http://twin.example/v1"
    assert data["raw_header"] == "http://twin.example/v1"


def test_twin_header_stripped_in_production():
    app = _build_app(env="production")
    client = TestClient(app)
    resp = client.get(
        "/_twin",
        headers={"X-Twin-Grok-Base-Url": "http://attacker.example/"},
    )
    data = resp.json()
    assert data["override"] is None
    assert data["raw_header"] is None


def test_twin_header_stripped_in_staging():
    app = _build_app(env="staging")
    client = TestClient(app)
    resp = client.get(
        "/_twin",
        headers={"X-Twin-Grok-Base-Url": "http://attacker.example/"},
    )
    data = resp.json()
    assert data["override"] is None


def test_twin_header_stripped_in_unknown_env():
    # Deny-by-default for new envs like "qa" / "preview" / etc.
    app = _build_app(env="qa")
    client = TestClient(app)
    resp = client.get(
        "/_twin",
        headers={"X-Twin-Grok-Base-Url": "http://x/"},
    )
    assert resp.json()["override"] is None


def test_body_size_limit_returns_413():
    app = _build_app(env="development")
    client = TestClient(app)
    big = b"x" * (2 * 1024 * 1024)
    resp = client.post("/_anything", content=big, headers={"content-type": "application/json"})
    assert resp.status_code == 413
