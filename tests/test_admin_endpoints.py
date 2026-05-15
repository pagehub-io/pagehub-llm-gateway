"""Admin endpoints: separate ADMIN_AUTH_TOKEN, listing, detail, content-redaction switch."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.engine import Engine
from api.main import create_app
from api.shared.recorder import MemoryRecorder
from tests.fakes import ScriptedXAIProvider


def _make_app(*, log_content: bool):
    app = create_app()
    provider = ScriptedXAIProvider(
        non_streaming_response={
            "id": "chatcmpl",
            "model": "grok-4",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
    )
    recorder = MemoryRecorder(log_content=log_content)
    app.state.provider = provider
    app.state.recorder = recorder
    app.state.engine = Engine(provider=provider, recorder=recorder)
    return app


def _basic_request(c: TestClient):
    return c.post(
        "/v1/messages",
        json={
            "model": "grok-4",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hello secret prompt"}],
        },
        headers={"x-api-key": "test-token"},
    )


def test_admin_requires_admin_token():
    app = _make_app(log_content=True)
    client = TestClient(app)
    resp = client.get("/v1/admin/requests")
    assert resp.status_code == 401


def test_admin_rejects_gateway_token():
    """A leaked caller token must not be enough to read history."""
    app = _make_app(log_content=True)
    client = TestClient(app)
    resp = client.get("/v1/admin/requests", headers={"x-api-key": "test-token"})
    assert resp.status_code == 401


def test_admin_list_returns_recent_requests():
    app = _make_app(log_content=True)
    client = TestClient(app)
    _basic_request(client)
    _basic_request(client)
    resp = client.get("/v1/admin/requests", headers={"x-api-key": "test-admin-token"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["items"]) == 2
    assert body["items"][0]["provider"] == "xai"


def test_admin_detail_returns_full_record_with_events():
    app = _make_app(log_content=True)
    client = TestClient(app)
    _basic_request(client)
    body = client.get("/v1/admin/requests", headers={"x-api-key": "test-admin-token"}).json()
    rid = body["items"][0]["id"]
    detail = client.get(f"/v1/admin/requests/{rid}", headers={"x-api-key": "test-admin-token"}).json()
    assert detail["id"] == rid
    assert detail["inbound_body"] is not None
    assert detail["canonical_request"] is not None
    assert detail["canonical_response"] is not None
    kinds = [e["kind"] for e in detail["events"]]
    assert "decoded_canonical" in kinds
    assert "provider_request" in kinds


def test_admin_detail_redacts_content_when_log_content_off():
    app = _make_app(log_content=False)
    client = TestClient(app)
    _basic_request(client)
    listing = client.get("/v1/admin/requests", headers={"x-api-key": "test-admin-token"}).json()
    rid = listing["items"][0]["id"]
    detail = client.get(f"/v1/admin/requests/{rid}", headers={"x-api-key": "test-admin-token"}).json()

    # The user prompt was "hello secret prompt" — the redacted inbound_body MUST NOT
    # contain that string anywhere.
    import json

    serialized = json.dumps(detail)
    assert "hello secret prompt" not in serialized
    # Structure is preserved — token counts, stop_reason, model still present.
    assert detail["stop_reason"] == "end_turn"
    assert detail["input_tokens"] == 1
    assert detail["output_tokens"] == 1


def test_admin_detail_404_on_unknown_id():
    app = _make_app(log_content=True)
    client = TestClient(app)
    resp = client.get(
        "/v1/admin/requests/00000000-0000-0000-0000-000000000000",
        headers={"x-api-key": "test-admin-token"},
    )
    assert resp.status_code == 404
