from __future__ import annotations


def _basic_body():
    return {
        "model": "grok-4",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
    }


def test_messages_requires_auth(client):
    resp = client.post("/v1/messages", json=_basic_body())
    assert resp.status_code == 401
    body = resp.json()
    assert body["detail"]["error"]["type"] == "authentication_error"


def test_models_requires_auth(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 401


def test_messages_rejects_wrong_token(client):
    resp = client.post(
        "/v1/messages",
        json=_basic_body(),
        headers={"x-api-key": "not-the-token"},
    )
    assert resp.status_code == 401


def test_messages_accepts_x_api_key(client, auth_headers):
    resp = client.post("/v1/messages", json=_basic_body(), headers=auth_headers)
    assert resp.status_code == 200


def test_messages_accepts_bearer_authorization(client):
    resp = client.post(
        "/v1/messages",
        json=_basic_body(),
        headers={"authorization": "Bearer test-token"},
    )
    assert resp.status_code == 200


def test_models_returns_list_with_default(client, auth_headers):
    resp = client.get("/v1/models", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert any(m["id"] == "grok-4" for m in data["data"])
    assert data["has_more"] is False
