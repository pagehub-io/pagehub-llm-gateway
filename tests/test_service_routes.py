from __future__ import annotations


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["env"] == "development"
    assert data["service"] == "pagehub-llm-gateway"
    assert data["default_model"] == "grok-4"


def test_metrics_initial_shape(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    for key in (
        "requests_total",
        "errors_total",
        "input_tokens_total",
        "output_tokens_total",
        "ttfb_ms_histogram",
        "total_ms_histogram",
    ):
        assert key in data
    assert isinstance(data["requests_total"], dict)
    assert isinstance(data["ttfb_ms_histogram"]["counts"], list)


def test_openapi_schema_available(client):
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    paths = schema.get("paths", {})
    assert "/v1/messages" in paths
    assert "/v1/models" in paths
    assert "/health" in paths
