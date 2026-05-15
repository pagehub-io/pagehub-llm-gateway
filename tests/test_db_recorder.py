"""Integration: PostgresRecorder against a real database.

Skipped when ``DATABASE_URL`` is unset. CI runs it via a Postgres service; local
runs need ``docker-compose up postgres`` + an export.
"""

from __future__ import annotations

import os
import uuid

import pytest

pytestmark = pytest.mark.db

DATABASE_URL = os.environ.get("DATABASE_URL", "")

if not DATABASE_URL:
    pytest.skip("DATABASE_URL not set — postgres integration tests skipped", allow_module_level=True)


@pytest.fixture
async def pool():
    from api.shared.db import apply_schema, close_pool, init_pool

    pool = await init_pool(DATABASE_URL)
    await apply_schema()
    try:
        yield pool
    finally:
        # Cleanup rows from this run so subsequent runs start clean.
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM request_events")
            await conn.execute("DELETE FROM requests")
        await close_pool()


async def test_round_trip_through_postgres_recorder(pool):
    from api.shared.recorder import PostgresRecorder

    rec = PostgresRecorder(pool, log_content=True)
    rid = await rec.start_request(
        client_request_id="rid_x",
        inbound_protocol="anthropic-messages-v1",
        inbound_body={"model": "grok-4", "messages": [{"role": "user", "content": "hi"}]},
        canonical_request={"model": "grok-4"},
        provider="xai",
        provider_model_requested="grok-4",
    )
    assert isinstance(rid, uuid.UUID)

    await rec.add_event(rid, "inbound_received", {"size": 100})
    await rec.add_event(rid, "provider_request", {"foo": "bar"})
    await rec.finalize_request(
        rid,
        outbound_body={"x": 1},
        provider_response={"ok": True},
        canonical_response={"id": "msg_y", "model": "grok-4"},
        outbound_body_returned={"id": "msg_y"},
        input_tokens=10,
        output_tokens=4,
        stop_reason="end_turn",
    )

    rows = await rec.list_requests(limit=10, offset=0)
    assert len(rows) == 1
    assert rows[0]["client_request_id"] == "rid_x"
    assert rows[0]["input_tokens"] == 10
    assert rows[0]["stop_reason"] == "end_turn"
    assert rows[0]["latency_ms"] is not None

    detail = await rec.get_request(rid)
    assert detail["canonical_request"]["model"] == "grok-4"
    kinds = [ev["kind"] for ev in detail["events"]]
    assert kinds == ["inbound_received", "provider_request"]
    seqs = [ev["seq"] for ev in detail["events"]]
    assert seqs == sorted(seqs)


async def test_redaction_applied_at_record_time(pool):
    from api.shared.recorder import PostgresRecorder

    rec = PostgresRecorder(pool, log_content=False)
    rid = await rec.start_request(
        client_request_id=None,
        inbound_protocol="anthropic-messages-v1",
        inbound_body={"messages": [{"role": "user", "content": "secret payload"}]},
        canonical_request={"messages": [{"role": "user", "content_blocks": [{"type": "text", "text": "secret payload"}]}]},
        provider="xai",
        provider_model_requested="grok-4",
    )
    await rec.finalize_request(rid)
    detail = await rec.get_request(rid)
    import json

    serialized = json.dumps(detail)
    assert "secret payload" not in serialized
