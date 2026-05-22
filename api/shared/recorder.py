"""Request/event recorder — Protocol + Postgres + Memory implementations.

Engine code talks to a ``Recorder``. In production it's a ``PostgresRecorder``
backed by asyncpg. In tests we use ``MemoryRecorder`` so the engine's recording
behavior is tested without a real database. The interface is intentionally
minimal: ``start_request``, ``add_event``, ``finalize_request``.

Redaction is applied at recording time using :func:`api.shared.redaction.redact`
so the stored JSON is already content-stripped (you can't accidentally leak by
opening the wrong row).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import uuid
from typing import Any, Protocol

from api.shared.redaction import redact

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class RequestRecord:
    id: uuid.UUID
    client_request_id: str | None
    inbound_protocol: str
    inbound_body: Any
    canonical_request: Any
    provider: str
    provider_model_requested: str | None
    outbound_body: Any = None
    provider_response: Any = None
    canonical_response: Any = None
    outbound_body_returned: Any = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    stop_reason: str | None = None
    error: str | None = None
    started_at: dt.datetime = dataclasses.field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))
    finished_at: dt.datetime | None = None
    latency_ms: int | None = None
    events: list[dict[str, Any]] = dataclasses.field(default_factory=list)


class Recorder(Protocol):
    async def start_request(
        self,
        *,
        client_request_id: str | None,
        inbound_protocol: str,
        inbound_body: Any,
        canonical_request: Any,
        provider: str,
        provider_model_requested: str | None,
    ) -> uuid.UUID:
        ...

    async def add_event(self, request_id: uuid.UUID, kind: str, payload: Any) -> None:
        ...

    async def finalize_request(
        self,
        request_id: uuid.UUID,
        *,
        outbound_body: Any = None,
        provider_response: Any = None,
        canonical_response: Any = None,
        outbound_body_returned: Any = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        stop_reason: str | None = None,
        error: str | None = None,
    ) -> None:
        ...

    async def list_requests(self, *, limit: int, offset: int) -> list[dict[str, Any]]:
        ...

    async def get_request(self, request_id: uuid.UUID) -> dict[str, Any] | None:
        ...


# ---------------------------------------------------------------------------
# MemoryRecorder — in-process, used by unit tests.
# ---------------------------------------------------------------------------


class MemoryRecorder:
    """Holds a small ring of recent requests in memory. Resets on process restart."""

    def __init__(self, *, log_content: bool, capacity: int = 200) -> None:
        self._log_content = log_content
        self._capacity = capacity
        self._records: dict[uuid.UUID, RequestRecord] = {}
        self._order: list[uuid.UUID] = []
        self._seq: dict[uuid.UUID, int] = {}

    @property
    def log_content(self) -> bool:
        return self._log_content

    def _redact(self, value: Any) -> Any:
        return redact(value, enabled=not self._log_content)

    async def start_request(
        self,
        *,
        client_request_id: str | None,
        inbound_protocol: str,
        inbound_body: Any,
        canonical_request: Any,
        provider: str,
        provider_model_requested: str | None,
    ) -> uuid.UUID:
        rid = uuid.uuid4()
        record = RequestRecord(
            id=rid,
            client_request_id=client_request_id,
            inbound_protocol=inbound_protocol,
            inbound_body=self._redact(inbound_body),
            canonical_request=self._redact(canonical_request),
            provider=provider,
            provider_model_requested=provider_model_requested,
        )
        self._records[rid] = record
        self._order.append(rid)
        self._seq[rid] = 0
        # Evict oldest beyond capacity.
        while len(self._order) > self._capacity:
            evicted = self._order.pop(0)
            self._records.pop(evicted, None)
            self._seq.pop(evicted, None)
        return rid

    async def add_event(self, request_id: uuid.UUID, kind: str, payload: Any) -> None:
        rec = self._records.get(request_id)
        if rec is None:
            return
        self._seq[request_id] += 1
        rec.events.append(
            {
                "seq": self._seq[request_id],
                "kind": kind,
                "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
                "payload": self._redact(payload),
            }
        )

    async def finalize_request(
        self,
        request_id: uuid.UUID,
        *,
        outbound_body: Any = None,
        provider_response: Any = None,
        canonical_response: Any = None,
        outbound_body_returned: Any = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        stop_reason: str | None = None,
        error: str | None = None,
    ) -> None:
        rec = self._records.get(request_id)
        if rec is None:
            return
        rec.outbound_body = self._redact(outbound_body)
        rec.provider_response = self._redact(provider_response)
        rec.canonical_response = self._redact(canonical_response)
        rec.outbound_body_returned = self._redact(outbound_body_returned)
        rec.input_tokens = input_tokens
        rec.output_tokens = output_tokens
        rec.stop_reason = stop_reason
        rec.error = error
        rec.finished_at = dt.datetime.now(dt.timezone.utc)
        rec.latency_ms = int((rec.finished_at - rec.started_at).total_seconds() * 1000)

    async def list_requests(self, *, limit: int, offset: int) -> list[dict[str, Any]]:
        rows = list(reversed(self._order))[offset : offset + limit]
        return [_record_to_summary(self._records[rid]) for rid in rows]

    async def get_request(self, request_id: uuid.UUID) -> dict[str, Any] | None:
        rec = self._records.get(request_id)
        if rec is None:
            return None
        return _record_to_detail(rec)


def _record_to_summary(rec: RequestRecord) -> dict[str, Any]:
    return {
        "id": str(rec.id),
        "ts": rec.started_at.isoformat(),
        "client_request_id": rec.client_request_id,
        "inbound_protocol": rec.inbound_protocol,
        "provider": rec.provider,
        "provider_model_requested": rec.provider_model_requested,
        "input_tokens": rec.input_tokens,
        "output_tokens": rec.output_tokens,
        "stop_reason": rec.stop_reason,
        "error": rec.error,
        "latency_ms": rec.latency_ms,
    }


def _record_to_detail(rec: RequestRecord) -> dict[str, Any]:
    return {
        **_record_to_summary(rec),
        "inbound_body": rec.inbound_body,
        "canonical_request": rec.canonical_request,
        "outbound_body": rec.outbound_body,
        "provider_response": rec.provider_response,
        "canonical_response": rec.canonical_response,
        "outbound_body_returned": rec.outbound_body_returned,
        "started_at": rec.started_at.isoformat(),
        "finished_at": rec.finished_at.isoformat() if rec.finished_at else None,
        "events": rec.events,
    }


# ---------------------------------------------------------------------------
# PostgresRecorder — production. Uses an asyncpg pool injected at construction.
# ---------------------------------------------------------------------------


class PostgresRecorder:
    def __init__(self, pool, *, log_content: bool) -> None:
        self._pool = pool
        self._log_content = log_content
        # Per-request monotonic event sequence counters live in-process; the unique
        # index on (request_id, seq) defends against duplicates if a request is
        # somehow handled by two workers (we don't share state across workers).
        self._seq: dict[uuid.UUID, int] = {}

    @property
    def log_content(self) -> bool:
        return self._log_content

    def _redact(self, value: Any) -> Any:
        return redact(value, enabled=not self._log_content)

    async def start_request(
        self,
        *,
        client_request_id: str | None,
        inbound_protocol: str,
        inbound_body: Any,
        canonical_request: Any,
        provider: str,
        provider_model_requested: str | None,
    ) -> uuid.UUID:
        rid = uuid.uuid4()
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO requests (
                    id, client_request_id, inbound_protocol, inbound_body,
                    canonical_request, provider, provider_model_requested, started_at
                ) VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6, $7, NOW())
                """,
                rid,
                client_request_id,
                inbound_protocol,
                _to_json(self._redact(inbound_body)),
                _to_json(self._redact(canonical_request)),
                provider,
                provider_model_requested,
            )
        self._seq[rid] = 0
        return rid

    async def add_event(self, request_id: uuid.UUID, kind: str, payload: Any) -> None:
        self._seq[request_id] = self._seq.get(request_id, 0) + 1
        seq = self._seq[request_id]
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO request_events (request_id, seq, kind, payload)
                    VALUES ($1, $2, $3, $4::jsonb)
                    """,
                    request_id,
                    seq,
                    kind,
                    _to_json(self._redact(payload)),
                )
        except Exception:  # noqa: BLE001
            logger.exception("add_event failed (rid=%s seq=%s kind=%s)", request_id, seq, kind)

    async def finalize_request(
        self,
        request_id: uuid.UUID,
        *,
        outbound_body: Any = None,
        provider_response: Any = None,
        canonical_response: Any = None,
        outbound_body_returned: Any = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        stop_reason: str | None = None,
        error: str | None = None,
    ) -> None:
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE requests SET
                        outbound_body = $2::jsonb,
                        provider_response = $3::jsonb,
                        canonical_response = $4::jsonb,
                        outbound_body_returned = $5::jsonb,
                        input_tokens = $6,
                        output_tokens = $7,
                        stop_reason = $8,
                        error = $9,
                        finished_at = NOW(),
                        latency_ms = (EXTRACT(EPOCH FROM (NOW() - started_at)) * 1000)::int
                    WHERE id = $1
                    """,
                    request_id,
                    _to_json(self._redact(outbound_body)),
                    _to_json(self._redact(provider_response)),
                    _to_json(self._redact(canonical_response)),
                    _to_json(self._redact(outbound_body_returned)),
                    input_tokens,
                    output_tokens,
                    stop_reason,
                    error,
                )
        finally:
            self._seq.pop(request_id, None)

    async def list_requests(self, *, limit: int, offset: int) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, ts, client_request_id, inbound_protocol, provider,
                       provider_model_requested, input_tokens, output_tokens,
                       stop_reason, error, latency_ms
                FROM requests
                ORDER BY ts DESC
                LIMIT $1 OFFSET $2
                """,
                limit,
                offset,
            )
        return [
            {
                "id": str(r["id"]),
                "ts": r["ts"].isoformat() if r["ts"] else None,
                "client_request_id": r["client_request_id"],
                "inbound_protocol": r["inbound_protocol"],
                "provider": r["provider"],
                "provider_model_requested": r["provider_model_requested"],
                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
                "stop_reason": r["stop_reason"],
                "error": r["error"],
                "latency_ms": r["latency_ms"],
            }
            for r in rows
        ]

    async def get_request(self, request_id: uuid.UUID) -> dict[str, Any] | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM requests WHERE id = $1", request_id)
            if row is None:
                return None
            events = await conn.fetch(
                "SELECT seq, ts, kind, payload FROM request_events WHERE request_id = $1 ORDER BY seq ASC",
                request_id,
            )
        detail = {k: (v.isoformat() if isinstance(v, dt.datetime) else v) for k, v in row.items()}
        detail["id"] = str(row["id"])
        detail["inbound_body"] = _from_json(row["inbound_body"])
        detail["canonical_request"] = _from_json(row["canonical_request"])
        detail["outbound_body"] = _from_json(row["outbound_body"])
        detail["provider_response"] = _from_json(row["provider_response"])
        detail["canonical_response"] = _from_json(row["canonical_response"])
        detail["outbound_body_returned"] = _from_json(row["outbound_body_returned"])
        detail["events"] = [
            {
                "seq": ev["seq"],
                "ts": ev["ts"].isoformat() if ev["ts"] else None,
                "kind": ev["kind"],
                "payload": _from_json(ev["payload"]),
            }
            for ev in events
        ]
        return detail


def _to_json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, default=_json_default, ensure_ascii=False)


def _from_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value
    return value


def _json_default(o: Any) -> Any:
    if hasattr(o, "model_dump"):
        return o.model_dump()
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    if isinstance(o, (uuid.UUID,)):
        return str(o)
    if isinstance(o, dt.datetime):
        return o.isoformat()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON-serializable")
