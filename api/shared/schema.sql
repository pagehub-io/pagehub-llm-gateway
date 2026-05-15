-- pagehub-llm-gateway — applied idempotently on lifespan boot under an advisory lock.
-- Two tables: one summary row per request, one row per intermediate event.

CREATE TABLE IF NOT EXISTS requests (
    id                          UUID PRIMARY KEY,
    ts                          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    client_request_id           TEXT,
    inbound_protocol            TEXT NOT NULL,
    inbound_body                JSONB,
    canonical_request           JSONB,
    provider                    TEXT NOT NULL,
    provider_model_requested    TEXT,
    outbound_body               JSONB,
    provider_response           JSONB,
    canonical_response          JSONB,
    outbound_body_returned      JSONB,
    input_tokens                INTEGER,
    output_tokens               INTEGER,
    stop_reason                 TEXT,
    error                       TEXT,
    started_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at                 TIMESTAMPTZ,
    latency_ms                  INTEGER
);

CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests (ts DESC);
CREATE INDEX IF NOT EXISTS idx_requests_provider ON requests (provider);
CREATE INDEX IF NOT EXISTS idx_requests_client_request_id ON requests (client_request_id);

CREATE TABLE IF NOT EXISTS request_events (
    id          BIGSERIAL PRIMARY KEY,
    request_id  UUID NOT NULL REFERENCES requests(id) ON DELETE CASCADE,
    ts          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    seq         INTEGER NOT NULL,
    kind        TEXT NOT NULL,
    payload     JSONB
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_request_events_request_id_seq
    ON request_events (request_id, seq);
