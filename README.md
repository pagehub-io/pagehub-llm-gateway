# pagehub-llm-gateway

Anthropic-compatible HTTP gateway that routes Claude-style `/v1/messages` calls
to non-Anthropic backends — via a canonical, provider-neutral middle layer.

Point Claude Code's `ANTHROPIC_BASE_URL` + `ANTHROPIC_AUTH_TOKEN` at this
gateway and the CLI thinks it's talking to Anthropic. Under the hood every
request is decoded into a `CanonicalRequest`, dispatched to whichever provider
adapter is registered (v1: xAI Grok), and the response is decoded back into a
`CanonicalResponse` before being re-encoded to the inbound protocol.

## Architecture: hub-and-spoke, not pairwise

```
   inbound HTTP (Anthropic-shaped)
         │
         ▼  AnthropicInbound.decode_request()    api/protocols/anthropic.py
   CanonicalRequest                              ───── persisted to `requests` ──┐
         │                                                                       │
         ▼  ProviderAdapter.encode_request()     api/providers/xai.py            │
   outbound HTTP (xAI / OpenAI-shaped)                                           │
         │                                                                       │
         ▼  external HTTPS call to provider                                      │
   provider response                                                             │
         │                                                                       │
         ▼  ProviderAdapter.decode_response()                                    │
   CanonicalResponse                             ───── persisted to `requests` ──┤
         │                                                                       │
         ▼  AnthropicInbound.encode_response()                                   │
   outbound HTTP (Anthropic-shaped) → caller                                     │
                                                                                 │
   [every canonical step ─────► `request_events` row ─────────────────────────► ]
```

**Why hub-and-spoke.** With N inbound protocols and M providers, pairwise
translators cost N×M implementations. A canonical middle layer costs N + M:
one inbound adapter per protocol, one provider adapter per backend. Adding
GPT-5 / Bedrock / Gemini / OpenRouter is then **one** new `ProviderAdapter`
that encodes `CanonicalRequest` → their shape and decodes their response →
`CanonicalResponse`. Adding an OpenAI-compatible inbound surface for other
harnesses is **one** new `Inbound` adapter. The canonical types in
`api/canonical/types.py` and `api/canonical/events.py` are the source of
truth — wire formats on either side are just encodings of canonical objects.

### Layout

```
api/
  canonical/           # source of truth — CanonicalRequest/Response/StreamEvent
    types.py
    events.py
  protocols/
    anthropic.py       # Anthropic <-> canonical  (decode_request, encode_response,
                       #                           encode_stream)
  providers/
    base.py            # ProviderAdapter protocol
    xai.py             # xAI Grok adapter        (encode_request, decode_response,
                       #                           decode_stream + httpx transport)
  shared/
    db.py              # asyncpg pool + schema bootstrap
    schema.sql         # requests + request_events tables
    redaction.py       # content-stripping for the log
    recorder.py        # Recorder protocol + PostgresRecorder + MemoryRecorder
  engine.py            # orchestrator: runs the pipeline + records events
  v1/
    messages_router.py # POST /v1/messages
    admin_router.py    # GET  /v1/admin/requests, /v1/admin/requests/{id}
    service_router.py  # /health, /metrics
  middleware.py        # X-Twin-* dev-only gate + body-size limit
  auth.py              # require_auth (caller) + require_admin_auth (admin)
  config.py            # pydantic-settings
```

## Surface

| Inbound (this gateway, Anthropic shape)            | Outbound (xAI, OpenAI shape)                     |
|----------------------------------------------------|--------------------------------------------------|
| `POST /v1/messages` (non-stream)                   | `POST /chat/completions` (`stream: false`)       |
| `POST /v1/messages` (`Accept: text/event-stream`)  | `POST /chat/completions` (`stream: true`)        |
| `GET /v1/models`                                   | (static — returns the configured Grok model id)  |
| `GET /v1/admin/requests` (admin token)             | (local DB)                                       |
| `GET /v1/admin/requests/{id}` (admin token)        | (local DB)                                       |
| `GET /health`, `GET /metrics`                      | (local)                                          |

**Caller auth (`/v1/messages`, `/v1/models`):** `x-api-key: <token>` **or**
`Authorization: Bearer <token>` against `GATEWAY_AUTH_TOKEN`. Claude Code's
`ANTHROPIC_AUTH_TOKEN` sends Bearer; the Anthropic SDKs send `x-api-key`.
Both work.

**Admin auth (`/v1/admin/*`):** same header schemes but compared against
`ADMIN_AUTH_TOKEN` — a separate token by design, so a leaked caller token
can't read request history.

## Logging

Every request lands in two tables:

- **`requests`** — one summary row: inbound body, canonical request, outbound
  body, provider response, canonical response, outbound body returned, token
  counts, stop_reason, error, latency.
- **`request_events`** — one row per intermediate step (`inbound_received`,
  `decoded_canonical`, `provider_request`, `provider_response`,
  `decoded_canonical_response`, `encoded_outbound`, plus one `canonical_event`
  row per streamed event). Ordered by `(request_id, seq)` for replay-exact
  reconstruction.

### Redaction

When `LOG_CONTENT=false` (default for production / staging), caller-controlled
content is redacted **at record time** — you can't accidentally leak by
opening a row. Stripped:

- Text-block `text`
- Plain-string message `content`
- Tool-call `input` (replaced with `{"redacted": true, "keys": [...]}` —
  keys preserved so debugging stays possible)
- `tool_result` content
- OpenAI tool_call `function.arguments` strings
- `partial_json` / `partial_input_json` streaming deltas
- Image `data` (base64 payload)
- Extended-thinking `thinking` text

Kept regardless of the flag (essential debugging signals, non-content):

- Schema / structure of every body
- Model ids, tool names, tool_call ids
- `stop_reason`, token counts, latency
- All event timing and `kind` markers

Locally `LOG_CONTENT=true` is the default so debugging the canonical layer
is straightforward.

## Run locally

```bash
# 1. Drop secrets in your shell (~/.bashrc or ~/.zshrc):
export XAI_API_KEY=xai-...
export GATEWAY_AUTH_TOKEN=local-dev-token
export ADMIN_AUTH_TOKEN=local-dev-admin-token

# 2. Pull non-secret defaults:
cp .env.example .env

# 3. Start postgres + gateway:
make up
```

Health check:

```bash
curl -s http://localhost:4011/health | jq
```

Smoke `/v1/messages` (non-stream) via curl:

```bash
curl -sS http://localhost:4011/v1/messages \
  -H "content-type: application/json" \
  -H "x-api-key: $GATEWAY_AUTH_TOKEN" \
  -d '{
    "model": "grok-4",
    "max_tokens": 64,
    "messages": [{"role": "user", "content": "say hi in 5 words"}]
  }' | jq
```

Inspect what was logged:

```bash
curl -sS http://localhost:4011/v1/admin/requests \
  -H "x-api-key: $ADMIN_AUTH_TOKEN" | jq '.items[0]'
curl -sS http://localhost:4011/v1/admin/requests/<id> \
  -H "x-api-key: $ADMIN_AUTH_TOKEN" | jq
```

### Without docker (uvicorn only)

Set `DATABASE_URL=""` in your env (or just `unset DATABASE_URL`) and run
`make dev`. The gateway falls back to an in-memory recorder so the admin
endpoints still work — the log just doesn't survive a restart.

## Point Claude Code at it

```bash
export ANTHROPIC_BASE_URL=http://localhost:4011
export ANTHROPIC_AUTH_TOKEN=$GATEWAY_AUTH_TOKEN

# `--model grok-4` (or whatever GROK_DEFAULT_MODEL is set to) routes to xAI.
# Anything that doesn't start with `grok` falls back to GROK_DEFAULT_MODEL.
claude -p "create hello.py that prints hi" \
  --model grok-4 \
  --output-format json \
  --dangerously-skip-permissions
```

If Claude Code completes the tool dance (Read / Write / Bash all flowed
through the gateway), `hello.py` exists and the JSON output reports nonzero
`usage.input_tokens` / `usage.output_tokens`.

### Picking a Grok model id

xAI ships fast — check [the model list](https://docs.x.ai/docs/models) for
the current top model. Known ids at time of writing:

- `grok-4` (default in `.env.example`)
- `grok-4-latest`
- `grok-4-1-fast-reasoning`, `grok-4-1-fast-non-reasoning`
- `grok-3`, `grok-beta` (older)

The gateway passes any `grok*` id straight through; non-`grok` ids fall
back to `GROK_DEFAULT_MODEL`.

## Translation reference

| Anthropic                                                  | Canonical                              | xAI / OpenAI                                           |
|------------------------------------------------------------|----------------------------------------|--------------------------------------------------------|
| `system: "..."` / `system: [{type: "text", ...}]`          | `system: list[CanonicalText]`          | leading `role: "system"` message                       |
| User text block                                            | `CanonicalText`                        | user message `content` string / `content-parts`        |
| User image block                                           | `CanonicalImage`                       | user message `image_url` part                          |
| Assistant `tool_use` block                                 | `CanonicalToolCall`                    | assistant `tool_calls[]`                               |
| User `tool_result` block                                   | `CanonicalToolResult`                  | standalone `role: "tool"` message                      |
| `tools: [{name, description, input_schema}]`               | `CanonicalTool`                        | `tools: [{type: "function", function: {...}}]`        |
| `tool_choice: "auto" / "any" / {type: "tool", name}`       | `CanonicalToolChoice(mode=...)`        | `"auto" / "required" / {type:"function",function:...}` |
| `stop_sequences`                                           | `stop_sequences`                       | `stop`                                                 |
| stream `message_start → content_block_* → message_delta`   | `StreamStart → ... → StreamDone`       | OpenAI SSE `data: {choices:[{delta:...}]}` chunks      |

## Known limits

- **No outbound `cache_control`.** Grok has no prompt-cache equivalent.
  Inbound `cache_control` is accepted and silently dropped.
- **No extended thinking on Grok.** `thinking: {...}` and `type: "thinking"`
  blocks are accepted on inbound (so prior assistant turns round-trip) and
  preserved in `CanonicalRequest.metadata.anthropic_thinking`; they're not
  forwarded to xAI.
- **Single backend per gateway instance.** v1 routes everything to xAI.
  Multi-provider routing (by model id, by tenant) is a follow-on slice that
  plugs in at the `engine.run_anthropic`'s ``self.provider`` selection.

## End-to-end verification

Tests in CI use scripted providers — they never hit xAI. Before declaring a
deploy healthy, run the manual smoke against a real `XAI_API_KEY`:

```bash
# In one terminal:
cd ~/github/pagehub-io/pagehub-llm-gateway
make up      # postgres + gateway

# In another, in a scratch dir:
cd /tmp && rm -f hello-from-grok.py
ANTHROPIC_BASE_URL=http://localhost:4011 \
ANTHROPIC_AUTH_TOKEN=$GATEWAY_AUTH_TOKEN \
claude -p "create /tmp/hello-from-grok.py that prints hi" \
  --model grok-4 \
  --output-format json \
  --dangerously-skip-permissions
cat /tmp/hello-from-grok.py
# Then inspect the log:
curl -sS http://localhost:4011/v1/admin/requests \
  -H "x-api-key: $ADMIN_AUTH_TOKEN" | jq '.items[0:5]'
```

Acceptance bar: `/tmp/hello-from-grok.py` exists, the JSON output reports
nonzero `usage` tokens, and `/v1/admin/requests` shows one summary row per
turn with the full per-event timeline.

## Tests

Three independently-runnable layers:

| Layer | What it covers | Path |
|-------|----------------|------|
| (a) Canonical / engine | Engine pipeline + recording, against an `EchoProvider` that speaks canonical natively. No wire formats involved. | `tests/canonical/` |
| (b) AnthropicInbound | `decode_request` / `encode_response` / `encode_stream` against fixture bodies. No providers, no engine. | `tests/protocols/` |
| (c) XAIProvider | `encode_request` / `decode_response` / `decode_stream` + `httpx.MockTransport`. No inbound. | `tests/providers/` |
| (d) End-to-end | All four layers wired through the FastAPI app with a scripted HTTP transport. | `tests/test_e2e_app.py` |
| (e) Admin / redaction / DB | Admin endpoints, content-redaction switch, postgres integration. | `tests/test_admin_endpoints.py`, `tests/test_redaction.py`, `tests/test_db_recorder.py` |

```bash
make test       # full suite — uses scripted providers + MemoryRecorder; no real xAI; postgres optional
make lint       # ruff
```

The postgres-integration tests are marked `db` and skip when `DATABASE_URL`
is unset. CI exports a `DATABASE_URL` against the `services.postgres`
container so the full suite runs green.

## Deploy (Modal)

`modal_app.py` is a stub — running `modal deploy modal_app.py` will work, but
the operator must first provision the `pagehub-llm-gateway` Modal Secret
containing `XAI_API_KEY`, `GATEWAY_AUTH_TOKEN`, `ADMIN_AUTH_TOKEN`, and a
`DATABASE_URL` pointing at a managed Postgres (Supabase, RDS, ...). Without
those env vars the gateway will reject inbound requests OR fall back to
MemoryRecorder (which means a process restart drops the log). CI does **not**
deploy this slice.

## Twin overrides (dev only)

Per the pagehub global standard, dependency base URLs are overridable
per-request via `X-Twin-*` headers — but **only when `ENV=development`**.
Example:

```bash
curl -sS http://localhost:4011/v1/messages \
  -H "x-api-key: $GATEWAY_AUTH_TOKEN" \
  -H "X-Twin-Grok-Base-Url: http://localhost:5050/v1" \
  -d '...'
```

In any non-`development` env (`production`, `staging`, or anything new) the
middleware silently strips `X-Twin-*` before any handler sees them. Header
values are never logged.
