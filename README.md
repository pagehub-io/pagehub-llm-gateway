# pagehub-llm-gateway

Anthropic-compatible HTTP gateway that routes Claude-style `/v1/messages` calls
to non-Anthropic backends — via a canonical, provider-neutral middle layer.

Point Claude Code's `ANTHROPIC_BASE_URL` + `ANTHROPIC_AUTH_TOKEN` at this
gateway and the CLI thinks it's talking to Anthropic. Under the hood every
request is decoded into a `CanonicalRequest`, dispatched to whichever provider
adapter claims the canonical `model` id, and the response is decoded back
into a `CanonicalResponse` before being re-encoded to the inbound protocol.

**Backends today:** xAI Grok (`grok-*`) and OpenAI (`gpt-*`, `o1-*`, `o3-*`,
`o4-*`). Both share the OpenAI chat-completions wire format, so they sit
behind a common `OpenAICompatibleProvider` base — adding a third
OpenAI-compatible provider (Groq, Fireworks, Together, OpenRouter, ...) is
typically a one-file specialization.

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
    base.py                  # ProviderAdapter protocol
    openai_compatible.py     # OpenAICompatibleProvider base — all the Canonical
                             #   <-> /v1/chat/completions translation lives here
                             #   (encode_request, decode_response, decode_stream
                             #    + httpx transport)
    xai.py                   # XAIProvider — thin specialization (model_name_prefixes,
                             #   twin_override_env_var, settings defaults)
    openai.py                # OpenAIProvider — same shape, OpenAI bindings
    registry.py              # ProviderRegistry — model id -> provider dispatch
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

| Inbound (this gateway, Anthropic shape)            | Outbound (xAI / OpenAI / ...)                                            |
|----------------------------------------------------|--------------------------------------------------------------------------|
| `POST /v1/messages` (non-stream)                   | `POST /chat/completions` (`stream: false`) — provider picked by `model`  |
| `POST /v1/messages` (`Accept: text/event-stream`)  | `POST /chat/completions` (`stream: true`) — provider picked by `model`   |
| `GET /v1/models`                                   | (static — returns each registered provider's default model id)           |
| `GET /v1/admin/requests` (admin token)             | (local DB)                                                               |
| `GET /v1/admin/requests/{id}` (admin token)        | (local DB)                                                               |
| `GET /health`, `GET /metrics`                      | (local — `/health` lists every registered provider + its claim prefixes) |

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
# 1. Drop secrets in your shell (~/.bashrc or ~/.zshrc). The gateway tolerates
#    an unset key for any backend you don't intend to use — the provider just
#    raises 500 if/when called.
export XAI_API_KEY=xai-...
export OPENAI_API_KEY=sk-...     # optional, only if you'll call gpt-*/o*
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

# `--model grok-*` routes to xAI; `--model gpt-*` / `o1*` / `o3*` / `o4*` routes
# to OpenAI. Each provider passes a claimed id straight through and falls back
# to its own configured default for unclaimed ids. An id no provider claims
# (e.g. `claude-opus-4-7`) returns a 400 listing the available providers.
claude -p "create hello.py that prints hi" \
  --model grok-4 \
  --output-format json \
  --dangerously-skip-permissions
```

If Claude Code completes the tool dance (Read / Write / Bash all flowed
through the gateway), `hello.py` exists and the JSON output reports nonzero
`usage.input_tokens` / `usage.output_tokens`.

### Picking a model id

The gateway routes by the canonical `model` field. Each provider claims a
small set of prefixes and falls back to its configured default for anything
it claims but doesn't recognize.

**xAI (`grok-*`).** xAI ships fast — check
[the model list](https://docs.x.ai/docs/models) for the current top.
Recent ids:
- `grok-4`, `grok-4-latest`, `grok-4.3`, `grok-4.20-*` reasoning / non-reasoning
- `grok-4-1-fast-reasoning`, `grok-3`, `grok-beta` (older)

**OpenAI (`gpt-*`, `o1-*`, `o3-*`, `o4-*`).** Check
[the OpenAI models endpoint](https://platform.openai.com/docs/models). Recent
mainline tool-use ids: `gpt-5`, `gpt-5-mini`, `gpt-4o`, `o3-mini`, `o4-mini`.

An id no provider claims returns a 400 from the gateway with the available
providers' prefixes — that's the routing diagnostic.

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

- **No outbound `cache_control`.** Neither xAI nor OpenAI's chat-completions
  endpoint exposes a prompt-cache knob in the same shape Anthropic does.
  Inbound `cache_control` is accepted and silently dropped. (OpenAI ships
  automatic prompt caching for repeated prefixes — that's transparent and
  needs no `cache_control` to engage.)
- **No extended thinking forwarded.** `thinking: {...}` and `type: "thinking"`
  blocks are accepted on inbound (so prior assistant turns round-trip) and
  preserved in `CanonicalRequest.metadata.anthropic_thinking`; they're not
  forwarded to the provider. OpenAI's o-series and GPT-5 do their own
  reasoning internally; reasoning tokens come back in
  `usage.completion_tokens` and are propagated to canonical `output_tokens`.
- **Routing is one-pass, first claimant wins.** The gateway's registry walks
  registered providers in declaration order. If a model id matches multiple
  providers, the first-registered one wins — see `api/main.py::_build_registry`.

## End-to-end verification

Tests in CI use scripted providers — they never hit xAI or OpenAI. Before
declaring a deploy healthy, run the manual smoke against real keys. Two
recipes, one per provider; both exercise the same canonical pipeline so
running both gives you a routing-correctness diagnostic too.

### Provider A — xAI Grok

```bash
# In one terminal:
cd ~/github/pagehub-io/pagehub-llm-gateway
export XAI_API_KEY=xai-...
export GATEWAY_AUTH_TOKEN=local-dev-token
export ADMIN_AUTH_TOKEN=local-dev-admin-token
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
```

### Provider B — OpenAI (the diagnostic second provider)

```bash
# Same gateway, different backend. `OPENAI_API_KEY` is required; the gateway
# routes to OpenAI when the model id starts with `gpt`, `o1`, `o3`, or `o4`.
cd ~/github/pagehub-io/pagehub-llm-gateway
export OPENAI_API_KEY=sk-...
export GATEWAY_AUTH_TOKEN=local-dev-token
export ADMIN_AUTH_TOKEN=local-dev-admin-token
make up      # postgres + gateway (no-op if already running)

cd $(mktemp -d) && rm -f hello.py
ANTHROPIC_BASE_URL=http://localhost:4011 \
ANTHROPIC_AUTH_TOKEN=$GATEWAY_AUTH_TOKEN \
claude -p "create hello.py that prints hi" \
  --model gpt-5 \
  --output-format json \
  --dangerously-skip-permissions
cat hello.py
```

### Inspect what was logged (both)

```bash
curl -sS http://localhost:4011/v1/admin/requests \
  -H "x-api-key: $ADMIN_AUTH_TOKEN" | jq '.items[0:5]'
curl -sS http://localhost:4011/v1/admin/requests/<id> \
  -H "x-api-key: $ADMIN_AUTH_TOKEN" | jq
```

**Acceptance bar (each recipe):** the target file exists with `print` in it,
the JSON output reports nonzero `usage.input_tokens` / `usage.output_tokens`,
and `/v1/admin/requests` shows one summary row per turn with the full
per-event timeline. The `requests.provider` column distinguishes which
backend served each turn — useful for a quick "did routing work" check.

**Diagnostic value.** If a Grok smoke produces weak output (poor tool use,
spec-noncompliance, etc.), running the OpenAI recipe on a known-good tool-use
model (`gpt-5`, `o3-mini`, ...) tells you whether the gateway's canonical
translation is correct: working OpenAI + broken Grok means Grok is the weak
link, not the gateway. Both failing in the same shape would point at the
canonical translation layer.

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
containing `XAI_API_KEY` (if using Grok), `OPENAI_API_KEY` (if using OpenAI),
`GATEWAY_AUTH_TOKEN`, `ADMIN_AUTH_TOKEN`, and a
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
