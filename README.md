# pagehub-llm-gateway

Anthropic-compatible HTTP gateway that lets [Claude Code](https://claude.com/claude-code) (and any
other Anthropic-API client) talk to **non-Anthropic** model providers.

Point `ANTHROPIC_BASE_URL` + `ANTHROPIC_AUTH_TOKEN` at this gateway and the CLI
thinks it's talking to Anthropic. Under the hood, every `/v1/messages` call is
translated to the target provider's wire format, dispatched, and the response is
translated back to Anthropic shape — including streaming SSE and tool-use round
trips.

**v1 backend:** [xAI Grok](https://docs.x.ai/) (OpenAI-compatible
`/v1/chat/completions`). New providers slot in behind a `Provider` Protocol
(`api/providers/base.py`).

## What it speaks

| Inbound (this gateway, Anthropic shape)           | Outbound (xAI Grok, OpenAI shape)                |
|----------------------------------------------------|--------------------------------------------------|
| `POST /v1/messages` (non-stream)                   | `POST /chat/completions` (`stream: false`)       |
| `POST /v1/messages` (`Accept: text/event-stream`)  | `POST /chat/completions` (`stream: true`)        |
| `GET /v1/models`                                   | (static — returns the configured Grok model id)  |
| `GET /health`, `GET /metrics`                      | (local)                                          |

Auth: every request must present **either** `x-api-key: <token>` **or**
`Authorization: Bearer <token>`, where `<token>` matches the gateway's
`GATEWAY_AUTH_TOKEN` env var. The Anthropic Python/TS SDKs send `x-api-key`;
Claude Code (via `ANTHROPIC_AUTH_TOKEN`) sends Bearer. Both work.

## Translation notes

**Request mapping (Anthropic → OpenAI):**
- `system` (string or list-of-text-blocks) → leading `role: "system"` message
- User text/image content blocks → OpenAI user message (`content` string or content-parts)
- User `tool_result` blocks → standalone `role: "tool"` messages keyed by `tool_call_id`
- Assistant `tool_use` blocks → assistant message with `tool_calls[]`
- `tools[]` (`name`, `description`, `input_schema`) → `tools[]` (`function.{name, description, parameters}`)
- `tool_choice`: `"auto"` → `"auto"`, `"any"` → `"required"`, `{type: "tool", name}` → `{type: "function", function: {name}}`
- `stop_sequences` → `stop`

**Response mapping (OpenAI → Anthropic):**
- `choices[0].message.content` → `content[].type: "text"`
- `choices[0].message.tool_calls[]` → `content[].type: "tool_use"` with parsed `input`
- `usage.prompt_tokens` / `completion_tokens` → `usage.input_tokens` / `output_tokens`
- `finish_reason` map: `stop`→`end_turn`, `length`→`max_tokens`, `tool_calls`→`tool_use`, `content_filter`→`stop_sequence`

**Streaming:** OpenAI's `data: {choices: [{delta: ...}]}` chunks are reassembled into the
Anthropic event sequence — `message_start` → (per content block: `content_block_start` →
`content_block_delta` (`text_delta` or `input_json_delta`) → `content_block_stop`) →
`message_delta` (final usage + stop_reason) → `message_stop`. Tool-call argument deltas
keyed by OpenAI `index` get assembled into a single `tool_use` block with streaming
`input_json_delta` partials. We request `stream_options.include_usage: true` from xAI so
the final `message_delta.usage.output_tokens` reflects the provider's real count.

## Known limits

- **No `cache_control` outbound.** Grok has no prompt-cache equivalent; the gateway
  accepts `cache_control` on inbound blocks and silently drops it.
- **No extended thinking.** Grok has no `thinking` block equivalent. The gateway accepts
  `thinking: {...}` and `type: "thinking"` content blocks on inbound (so a caller can replay
  prior assistant turns without rewriting) and drops them outbound.
- **Single backend per gateway instance.** v1 routes everything to Grok. Multi-provider
  routing (by model id, by tenant, etc.) is a follow-on slice.
- **Token counts are provider-reported.** The gateway estimates input tokens cheaply
  (chars/4) only for the streaming `message_start` placeholder; the authoritative count
  lives in the final `message_delta.usage`.

## Run locally

```bash
cp .env.example .env
# Set GATEWAY_AUTH_TOKEN (any string) and XAI_API_KEY in .env.
make install   # pip install -e ".[dev]"
make dev       # uvicorn on :4011
# OR
make up        # docker-compose
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

## Point Claude Code at it

```bash
export ANTHROPIC_BASE_URL=http://localhost:4011
export ANTHROPIC_AUTH_TOKEN=<your GATEWAY_AUTH_TOKEN>

# Pass an explicit Grok model — anything not starting with `grok` falls back to
# GROK_DEFAULT_MODEL on the gateway side.
claude -p "create hello.py that prints hi" \
  --model grok-4 \
  --output-format json \
  --dangerously-skip-permissions
```

If Claude Code completes the tool dance (Read / Write / Bash all flowed through the
gateway), `hello.py` will exist in the current directory and the JSON output will report
nonzero `usage.input_tokens` / `usage.output_tokens`.

### Picking a Grok model id

xAI ships fast — check [the model list](https://docs.x.ai/docs/models) for the current
top model. Known ids at time of writing (set `GROK_DEFAULT_MODEL` accordingly):

- `grok-4` (default in `.env.example`)
- `grok-4-latest`
- `grok-4-1-fast-reasoning`, `grok-4-1-fast-non-reasoning`
- `grok-3`, `grok-beta` (older)

The gateway passes any `grok*` model id straight through to xAI; ids that don't start
with `grok` (Claude Code's SDK default, for instance) fall back to `GROK_DEFAULT_MODEL`.

## End-to-end verification

Tests in CI use a mock provider — they never hit xAI. Before declaring a deploy
healthy, run the manual smoke. Two options for getting the secrets into the container:

**Option A — keep secrets in your shell (recommended).** `docker-compose.yml`
forwards `XAI_API_KEY` and `GATEWAY_AUTH_TOKEN` from the host shell, so:

```bash
# In ~/.bashrc (or ~/.zshrc):
export XAI_API_KEY=xai-...
export GATEWAY_AUTH_TOKEN=local-dev-token

# Then:
cd ~/github/pagehub-io/pagehub-llm-gateway
cp .env.example .env   # non-secret defaults only; do NOT add the key here
make up                # docker-compose up
```

**Option B — local uvicorn (no docker).** Put both vars in `.env` (which is
gitignored), then `make dev`.

Either way, run the smoke from a scratch dir:

```bash
cd /tmp && rm -f hello-from-grok.py
ANTHROPIC_BASE_URL=http://localhost:4011 \
ANTHROPIC_AUTH_TOKEN=$GATEWAY_AUTH_TOKEN \
claude -p "create /tmp/hello-from-grok.py that prints hi" \
  --model grok-4 \
  --output-format json \
  --dangerously-skip-permissions
cat /tmp/hello-from-grok.py
```

The acceptance bar: `hello-from-grok.py` exists with reasonable contents, the JSON
output reports nonzero token usage, and the gateway's `/metrics` shows the increment.

## Tests

```bash
make test    # pytest — uses a fake in-process Provider; no real xAI calls
make lint    # ruff
```

CI runs both on every push.

## Deploy (Modal)

`modal_app.py` is a stub — running `modal deploy modal_app.py` will work, but the
operator must first provision the `pagehub-llm-gateway` Modal Secret containing
`XAI_API_KEY` and `GATEWAY_AUTH_TOKEN`. Without those env vars set, every inbound
request will be rejected. CI does **not** deploy this slice.

## Twin overrides (dev only)

Per the pagehub global standard, dependency base URLs are overridable per-request via
`X-Twin-*` headers — but **only when `ENV=development`**. Example:

```bash
# Point a single request at a local Grok-shaped fake without changing env vars:
curl -sS http://localhost:4011/v1/messages \
  -H "x-api-key: $GATEWAY_AUTH_TOKEN" \
  -H "X-Twin-Grok-Base-Url: http://localhost:5050/v1" \
  -d '...'
```

In any non-`development` env the middleware silently strips `X-Twin-*` headers before any
handler sees them. Header values are never logged.
