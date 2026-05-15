"""Modal deployment stub for pagehub-llm-gateway.

NOT auto-deployed by CI — this file is the operator handoff. To actually deploy:

    modal deploy modal_app.py

The deployer is responsible for provisioning the runtime secrets:
``XAI_API_KEY`` and ``GATEWAY_AUTH_TOKEN``. Don't ship without them — the gateway
will reject every request otherwise.
"""

from __future__ import annotations

import subprocess

import modal


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:  # noqa: BLE001 - deploy must not hard-fail over this
        return "unknown"


_GIT_COMMIT = _git_commit()

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi==0.136.1",
        "uvicorn[standard]==0.34.0",
        "pydantic==2.10.4",
        "pydantic-settings==2.7.0",
        "httpx==0.27.2",
    )
    .env({"GIT_COMMIT": _GIT_COMMIT, "ENV": "production"})
    .add_local_dir("api", remote_path="/root/api")
)

app = modal.App("pagehub-llm-gateway", image=image)

MODAL_FUNCTION_TIMEOUT = 700  # >= PROVIDER_TIMEOUT_SECONDS + headroom for slow Grok responses
CONCURRENT_INPUTS = 32


@app.function(
    timeout=MODAL_FUNCTION_TIMEOUT,
    secrets=[modal.Secret.from_name("pagehub-llm-gateway")],
)
@modal.concurrent(max_inputs=CONCURRENT_INPUTS)
@modal.asgi_app()
def fastapi_app():
    from api.main import create_app

    return create_app()
