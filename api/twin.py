"""Per-request twin-override store.

Mechanics match the global standard (CLAUDE.md): inbound `X-Twin-{Dep}-Base-Url`
header value is the full URL to use for that dependency for the lifetime of the
request. Middleware populates the contextvar; outbound HTTP helpers read it before
falling back to the env-var default. Only active when settings.env == "development".
"""

from __future__ import annotations

from contextvars import ContextVar

_TWIN_OVERRIDES: ContextVar[dict[str, str] | None] = ContextVar("twin_overrides", default=None)


def set_overrides(overrides: dict[str, str]) -> None:
    _TWIN_OVERRIDES.set(overrides)


def get_override(env_var_name: str) -> str | None:
    """Look up the twin override for an env-var name (e.g. ``GROK_BASE_URL``)."""
    overrides = _TWIN_OVERRIDES.get()
    if not overrides:
        return None
    return overrides.get(env_var_name)


def reset() -> None:
    _TWIN_OVERRIDES.set(None)


def header_to_env_var(header_name: str) -> str | None:
    """Map ``X-Twin-Grok-Base-Url`` -> ``GROK_BASE_URL``. Returns None if not an X-Twin- header."""
    lower = header_name.lower()
    if not lower.startswith("x-twin-"):
        return None
    rest = lower[len("x-twin-") :]
    return rest.replace("-", "_").upper()
