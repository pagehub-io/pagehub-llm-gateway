"""Model-name → provider dispatch tests.

The gateway runs one engine with multiple provider adapters. A request arrives
with a canonical ``model`` id; the registry walks its providers in order and
returns the first one whose ``claims_model(model)`` is true. Unknown models
turn into a 400 with a message listing the available providers — these tests
pin both halves.
"""

from __future__ import annotations

import pytest

from api.providers.base import ProviderError
from api.providers.openai import OpenAIProvider
from api.providers.registry import ProviderRegistry
from api.providers.xai import XAIProvider


def _registry() -> ProviderRegistry:
    # Production-shape wiring: xAI first, OpenAI second. Order is the public
    # contract — the first claimant wins — so we exercise it that way too.
    return ProviderRegistry(
        [
            XAIProvider(api_key="xai-k", base_url="http://xai/v1", default_model="grok-4"),
            OpenAIProvider(
                api_key="oai-k", base_url="http://openai/v1", default_model="gpt-5"
            ),
        ]
    )


def test_grok_routes_to_xai():
    reg = _registry()
    p = reg.pick("grok-4")
    assert isinstance(p, XAIProvider)


def test_grok_variant_routes_to_xai():
    reg = _registry()
    p = reg.pick("grok-4.3")
    assert isinstance(p, XAIProvider)


def test_gpt5_routes_to_openai():
    reg = _registry()
    p = reg.pick("gpt-5")
    assert isinstance(p, OpenAIProvider)


def test_o_series_routes_to_openai():
    reg = _registry()
    for mid in ("o1", "o1-pro", "o3-mini", "o4-mini"):
        p = reg.pick(mid)
        assert isinstance(p, OpenAIProvider), f"{mid} routed to {type(p).__name__}"


def test_gpt_variant_routes_to_openai():
    reg = _registry()
    for mid in ("gpt-4o", "gpt-4o-mini", "gpt-5-mini", "gpt-5-thinking"):
        p = reg.pick(mid)
        assert isinstance(p, OpenAIProvider), f"{mid} routed to {type(p).__name__}"


def test_unknown_model_400_lists_available_providers():
    reg = _registry()
    with pytest.raises(ProviderError) as ei:
        reg.pick("claude-opus-4-7")
    assert ei.value.status_code == 400
    msg = ei.value.message
    assert "claude-opus-4-7" in msg
    # The error message must enumerate what's available so the caller can
    # diagnose without reading source — protect that contract.
    assert "xai" in msg.lower()
    assert "openai" in msg.lower()
    assert "grok" in msg.lower()
    assert "gpt" in msg.lower()


def test_empty_model_id_400():
    reg = _registry()
    with pytest.raises(ProviderError) as ei:
        reg.pick("")
    assert ei.value.status_code == 400


def test_empty_registry_disallowed():
    """A registry with no providers cannot serve traffic — fail fast at
    construction time, not on the first request."""
    with pytest.raises(ValueError):
        ProviderRegistry([])


def test_pick_walks_in_order_first_claimant_wins():
    """If two providers both claim a model id (hypothetical conflict), the
    earlier one wins. Pin this so adding a third provider later doesn't
    silently steal routing from an established one."""
    # Build a one-off scenario: a fake provider that claims everything,
    # registered AHEAD of the real OpenAIProvider.
    class _ClaimAll(OpenAIProvider):
        name = "fake-claim-all"
        model_name_prefixes = ("",)  # empty prefix matches anything

    fake = _ClaimAll(api_key="k", base_url="x", default_model="any")
    reg = ProviderRegistry(
        [
            fake,
            OpenAIProvider(api_key="k", base_url="x", default_model="gpt-5"),
        ]
    )
    assert reg.pick("gpt-5") is fake


def test_providers_property_returns_a_copy():
    """Defensive: the registry's internal list must not leak — mutating the
    returned list shouldn't change routing."""
    reg = _registry()
    snapshot = reg.providers
    snapshot.clear()
    # routing still works after caller mutates the returned list
    assert isinstance(reg.pick("grok-4"), XAIProvider)
