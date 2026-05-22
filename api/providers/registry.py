"""Provider routing — pick the right :class:`ProviderAdapter` for a canonical
model id.

The gateway holds one engine, but multiple provider adapters. A request comes
in with a canonical ``model`` field; the registry walks the registered
providers in order and returns the first one whose ``claims_model(model)`` is
true. Unknown models raise a 400-level :class:`ProviderError` with a message
listing what each registered provider claims, so the caller can tell why their
model id didn't route.

Design choice: a simple in-order list, not a richer dispatch tree. Adding a
new provider means appending one entry in ``api/main.py``; we keep the
routing rule visible there rather than hidden behind a registration
side-effect.
"""

from __future__ import annotations

from typing import Sequence

from api.providers.base import ProviderAdapter, ProviderError


class ProviderRegistry:
    """Routes a canonical model id to the right :class:`ProviderAdapter`."""

    def __init__(self, providers: Sequence[ProviderAdapter]) -> None:
        if not providers:
            raise ValueError("ProviderRegistry requires at least one provider")
        self._providers: list[ProviderAdapter] = list(providers)

    @property
    def providers(self) -> list[ProviderAdapter]:
        return list(self._providers)

    def pick(self, canonical_model: str) -> ProviderAdapter:
        for p in self._providers:
            claims = getattr(p, "claims_model", None)
            if claims is not None and claims(canonical_model):
                return p
        raise ProviderError(
            status_code=400,
            message=(
                f"model {canonical_model!r} not handled by any registered provider; "
                f"available providers: {self._describe_providers()}"
            ),
        )

    def _describe_providers(self) -> str:
        parts: list[str] = []
        for p in self._providers:
            prefixes = getattr(p, "model_name_prefixes", ()) or ()
            claim_str = "/".join(f"{pre}-*" for pre in prefixes) if prefixes else "(no prefixes)"
            parts.append(f"{getattr(p, 'name', type(p).__name__)} for {claim_str}")
        return ", ".join(parts)
