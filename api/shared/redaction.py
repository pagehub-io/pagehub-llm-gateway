"""Content redaction for request logging.

When ``LOG_CONTENT=false``, drop all caller-controlled text/tool-input/tool-result
content from logged payloads — keep schema, model, token counts, stop_reason,
tool names, tool_call ids, and timing.

We don't try to be clever: we walk the JSON tree and redact any string at the
known "content-carrying" keys, regardless of where in the document it appears.
The same logic applies whether the payload is an Anthropic body, an xAI body, a
CanonicalRequest, or a canonical stream event — they all use overlapping key
names for content.
"""

from __future__ import annotations

from typing import Any

# Keys whose STRING values are caller-content and should be redacted.
_CONTENT_STRING_KEYS = {
    "text",
    "data",  # base64 image data
    "partial_json",  # input_json_delta partials
    "partial_input_json",  # canonical tool_call_delta partials
    "arguments",  # OpenAI assistant tool_call.function.arguments (JSON-encoded string)
    "thinking",  # anthropic thinking_delta text
}

# Keys whose values may be dicts/lists with content nested inside but ARE NOT
# themselves the leaf content. We descend into these without redacting them.
# (Just here as documentation — descent is the default.)


def redact(value: Any, *, enabled: bool) -> Any:
    """Return ``value`` with caller-content redacted iff ``enabled`` is True.

    Pure / non-mutating: returns a new dict/list/tuple where any node was rewritten.
    """
    if not enabled:
        return value
    return _walk(value, parent_key=None)


def _walk(value: Any, *, parent_key: str | None) -> Any:
    if isinstance(value, dict):
        return _walk_dict(value)
    if isinstance(value, list):
        return [_walk(v, parent_key=parent_key) for v in value]
    if isinstance(value, str) and parent_key in _CONTENT_STRING_KEYS:
        return {"redacted": True, "chars": len(value)}
    return value


def _walk_dict(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    block_type = d.get("type") if isinstance(d, dict) else None
    for k, v in d.items():
        # Tool input is a dict of caller-supplied keys/values; redact the whole input
        # for tool_call / tool_use blocks but KEEP the tool name + id which live as
        # siblings.
        if k == "input" and block_type in {"tool_call", "tool_use"} and isinstance(v, dict):
            out[k] = {"redacted": True, "keys": sorted(v.keys())}
            continue
        # Anthropic tool_result has `content` that's either a string OR a list of
        # blocks. Either way, redact the content but keep the block structure intact
        # so we still see whether it was a list-of-text vs. plain string.
        if k == "content" and block_type == "tool_result":
            if isinstance(v, str):
                out[k] = {"redacted": True, "chars": len(v)}
            elif isinstance(v, list):
                out[k] = [{"redacted_block": True, "type": (x or {}).get("type")} for x in v]
            else:
                out[k] = v
            continue
        # OpenAI role:"tool" message: `content` is a plain string. Redact.
        if k == "content" and d.get("role") == "tool" and isinstance(v, str):
            out[k] = {"redacted": True, "chars": len(v)}
            continue
        # System prompt as plain string -> redact. As list-of-blocks -> walk normally
        # (the inner text/* leaves get redacted by the leaf rule).
        if k == "system" and isinstance(v, str):
            out[k] = {"redacted": True, "chars": len(v)}
            continue
        # Anthropic user message `content` as plain string -> redact.
        if (
            k == "content"
            and d.get("role") in {"user", "assistant"}
            and isinstance(v, str)
        ):
            out[k] = {"redacted": True, "chars": len(v)}
            continue
        # OpenAI user-message content as plain string -> redact.
        if k == "content" and isinstance(v, str) and d.get("role") in {"user", "assistant", "system"}:
            out[k] = {"redacted": True, "chars": len(v)}
            continue
        out[k] = _walk(v, parent_key=k)
    return out
