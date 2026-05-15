"""Translate an OpenAI chat-completions response back to Anthropic Messages shape."""

from __future__ import annotations

import json
import uuid
from typing import Any

_FINISH_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "stop_sequence",
}


def map_finish_reason(finish_reason: str | None) -> str | None:
    if finish_reason is None:
        return None
    return _FINISH_REASON_MAP.get(finish_reason, "end_turn")


def _safe_parse_arguments(args: Any) -> dict[str, Any]:
    if args is None:
        return {}
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        if not args.strip():
            return {}
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                return parsed
            return {"value": parsed}
        except json.JSONDecodeError:
            # Malformed JSON should not 500 — surface a structured fallback so the
            # caller at least sees the raw payload and can recover.
            return {"_unparsed_arguments": args}
    return {"value": args}


def translate_response(
    openai_response: dict[str, Any],
    *,
    anthropic_model: str,
) -> dict[str, Any]:
    """Build the Anthropic /v1/messages response dict from an OpenAI completion dict.

    ``anthropic_model`` is the value to echo back in the response's ``model`` field;
    callers should pass the Anthropic-side id the request asked for, NOT the
    provider-side id (Claude Code uses this to confirm its request was routed).
    """

    choices = openai_response.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    finish_reason = choice.get("finish_reason")

    content_blocks: list[dict[str, Any]] = []
    text = message.get("content")
    if isinstance(text, str) and text:
        content_blocks.append({"type": "text", "text": text})
    elif isinstance(text, list):
        for part in text:
            if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                content_blocks.append({"type": "text", "text": part["text"]})

    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        content_blocks.append(
            {
                "type": "tool_use",
                "id": call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                "name": fn.get("name") or "",
                "input": _safe_parse_arguments(fn.get("arguments")),
            }
        )

    # Anthropic always returns at least one content block — empty content + tool-only
    # responses are fine, but if literally nothing came back we emit an empty text block
    # so the SDK doesn't blow up on `content[0]`.
    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    usage = openai_response.get("usage") or {}
    response_id = openai_response.get("id") or f"msg_{uuid.uuid4().hex[:24]}"
    if not response_id.startswith("msg_"):
        response_id = f"msg_{response_id}"

    return {
        "id": response_id,
        "type": "message",
        "role": "assistant",
        "model": anthropic_model,
        "content": content_blocks,
        "stop_reason": map_finish_reason(finish_reason),
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
        },
    }
