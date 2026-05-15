"""Translate an Anthropic POST /v1/messages request body to an OpenAI chat-completions body.

xAI Grok speaks the OpenAI chat-completions shape, so the two together effectively
form Anthropic -> Grok. Stays deliberately small: no defensive transformations, no
re-validation of inputs the inbound Pydantic model already validated.
"""

from __future__ import annotations

import json
from typing import Any

from api.v1.schemas import MessagesRequest


def _system_to_string(system: str | list[dict[str, Any]] | None) -> str | None:
    if system is None:
        return None
    if isinstance(system, str):
        return system or None
    parts: list[str] = []
    for block in system:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text") or "")
    text = "\n".join(p for p in parts if p)
    return text or None


def _content_blocks(raw_content: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize an Anthropic message ``content`` field into a list of block dicts."""
    if isinstance(raw_content, str):
        return [{"type": "text", "text": raw_content}]
    return raw_content or []


def _image_block_to_openai(block: dict[str, Any]) -> dict[str, Any] | None:
    source = block.get("source") or {}
    src_type = source.get("type")
    if src_type == "url":
        url = source.get("url")
        if not url:
            return None
        return {"type": "image_url", "image_url": {"url": url}}
    if src_type == "base64":
        data = source.get("data")
        media_type = source.get("media_type") or "image/png"
        if not data:
            return None
        return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}
    return None


def _tool_result_content_to_string(content: str | list[dict[str, Any]]) -> str:
    """OpenAI's `role:"tool"` `content` is plain text. Flatten Anthropic tool_result blocks."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            parts.append(block.get("text") or "")
        else:
            # Images / other block types inside tool_result get dropped to text;
            # Grok wouldn't see them anyway.
            parts.append(json.dumps(block, ensure_ascii=False))
    return "\n".join(p for p in parts if p)


def _convert_user_message(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A user-role Anthropic message may contain a mix of text/image blocks AND tool_result
    blocks. Tool results become standalone OpenAI ``role:"tool"`` messages — text/image
    blocks become a single ``role:"user"`` message. Return them in the original order so
    the assistant sees tool replies before the user's follow-up text.
    """

    out: list[dict[str, Any]] = []
    pending_user_parts: list[dict[str, Any]] = []

    def flush_user() -> None:
        if not pending_user_parts:
            return
        if len(pending_user_parts) == 1 and pending_user_parts[0].get("type") == "text":
            out.append({"role": "user", "content": pending_user_parts[0]["text"]})
        else:
            out.append({"role": "user", "content": list(pending_user_parts)})
        pending_user_parts.clear()

    for block in blocks:
        btype = block.get("type")
        if btype == "tool_result":
            flush_user()
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id") or "",
                    "content": _tool_result_content_to_string(block.get("content") or ""),
                }
            )
            continue
        if btype == "text":
            pending_user_parts.append({"type": "text", "text": block.get("text") or ""})
            continue
        if btype == "image":
            img = _image_block_to_openai(block)
            if img:
                pending_user_parts.append(img)
            continue
        # thinking / unknown -> drop silently (a user-role thinking block is malformed
        # input; xAI would reject it anyway).
    flush_user()
    return out


def _convert_assistant_message(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """An assistant message becomes a single OpenAI assistant message with optional
    ``content`` text and optional ``tool_calls`` array.
    """

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in blocks:
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text") or "")
        elif btype == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id") or "",
                    "type": "function",
                    "function": {
                        "name": block.get("name") or "",
                        "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                    },
                }
            )
        # thinking blocks dropped — Grok has no equivalent and they'd just confuse it.
    msg: dict[str, Any] = {"role": "assistant"}
    text = "".join(text_parts)
    if text:
        msg["content"] = text
    else:
        msg["content"] = None
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _convert_tools(tools: list[Any] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    out: list[dict[str, Any]] = []
    for t in tools:
        # Tool is a pydantic model
        td = t.model_dump(exclude_none=True) if hasattr(t, "model_dump") else dict(t)
        out.append(
            {
                "type": "function",
                "function": {
                    "name": td.get("name"),
                    "description": td.get("description") or "",
                    "parameters": td.get("input_schema") or {"type": "object", "properties": {}},
                },
            }
        )
    return out


def _convert_tool_choice(tc: dict[str, Any] | None) -> str | dict[str, Any] | None:
    if tc is None:
        return None
    t = tc.get("type")
    if t == "auto":
        return "auto"
    if t == "any":
        return "required"
    if t == "none":
        return "none"
    if t == "tool":
        return {"type": "function", "function": {"name": tc.get("name") or ""}}
    return None


def translate_request(req: MessagesRequest, target_model: str) -> dict[str, Any]:
    """Build the OpenAI chat-completions body for ``req``, using ``target_model`` (the
    provider-side id — caller maps the Anthropic-side ``req.model`` to it).
    """

    messages: list[dict[str, Any]] = []
    system_text = _system_to_string(req.system)
    if system_text:
        messages.append({"role": "system", "content": system_text})

    for m in req.messages:
        blocks = _content_blocks(m.content)
        if m.role == "user":
            messages.extend(_convert_user_message(blocks))
        else:
            messages.append(_convert_assistant_message(blocks))

    body: dict[str, Any] = {
        "model": target_model,
        "messages": messages,
        "stream": req.stream,
        "max_tokens": req.max_tokens,
    }
    if req.temperature is not None:
        body["temperature"] = req.temperature
    if req.top_p is not None:
        body["top_p"] = req.top_p
    if req.stop_sequences:
        body["stop"] = req.stop_sequences

    openai_tools = _convert_tools(req.tools)
    if openai_tools:
        body["tools"] = openai_tools
    tool_choice = _convert_tool_choice(req.tool_choice)
    if tool_choice is not None:
        body["tool_choice"] = tool_choice

    if req.stream:
        body["stream_options"] = {"include_usage": True}

    return body
