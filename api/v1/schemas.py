"""Pydantic schemas for the Anthropic-compatible surface.

These mirror the Anthropic Messages API shape (the inbound contract). Outbound
OpenAI/xAI shapes are handled in :mod:`api.translation` as plain dicts — the
gateway never serves OpenAI directly, so locking down a Pydantic model for the
xAI side would be ceremony without benefit.
"""

from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Inbound content blocks (Anthropic shape)
# ---------------------------------------------------------------------------


class TextBlock(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["text"] = "text"
    text: str
    # cache_control is accepted-and-ignored — Grok has no equivalent.
    cache_control: dict | None = None


class ImageSource(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["base64", "url"]
    media_type: str | None = None
    data: str | None = None
    url: str | None = None


class ImageBlock(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["image"] = "image"
    source: ImageSource
    cache_control: dict | None = None


class ToolUseBlock(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)
    cache_control: dict | None = None


class ToolResultBlock(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    # Anthropic allows content to be a plain string OR a list of TextBlock / ImageBlock dicts.
    content: Union[str, list[dict[str, Any]]] = ""
    is_error: bool | None = None
    cache_control: dict | None = None


class ThinkingBlock(BaseModel):
    """Accepted-and-ignored. Grok has no extended-thinking equivalent.

    We accept it so a caller round-tripping previous assistant turns isn't rejected.
    """

    model_config = ConfigDict(extra="allow")
    type: Literal["thinking"] = "thinking"
    thinking: str = ""
    signature: str | None = None


ContentBlock = Union[TextBlock, ImageBlock, ToolUseBlock, ToolResultBlock, ThinkingBlock]


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Literal["user", "assistant"]
    # Per Anthropic spec, content is EITHER a plain string OR a list of blocks.
    content: Union[str, list[dict[str, Any]]]


# ---------------------------------------------------------------------------
# Tools (Anthropic shape)
# ---------------------------------------------------------------------------


class Tool(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    description: str | None = None
    input_schema: dict[str, Any] = Field(default_factory=dict)
    cache_control: dict | None = None


class ToolChoiceAuto(BaseModel):
    type: Literal["auto"] = "auto"
    disable_parallel_tool_use: bool | None = None


class ToolChoiceAny(BaseModel):
    type: Literal["any"] = "any"
    disable_parallel_tool_use: bool | None = None


class ToolChoiceTool(BaseModel):
    type: Literal["tool"] = "tool"
    name: str
    disable_parallel_tool_use: bool | None = None


class ToolChoiceNone(BaseModel):
    type: Literal["none"] = "none"


ToolChoice = Union[ToolChoiceAuto, ToolChoiceAny, ToolChoiceTool, ToolChoiceNone]


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class MessagesRequest(BaseModel):
    """Mirror of Anthropic POST /v1/messages request body.

    `extra="allow"` so unknown fields a future Anthropic SDK passes through don't
    400 the caller — we just ignore them.
    """

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[Message]
    # Anthropic allows `system` as either a plain string or a list of content blocks.
    system: Union[str, list[dict[str, Any]], None] = None
    max_tokens: int = 1024
    metadata: dict[str, Any] | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    tools: list[Tool] | None = None
    tool_choice: dict[str, Any] | None = None
    # Accepted-and-ignored: Grok has no extended thinking. Documented in README.
    thinking: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Response (non-streaming)
# ---------------------------------------------------------------------------


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


class MessagesResponse(BaseModel):
    """Mirror of the Anthropic non-streaming response shape."""

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str
    content: list[dict[str, Any]]
    stop_reason: Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None = None
    stop_sequence: str | None = None
    usage: Usage


# ---------------------------------------------------------------------------
# /v1/models response (Anthropic shape)
# ---------------------------------------------------------------------------


class ModelInfo(BaseModel):
    id: str
    type: Literal["model"] = "model"
    display_name: str
    created_at: str


class ModelListResponse(BaseModel):
    data: list[ModelInfo]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool = False
