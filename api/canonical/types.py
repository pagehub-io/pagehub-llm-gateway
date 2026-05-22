"""Canonical request / response / content / tool types.

These are deliberately Anthropic-and-OpenAI agnostic. Where the two protocols
disagree, we pick a third name. Where one protocol has a concept the other
doesn't (cache_control, extended thinking), we leave it out of canonical — it
travels in ``metadata`` if it travels at all.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class CanonicalText(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["text"] = "text"
    text: str = ""


class CanonicalImage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["image"] = "image"
    media_type: str = "image/png"
    data: str | None = None  # base64 payload OR
    url: str | None = None  # an absolute URL — exactly one of the two


class CanonicalToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class CanonicalToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    content: str = ""  # canonicalized to plain text — multimodal tool replies are flattened
    is_error: bool = False


CanonicalContent = Annotated[
    Union[CanonicalText, CanonicalImage, CanonicalToolCall, CanonicalToolResult],
    Field(discriminator="type"),
]


class CanonicalMessage(BaseModel):
    """One conversational turn.

    Note: a ``tool`` role here is a *future* generalization for protocols that
    expose tool replies as standalone messages. AnthropicInbound currently models
    tool replies as ``CanonicalToolResult`` blocks inside a ``user`` message; it
    never emits a ``tool`` role. Provider adapters MAY surface ``tool`` if their
    wire format demands it (xAI does), but the canonical convention is: tool
    replies live in user-message ``content_blocks`` as ToolResult blocks.
    """

    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant", "tool"]
    content_blocks: list[CanonicalContent] = Field(default_factory=list)


class CanonicalTool(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str = ""
    json_schema: dict[str, Any] = Field(default_factory=dict)


class CanonicalToolChoice(BaseModel):
    """Provider-neutral tool-choice.

    ``mode``: ``auto`` (provider decides), ``required`` (must call at least one
    tool), ``none`` (no tools), ``tool`` (must call the named tool — ``name``
    required iff ``mode == "tool"``).
    """

    model_config = ConfigDict(extra="forbid")
    mode: Literal["auto", "required", "none", "tool"]
    name: str | None = None


class CanonicalUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_tokens: int = 0
    output_tokens: int = 0
    cache_tokens: int | None = None  # optional — only some providers expose this


CanonicalStopReason = Literal["end_turn", "max_tokens", "tool_use", "stop_sequence", "error"]


class CanonicalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    messages: list[CanonicalMessage] = Field(default_factory=list)
    system: list[CanonicalContent] | None = None
    max_tokens: int = 1024
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False
    tools: list[CanonicalTool] = Field(default_factory=list)
    tool_choice: CanonicalToolChoice | None = None
    stop_sequences: list[str] = Field(default_factory=list)
    # Free-form metadata for protocol-specific extras that don't fit in canonical
    # (e.g. anthropic ``thinking`` config, openai ``user`` id). Provider adapters
    # MAY read keys they understand and MUST ignore everything else.
    metadata: dict[str, Any] = Field(default_factory=dict)


class CanonicalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    model: str
    content_blocks: list[CanonicalContent] = Field(default_factory=list)
    stop_reason: CanonicalStopReason | None = None
    usage: CanonicalUsage = Field(default_factory=CanonicalUsage)
