"""Canonical streaming events.

Provider adapters decode their wire SSE into a stream of these. Inbound protocol
adapters encode this stream into their wire SSE. The engine just shuttles events
between the two without needing to know what either wire looks like.

Event ordering produced by every well-behaved provider adapter:

    StreamStart
    ( ContentBlockStart(index=N) -> ContentTextDelta|ContentToolCallDelta(index=N)* -> ContentBlockDone(index=N) )*
    MessageDelta(stop_reason=..., usage=...)
    StreamDone

If anything goes wrong upstream, a single ``StreamError`` is emitted in place of
the remaining events. Inbound adapters MUST emit a clean terminator even after
an Error so the consumer's parser doesn't hang.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from api.canonical.types import (
    CanonicalContent,
    CanonicalStopReason,
    CanonicalUsage,
)


class StreamStart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["start"] = "start"
    message_id: str
    model: str
    usage_estimate: CanonicalUsage = Field(default_factory=CanonicalUsage)


class ContentBlockStart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["block_start"] = "block_start"
    index: int
    block: CanonicalContent


class ContentTextDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["text_delta"] = "text_delta"
    index: int
    text: str


class ContentToolCallDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["tool_call_delta"] = "tool_call_delta"
    index: int
    # ``id`` and ``name`` are set on the first delta for this index, once known.
    # Subsequent deltas carry only ``partial_input_json`` increments.
    id: str | None = None
    name: str | None = None
    partial_input_json: str = ""


class ContentBlockDone(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["block_done"] = "block_done"
    index: int


class MessageDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["message_delta"] = "message_delta"
    stop_reason: CanonicalStopReason | None = None
    usage: CanonicalUsage = Field(default_factory=CanonicalUsage)


class StreamDone(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["done"] = "done"


class StreamError(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["error"] = "error"
    message: str


CanonicalStreamEvent = Annotated[
    Union[
        StreamStart,
        ContentBlockStart,
        ContentTextDelta,
        ContentToolCallDelta,
        ContentBlockDone,
        MessageDelta,
        StreamDone,
        StreamError,
    ],
    Field(discriminator="type"),
]
