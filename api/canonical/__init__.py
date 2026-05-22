"""Canonical, provider-neutral request/response/event vocabulary.

This package is the load-bearing middle layer of the gateway. Every inbound
protocol (AnthropicInbound today, OpenAIInbound tomorrow) decodes to / encodes
from these types. Every provider adapter (XAIProvider today, GPT5Provider /
BedrockProvider / GeminiProvider tomorrow) encodes from / decodes to these
types. Adding an N-th provider is then one new adapter, not N x N pairwise
translators.

The canonical layer is the source of truth for what the request "means" — the
wire formats on either side are just encodings.
"""

from api.canonical.events import (
    CanonicalStreamEvent,
    ContentBlockDone,
    ContentBlockStart,
    ContentTextDelta,
    ContentToolCallDelta,
    MessageDelta,
    StreamDone,
    StreamError,
    StreamStart,
)
from api.canonical.types import (
    CanonicalContent,
    CanonicalImage,
    CanonicalMessage,
    CanonicalRequest,
    CanonicalResponse,
    CanonicalStopReason,
    CanonicalText,
    CanonicalTool,
    CanonicalToolCall,
    CanonicalToolChoice,
    CanonicalToolResult,
    CanonicalUsage,
)

__all__ = [
    "CanonicalContent",
    "CanonicalImage",
    "CanonicalMessage",
    "CanonicalRequest",
    "CanonicalResponse",
    "CanonicalStopReason",
    "CanonicalStreamEvent",
    "CanonicalText",
    "CanonicalTool",
    "CanonicalToolCall",
    "CanonicalToolChoice",
    "CanonicalToolResult",
    "CanonicalUsage",
    "ContentBlockDone",
    "ContentBlockStart",
    "ContentTextDelta",
    "ContentToolCallDelta",
    "MessageDelta",
    "StreamDone",
    "StreamError",
    "StreamStart",
]
