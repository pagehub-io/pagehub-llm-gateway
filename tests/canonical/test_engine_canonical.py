"""Layer (a): CanonicalRequest -> CanonicalResponse round-trips against EchoProvider.

These tests exercise the engine's pipeline + recording behavior without ever
touching a wire format. If the canonical layer is sound, every event below
is the same regardless of which inbound protocol or provider plugs in later.
"""

from __future__ import annotations

import pytest

from api.canonical.events import (
    ContentBlockDone,
    ContentBlockStart,
    ContentTextDelta,
    ContentToolCallDelta,
    MessageDelta,
    StreamDone,
    StreamStart,
)
from api.canonical.types import (
    CanonicalMessage,
    CanonicalRequest,
    CanonicalResponse,
    CanonicalText,
    CanonicalToolCall,
    CanonicalUsage,
)
from api.engine import Engine
from api.providers.base import ProviderError
from api.shared.recorder import MemoryRecorder
from tests.fakes import EchoProvider


@pytest.fixture
def engine_with_echo():
    provider = EchoProvider(
        response=CanonicalResponse(
            id="msg_echo",
            model="echo-model",
            content_blocks=[CanonicalText(text="hi back")],
            stop_reason="end_turn",
            usage=CanonicalUsage(input_tokens=5, output_tokens=2),
        )
    )
    recorder = MemoryRecorder(log_content=True)
    return Engine(provider=provider, recorder=recorder), provider, recorder


def _canonical_user_text(text: str) -> CanonicalRequest:
    return CanonicalRequest(
        model="echo-model",
        messages=[CanonicalMessage(role="user", content_blocks=[CanonicalText(text=text)])],
        max_tokens=64,
    )


async def test_non_streaming_pipeline_runs_through(engine_with_echo):
    engine, provider, recorder = engine_with_echo
    canonical = _canonical_user_text("hello")
    body = canonical.model_dump(exclude_none=True)

    result, rid = await engine.run_anthropic(
        client_request_id="rid_test",
        raw_inbound_body=body,
        canonical_request=canonical,
    )

    assert result["type"] == "message"
    assert result["model"] == "echo-model"
    assert result["content"][0]["text"] == "hi back"
    assert result["usage"]["input_tokens"] == 5
    assert result["usage"]["output_tokens"] == 2

    # Provider got the same canonical request we passed in.
    assert provider.last_canonical_request.model == "echo-model"
    assert provider.last_canonical_request.messages[0].content_blocks[0].text == "hello"

    # And the recorder captured the pipeline.
    detail = await recorder.get_request(rid)
    kinds = [e["kind"] for e in detail["events"]]
    assert kinds[0] == "inbound_received"
    assert "decoded_canonical" in kinds
    assert "provider_request" in kinds
    assert "provider_response" in kinds
    assert "decoded_canonical_response" in kinds
    assert kinds[-1] == "encoded_outbound"
    assert detail["stop_reason"] == "end_turn"
    assert detail["input_tokens"] == 5
    assert detail["output_tokens"] == 2


async def test_provider_error_records_error_event_and_propagates(engine_with_echo):
    _engine, _provider, recorder = engine_with_echo
    failing_provider = EchoProvider(error=ProviderError(status_code=503, message="down"))
    engine = Engine(provider=failing_provider, recorder=recorder)
    canonical = _canonical_user_text("hello")
    with pytest.raises(ProviderError):
        await engine.run_anthropic(
            client_request_id=None,
            raw_inbound_body=canonical.model_dump(exclude_none=True),
            canonical_request=canonical,
        )
    # The single recorded request should carry an error.
    rows = await recorder.list_requests(limit=10, offset=0)
    assert rows
    assert rows[0]["error"] == "down"


async def test_streaming_pipeline_emits_anthropic_sse_and_records_events():
    provider = EchoProvider(
        stream_events=[
            StreamStart(message_id="msg_stream", model="echo-model"),
            ContentBlockStart(index=0, block=CanonicalText()),
            ContentTextDelta(index=0, text="Hel"),
            ContentTextDelta(index=0, text="lo"),
            ContentBlockDone(index=0),
            MessageDelta(stop_reason="end_turn", usage=CanonicalUsage(output_tokens=2)),
            StreamDone(),
        ]
    )
    recorder = MemoryRecorder(log_content=True)
    engine = Engine(provider=provider, recorder=recorder)
    canonical = CanonicalRequest(
        model="echo-model",
        messages=[CanonicalMessage(role="user", content_blocks=[CanonicalText(text="hi")])],
        stream=True,
    )

    chunks: list[bytes] = []
    async for c in engine.stream_anthropic(
        client_request_id="rid_stream",
        raw_inbound_body=canonical.model_dump(exclude_none=True),
        canonical_request=canonical,
    ):
        chunks.append(c)

    raw = b"".join(chunks)
    assert b"event: message_start" in raw
    assert b"event: content_block_start" in raw
    assert b"event: content_block_delta" in raw
    assert b"event: content_block_stop" in raw
    assert b"event: message_delta" in raw
    assert b"event: message_stop" in raw

    # Recorder caught a canonical_event row per emitted event.
    rows = await recorder.list_requests(limit=10, offset=0)
    import uuid as _uuid
    detail = await recorder.get_request(_uuid.UUID(rows[0]["id"]))
    canonical_events = [e for e in detail["events"] if e["kind"] == "canonical_event"]
    assert len(canonical_events) == 7
    # The final canonical_response on the request row reflects the accumulated stream.
    assert detail["canonical_response"]["content_blocks"][0]["text"] == "Hello"
    assert detail["canonical_response"]["stop_reason"] == "end_turn"
    assert detail["output_tokens"] == 2


async def test_streaming_tool_call_accumulates_input_json():
    provider = EchoProvider(
        stream_events=[
            StreamStart(message_id="m1", model="echo-model"),
            ContentBlockStart(
                index=0, block=CanonicalToolCall(id="c1", name="get_weather", input={})
            ),
            ContentToolCallDelta(
                index=0, id="c1", name="get_weather", partial_input_json='{"loc":'
            ),
            ContentToolCallDelta(index=0, partial_input_json='"SF"}'),
            ContentBlockDone(index=0),
            MessageDelta(stop_reason="tool_use", usage=CanonicalUsage(output_tokens=3)),
            StreamDone(),
        ]
    )
    recorder = MemoryRecorder(log_content=True)
    engine = Engine(provider=provider, recorder=recorder)
    canonical = CanonicalRequest(model="echo-model", stream=True)
    async for _ in engine.stream_anthropic(
        client_request_id=None,
        raw_inbound_body={},
        canonical_request=canonical,
    ):
        pass
    rows = await recorder.list_requests(limit=1, offset=0)
    import uuid as _uuid
    detail = await recorder.get_request(_uuid.UUID(rows[0]["id"]))
    block = detail["canonical_response"]["content_blocks"][0]
    assert block["type"] == "tool_call"
    assert block["input"] == {"loc": "SF"}
    assert detail["canonical_response"]["stop_reason"] == "tool_use"
