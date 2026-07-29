"""Tests for transcript persistence wiring.

v1's TranscriptPublisher had zero call sites — it was instantiated,
assigned, and never invoked, so transcripts stayed [] forever. These tests
drive the real AgentSession event pipeline with a stub publisher and prove
utterances are actually captured now.
"""
import httpx
import pytest
from livekit.agents.voice.agent_session import AgentSession

from agent import EnglishTutorAgent, wire_transcript_persistence
from transcript_publisher import TranscriptPublisher


class StubPublisher:
    def __init__(self):
        self.messages = []

    def enqueue(self, **kwargs):
        self.messages.append(kwargs)


async def test_both_speakers_are_captured_in_order(fake_llm, topic_context):
    agent = EnglishTutorAgent(context=topic_context)
    publisher = StubPublisher()

    session = AgentSession(llm=fake_llm)
    await session.start(agent=agent)
    wire_transcript_persistence(session, publisher)

    await session.run(user_input="I am working with software engineer")

    speakers = [m["speaker"] for m in publisher.messages]
    assert "learner" in speakers
    assert "ai" in speakers

    learner_msg = next(m for m in publisher.messages if m["speaker"] == "learner")
    assert learner_msg["text"] == "I am working with software engineer"
    assert learner_msg["external_id"]  # ChatMessage.id → Rails dedupe key


async def test_ai_messages_carry_a_timestamp(fake_llm, topic_context):
    agent = EnglishTutorAgent(context=topic_context)
    publisher = StubPublisher()

    session = AgentSession(llm=fake_llm)
    await session.start(agent=agent)
    wire_transcript_persistence(session, publisher)
    await session.run(user_input="Hello")

    ai_msg = next(m for m in publisher.messages if m["speaker"] == "ai")
    assert ai_msg["spoke_started_at_ms"] is not None
    assert ai_msg["spoke_started_at_ms"] > 1_500_000_000_000  # epoch ms, not seconds


class _FailOnceTransport(httpx.AsyncBaseTransport):
    """First request fails at the network level, later ones succeed."""

    def __init__(self):
        self.calls = 0
        self.batches = []

    async def handle_async_request(self, request):
        self.calls += 1
        if self.calls == 1:
            raise httpx.ConnectError("rails is restarting", request=request)
        import json

        self.batches.append(json.loads(request.content)["messages"])
        return httpx.Response(201, json={"created": 1})


@pytest.fixture
def publisher_with_flaky_rails(monkeypatch):
    transport = _FailOnceTransport()
    publisher = TranscriptPublisher(session_id=1)
    # Swap the client for one whose transport we control — no real network.
    publisher.client = httpx.AsyncClient(
        base_url="http://test", transport=transport, timeout=1.0
    )
    return publisher, transport


async def test_failed_flush_buffers_and_retries(publisher_with_flaky_rails):
    """A 3-second Rails restart must not eat the learner's conversation."""
    publisher, transport = publisher_with_flaky_rails

    publisher.enqueue(external_id="m1", speaker="learner", text="hello there")
    await publisher.flush()  # first attempt → ConnectError → re-buffered
    assert publisher._pending, "message should be re-buffered after failure"

    await publisher.flush()  # Rails is back
    assert not publisher._pending
    assert transport.batches[-1][0]["external_id"] == "m1"

    await publisher.aclose()


async def test_aclose_flushes_whatever_is_left(publisher_with_flaky_rails):
    publisher, transport = publisher_with_flaky_rails

    publisher.enqueue(external_id="m2", speaker="ai", text="goodbye")
    await publisher.flush()  # fails, buffered
    await publisher.aclose()  # shutdown callback path → final flush succeeds

    assert transport.batches[-1][0]["external_id"] == "m2"
