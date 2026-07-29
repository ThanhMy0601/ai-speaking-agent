"""Shared fixtures for the agent test suite.

FakeLLM lets us drive the real livekit-agents AgentSession/eval harness
end-to-end (on_enter, chat_ctx, session.run()) without hitting Gemini or
needing an API key — deterministic and free to run in CI.
"""
import pytest

from livekit.agents import llm as lk_llm
from livekit.agents.llm.llm import LLM, LLMStream
from livekit.agents.types import APIConnectOptions

from context_client import AgentContext, Learner


class FakeLLMStream(LLMStream):
    def __init__(self, llm_instance, *, chat_ctx, tools, conn_options, reply):
        super().__init__(llm_instance, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._reply = reply

    async def _run(self) -> None:
        self._event_ch.send_nowait(
            lk_llm.ChatChunk(
                id="fake-chunk-1",
                delta=lk_llm.ChoiceDelta(role="assistant", content=self._reply),
            )
        )


class FakeLLM(LLM):
    """A scripted, non-network LLM for deterministic agent tests."""

    def __init__(self, reply: str = "That's interesting, tell me more."):
        super().__init__()
        self._reply = reply

    @property
    def model(self) -> str:
        return "fake-model"

    def chat(self, *, chat_ctx, tools=None, conn_options=None, **kwargs):
        return FakeLLMStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options or APIConnectOptions(),
            reply=self._reply,
        )


@pytest.fixture
def fake_llm():
    return FakeLLM()


# Mirrors AgentContextBuilder::TURN_TUNING in Rails. Rails computes the
# tuning and sends it down; the agent only applies it.
RAILS_TURN_TUNING = {
    "beginner": {"min_delay": 1.2, "max_delay": 8.0},
    "intermediate": {"min_delay": 0.9, "max_delay": 6.0},
    "advanced": {"min_delay": 0.6, "max_delay": 4.0},
}


def build_context(
    *,
    topic_title="Work & Career",
    opening_line="Hello! Let's talk about work today. What do you do day to day?",
    conversation_guide="Focus on professional English and workplace stories.",
    proficiency_level="intermediate",
    attempt_number=1,
    level=None,
    display_name="Test Learner",
):
    """Builds the context Rails would return, without touching HTTP or a DB."""
    return AgentContext(
        learner=Learner(
            display_name=display_name,
            proficiency_level=proficiency_level,
            learning_goal="business_english",
            attempt_number=attempt_number,
        ),
        topic={
            "id": 2,
            "title": topic_title,
            "description": "Excel in job interviews and workplace communication",
            "conversation_guide": conversation_guide,
            "opening_line": opening_line,
            "target_vocabulary": ["take on", "deadline", "stakeholder"],
            "target_grammar": ["Present perfect for experience"],
            "cefr_level": "b1",
        },
        level=level,
        turn_tuning=dict(
            RAILS_TURN_TUNING.get(proficiency_level, RAILS_TURN_TUNING["intermediate"])
        ),
    )


@pytest.fixture
def topic_context():
    return build_context()
