"""Shared fixtures for the agent test suite.

FakeLLM lets us drive the real livekit-agents AgentSession/eval harness
end-to-end (on_enter, chat_ctx, session.run()) without hitting Gemini or
needing an API key — deterministic and free to run in CI.
"""
import pytest

from livekit.agents import llm as lk_llm
from livekit.agents.llm.llm import LLM, LLMStream
from livekit.agents.types import APIConnectOptions


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


class FakeRag:
    """Duck-typed stand-in for RAGContextProvider — no DB connection.

    build_system_prompt() only calls get_topic_context, so tests don't need
    a real Postgres connection to exercise prompt-building logic.
    """

    def get_topic_context(self, topic_id) -> str:
        return f"[fake topic context for topic {topic_id}]"


@pytest.fixture
def fake_rag():
    return FakeRag()
