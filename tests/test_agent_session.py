"""End-to-end smoke test for EnglishTutorAgent via the livekit-agents eval
harness (AgentSession + a scripted FakeLLM — no network, no API keys).

This proves the harness itself works against the CURRENT agent, so Phase 3
(topic-aware greeting from DB) and Phase 4 (TTS-contract / honesty prompt
assertions) can extend it with confidence instead of discovering the harness
API from scratch.
"""
from livekit.agents.voice.agent_session import AgentSession

from agent import EnglishTutorAgent


async def test_greeting_is_spoken_on_enter(fake_llm):
    agent = EnglishTutorAgent(system_prompt="You are a helpful tutor.", publisher=None)

    session = AgentSession(llm=fake_llm)
    await session.start(agent=agent)
    await session.run(user_input="Hi there")

    greetings = [
        item.text_content
        for item in session.history.items
        if getattr(item, "role", None) == "assistant"
    ]
    assert (
        "Hello! I'm your English practice partner. "
        "What would you like to talk about today?"
    ) in greetings


async def test_current_greeting_ignores_topic_id(fake_llm):
    """Characterization test for a known gap (see REBUILD_PLAN_V2.md §5).

    on_enter() only branches on session_type, never on topic_id — so even
    when the learner already picked a topic, the agent still asks them what
    they want to talk about. Phase 3 replaces this with a DB-driven,
    topic-aware opening line; when that lands, this test's assertion flips
    and should be updated deliberately, not treated as a regression.
    """
    agent = EnglishTutorAgent(system_prompt="You are a helpful tutor.", publisher=None)
    agent._topic_id = 3  # learner already chose "Travel & Tourism"

    session = AgentSession(llm=fake_llm)
    await session.start(agent=agent)
    await session.run(user_input="Hi there")

    greetings = " ".join(
        item.text_content or ""
        for item in session.history.items
        if getattr(item, "role", None) == "assistant"
    )
    assert "what would you like to talk about" in greetings.lower()


async def test_llm_reply_after_greeting_is_captured_in_run_result(fake_llm):
    """result.events from session.run() only contains events produced
    during that run (the assistant's reply) — the user's own input isn't
    re-emitted as an event, it's the run's input."""
    agent = EnglishTutorAgent(system_prompt="You are a helpful tutor.", publisher=None)

    session = AgentSession(llm=fake_llm)
    await session.start(agent=agent)
    result = await session.run(user_input="Let's talk about travel")

    assistant_msg = result.expect.next_event(type="message")
    assert assistant_msg.event().item.role == "assistant"
    assert assistant_msg.event().item.text_content == fake_llm._reply
    result.expect.no_more_events()
