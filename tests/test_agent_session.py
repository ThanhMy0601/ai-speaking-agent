"""End-to-end agent tests via the livekit-agents eval harness.

AgentSession + a scripted FakeLLM — no network, no API keys, no LiveKit room.
"""
from livekit.agents.voice.agent_session import AgentSession

from agent import EnglishTutorAgent, build_turn_handling
from context_client import AgentContext

from conftest import build_context


def assistant_turns(session) -> str:
    return " ".join(
        item.text_content or ""
        for item in session.history.items
        if getattr(item, "role", None) == "assistant"
    )


async def test_greeting_opens_in_the_chosen_topic(fake_llm, topic_context):
    """The behaviour this whole phase exists to fix.

    Previously on_enter() spoke one hardcoded line for every session —
    "What would you like to talk about today?" — even though the learner had
    just chosen a topic on the previous screen.
    """
    agent = EnglishTutorAgent(context=topic_context)

    session = AgentSession(llm=fake_llm)
    await session.start(agent=agent)
    await session.run(user_input="Hi")

    spoken = assistant_turns(session)
    assert topic_context.topic["opening_line"] in spoken
    assert "what would you like to talk about" not in spoken.lower()


async def test_greeting_uses_the_level_opening_line_on_a_repeat_attempt(fake_llm):
    """Attempt 3 gets level 3's opening line, not the topic's."""
    ctx = build_context(
        attempt_number=3,
        level={
            "id": 9,
            "level": 3,
            "title": "Interviews and career goals",
            "conversation_guide": "Ask the harder interview questions.",
            "opening_line": "Welcome back. Where do you want your career to be in a few years?",
            "target_vocabulary": [],
            "target_grammar": [],
        },
    )
    agent = EnglishTutorAgent(context=ctx)

    session = AgentSession(llm=fake_llm)
    await session.start(agent=agent)
    await session.run(user_input="Hi")

    spoken = assistant_turns(session)
    assert "Where do you want your career to be" in spoken
    assert ctx.topic["opening_line"] not in spoken


async def test_greeting_substitutes_the_learner_name(fake_llm):
    ctx = build_context(
        opening_line="Hi {name}! Ready to talk about work?",
        display_name="Mai",
    )
    agent = EnglishTutorAgent(context=ctx)

    session = AgentSession(llm=fake_llm)
    await session.start(agent=agent)
    await session.run(user_input="Hi")

    assert "Hi Mai! Ready to talk about work?" in assistant_turns(session)


async def test_agent_without_context_still_starts(fake_llm):
    """Rails being unreachable degrades to a generic conversation rather
    than leaving the learner in a silent room."""
    agent = EnglishTutorAgent(context=AgentContext())

    session = AgentSession(llm=fake_llm)
    await session.start(agent=agent)
    result = await session.run(user_input="Hello?")

    reply = result.expect.next_event(type="message")
    assert reply.event().item.role == "assistant"


def test_endpointing_waits_longer_for_beginners():
    """The SDK default is a fixed 0.5s, tuned for native speakers. Learners
    pause mid-sentence to search for words and get cut off."""
    beginner = build_turn_handling(build_context(proficiency_level="beginner"))
    advanced = build_turn_handling(build_context(proficiency_level="advanced"))

    assert beginner["endpointing"]["mode"] == "dynamic"
    assert beginner["endpointing"]["min_delay"] > 0.5
    assert beginner["endpointing"]["min_delay"] > advanced["endpointing"]["min_delay"]


def test_interruption_ignores_a_single_filler_word():
    opts = build_turn_handling(build_context())

    assert opts["interruption"]["min_words"] >= 2
    assert opts["interruption"]["resume_false_interruption"] is True


def test_interruption_mode_is_pinned_to_vad_by_default():
    """Regression guard.

    This test previously asserted the opposite — that `mode` was left unset
    so the SDK could "pick its adaptive ML classifier". Adaptive is not a
    local model: it is a LiveKit Cloud inference service at
    agent-gateway.livekit.cloud. Against a self-hosted server it
    authenticates with the local devkey, gets a 401, and kills the job with
    "failed to detect interruption after 3 attempts" — the agent joins the
    room, publishes a track, and immediately leaves.

    Leaving mode unset let the SDK auto-select it whenever a streaming STT
    and a VAD were present, which is always here. So it stays pinned.
    """
    opts = build_turn_handling(build_context())

    assert opts["interruption"]["mode"] == "vad"


def test_adaptive_interruption_is_opt_in_for_livekit_cloud(monkeypatch):
    import agent as agent_module

    monkeypatch.setattr(agent_module, "ADAPTIVE_INTERRUPTION", True)
    opts = agent_module.build_turn_handling(build_context())

    assert opts["interruption"]["mode"] == "adaptive"
