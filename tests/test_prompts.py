"""Unit tests for the pure prompt-building functions in agent.py.

These don't need AgentSession/AudioStream/DB at all — build_system_prompt
and get_roleplay_greeting are plain functions, so we test them directly.
"""
from agent import build_system_prompt, get_roleplay_greeting


def test_ielts_prompt_is_examiner_only_and_ignores_base_prompt(fake_rag):
    prompt = build_system_prompt(
        session_type="ielts_mock_test",
        lesson_id=None,
        topic_id=None,
        scenario="job_interview",
        rag=fake_rag,
    )
    assert "IELTS Speaking examiner" in prompt
    assert "[fake ielts context part 1]" in prompt
    # documents current behavior: IELTS is hardcoded to part 1
    assert "part 2" not in prompt.lower()


def test_role_play_prompt_includes_scenario_context(fake_rag):
    prompt = build_system_prompt(
        session_type="role_play",
        lesson_id=None,
        topic_id=None,
        scenario="salary_negotiation",
        rag=fake_rag,
    )
    assert "playing a specific professional role" in prompt
    assert "[fake roleplay context for salary_negotiation]" in prompt


def test_topic_practice_prompt_includes_topic_context_when_topic_id_present(fake_rag):
    prompt = build_system_prompt(
        session_type="free_practice",
        lesson_id=None,
        topic_id=42,
        scenario="job_interview",
        rag=fake_rag,
    )
    assert "[fake topic context for topic 42]" in prompt
    assert "Guide the conversation around the following topic" in prompt


def test_free_practice_without_topic_falls_back_to_legacy_lesson_context(fake_rag):
    prompt = build_system_prompt(
        session_type="free_practice",
        lesson_id=7,
        topic_id=None,
        scenario="job_interview",
        rag=fake_rag,
    )
    assert "[fake lesson context for lesson 7]" in prompt
    assert "Incorporate the following lesson context" in prompt


def test_roleplay_greeting_known_scenario():
    assert "salary" not in get_roleplay_greeting("salary_negotiation").lower()
    assert "compensation package" in get_roleplay_greeting("salary_negotiation")


def test_roleplay_greeting_unknown_scenario_falls_back_to_job_interview():
    assert get_roleplay_greeting("no_such_scenario") == get_roleplay_greeting("job_interview")
