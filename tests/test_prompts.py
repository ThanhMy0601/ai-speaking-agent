"""Unit tests for the pure prompt-building function in agent.py.

These don't need AgentSession/AudioStream/DB — build_system_prompt is a
plain function that takes a duck-typed context provider.
"""
from agent import build_system_prompt


def test_prompt_includes_topic_context(fake_rag):
    prompt = build_system_prompt(topic_id=42, rag=fake_rag)

    assert "[fake topic context for topic 42]" in prompt
    assert "Guide the conversation around the following topic" in prompt


def test_prompt_keeps_the_tutor_base_instructions(fake_rag):
    prompt = build_system_prompt(topic_id=1, rag=fake_rag)

    assert "AI English speaking tutor" in prompt
    assert "2-3 sentences" in prompt


def test_prompt_without_topic_still_builds(fake_rag):
    """topic_id is None for a session created without one — the provider
    returns generic context rather than the prompt blowing up."""
    prompt = build_system_prompt(topic_id=None, rag=fake_rag)

    assert "[fake topic context for topic None]" in prompt


def test_prompt_has_no_ielts_or_roleplay_traces(fake_rag):
    """IELTS mock test and role-play were removed from the product in
    Phase 2. Nothing in the prompt path should reference them."""
    prompt = build_system_prompt(topic_id=1, rag=fake_rag).lower()

    assert "ielts" not in prompt
    assert "examiner" not in prompt
    assert "role-play" not in prompt
    assert "in character" not in prompt
