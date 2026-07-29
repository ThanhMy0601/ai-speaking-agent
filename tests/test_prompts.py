"""Tests for system-prompt construction.

These are pure functions over an AgentContext, so they need no DB, no HTTP
and no LLM.
"""
from context_client import AgentContext
from prompts import build_instructions

from conftest import build_context


def test_prompt_includes_topic_content_from_the_database(topic_context):
    prompt = build_instructions(topic_context)

    assert "Work & Career" in prompt
    assert "Focus on professional English" in prompt
    assert "take on" in prompt
    assert "Present perfect for experience" in prompt


def test_prompt_tells_the_model_not_to_ask_for_a_topic(topic_context):
    """The learner already picked a topic on the previous screen."""
    prompt = build_instructions(topic_context)

    assert "never ask them what they would like to talk about" in prompt.lower()


def test_prompt_never_claims_to_hear_pronunciation(topic_context):
    """The model only ever sees an ASR transcript. Instructing it to correct
    pronunciation made it confabulate feedback it could not possibly have."""
    prompt = build_instructions(topic_context)

    assert "You do not hear audio" in prompt
    assert "cannot assess pronunciation" in prompt


def test_prompt_forbids_markdown_because_output_is_spoken(topic_context):
    prompt = build_instructions(topic_context)

    assert "spoken aloud" in prompt
    assert "No markdown" in prompt


def test_prompt_adapts_to_beginner_level():
    prompt = build_instructions(build_context(proficiency_level="beginner"))

    assert "beginner" in prompt
    assert "ten words" in prompt


def test_prompt_adapts_to_advanced_level():
    prompt = build_instructions(build_context(proficiency_level="advanced"))

    assert "advanced" in prompt
    assert "idiom and collocation" in prompt


def test_prompt_falls_back_to_intermediate_for_unknown_level():
    prompt = build_instructions(build_context(proficiency_level=None))

    assert "eighteen words" in prompt


def test_repeat_attempts_tell_the_model_not_to_re_explain_basics():
    prompt = build_instructions(build_context(attempt_number=3))

    assert "session number 3" in prompt
    assert "don't re-explain the basics" in prompt


def test_level_content_overrides_topic_content():
    ctx = build_context(
        level={
            "id": 9,
            "level": 3,
            "title": "Interviews and career goals",
            "conversation_guide": "Ask the harder interview questions.",
            "opening_line": "Where do you want your career to be in a few years?",
            "target_vocabulary": ["career path"],
            "target_grammar": ["Future forms for goals"],
        }
    )
    prompt = build_instructions(ctx)

    assert "Level 3: Interviews and career goals" in prompt
    assert "Ask the harder interview questions." in prompt
    assert "Focus on professional English" not in prompt


def test_prompt_without_a_topic_still_builds():
    prompt = build_instructions(AgentContext())

    assert "No specific topic was selected" in prompt
    assert "You do not hear audio" in prompt
