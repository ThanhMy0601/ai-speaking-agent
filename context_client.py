"""Fetches session context from Rails over HTTP.

Replaces rag_context.py, which opened its own psycopg2 connection and looked
up teaching content in a hardcoded Python dict keyed on the topic's English
title — so renaming a topic in Rails silently degraded it to a generic
fallback with no log and no error.

Going through Rails kills that coupling by construction: there is no longer
anywhere in Python that can key on a title. It also means admin edits to a
topic take effect on the next session with no coordination between the two
repos, and the agent no longer needs database credentials.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import httpx

RAILS_API_URL = os.getenv("RAILS_API_URL", "http://localhost:8000")
AGENT_SERVICE_TOKEN = os.getenv("AGENT_SERVICE_TOKEN", "dev_agent_service_token")

# Endpointing defaults if Rails is unreachable. Deliberately the
# intermediate profile rather than the SDK's 0.5s, which cuts off learners
# who pause mid-sentence to find a word.
DEFAULT_TURN_TUNING = {"min_delay": 0.9, "max_delay": 6.0}


@dataclass
class Learner:
    display_name: str | None = None
    proficiency_level: str | None = None
    learning_goal: str | None = None
    attempt_number: int = 1
    topics_practised: int = 0


@dataclass
class AgentContext:
    """Everything the agent needs for one session, already rendered."""

    learner: Learner = field(default_factory=Learner)
    topic: dict[str, Any] | None = None
    level: dict[str, Any] | None = None
    turn_tuning: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TURN_TUNING))

    @property
    def has_topic(self) -> bool:
        return self.topic is not None

    @property
    def opening_line(self) -> str | None:
        """The level's opening line if this attempt has one, else the topic's."""
        if self.level and self.level.get("opening_line"):
            return self.level["opening_line"]
        if self.topic:
            return self.topic.get("opening_line")
        return None

    @property
    def conversation_guide(self) -> str | None:
        if self.level and self.level.get("conversation_guide"):
            return self.level["conversation_guide"]
        if self.topic:
            return self.topic.get("conversation_guide")
        return None

    @property
    def target_vocabulary(self) -> list[str]:
        if self.level and self.level.get("target_vocabulary"):
            return self.level["target_vocabulary"]
        if self.topic:
            return self.topic.get("target_vocabulary") or []
        return []

    @property
    def target_grammar(self) -> list[str]:
        if self.level and self.level.get("target_grammar"):
            return self.level["target_grammar"]
        if self.topic:
            return self.topic.get("target_grammar") or []
        return []

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> AgentContext:
        learner_data = payload.get("learner") or {}
        return cls(
            learner=Learner(
                display_name=learner_data.get("display_name"),
                proficiency_level=learner_data.get("proficiency_level"),
                learning_goal=learner_data.get("learning_goal"),
                attempt_number=learner_data.get("attempt_number") or 1,
                topics_practised=learner_data.get("topics_practised") or 0,
            ),
            topic=payload.get("topic"),
            level=payload.get("level"),
            turn_tuning=payload.get("turn_tuning") or dict(DEFAULT_TURN_TUNING),
        )


class ContextClient:
    """Async client for /internal/agent_contexts."""

    def __init__(self, base_url: str | None = None, token: str | None = None):
        self._client = httpx.AsyncClient(
            base_url=base_url or RAILS_API_URL,
            headers={"Authorization": f"Bearer {token or AGENT_SERVICE_TOKEN}"},
            timeout=5.0,
        )

    async def fetch(self, session_id) -> AgentContext:
        """Fetch context for a session.

        Returns an empty AgentContext rather than raising if Rails is
        unreachable: a degraded generic conversation beats dropping the
        learner into a room where nobody ever speaks.
        """
        if not session_id:
            return AgentContext()

        try:
            response = await self._client.get(f"/internal/agent_contexts/{session_id}")
            response.raise_for_status()
            return AgentContext.from_payload(response.json())
        except (httpx.HTTPError, ValueError):
            return AgentContext()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> ContextClient:
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.aclose()
