"""Publishes transcript events to the Rails API for persistence.

NOT YET WIRED UP. The route this posts to does not exist, and nothing calls
publish_utterance — Phase 4 replaces both sides: a transcript_messages child
table in Rails (the current jsonb blob is a lost-update race) and real
conversation_item_added / user_state_changed handlers here.

It is kept, and closed properly on shutdown, so the plumbing is in place for
that change rather than being reinvented.
"""

import logging
import os

import httpx

logger = logging.getLogger(__name__)

RAILS_API_URL = os.getenv("RAILS_API_URL", "http://localhost:8000")
AGENT_SERVICE_TOKEN = os.getenv("AGENT_SERVICE_TOKEN", "dev_agent_service_token")


class TranscriptPublisher:
    """Sends transcript updates to the Rails backend over HTTP."""

    def __init__(self, session_id: int):
        self.session_id = session_id
        self.client = httpx.AsyncClient(
            base_url=RAILS_API_URL,
            headers={"Authorization": f"Bearer {AGENT_SERVICE_TOKEN}"},
            timeout=10.0,
        )

    async def publish_utterance(
        self,
        speaker: str,
        text: str,
        pronunciation_score: int | None = None,
    ):
        """Publish a single utterance to the Rails API."""
        try:
            payload = {"speaker": speaker, "text": text}
            if pronunciation_score is not None:
                payload["pronunciation_score"] = pronunciation_score

            await self.client.post(
                f"/internal/practice_sessions/{self.session_id}/transcript_messages",
                json=payload,
            )
        except Exception as e:
            logger.error("Failed to publish transcript: %s", e)

    async def aclose(self):
        await self.client.aclose()
