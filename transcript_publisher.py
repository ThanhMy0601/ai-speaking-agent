"""Publishes transcript events to the Rails API for persistence."""

import os
import httpx
import logging

logger = logging.getLogger(__name__)

RAILS_API_URL = os.getenv("RAILS_API_URL", "http://localhost:3000/api/v1")


class TranscriptPublisher:
    """Sends transcript updates to the Rails backend via HTTP webhook."""

    def __init__(self, session_id: int):
        self.session_id = session_id
        self.client = httpx.AsyncClient(base_url=RAILS_API_URL, timeout=10.0)

    async def publish_utterance(
        self,
        speaker: str,
        text: str,
        pronunciation_score: int | None = None,
    ):
        """Publish a single utterance to the Rails API."""
        try:
            payload = {
                "speaker": speaker,
                "text": text,
            }
            if pronunciation_score is not None:
                payload["pronunciation_score"] = pronunciation_score

            await self.client.post(
                f"/practice_sessions/{self.session_id}/transcript/append",
                json=payload,
            )
        except Exception as e:
            logger.error(f"Failed to publish transcript: {e}")

    async def publish_pronunciation_score(
        self,
        utterance_text: str,
        utterance_index: int,
        audio_data: bytes,
    ):
        """Forward utterance audio to Rails for async Speechace scoring."""
        try:
            await self.client.post(
                f"/practice_sessions/{self.session_id}/pronunciation_scores",
                json={
                    "utterance_text": utterance_text,
                    "utterance_index": utterance_index,
                },
            )
        except Exception as e:
            logger.error(f"Failed to publish pronunciation score: {e}")

    async def close(self):
        await self.client.aclose()
