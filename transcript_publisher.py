"""Persists transcript utterances to Rails, one batch POST at a time.

Design constraints this encodes:

- Rails owns sequence numbers (assigned under a row lock server-side), so an
  agent-worker crash and rejoin cannot collide with rows that already exist.
  We send ChatMessage.id as external_id and Rails deduplicates on it, which
  makes every retry a no-op instead of a duplicate.
- Failures buffer in memory and are retried on the next enqueue and again in
  aclose() — a 3-second Rails restart must not silently eat a learner's
  conversation. aclose() runs from the job's shutdown callback.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

RAILS_API_URL = os.getenv("RAILS_API_URL", "http://localhost:8000")
AGENT_SERVICE_TOKEN = os.getenv("AGENT_SERVICE_TOKEN", "dev_agent_service_token")


class TranscriptPublisher:
    """Buffers utterances and ships them to Rails over HTTP."""

    def __init__(self, session_id: int):
        self.session_id = session_id
        self.client = httpx.AsyncClient(
            base_url=RAILS_API_URL,
            headers={"Authorization": f"Bearer {AGENT_SERVICE_TOKEN}"},
            timeout=10.0,
        )
        self._pending: list[dict[str, Any]] = []
        self._flush_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()

    def enqueue(
        self,
        *,
        external_id: str,
        speaker: str,
        text: str,
        spoke_started_at_ms: int | None = None,
        spoke_ended_at_ms: int | None = None,
        interrupted: bool = False,
        stt_confidence: float | None = None,
    ) -> None:
        """Queue one utterance and kick off an async flush.

        Sync on purpose: it is called from livekit event handlers, which are
        plain callbacks.
        """
        self._pending.append(
            {
                "external_id": external_id,
                "speaker": speaker,
                "text": text,
                "spoke_started_at_ms": spoke_started_at_ms,
                "spoke_ended_at_ms": spoke_ended_at_ms,
                "interrupted": interrupted,
                "stt_confidence": stt_confidence,
            }
        )
        task = asyncio.create_task(self.flush())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def flush(self) -> None:
        """Send everything pending. On failure, keep it for the next try."""
        async with self._flush_lock:
            if not self._pending:
                return

            batch, self._pending = self._pending, []
            try:
                response = await self.client.post(
                    f"/internal/practice_sessions/{self.session_id}/transcript_messages",
                    json={"messages": batch},
                )
                response.raise_for_status()
            except httpx.HTTPError as e:
                # Re-buffer at the front so ordering survives the retry.
                self._pending = batch + self._pending
                logger.warning(
                    "transcript flush failed (%s) — %d message(s) buffered",
                    e,
                    len(self._pending),
                )

    async def aclose(self) -> None:
        """Final flush, then close. Called from the shutdown callback."""
        try:
            await self.flush()
            if self._pending:
                logger.error(
                    "dropping %d unpersisted transcript message(s) for session %s",
                    len(self._pending),
                    self.session_id,
                )
        finally:
            await self.client.aclose()
