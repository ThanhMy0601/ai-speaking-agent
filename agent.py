"""
AI English Speaking App — LiveKit Voice Agent (v1.5 API)

Joins LiveKit rooms as an AI participant and orchestrates
the STT → LLM (RAG) → TTS voice pipeline.
"""

import os
import json
import logging
from dotenv import load_dotenv

load_dotenv()

from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    InterruptionOptions,
    JobContext,
    JobProcess,
    TurnHandlingOptions,
    WorkerOptions,
    cli,
    llm,
)
from livekit.plugins import deepgram, elevenlabs, silero
from livekit.plugins import google as google_plugins

from rag_context import RAGContextProvider
from transcript_publisher import TranscriptPublisher

logger = logging.getLogger("voice-agent")
logger.setLevel(logging.INFO)


class EnglishTutorAgent(Agent):
    """AI English speaking tutor agent."""

    def __init__(
        self,
        *,
        system_prompt: str,
        publisher: TranscriptPublisher | None = None,
    ) -> None:
        super().__init__(
            instructions=system_prompt,
        )
        self._publisher = publisher

    async def on_enter(self):
        """Called when the agent enters the session."""
        # Phase 3 replaces this with a per-topic opening line read from the
        # database, so the tutor opens *in* the topic the learner already
        # picked instead of asking them to pick one again.
        self.session.say(
            "Hello! I'm your English practice partner. What would you like to talk about today?"
        )


def prewarm(proc: JobProcess):
    """Preload the Silero VAD model for faster startup."""
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    """Main agent entrypoint — called when a new room is created."""

    # Extract session metadata from room metadata
    room_metadata = {}
    if ctx.room.metadata:
        try:
            room_metadata = json.loads(ctx.room.metadata)
        except json.JSONDecodeError:
            pass

    topic_id = room_metadata.get("topic_id")
    session_id = room_metadata.get("session_id")

    # Initialize topic context
    rag = RAGContextProvider()

    system_prompt = build_system_prompt(topic_id=topic_id, rag=rag)

    # Initialize transcript publisher
    publisher = TranscriptPublisher(session_id) if session_id else None

    # Connect to the room
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Create the agent
    agent = EnglishTutorAgent(
        system_prompt=system_prompt,
        publisher=publisher,
    )
    agent._topic_id = topic_id

    # Create and start the session with the voice pipeline
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(
            model="nova-2",
            language="en",
        ),
        llm=google_plugins.LLM(
            model="gemini-2.5-flash",
            temperature=0.7,
        ),
        tts=elevenlabs.TTS(
            model="eleven_turbo_v2",
            voice_id="EXAVITQu4vr4xnSDxMaL",  # Sarah - premade, works on free tier
        ),
        turn_handling=TurnHandlingOptions(
            interruption=InterruptionOptions(mode="vad", enabled=True),
        ),
    )

    await session.start(
        room=ctx.room,
        agent=agent,
    )


def build_system_prompt(
    topic_id: int | None,
    rag: RAGContextProvider,
) -> str:
    """Build the LLM system prompt from the selected topic's context."""

    base_prompt = (
        "You are an AI English speaking tutor. Your role is to help learners "
        "practice spoken English in a natural, encouraging way. "
        "Speak clearly and at a moderate pace. "
        "Gently correct grammar and pronunciation errors. "
        "Keep responses concise (2-3 sentences) to maintain conversation flow.\n\n"
    )

    return (
        base_prompt
        + "Guide the conversation around the following topic. Keep it engaging and educational:\n\n"
        + rag.get_topic_context(topic_id)
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
