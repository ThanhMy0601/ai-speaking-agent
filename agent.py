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
        session_type = getattr(self, "_session_type", "free_practice")
        scenario = getattr(self, "_scenario", "job_interview")

        greetings = {
            "free_practice": "Hello! I'm your English practice partner. What would you like to talk about today?",
            "ielts_mock_test": "Welcome to your IELTS Speaking mock test. I'll be your examiner today. Let's begin with Part 1. Could you tell me your full name, please?",
            "role_play": f"Let's begin our role-play scenario. {get_roleplay_greeting(scenario)}",
        }
        greeting = greetings.get(session_type, greetings["free_practice"])
        self.session.say(greeting)


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

    session_type = room_metadata.get("session_type", "free_practice")
    lesson_id = room_metadata.get("lesson_id")
    scenario = room_metadata.get("scenario", "job_interview")
    session_id = room_metadata.get("session_id")

    # Initialize RAG context
    rag = RAGContextProvider()

    # Build system prompt based on session type
    system_prompt = build_system_prompt(
        session_type=session_type,
        lesson_id=lesson_id,
        scenario=scenario,
        rag=rag,
    )

    # Initialize transcript publisher
    publisher = TranscriptPublisher(session_id) if session_id else None

    # Connect to the room
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Create the agent
    agent = EnglishTutorAgent(
        system_prompt=system_prompt,
        publisher=publisher,
    )
    agent._session_type = session_type
    agent._scenario = scenario

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
    session_type: str,
    lesson_id: int | None,
    scenario: str,
    rag: RAGContextProvider,
) -> str:
    """Build the LLM system prompt based on session type and RAG context."""

    base_prompt = (
        "You are an AI English speaking tutor. Your role is to help learners "
        "practice spoken English in a natural, encouraging way. "
        "Speak clearly and at a moderate pace. "
        "Gently correct grammar and pronunciation errors. "
        "Keep responses concise (2-3 sentences) to maintain conversation flow.\n\n"
    )

    if session_type == "ielts_mock_test":
        return (
            "You are an IELTS Speaking examiner. Follow the official IELTS Speaking test format strictly.\n"
            "Maintain a professional, neutral tone. Do not help the candidate with answers.\n"
            "Ask follow-up questions to probe deeper. Assess fluency, coherence, lexical resource, "
            "grammatical range, and pronunciation.\n\n"
            + rag.get_ielts_context(1)
        )

    if session_type == "role_play":
        return (
            base_prompt
            + "You are playing a specific professional role in a business scenario.\n"
            "Stay in character throughout the conversation.\n"
            "Use professional vocabulary appropriate to the scenario.\n\n"
            + rag.get_roleplay_context(scenario)
        )

    # Free practice
    lesson_context = rag.get_lesson_context(lesson_id)
    return (
        base_prompt
        + "Incorporate the following lesson context into the conversation naturally:\n\n"
        + lesson_context
    )


def get_roleplay_greeting(scenario: str) -> str:
    """Get the opening line for a role-play scenario."""
    greetings = {
        "salary_negotiation": "Thank you for coming in. I understand you'd like to discuss the compensation package we offered. What are your thoughts?",
        "client_presentation": "Thank you for scheduling this demo. We're evaluating several solutions. Please go ahead with your presentation.",
        "job_interview": "Welcome, please have a seat. Thank you for coming in today. Let's start — can you tell me a bit about yourself?",
        "team_meeting_facilitation": "Hi everyone, thanks for joining. I believe you're leading today's meeting?",
        "conflict_resolution": "I wanted to talk about the project direction. I have some concerns about the current approach.",
    }
    return greetings.get(scenario, greetings["job_interview"])


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
