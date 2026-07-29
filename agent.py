"""
AI English Speaking App — LiveKit Voice Agent (v1.5 API)

Joins LiveKit rooms as an AI participant and orchestrates the
STT → LLM → TTS voice pipeline. All teaching content comes from Rails via
ContextClient; nothing about what to teach lives in this repo.
"""

import json
import logging
import os
import time

from dotenv import load_dotenv

load_dotenv()

from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess,
    TurnHandlingOptions,
    WorkerOptions,
    cli,
)
from livekit.plugins import deepgram, elevenlabs, silero
from livekit.plugins import google as google_plugins

from context_client import AgentContext, ContextClient
from prompts import build_instructions
from transcript_publisher import TranscriptPublisher

logger = logging.getLogger("voice-agent")
logger.setLevel(logging.INFO)

# Adaptive interruption is a LiveKit Cloud service, not a local model, and it
# bills per use. Off by default so a self-hosted server works out of the box;
# set ADAPTIVE_INTERRUPTION=true only when running against LiveKit Cloud with
# credentials that are valid there.
ADAPTIVE_INTERRUPTION = os.getenv("ADAPTIVE_INTERRUPTION", "").lower() in ("1", "true", "yes")


class EnglishTutorAgent(Agent):
    """AI English speaking tutor agent."""

    def __init__(
        self,
        *,
        context: AgentContext | None = None,
        publisher: TranscriptPublisher | None = None,
    ) -> None:
        ctx = context or AgentContext()
        super().__init__(instructions=build_instructions(ctx))
        self._context = ctx
        self._publisher = publisher

    async def on_enter(self):
        """Speak the topic's opening line as soon as the agent joins.

        The learner already picked a topic on the previous screen, so the old
        behaviour — a single hardcoded "What would you like to talk about
        today?" for every session — made them choose twice.

        session.say() speaks the database text verbatim rather than asking the
        LLM to improvise a greeting, which is what makes "each topic has its
        own opening line" mean something. add_to_chat_ctx keeps the model
        aware of what it already said so it doesn't greet twice.
        """
        opening = self._context.opening_line

        if opening:
            if self._context.learner.display_name:
                opening = opening.replace("{name}", self._context.learner.display_name)
            self.session.say(opening, add_to_chat_ctx=True)
        else:
            self.session.generate_reply(
                instructions=(
                    "Greet the learner warmly in one or two sentences and open "
                    "the conversation with a question."
                )
            )


def prewarm(proc: JobProcess):
    """Preload the Silero VAD model for faster startup."""
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    """Main agent entrypoint — called when a new room is created."""

    # Connect first: no blocking work before the room is joined, so a slow
    # dependency can never leave the learner in a room with no agent.
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    room_metadata = {}
    if ctx.room.metadata:
        try:
            room_metadata = json.loads(ctx.room.metadata)
        except json.JSONDecodeError:
            logger.warning("room metadata is not valid JSON", extra={"room": ctx.room.name})

    session_id = room_metadata.get("session_id")
    log_ctx = {
        "session_id": session_id,
        "user_id": room_metadata.get("user_id"),
        "topic_id": room_metadata.get("topic_id"),
    }

    context_client = ContextClient()
    ctx.add_shutdown_callback(context_client.aclose)

    agent_context = await context_client.fetch(session_id)

    logger.info(
        "session context loaded: topic=%s level=%s attempt=%s proficiency=%s",
        (agent_context.topic or {}).get("title"),
        (agent_context.level or {}).get("level"),
        agent_context.learner.attempt_number,
        agent_context.learner.proficiency_level,
        extra=log_ctx,
    )

    publisher = TranscriptPublisher(session_id) if session_id else None
    if publisher:
        ctx.add_shutdown_callback(publisher.aclose)

    agent = EnglishTutorAgent(context=agent_context, publisher=publisher)

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(
            model="nova-2",
            language="en",
            # Explicit even though these match the plugin defaults: filler
            # words are what make a later hesitation analysis real data, and
            # smart_format would inject punctuation the LLM would then try to
            # "correct" as a learner error.
            filler_words=True,
            smart_format=False,
        ),
        llm=google_plugins.LLM(
            model="gemini-2.5-flash",
            temperature=0.7,
        ),
        tts=elevenlabs.TTS(
            model="eleven_turbo_v2",
            voice_id="EXAVITQu4vr4xnSDxMaL",  # Sarah - premade, works on free tier
        ),
        turn_handling=build_turn_handling(agent_context),
    )

    await session.start(room=ctx.room, agent=agent)

    if publisher:
        wire_transcript_persistence(session, publisher)


def wire_transcript_persistence(session: AgentSession, publisher: TranscriptPublisher) -> None:
    """Persist the conversation as it happens.

    conversation_item_added is the transcript source of truth: it fires for
    BOTH speakers with final text, in order. (user_input_transcribed fires
    per interim chunk and would persist half-sentences.)

    user_state_changed supplies the wall-clock speech timestamps.
    ChatMessage.created_at alone is the moment the item was added — after
    endpointing, i.e. 0.9-6s late — useless for slicing audio at these
    times in Phase 5. The agent sends ABSOLUTE epoch-ms; it cannot know
    "ms from recording start" because egress starts in another process at
    an unknown moment, so Rails computes offsets from egress T0 later.
    """
    speech_window: dict[str, int] = {}

    @session.on("user_state_changed")
    def _on_user_state(ev) -> None:
        now_ms = int(time.time() * 1000)
        if ev.new_state == "speaking":
            speech_window["started_at"] = now_ms
        elif ev.old_state == "speaking":
            speech_window["ended_at"] = now_ms

    @session.on("conversation_item_added")
    def _on_item(ev) -> None:
        item = ev.item
        text = item.text_content
        if item.type != "message" or not text:
            return

        if item.role == "user":
            started = speech_window.pop("started_at", None)
            ended = speech_window.pop("ended_at", None)
        else:
            started = int(item.created_at * 1000)
            ended = None

        publisher.enqueue(
            external_id=item.id,
            speaker="learner" if item.role == "user" else "ai",
            text=text,
            spoke_started_at_ms=started,
            spoke_ended_at_ms=ended,
            interrupted=bool(item.interrupted),
            stt_confidence=item.transcript_confidence,
        )


def build_turn_handling(context: AgentContext) -> TurnHandlingOptions:
    """Turn-taking tuned for hesitant second-language speakers.

    The SDK default is fixed endpointing at 0.5s of silence, which is tuned
    for native speakers and chronically cuts off learners who pause
    mid-sentence to find a word.

    On interruption mode: "adaptive" is NOT a local ML classifier, it is a
    LiveKit Cloud inference service (agent-gateway.livekit.cloud). Leaving
    `mode` unset lets the SDK auto-select it whenever a streaming STT and a
    VAD are present, which against a self-hosted server authenticates with
    the local devkey, gets a 401, and kills the job with
    "failed to detect interruption after 3 attempts" — the agent joins the
    room, publishes a track, and leaves. So the mode is pinned explicitly,
    and only opts into the cloud service when told to.
    """
    tuning = context.turn_tuning
    mode = "adaptive" if ADAPTIVE_INTERRUPTION else "vad"

    return TurnHandlingOptions(
        endpointing={
            # "dynamic" extends the wait when an utterance sounds unfinished
            # — a trailing conjunction, rising intonation — which is exactly
            # the "I went to the... um..." case.
            "mode": "dynamic",
            "min_delay": tuning.get("min_delay", 0.9),
            "max_delay": tuning.get("max_delay", 6.0),
        },
        interruption={
            "enabled": True,
            "mode": mode,
            # These three are what keep plain VAD usable for learners: a
            # cough or a lone "umm" is too short and too few words to count,
            # and if the learner starts then stalls, the tutor resumes its
            # sentence instead of abandoning it.
            "min_duration": 0.6,
            "min_words": 2,
            "resume_false_interruption": True,
            "false_interruption_timeout": 3.0,
        },
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
