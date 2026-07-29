"""Joins a practice room as the learner and speaks a WAV file.

Exists because the recording path cannot be verified without a real
publisher: a room containing only the agent is treated as empty, LiveKit
reaps it, and the egress aborts with "Start signal not received". Every
piece downstream — the MP4 in object storage, the channel split, the
transcript offsets — needs actual audio on the wire.

Usage (from the agent venv):

    python tools/fake_learner.py --token "$JWT" --url ws://localhost:7880 \
        --wav /tmp/learner_speech.wav --hold 25

Generate speech on macOS with:

    say -o /tmp/learner_speech.wav --data-format=LEI16@48000 "..."
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import wave

from livekit import rtc

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("fake-learner")

# 10ms of audio per frame is what LiveKit's examples use and what the SFU
# expects for smooth pacing.
FRAME_MS = 10


async def publish_wav(source: rtc.AudioSource, path: str, *, lead_in_s: float) -> None:
    with wave.open(path, "rb") as wav:
        channels = wav.getnchannels()
        rate = wav.getframerate()
        width = wav.getsampwidth()

        if width != 2:
            raise SystemExit(
                f"{path} is {width * 8}-bit; this needs signed 16-bit PCM "
                "(say --data-format=LEI16@48000)"
            )

        samples_per_frame = int(rate * FRAME_MS / 1000)

        # Silence first: the agent greets on join, and talking over its
        # opening line makes the transcript a mess to read back.
        logger.info("holding %.1fs of silence before speaking", lead_in_s)
        silence = b"\x00" * (samples_per_frame * channels * width)
        for _ in range(int(lead_in_s * 1000 / FRAME_MS)):
            await source.capture_frame(
                rtc.AudioFrame(silence, rate, channels, samples_per_frame)
            )

        logger.info("speaking %s (%dHz, %dch)", path, rate, channels)
        while True:
            data = wav.readframes(samples_per_frame)
            if not data:
                break
            # A short final frame would desync the sample counter; pad it.
            expected = samples_per_frame * channels * width
            if len(data) < expected:
                data += b"\x00" * (expected - len(data))
            await source.capture_frame(
                rtc.AudioFrame(data, rate, channels, samples_per_frame)
            )
        logger.info("finished speaking")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://localhost:7880")
    parser.add_argument("--token", required=True)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--lead-in", type=float, default=6.0)
    parser.add_argument(
        "--hold",
        type=float,
        default=20.0,
        help="seconds to stay in the room after the audio ends, so the "
        "agent's reply is captured by the recording too",
    )
    args = parser.parse_args()

    with wave.open(args.wav, "rb") as wav:
        rate, channels = wav.getframerate(), wav.getnchannels()

    room = rtc.Room()
    await room.connect(args.url, args.token)
    logger.info("connected as %s to %s", room.local_participant.identity, room.name)

    source = rtc.AudioSource(rate, channels)
    track = rtc.LocalAudioTrack.create_audio_track("learner-mic", source)
    await room.local_participant.publish_track(
        track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    )
    logger.info("microphone published")

    try:
        await publish_wav(source, args.wav, lead_in_s=args.lead_in)
        logger.info("holding the room open for %.1fs", args.hold)
        await asyncio.sleep(args.hold)
    finally:
        with contextlib.suppress(Exception):
            await room.disconnect()
        logger.info("disconnected")


if __name__ == "__main__":
    asyncio.run(main())
