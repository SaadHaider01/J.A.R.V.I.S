# ==============================================================================
# J.A.R.V.I.S — DUPLEX CONVERSATION CONSTANTS
# ==============================================================================
# WHAT THIS MODULE DOES:
# This file serves as the single source of truth (SSOT) for all configuration
# parameters, timeouts, queue sizes, and thresholds governing the real-time,
# full-duplex conversational engine.
#
# WHY IT EXISTS:
# In real-time systems, hardcoded values (also known as "magic numbers") spread
# throughout multiple files lead to bugs, race conditions, and massive tuning
# headaches. Centralizing them here ensures that tuning latency vs. accuracy
# can be done in one single place.
#
# WHAT PROBLEM IT PREVENTS:
# Prevents state mismatch where one module expects 30ms audio chunks but another
# expects 50ms, causing buffer underruns or audio processing clipping.
# ==============================================================================

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# ── AUDIO STREAM SETTINGS ─────────────────────────────────────────────────────
# We use 16000Hz (16kHz) as it is the standard sample rate required by both the
# Whisper Speech-to-Text model and the openWakeWord inference engine.
SAMPLE_RATE = 16000

# Chunk duration determines the size of each audio packet processed by the system.
# 30 milliseconds offers the perfect balance:
#   - Short enough to minimize latency (we check for interrupts every 30ms).
#   - Long enough to avoid high CPU overhead from processing too many small packets.
CHUNK_DURATION_MS = 30
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # = 480 samples

# ── CENTRAL AUDIO BUS QUEUE SETTINGS ──────────────────────────────────────────
# Queue overflow protection limits memory growth if the system falls behind.
# 200 chunks = 6.0 seconds of audio buffer.
# If the queue exceeds this limit, we drop the oldest frames to preserve real-time
# responsiveness.
QUEUE_MAXSIZE = 200

# Queue pressure alert percentage. If the queue length exceeds 80% of its maxsize
# (160 chunks), we issue warnings because processing latency is about to spike.
QUEUE_PRESSURE_WARNING_PCT = 0.8

# ── ENERGY THRESHOLDS (BARGE-IN DETECTION) ────────────────────────────────────
# Root Mean Square (RMS) threshold for silence/speech detection.
# NORMAL_ENERGY_THRESHOLD is lowered to 0.003 (from 0.01) when the assistant is silent.
# WHY THIS WAS LOWERED:
# Human speech at startup or at a distance might not hit 0.01 RMS consistently.
# Setting it to 0.003 ensures soft voices or low microphone gains still trigger VAD
# without being discarded as "short non-speech events", while keeping it high enough
# to reject ambient background noise.
#
# INTERRUPT_ENERGY_THRESHOLD is lowered to 0.02 (from 0.05) when speaking.
# WHY THIS WAS TUNED:
# 0.05 required the user to shout to trigger an interruption. A threshold of 0.02
# allows natural speaking volume to trigger barge-in while remaining above the typical
# room echo level of speaker playback.
NORMAL_ENERGY_THRESHOLD = 0.003
INTERRUPT_ENERGY_THRESHOLD = 0.02

# ── TIMING WINDOWS & COOLDOWNS ────────────────────────────────────────────────
# Cooldown window in seconds after speech playback stops.
# Sound waves physically bounce around the room for a few hundred milliseconds,
# and the Windows audio driver takes time to release the play handle.
# A 0.5s cooldown prevents the system from immediately falsely interrupting itself
# due to lingering echo.
INTERRUPT_COOLDOWN_S = 0.5

# Delay before THINKING-state interruption monitoring begins (seconds).
# The user's own voice decays in the room for ~0.5–0.8s after they stop speaking.
# 0.8s is enough to let tail-decay settle without creating a dead zone on fast
# Groq responses that complete in under 1.5s.
THINKING_INTERRUPT_GUARD_S = 0.8

# Soft interrupt timing in milliseconds.
# Require consistent speech for 90ms (3 consecutive 30ms chunks) before triggering.
# WHY REDUCED FROM 150ms:
# Short commands like "stop", "hey", or "wait" are 80–100ms of peak energy.
# At 150ms (5 chunks) those were being missed. 3 chunks still rejects single
# pop/click transients while catching real speech reliably.
SOFT_INTERRUPT_MS = 90
SOFT_INTERRUPT_CHUNKS = int(SOFT_INTERRUPT_MS / CHUNK_DURATION_MS)  # = 3 chunks

# Auto-stop recording (VAD) timing.
# Stop recording user input after 1.5 seconds of silence (50 chunks of 30ms).
SILENCE_CHUNKS_TO_STOP = int(1500 / CHUNK_DURATION_MS)

# Minimum speech duration to filter out quick background pops/clicks (0.5s).
MIN_SPEECH_DURATION_S = 0.5

# Hard ceiling limit for any single recording session (10 seconds) to prevent infinite loops.
MAX_RECORDING_S = 10

# ── PRE-SPEECH RING BUFFER ────────────────────────────────────────────────────
# Ring buffer size in milliseconds.
# We buffer the last 1.0 second of audio chunks at all times.
# When an interruption is triggered, we prepend these pre-speech chunks to the
# recording. This prevents the beginning of the user's spoken phrase from being
# clipped off during the time it takes the system to transition states.
PRE_SPEECH_BUFFER_MS = 1000
PRE_SPEECH_BUFFER_CHUNKS = int(PRE_SPEECH_BUFFER_MS / CHUNK_DURATION_MS)  # ~33 chunks

# ── STATE TIMEOUTS (WATCHDOG RECOVERY) ────────────────────────────────────────
# Protection against stuck threads/APIs. If the assistant stays in a state
# longer than these limits, the Watchdog thread will force-recover the system.
# THINKING timeout (30s) handles network timeouts to Groq.
# SPEAKING timeout (60s) handles audio stream/MCI lockups.
STATE_TIMEOUTS = {
    "THINKING": 30.0,
    "SPEAKING": 60.0
}

# ── LATENCY BUDGET TARGETS (MILLISECONDS) ─────────────────────────────────────
# Targets for assessing perceived conversational fluidity:
# - Interrupt detection: < 150ms (the delay between speaking and system acknowledging).
# - TTS Stop: < 100ms (how fast audio stops playing physically).
# - State Transition: < 50ms (thread-safe state lock update speed).
GOAL_INTERRUPT_DETECTION_MS = 150.0
GOAL_TTS_STOP_MS = 100.0
GOAL_STATE_TRANSITION_MS = 50.0
