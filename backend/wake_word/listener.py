import sounddevice as sd
import numpy as np
import queue
import logging
import collections
import random
import subprocess
import time
import webbrowser
from config import CONVERSATION_TIMEOUT, WAKE_WORD_NAME, DEMO_MODE
from backend.wake_word.detector import WakeWordDetector
from backend.voice.tts import speak
from backend.voice.stt import SpeechToText
from backend.nlp.agent import ZytrixAgent

logger = logging.getLogger("ZYTRIX.Listener")

# ───────────────────────────────────────────────────────────────────────────────
# DEMO MODE helper
# Opens browser → Notepad → CMD with a natural 0.5–1 s stagger.
# Called by listen_for_wake_word() when DEMO_MODE=True in config.py.
# ───────────────────────────────────────────────────────────────────────────────

def _run_demo_sequence():
    """Demo: browser → Notepad → CMD with staggered delays."""
    logger.info("[DEMO] Starting demo sequence...")
    # NOTE: No speak() here — TTS is non-blocking and its audio would
    # echo back into the mic after ignore_mic=False, re-triggering the
    # wake word in a cascade loop.

    # 1. Browser
    logger.info("[DEMO] Opening browser...")
    webbrowser.open("https://www.google.com")

    # 2. Notepad
    delay = random.uniform(0.5, 1.0)
    time.sleep(delay)
    logger.info(f"[DEMO] (+{delay:.2f}s) Opening Notepad...")
    subprocess.Popen(["notepad.exe"])

    # 3. CMD — CREATE_NEW_CONSOLE ensures it gets its own visible window
    # and taskbar entry instead of inheriting the parent ZYTRIX console.
    delay = random.uniform(0.5, 1.0)
    time.sleep(delay)
    logger.info(f"[DEMO] (+{delay:.2f}s) Opening Command Prompt...")
    subprocess.Popen(["cmd.exe"], creationflags=subprocess.CREATE_NEW_CONSOLE)

    logger.info("[DEMO] Sequence complete. Resuming wake-word detection.")


# Words Whisper hallucinates when it hears silence
SILENCE_PHRASES = {
    "", ".", "thank you.", "bye.", "stop.", "never mind.",
    "never mind", "stop", "dismissed", "go to sleep", "goodbye.", "goodbye"
}

# ─────────────────────────────────────────────────────────────────────────────
# VAD (Voice Activity Detection) settings
# We record in small chunks and stop as soon as the user stops speaking.
# This eliminates the fixed 10-second wait and makes commands feel instant.
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 30          # Size of each audio chunk in milliseconds
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # = 480 samples
MIN_SPEECH_DURATION_S = 0.5     # Ignore anything shorter than 0.5s (noise)
MAX_RECORDING_S = 10            # Hard ceiling: never record more than 10s
# Stop recording after this many consecutive silent chunks
SILENCE_CHUNKS_TO_STOP = int(1500 / CHUNK_DURATION_MS)  # 1.5s of silence = 50 chunks


# Phrases that activate / deactivate demo mode at runtime
_DEMO_ON_PHRASES  = {"activate demo mode", "enable demo mode", "start demo mode", "demo mode on"}
_DEMO_OFF_PHRASES = {"deactivate demo mode", "disable demo mode", "stop demo mode", "demo mode off"}

# Seconds after a demo run during which wake word re-triggers are suppressed.
# The openWakeWord model keeps an internal rolling activation window that
# doesn't clear when audio stops, so false positives fire the instant the mic
# reopens. A timestamp guard is the only reliable way to block them.
DEMO_RETRIGGER_GUARD_S = 8


class MicrophoneListener:
    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.chunk_size = 1280
        self.audio_queue = queue.Queue()

        self.detector = WakeWordDetector(WAKE_WORD_NAME)
        self.stt = SpeechToText()
        self.agent = ZytrixAgent()
        self.is_listening = False
        self.ignore_mic = False  # Safety lock: prevents ZYTRIX from hearing himself
        # Runtime demo-mode flag — toggled by voice command, not a restart
        self.demo_mode = DEMO_MODE
        # Timestamp of the last completed demo run (0 = never ran)
        self._demo_last_run: float = 0.0

    def audio_callback(self, indata, frames, time, status):
        """Streams raw mic bytes into a queue for the Wake Word detector."""
        if self.ignore_mic:
            return
        # Stream is dtype='int16' — pass directly to openWakeWord (no conversion needed)
        self.audio_queue.put(indata.copy().flatten())

    def _record_with_vad(self) -> np.ndarray:
        """
        Records audio from the microphone using Voice Activity Detection.
        Instead of always waiting CONVERSATION_TIMEOUT seconds, this function
        stops recording automatically after 1.5 seconds of silence following
        the user's speech, capped at MAX_RECORDING_S total.

        Returns a float32 numpy array of the recorded audio.
        """
        logger.info("Mic open — listening for speech (VAD active)...")

        frames_collected = []
        silent_chunk_count = 0
        max_chunks = int(MAX_RECORDING_S * 1000 / CHUNK_DURATION_MS)
        min_chunks_for_speech = int(MIN_SPEECH_DURATION_S * 1000 / CHUNK_DURATION_MS)
        speech_detected = False

        # Energy threshold: frames with RMS below this are considered silent.
        # 0.01 works well for typical mic levels; adjust if needed.
        energy_threshold = 0.01

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            blocksize=CHUNK_SAMPLES,
        ) as stream:
            for _ in range(max_chunks):
                chunk, _ = stream.read(CHUNK_SAMPLES)
                chunk = chunk.flatten()
                frames_collected.append(chunk)

                # Calculate RMS energy of this chunk
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                is_silent = rms < energy_threshold

                if not is_silent:
                    speech_detected = True
                    silent_chunk_count = 0
                else:
                    if speech_detected:
                        silent_chunk_count += 1

                # Stop if we've had enough silence after speech was detected
                if speech_detected and silent_chunk_count >= SILENCE_CHUNKS_TO_STOP:
                    logger.info(
                        f"VAD: silence detected after speech — stopping early "
                        f"({len(frames_collected) * CHUNK_DURATION_MS / 1000:.1f}s recorded)."
                    )
                    break

        if not frames_collected:
            return np.array([], dtype=np.float32)

        audio = np.concatenate(frames_collected)

        # If we never detected any real speech, return empty so it's treated as silence
        if not speech_detected or len(frames_collected) < min_chunks_for_speech:
            logger.debug("VAD: no meaningful speech found in recording.")
            return np.array([], dtype=np.float32)

        return audio

    def listen_for_wake_word(self):
        """
        The master event loop. Sits silently until 'Hey Mycroft' is detected,
        then enters a continuous conversation loop until the user goes silent.
        """
        self.is_listening = True

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='int16',
            blocksize=self.chunk_size,
            callback=self.audio_callback
        ):
                # Dynamic log message based on WAKE_WORD_NAME
            from config import WAKE_WORD_NAME
            import os
            ww_display = os.path.basename(WAKE_WORD_NAME).split('.')[0].replace('_', ' ')
            logger.info(f"ZYTRIX is actively listening for '{ww_display}'...")

            while self.is_listening:
                audio_chunk = self.audio_queue.get()

                if self.detector.process_audio(audio_chunk):

                    # ── Timestamp guard: suppress re-triggers after a demo ───────
                    # The detector's internal rolling window keeps stale signal
                    # even when the mic is off, causing instant false positives
                    # the moment ignore_mic goes False. We skip any trigger that
                    # fires within DEMO_RETRIGGER_GUARD_S of the last demo run.
                    if self.demo_mode and (time.time() - self._demo_last_run) < DEMO_RETRIGGER_GUARD_S:
                        logger.debug(
                            f"[DEMO] Re-trigger suppressed "
                            f"({time.time() - self._demo_last_run:.1f}s < {DEMO_RETRIGGER_GUARD_S}s guard)."
                        )
                        continue
                    # ───────────────────────────────────────────────────────

                    logger.info("Wake word detected! Entering conversation mode.")
                    self.ignore_mic = True

                    # ── DEMO MODE: skip STT/agent, just open the 3 apps ───────────
                    if self.demo_mode:
                        _run_demo_sequence()
                        # Record the finish time so the timestamp guard above
                        # can block the detector's false-positive re-triggers.
                        self._demo_last_run = time.time()
                        # Brief drain then re-open mic; the timestamp guard
                        # handles the rest — no long sleep needed here.
                        while not self.audio_queue.empty():
                            self.audio_queue.get()
                        self.ignore_mic = False
                        logger.info("[DEMO] Mic re-enabled. Re-trigger guard active for "
                                    f"{DEMO_RETRIGGER_GUARD_S}s.")
                        continue
                    # ───────────────────────────────────────────────────────

                    in_conversation = True
                    while in_conversation:
                        # ── Record the user's spoken command (VAD-aware) ──────
                        command_audio = self._record_with_vad()

                        # ── Transcribe audio to text ───────────────────────
                        logger.info("Transcribing...")
                        command_text = self.stt.transcribe(command_audio)
                        logger.info(f"User said: '{command_text}'")

                        clean = command_text.lower().strip()
                        # Strip trailing punctuation Whisper commonly adds
                        # e.g. "activate demo mode." → "activate demo mode"
                        clean_nopunct = clean.rstrip(".,!?;:")

                        # ── Check for silence or exit command ──────────────
                        if clean in SILENCE_PHRASES or not clean:
                            logger.info("Silence or exit command detected. Sleeping.")
                            speak("Standing by.")
                            in_conversation = False
                            continue

                        # ── Demo mode toggle (intercepted before the agent) ─
                        # Match exact phrase OR substring so Whisper variants
                        # like "activate demo mode, please" are still caught.
                        _demo_on  = (clean_nopunct in _DEMO_ON_PHRASES  or
                                     any(p in clean_nopunct for p in _DEMO_ON_PHRASES))
                        _demo_off = (clean_nopunct in _DEMO_OFF_PHRASES or
                                     any(p in clean_nopunct for p in _DEMO_OFF_PHRASES))

                        if _demo_on:
                            self.demo_mode = True
                            logger.info(f"[DEMO] Demo mode ENABLED by voice command. (heard: '{clean}')")
                            speak("Demo mode activated. I'll open browser, Notepad, and Command Prompt on the next wake word.")
                            in_conversation = False
                            continue

                        if _demo_off:
                            self.demo_mode = False
                            logger.info(f"[DEMO] Demo mode DISABLED by voice command. (heard: '{clean}')")
                            speak("Demo mode deactivated. Resuming normal command mode.")
                            in_conversation = False
                            continue
                        # ───────────────────────────────────────────────────

                        # ── Hand ALL commands to the Agent Brain ───────────
                        logger.info("Routing to Agent Brain...")
                        reply = self.agent.think(command_text)
                        speak(reply)

                        # Flush stale audio from wake-word queue
                        while not self.audio_queue.empty():
                            self.audio_queue.get()

                    # ── Return to Wake Word detection ──────────────────────
                    self.ignore_mic = False
                    logger.info(f"Resuming wake word detection. Waiting for '{ww_display}'...")

    def stop(self):
        self.is_listening = False
