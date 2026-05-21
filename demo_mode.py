"""
demo_mode.py  ──  JARVIS Temporary Demo Mode
=============================================
Run this instead of main.py when you want to record a demo.

On wake-word detection it will:
  1. Open the default browser          (immediately)
  2. Open Notepad                       (after 0.75 s)
  3. Open Command Prompt (cmd)          (after another 0.75 s)

Each delay is randomly sampled from [0.5, 1.0] seconds so the
sequence looks natural on camera.

Exit with  Ctrl+C.
"""

import logging
import queue
import random
import subprocess
import sys
import time
import webbrowser

import sounddevice as sd

from config import WAKE_WORD_NAME
from backend.wake_word.detector import WakeWordDetector

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("JARVIS.Demo")

# ── Audio settings (must match the detector's expected rate) ────────────────
SAMPLE_RATE = 16_000
CHUNK_SIZE  = 1_280          # same as MicrophoneListener

# ── Demo sequence ───────────────────────────────────────────────────────────
DELAY_MIN = 0.5              # seconds
DELAY_MAX = 1.0              # seconds


def _random_delay() -> float:
    """Return a delay in [DELAY_MIN, DELAY_MAX] and sleep for it."""
    delay = random.uniform(DELAY_MIN, DELAY_MAX)
    time.sleep(delay)
    return delay


def run_demo_sequence():
    """
    Open browser → Notepad → CMD with natural staggered delays.
    Called once each time the wake word fires.
    """
    logger.info("─── Demo sequence starting ───")

    # 1. Browser
    logger.info("Opening browser...")
    webbrowser.open("https://www.google.com")

    # 2. Notepad
    d = _random_delay()
    logger.info(f"[+{d:.2f}s] Opening Notepad...")
    subprocess.Popen(["notepad.exe"])

    # 3. CMD
    d = _random_delay()
    logger.info(f"[+{d:.2f}s] Opening Command Prompt...")
    subprocess.Popen(["cmd.exe"])

    logger.info("─── Demo sequence complete ───")


# ── Main listener loop ──────────────────────────────────────────────────────
def main():
    import os
    ww_display = (
        os.path.basename(WAKE_WORD_NAME).split(".")[0].replace("_", " ").title()
    )

    logger.info(f"JARVIS Demo Mode — listening for '{ww_display}'")
    logger.info("Press Ctrl+C to exit.\n")

    detector   = WakeWordDetector(WAKE_WORD_NAME)
    audio_q: queue.Queue = queue.Queue()

    def _callback(indata, frames, time_info, status):
        audio_q.put(indata.copy().flatten())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=CHUNK_SIZE,
        callback=_callback,
    ):
        try:
            while True:
                chunk = audio_q.get()
                if detector.process_audio(chunk):
                    logger.info("✅  Wake word detected!")
                    run_demo_sequence()
                    # Drain stale audio so a second trigger doesn't fire instantly
                    while not audio_q.empty():
                        audio_q.get()
                    logger.info(f"\nBack to listening for '{ww_display}'...\n")
        except KeyboardInterrupt:
            logger.info("Demo mode stopped. Goodbye.")
            sys.exit(0)


if __name__ == "__main__":
    main()
