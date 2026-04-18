import whisper
import numpy as np
from config import WHISPER_MODEL_SIZE
import logging

logger = logging.getLogger("JARVIS.STT")

# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary hint primes Whisper to expect JARVIS-specific words.
# This drastically reduces misrecognition of app names, commands, etc.
# ─────────────────────────────────────────────────────────────────────────────
_INITIAL_PROMPT = (
    "JARVIS assistant commands: open Brave, open Notepad, open WhatsApp, open Calculator, "
    "open Spotify, open YouTube, search for, play music, shut down, sleep, "
    "terminate, close, type, write, Google, Spider-Man, Iron Man, weather, "
    "news, volume up, volume down, screenshot, open file explorer."
)


class SpeechToText:
    def __init__(self):
        logger.info(f"Loading Whisper model ({WHISPER_MODEL_SIZE})... This may take a moment.")
        # fp16=False prevents warnings when running on CPU instead of GPU.
        self.model = whisper.load_model(WHISPER_MODEL_SIZE)
        logger.info("Whisper model loaded successfully.")

    def transcribe(self, audio_data: np.ndarray) -> str:
        """
        Takes raw audio data (numpy array) and translates it to text.
        - Language is locked to English to avoid misdetection with accented speech.
        - initial_prompt seeds Whisper with JARVIS-specific vocabulary.
        - condition_on_previous_text=False prevents hallucination chaining.
        - Silence is stripped from both ends before transcription.
        """
        # Ensure audio_data is a flat 32-bit float array as required by Whisper
        audio_data = audio_data.flatten().astype(np.float32)

        # Strip leading/trailing silence to avoid hallucinations from quiet segments
        audio_data = self._trim_silence(audio_data)

        if audio_data.size == 0:
            logger.debug("Audio was empty after silence trim — ignoring.")
            return ""

        result = self.model.transcribe(
            audio_data,
            fp16=False,
            language="en",                       # Lock to English — prevents language misdetection
            initial_prompt=_INITIAL_PROMPT,      # Seed with JARVIS vocabulary
            condition_on_previous_text=False,    # Prevent hallucination chaining
        )
        return result["text"].strip()

    def _trim_silence(self, audio: np.ndarray, top_db: float = 20.0) -> np.ndarray:
        """
        Strips silence from both ends of audio using a simple RMS energy threshold.
        top_db: How many dB below the peak RMS to consider as silence.
        """
        if audio.size == 0:
            return audio

        frame_length = 512
        hop_length = 256

        # Compute per-frame RMS energy
        frames = [
            audio[i: i + frame_length]
            for i in range(0, len(audio) - frame_length, hop_length)
        ]
        if not frames:
            return audio

        rms = np.array([np.sqrt(np.mean(f ** 2)) for f in frames])
        max_rms = rms.max()
        if max_rms == 0:
            return np.array([], dtype=np.float32)

        # Threshold: anything below (max / 10^(top_db/20)) is silence
        threshold = max_rms / (10 ** (top_db / 20))
        non_silent = np.where(rms > threshold)[0]

        if non_silent.size == 0:
            return np.array([], dtype=np.float32)

        start_sample = non_silent[0] * hop_length
        end_sample = min((non_silent[-1] + 1) * hop_length + frame_length, len(audio))
        return audio[start_sample:end_sample]
