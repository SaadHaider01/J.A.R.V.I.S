import whisper
import numpy as np
from config import WHISPER_MODEL_SIZE
import logging

logger = logging.getLogger("JARVIS.STT")

class SpeechToText:
    def __init__(self):
        logger.info(f"Loading Whisper model ({WHISPER_MODEL_SIZE})... This may take a moment.")
        # fp16=False prevents warnings if running on CPU instead of GPU.
        self.model = whisper.load_model(WHISPER_MODEL_SIZE)
        logger.info("Whisper model loaded successfully.")

    def transcribe(self, audio_data: np.ndarray) -> str:
        """
        Takes raw audio data (numpy array) and translates it to text.
        """
        # Ensure audio_data is a flat 32-bit float array as required by Whisper
        audio_data = audio_data.flatten().astype(np.float32)
        
        # We let whisper's internal padding handle the exact sizing automatically
        result = self.model.transcribe(audio_data, fp16=False)
        return result["text"].strip()
