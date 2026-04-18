import openwakeword
from openwakeword.model import Model
import numpy as np
import logging

logger = logging.getLogger("JARVIS.WakeWord")

class WakeWordDetector:
    def __init__(self, wake_word="hey_mycroft"):
        """
        Initializes the openwakeword model.
        Available default models: 'hey_mycroft', 'alexa', 'hey_siri', 'timer', 'weather'
        """
        logger.info(f"Loading Wake Word model: {wake_word}")
        openwakeword.utils.download_models() # Ensures the models are downloaded locally
        self.model = Model(wakeword_models=[wake_word], inference_framework="onnx")
        self.wake_word = wake_word
        logger.info("Wake Word model loaded.")

    def process_audio(self, audio_chunk: np.ndarray) -> bool:
        """
        Feeds audio chunks into the wake word model.
        Returns True if the wake word was detected in this chunk.
        """
        # openwakeword expects 16khz, integer data (int16)
        prediction = self.model.predict(audio_chunk)
        
        # openwakeword stores predictions in a dictionary keyed by model name.
        # The value is a confidence score from 0.0 to 1.0.
        score = list(prediction.values())[0]
        
        # 0.5 is a generic sensitivity threshold. Higher means fewer false positives
        # but requires you to speak more clearly.
        if score > 0.5: 
            return True
        return False
