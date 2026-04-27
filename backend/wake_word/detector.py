import openwakeword
from openwakeword.model import Model
import numpy as np
import logging

logger = logging.getLogger("JARVIS.WakeWord")

class WakeWordDetector:
    def __init__(self, wake_word="jarvis"):
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
        from config import WAKE_WORD_SENSITIVITY
        
        # openwakeword expects 16khz, integer data (int16)
        prediction = self.model.predict(audio_chunk)
        
        # Extract the highest score from the prediction dictionary
        if not prediction:
            return False
            
        score = max(prediction.values())
        
        # Use the sensitivity threshold from config.py
        if score > WAKE_WORD_SENSITIVITY: 
            logger.info(f"Wake word detected with confidence: {score:.2f}")
            return True
        return False
