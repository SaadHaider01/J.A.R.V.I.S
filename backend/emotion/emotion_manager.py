"""
=============================================================================
backend/emotion/emotion_manager.py
=============================================================================

WHAT THIS FILE DOES:
    Orchestrates the entire Adaptive Conversation subsystem. It receives 
    audio from the Duplex Manager, pipes it to the Feature Extractor, 
    passes the features to the Classifier, and updates the Emotion Context.

WHY THIS ARCHITECTURE?
    This is the Facade Pattern. The rest of the application (Agent, Duplex 
    Manager, TTS) only needs to interact with this one class. They don't 
    need to know about DSP extraction, heuristics, or rolling smoothing 
    windows. This keeps the subsystem completely decoupled.
=============================================================================
"""

import time
import numpy as np
from typing import Optional
from backend.emotion.emotion_models import EmotionState, EmotionResult
from backend.emotion.feature_extractor import FeatureExtractor
from backend.emotion.classifier import RuleBasedClassifier
from backend.emotion.emotion_context import EmotionContext
from backend.emotion.logger import log_emotion
from backend.emotion.metrics import emotion_metrics
from backend.duplex.constants import SAMPLE_RATE

class EmotionManager:
    """Facade for the Adaptive Conversation Subsystem."""
    
    def __init__(self):
        self.extractor = FeatureExtractor()
        self.classifier = RuleBasedClassifier()
        self.context = EmotionContext(history_size=3, time_decay_s=120.0)
        
        log_emotion("INIT", "Emotion Manager (Adaptive Conversation Engine) initialized.")

    def analyze_audio(self, audio_data: np.ndarray) -> Optional[EmotionResult]:
        """
        Analyzes a chunk of user speech, updates the rolling context, 
        and logs metrics.
        
        This is called by the Duplex Manager right after STT finishes.
        """
        if audio_data is None or len(audio_data) == 0:
            return None
            
        start_time = time.time()
        
        try:
            # 1. Feature Extraction (DSP)
            duration_s = len(audio_data) / SAMPLE_RATE
            features = self.extractor.extract_features(audio_data, duration_s)
            
            # 2. Classification (Heuristics)
            raw_result = self.classifier.classify(features)
            
            # 3. Context Updating (Smoothing)
            old_state = self.context.get_current_state()
            self.context.add_result(raw_result)
            new_state = self.context.get_current_state()
            
            # 4. Metrics Recording
            latency_s = time.time() - start_time
            emotion_metrics.record_analysis(
                emotion=raw_result.emotion.name, 
                confidence=raw_result.confidence, 
                latency_s=latency_s
            )
            
            if old_state != new_state:
                emotion_metrics.record_transition()
                
            log_emotion("ANALYSIS", f"Raw: {raw_result.emotion.name} ({raw_result.confidence:.2f}) | Smoothed: {new_state.name}")
            return raw_result
            
        except Exception as e:
            log_emotion("ERROR", f"Failed to analyze audio: {e}", level=40)
            return None

    def get_current_state(self) -> EmotionState:
        """Returns the smoothed conversational state."""
        return self.context.get_current_state()

    def get_prompt_injection(self) -> str:
        """Returns the formatted prompt block for the NLP Agent."""
        return self.context.format_prompt_injection()
        
    def get_tts_adaptation(self) -> dict:
        """
        Returns TTS configuration tweaks based on the current state.
        Prioritizes subtlety over extreme voice modulation.
        """
        state = self.get_current_state()
        
        if state in (EmotionState.FRUSTRATED, EmotionState.TIRED):
            # Slow down slightly to appear more patient/gentle
            return {"rate": "-10%"}
        elif state == EmotionState.EXCITED:
            # Speed up slightly to match energy
            return {"rate": "+5%"}
            
        return {} # Default for Calm, Neutral, Unknown
