"""
=============================================================================
backend/emotion/classifier.py
=============================================================================

WHAT THIS FILE DOES:
    Implements a Rule-Based Heuristic Classifier to map acoustic features
    (energy, pitch, tempo) into a conversational state (Excited, Tired, etc).

WHY THIS ARCHITECTURE?
    By separating the Classifier from the Feature Extractor, we can easily 
    rip this file out in the future and replace it with a PyTorch or 
    TensorFlow ML model, without touching the rest of the audio pipeline.

HOW IT WORKS:
    It uses simple boundary thresholds.
    - Loud + Fast = Excited
    - Loud + Slow/Abrupt = Frustrated
    - Quiet + Slow = Tired
    If the audio doesn't clearly match a heuristic, it falls back to Neutral.
    If the audio is too short or noisy, it returns Unknown.
=============================================================================
"""

import time
from backend.emotion.emotion_models import AudioFeatures, EmotionState, EmotionResult

class RuleBasedClassifier:
    """Maps extracted DSP features to a conversational state."""
    
    def __init__(self):
        # Base global thresholds (Future TODO: load from user baseline)
        self.HIGH_ENERGY_THRESH = 0.08
        self.LOW_ENERGY_THRESH  = 0.02
        
        self.HIGH_ZCR_THRESH    = 1500  # Crossings per sec (rough proxy)
        self.LOW_ZCR_THRESH     = 500
        
        self.FAST_SPEECH_RATIO  = 0.85  # Highly active buffer
        self.SLOW_SPEECH_RATIO  = 0.40  # Lots of pauses
        
    def classify(self, features: AudioFeatures) -> EmotionResult:
        """
        Runs heuristics to guess the conversational state.
        Returns an EmotionResult with a confidence score.
        """
        # 1. Reject garbage/too short audio
        if features.rms_energy == 0.0:
            return EmotionResult(EmotionState.UNKNOWN, 0.0, features, time.time())
            
        # 2. Heuristic Rules
        state = EmotionState.NEUTRAL
        confidence = 0.5 # Base confidence
        
        # Rule: EXCITED (Loud, fast, high pitch)
        if features.rms_energy > self.HIGH_ENERGY_THRESH and features.voice_activity_ratio > self.FAST_SPEECH_RATIO:
            state = EmotionState.EXCITED
            confidence = 0.85
            if features.pitch_estimate > self.HIGH_ZCR_THRESH:
                confidence = 0.95
                
        # Rule: FRUSTRATED (Loud, but slower/abrupt, low pitch)
        elif features.rms_energy > self.HIGH_ENERGY_THRESH and features.voice_activity_ratio < self.FAST_SPEECH_RATIO:
            state = EmotionState.FRUSTRATED
            confidence = 0.80
            if features.pitch_estimate < self.LOW_ZCR_THRESH:
                confidence = 0.90
                
        # Rule: TIRED (Quiet, slow, lots of pauses)
        elif features.rms_energy < self.LOW_ENERGY_THRESH and features.voice_activity_ratio < self.SLOW_SPEECH_RATIO:
            state = EmotionState.TIRED
            confidence = 0.80
            
        # Rule: CALM (Normal energy, steady/moderate pace)
        elif self.LOW_ENERGY_THRESH <= features.rms_energy <= self.HIGH_ENERGY_THRESH:
            # If the features are right in the middle, it's a calm/steady voice
            if 0.4 <= features.voice_activity_ratio <= 0.8:
                state = EmotionState.CALM
                confidence = 0.75
                
        # 3. Fallback and Confidence thresholds
        if confidence < 0.60:
            # Not enough strong signal to confidently assert an emotion
            state = EmotionState.UNKNOWN
            
        return EmotionResult(
            emotion=state,
            confidence=confidence,
            features=features,
            timestamp=time.time()
        )
