"""
=============================================================================
test_emotion_system.py
=============================================================================
Tests the Adaptive Conversation Engine (Emotion Subsystem).
We mock numpy arrays to simulate different conversational states and ensure
the rule-based classifier and rolling context smoothing behave correctly.
=============================================================================
"""

import unittest
import numpy as np
import time
from backend.emotion.emotion_models import EmotionState, AudioFeatures, EmotionResult
from backend.emotion.feature_extractor import FeatureExtractor
from backend.emotion.classifier import RuleBasedClassifier
from backend.emotion.emotion_context import EmotionContext
from backend.emotion.emotion_manager import EmotionManager

class TestEmotionSystem(unittest.TestCase):

    def setUp(self):
        self.extractor = FeatureExtractor()
        self.classifier = RuleBasedClassifier()
        self.context = EmotionContext(history_size=3)

    def test_feature_extractor_silence(self):
        """Empty or silent audio should yield zero energy."""
        audio = np.zeros(16000, dtype=np.float32)
        features = self.extractor.extract_features(audio, 1.0)
        self.assertEqual(features.rms_energy, 0.0)
        self.assertEqual(features.pitch_estimate, 0.0)

    def test_classifier_excited(self):
        """High energy + high voice activity + high pitch -> EXCITED"""
        features = AudioFeatures(
            rms_energy=0.1,             # > 0.08 (HIGH)
            pitch_estimate=1600.0,      # > 1500 (HIGH)
            speech_rate=1.0, 
            voice_activity_ratio=0.9    # > 0.85 (FAST)
        )
        result = self.classifier.classify(features)
        self.assertEqual(result.emotion, EmotionState.EXCITED)
        self.assertTrue(result.confidence > 0.8)

    def test_classifier_tired(self):
        """Low energy + low voice activity -> TIRED"""
        features = AudioFeatures(
            rms_energy=0.01,            # < 0.02 (LOW)
            pitch_estimate=200.0,
            speech_rate=1.0,
            voice_activity_ratio=0.3    # < 0.40 (SLOW)
        )
        result = self.classifier.classify(features)
        self.assertEqual(result.emotion, EmotionState.TIRED)

    def test_classifier_frustrated(self):
        """High energy + low voice activity + low pitch -> FRUSTRATED"""
        features = AudioFeatures(
            rms_energy=0.1,             # > 0.08 (HIGH)
            pitch_estimate=400.0,       # < 500 (LOW)
            speech_rate=1.0,
            voice_activity_ratio=0.5    # < 0.85 (ABRUPT/SLOW)
        )
        result = self.classifier.classify(features)
        self.assertEqual(result.emotion, EmotionState.FRUSTRATED)

    def test_emotion_context_smoothing(self):
        """Tests rolling average confidence smoothing."""
        # 1. Start neutral
        self.assertEqual(self.context.get_current_state(), EmotionState.NEUTRAL)
        
        # 2. Add one excited (confidence 0.9)
        res1 = EmotionResult(EmotionState.EXCITED, 0.9, None, time.time())
        self.context.add_result(res1)
        self.assertEqual(self.context.get_current_state(), EmotionState.EXCITED)
        self.assertEqual(self.context.get_current_confidence(), 0.9)
        
        # 3. Add a low confidence calm (confidence 0.7)
        res2 = EmotionResult(EmotionState.CALM, 0.7, None, time.time())
        self.context.add_result(res2)
        
        # Because Excited has 0.9 score and Calm has 0.7, Excited still wins!
        # This prevents erratic jumping.
        self.assertEqual(self.context.get_current_state(), EmotionState.EXCITED)
        self.assertEqual(self.context.get_current_confidence(), 0.9)
        
        # 4. Add another calm (confidence 0.75)
        res3 = EmotionResult(EmotionState.CALM, 0.75, None, time.time())
        self.context.add_result(res3)
        
        # Now Calm total = 1.45, Excited = 0.9. Calm takes over!
        self.assertEqual(self.context.get_current_state(), EmotionState.CALM)
        self.assertEqual(self.context.get_current_confidence(), 0.725) # 1.45 / 2

    def test_manager_tts_adaptation(self):
        """Tests TTS rate output based on state."""
        manager = EmotionManager()
        
        # Force the stable state by adding results
        res_frustrated = EmotionResult(EmotionState.FRUSTRATED, 0.9, None, time.time())
        manager.context.add_result(res_frustrated)
        tweaks = manager.get_tts_adaptation()
        self.assertEqual(tweaks["rate"], "-10%")
        
        # Override with excited
        res_excited = EmotionResult(EmotionState.EXCITED, 0.99, None, time.time())
        for _ in range(5):  # overwhelm the history
            manager.context.add_result(res_excited)
        
        tweaks = manager.get_tts_adaptation()
        self.assertEqual(tweaks["rate"], "+5%")

if __name__ == "__main__":
    unittest.main()
