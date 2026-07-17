"""
=============================================================================
backend/emotion/feature_extractor.py
=============================================================================

WHAT THIS FILE DOES:
    Extracts purely objective, measurable audio characteristics (features) 
    from a raw numpy audio array.

WHY THIS ARCHITECTURE?
    Separation of Concerns. The Feature Extractor has zero knowledge of 
    "Emotions". It only knows math and DSP (Digital Signal Processing).
    By isolating this, if we ever replace the Rule-Based Classifier with
    a Neural Network, this file might not need to change at all.

DSP CONCEPTS EXPLAINED:
    - RMS Energy: Root Mean Square. Measures the average loudness or 
      amplitude of the audio.
    - Zero-Crossing Rate (ZCR): How often the audio wave crosses the zero 
      axis. High ZCR generally correlates with higher pitch or fricatives.
    - Voice Activity Ratio: The percentage of chunks that contain speech 
      rather than silence.

FUTURE TODO:
    - User Baselines: Currently we extract absolute values. In the future, 
      this should load a user profile (average_pitch, average_energy) and 
      return relative values (e.g., energy_variance = +1.2).
=============================================================================
"""

import numpy as np
import time
from backend.emotion.emotion_models import AudioFeatures
from backend.duplex.constants import SAMPLE_RATE

class FeatureExtractor:
    """
    Extracts measurable audio characteristics from the voice buffer.
    """
    
    def __init__(self):
        # We will use this to implement user baselines in the future.
        self._user_baseline = None

    def extract_features(self, audio_data: np.ndarray, duration_s: float) -> AudioFeatures:
        """
        Analyzes the full audio array of a user's speech utterance.
        
        Parameters:
            audio_data: Float32 numpy array representing the speech waveform.
            duration_s: The length of the audio in seconds.
        """
        if audio_data.size == 0 or duration_s <= 0:
            return AudioFeatures(0.0, 0.0, 0.0, 0.0)

        # 1. RMS Energy (Loudness)
        # We square the samples, find the mean, and take the square root.
        rms = float(np.sqrt(np.mean(audio_data**2)))
        
        # 2. Pitch Estimate (Zero-Crossing Rate Proxy)
        # We count how many times the signal crosses 0 (changes sign).
        # A higher rate loosely indicates higher frequency components.
        zero_crossings = np.nonzero(np.diff(audio_data > 0))[0]
        zcr = len(zero_crossings) / duration_s
        
        # 3. Speech Rate Proxy
        # For a true speech rate (words per minute), we would need the STT transcript.
        # As an acoustic proxy, we can measure amplitude envelope peaks (syllables).
        # Here we use a simplified metric based on duration.
        # (A more complex implementation would count intensity peaks).
        # We will expose it as a raw duration-based metric for now.
        # If duration is very short with high energy, it's fast.
        # We'll normalize it later in the classifier.
        speech_rate = duration_s
        
        # 4. Voice Activity Ratio
        # What percentage of the audio is actually above a tiny noise floor?
        # This helps differentiate slow speech with long pauses vs fast speech.
        noise_floor = 0.005
        active_samples = np.sum(np.abs(audio_data) > noise_floor)
        voice_activity_ratio = active_samples / len(audio_data)
        
        return AudioFeatures(
            rms_energy=rms,
            pitch_estimate=zcr,
            speech_rate=speech_rate,
            voice_activity_ratio=float(voice_activity_ratio)
        )
