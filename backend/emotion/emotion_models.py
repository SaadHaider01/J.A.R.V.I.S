"""
=============================================================================
backend/emotion/emotion_models.py
=============================================================================

WHAT THIS FILE DOES:
    Defines the structured data models (Enums and Dataclasses) for the
    Emotion Subsystem.

WHY IT EXISTS:
    Using strongly typed Enums and Dataclasses prevents typos and bugs that
    arise from passing around raw strings and dictionaries. It ensures the
    Feature Extractor, Classifier, and Manager all speak the exact same
    language.

EDUCATIONAL CONCEPT — ENUMS & DATACLASSES:
    - Enum (Enumeration): A set of symbolic names bound to unique values. 
      It guarantees that 'emotion' can ONLY be one of the defined states.
    - Dataclass: A lightweight way to create classes that are primarily used
      to store data. It automatically generates __init__ and __repr__ methods.
=============================================================================
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional

class EmotionState(Enum):
    """
    The discrete conversational states the system can classify.
    Note: These are observable conversational styles, not medical diagnoses.
    """
    NEUTRAL    = "neutral"
    CALM       = "calm"
    FRUSTRATED = "frustrated"
    EXCITED    = "excited"
    TIRED      = "tired"
    UNKNOWN    = "unknown"  # Used when confidence is too low


@dataclass
class AudioFeatures:
    """
    Raw DSP features extracted from the user's speech audio.
    """
    rms_energy: float         # Loudness/Amplitude
    pitch_estimate: float     # Proxy for voice pitch (e.g., zero-crossing rate)
    speech_rate: float        # Proxy for tempo (how fast they are speaking)
    voice_activity_ratio: float # Percentage of the buffer containing actual speech
    

@dataclass
class EmotionResult:
    """
    The final classified output produced by the subsystem.
    """
    emotion: EmotionState
    confidence: float
    features: AudioFeatures
    timestamp: float
    
    def __str__(self):
        return f"{self.emotion.name} (Conf: {self.confidence:.2f})"
