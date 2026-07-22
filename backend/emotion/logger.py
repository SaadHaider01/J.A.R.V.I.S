"""
=============================================================================
backend/emotion/logger.py
=============================================================================

WHAT THIS FILE DOES:
    Provides a standardized logger specifically for the Emotion Subsystem.

WHY IT EXISTS:
    Separation of concerns. By giving the Emotion Subsystem its own logger
    ('zytrix.emotion'), we can easily filter, debug, and monitor logs 
    related purely to conversational state tracking without noise from 
    the audio bus or NLP agent.

EDUCATIONAL CONCEPT — NAMESPACED LOGGING:
    Using hierarchical namespaces (like `zytrix.emotion`) allows us to
    control log levels for specific parts of an application independently.
=============================================================================
"""

import logging

# Initialize the subsystem logger
emotion_logger = logging.getLogger("zytrix.emotion")

def log_emotion(event: str, message: str, level: int = logging.INFO):
    """
    Standardized logging format for emotion events.
    
    Format: [EMOTION:EVENT] Message
    """
    formatted_msg = f"\033[1;35m[EMOTION:{event.upper()}]\033[0m \033[36m{message}\033[0m"
    emotion_logger.log(level, formatted_msg)
