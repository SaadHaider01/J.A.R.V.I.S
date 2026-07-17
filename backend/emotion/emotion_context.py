"""
=============================================================================
backend/emotion/emotion_context.py
=============================================================================

WHAT THIS FILE DOES:
    Maintains a rolling window of recent conversational states (emotions)
    and applies a Confidence Smoothing algorithm to prevent erratic behavior.

WHY THIS ARCHITECTURE?
    If the user speaks one sentence excitedly, but is generally calm, we 
    don't want the assistant to instantly jump to "Excited" mode, only to
    jump back to "Calm" a second later. That creates a very unnatural, 
    robotic conversational experience.
    
    By smoothing the last N interactions (e.g., 3 interactions), we only
    change the assistant's behavior when a conversational state trend is 
    clearly established.

EDUCATIONAL CONCEPT — ROLLING AVERAGE / SMOOTHING:
    A common technique in signal processing. We keep a buffer of the 
    most recent data points and average them. If a new, highly confident 
    signal arrives, it might sway the average. If a low-confidence signal 
    arrives, it has less impact.
=============================================================================
"""

import time
from collections import deque
from typing import List, Optional
from backend.emotion.emotion_models import EmotionResult, EmotionState
from backend.emotion.logger import log_emotion

class EmotionContext:
    """Maintains transient conversational state with smoothing."""
    
    def __init__(self, history_size: int = 3, time_decay_s: float = 120.0):
        # We keep only the last N results
        self.history_size = history_size
        # If a result is older than this, it is discarded
        self.time_decay_s = time_decay_s
        self._history: deque[EmotionResult] = deque(maxlen=history_size)
        
        # The current stable state after smoothing
        self._stable_state: EmotionState = EmotionState.NEUTRAL
        self._stable_confidence: float = 0.5
        
    def add_result(self, result: EmotionResult):
        """Adds a new raw classification and updates the stable state."""
        self._prune_old_history()
        
        # We don't add UNKNOWN to history, as it dilutes the actual signal.
        if result.emotion != EmotionState.UNKNOWN:
            self._history.append(result)
            
        self._recalculate_stable_state()
        
    def _prune_old_history(self):
        """Removes results that are too old to be relevant."""
        now = time.time()
        while self._history:
            if now - self._history[0].timestamp > self.time_decay_s:
                self._history.popleft()
            else:
                break
                
    def _recalculate_stable_state(self):
        """
        Calculates the smoothed conversational state based on recent history.
        Uses a weighted voting mechanism based on confidence.
        """
        if not self._history:
            self._stable_state = EmotionState.NEUTRAL
            self._stable_confidence = 0.5
            return
            
        # Tally the weighted scores for each emotion in the history buffer
        scores = {}
        for res in self._history:
            # We can weigh recent events slightly higher if desired, 
            # but for now simple confidence weighting is fine.
            scores[res.emotion] = scores.get(res.emotion, 0.0) + res.confidence
            
        # Find the emotion with the highest accumulated score
        best_emotion = max(scores.keys(), key=lambda e: scores[e])
        best_score = scores[best_emotion]
        
        # Average confidence is the score divided by the number of times it appeared
        count = sum(1 for res in self._history if res.emotion == best_emotion)
        avg_confidence = best_score / count
        
        old_state = self._stable_state
        self._stable_state = best_emotion
        self._stable_confidence = avg_confidence
        
        if old_state != self._stable_state:
            log_emotion("TRANSITION", f"State changed: {old_state.name} -> {self._stable_state.name} (Conf: {avg_confidence:.2f})")

    def get_current_state(self) -> EmotionState:
        """Returns the current smoothed emotional state."""
        self._prune_old_history()
        # If pruning emptied the history, it will stay on the last stable state
        # until a new one arrives, or we can force it to neutral.
        # It's better to decay to neutral if we haven't spoken in a long time.
        if not self._history:
            return EmotionState.NEUTRAL
        return self._stable_state
        
    def get_current_confidence(self) -> float:
        """Returns the confidence of the current smoothed state."""
        return self._stable_confidence

    def format_prompt_injection(self) -> str:
        """
        Formats the current state into a string block for the LLM System Prompt.
        This focuses on adapting the conversational style.
        """
        state = self.get_current_state()
        conf = self.get_current_confidence()
        
        # If we are neutral or unknown, don't clutter the prompt with instructions.
        if state in (EmotionState.NEUTRAL, EmotionState.UNKNOWN):
            return ""
            
        instructions = {
            EmotionState.CALM: "Speak at a normal pace. Keep the conversation steady and relaxed.",
            EmotionState.FRUSTRATED: "Be concise. Remain highly patient. Avoid unnecessary explanations or overly cheerful tones.",
            EmotionState.EXCITED: "Match the user's enthusiasm! Celebrate progress and use energetic language.",
            EmotionState.TIRED: "Speak gently. Use shorter responses. Reduce the user's cognitive load."
        }
        
        instruction = instructions.get(state, "Maintain standard conversational style.")
        
        # The prompt block
        block = (
            "\n==================================================\n"
            "ADAPTIVE CONVERSATION CONTEXT\n"
            "==================================================\n"
            f"User's Current Conversational State: {state.name.title()} (Confidence: {int(conf * 100)}%)\n\n"
            f"ADAPTIVE INSTRUCTION: {instruction}\n"
            "==================================================\n"
        )
        return block
