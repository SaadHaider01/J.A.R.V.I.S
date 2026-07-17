"""
=============================================================================
backend/emotion/metrics.py
=============================================================================

WHAT THIS FILE DOES:
    Tracks statistics and metrics for the Emotion Subsystem.

WHY IT EXISTS:
    We need observable metrics (telemetry) to understand how often the 
    system detects emotions, what its confidence levels are, and how 
    long analysis takes. Without metrics, tuning the rule-based heuristics 
    would be pure guesswork.

EDUCATIONAL CONCEPT — OBSERVABILITY:
    In production architectures, Observability answers the question:
    "Is the system behaving correctly in the wild?"
    Tracking metrics allows us to detect regressions (e.g., if "Unknown"
    suddenly spikes to 90%, our audio feature extraction might be broken).
=============================================================================
"""

import threading
from typing import Dict
from backend.emotion.logger import log_emotion

class EmotionMetricsTracker:
    """Tracks emotion-related metrics thread-safely."""
    
    def __init__(self):
        self._lock = threading.Lock()
        
        # Core Counters
        self.total_analyzed = 0
        self.emotion_counts: Dict[str, int] = {}
        self.transitions = 0
        
        # Rolling Averages
        self._total_confidence = 0.0
        self._total_latency = 0.0
        
    def record_analysis(self, emotion: str, confidence: float, latency_s: float):
        """Records a single emotion analysis pass."""
        with self._lock:
            self.total_analyzed += 1
            
            # Update distribution
            self.emotion_counts[emotion] = self.emotion_counts.get(emotion, 0) + 1
            
            # Update rolling sums
            self._total_confidence += confidence
            self._total_latency += latency_s
            
    def record_transition(self):
        """Records when the final smoothed conversation state changes."""
        with self._lock:
            self.transitions += 1

    def get_metrics_summary(self) -> dict:
        """Returns a snapshot of current metrics."""
        with self._lock:
            avg_conf = (self._total_confidence / self.total_analyzed) if self.total_analyzed > 0 else 0.0
            avg_lat = (self._total_latency / self.total_analyzed) if self.total_analyzed > 0 else 0.0
            unknown_pct = (self.emotion_counts.get("UNKNOWN", 0) / self.total_analyzed) if self.total_analyzed > 0 else 0.0
            
            return {
                "total_analyzed": self.total_analyzed,
                "transitions": self.transitions,
                "average_confidence": round(avg_conf, 2),
                "average_latency_ms": round(avg_lat * 1000, 2),
                "unknown_percentage": round(unknown_pct * 100, 1),
                "distribution": self.emotion_counts.copy()
            }
            
    def log_metrics(self):
        """Dumps metrics to the logger."""
        metrics = self.get_metrics_summary()
        log_emotion("METRICS", f"Summary: {metrics}")

# Global Singleton Tracker
emotion_metrics = EmotionMetricsTracker()
