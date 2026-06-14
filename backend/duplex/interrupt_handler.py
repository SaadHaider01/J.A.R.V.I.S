# ==============================================================================
# J.A.R.V.I.S — INTERRUPT DETECTION SYSTEM
# ==============================================================================
# WHAT THIS MODULE DOES:
# Monitors incoming audio levels and calculates Root Mean Square (RMS) energy
# values to determine if the user has interrupted (barge-in) the assistant.
#
# WHY IT EXISTS:
# If we simply cut off the speech on the very first loud sample, any small noise
# (mouse clicks, room echoes, coughing) would cause the assistant to stop talking
# and stand by. We implement a "soft interrupt" buffer to prevent false triggers.
#
# WHAT ADVANCED CONCEPTS ARE HERE:
#   - Root Mean Square (RMS) Energy: A statistical measure of the magnitude of a
#     varying signal. For audio, it represents the average volume/power.
#   - Soft Interruption: Requiring energy to exceed thresholds for multiple
#     consecutive chunks (e.g. 150ms) to ensure it represents actual human speech.
#   - Priority Levels: Abstracting interrupt triggers into priorities (LOW, MEDIUM,
#     HIGH) to support future emergency override keywords or wake word detections.
# ==============================================================================

import numpy as np
from enum import Enum
from backend.duplex.logger import log_event
from backend.duplex.constants import (
    NORMAL_ENERGY_THRESHOLD,
    INTERRUPT_ENERGY_THRESHOLD,
    SOFT_INTERRUPT_CHUNKS
)

class InterruptPriority(Enum):
    LOW = 1       # Minor room noises, ignored
    MEDIUM = 2    # Standard user speech / voice barge-in
    HIGH = 3      # Wake word trigger or emergency hard-stop word

class InterruptionDetector:
    def __init__(self):
        self.consecutive_chunks_above = 0

    def calculate_rms(self, chunk: np.ndarray) -> float:
        """Calculates root-mean-square energy of the float32 audio chunk."""
        if chunk.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(chunk ** 2)))

    def detect_interrupt(self, chunk: np.ndarray, is_speaking: bool) -> tuple[bool, InterruptPriority]:
        """
        Determines if an audio chunk represents an interruption.
        
        Args:
            chunk: The raw float32 audio array.
            is_speaking: True if assistant is currently speaking (requires higher threshold).
            
        Returns:
            A tuple of (is_interrupted, InterruptPriority)
            
        Educational Design Decision:
            By encapsulating this check inside a method rather than hardcoding it
            into the main event loop, we make the system modular. Later, we can
            replace the raw RMS calculation with a lightweight ML-based human voice
            classifier (VAD) without changing the duplex coordination code.
        """
        rms = self.calculate_rms(chunk)
        threshold = INTERRUPT_ENERGY_THRESHOLD if is_speaking else NORMAL_ENERGY_THRESHOLD
        
        if rms >= threshold:
            self.consecutive_chunks_above += 1
            # Debug tracking of consecutive levels
            if self.consecutive_chunks_above > 0 and is_speaking:
                log_event("INTERRUPT", f"Soft interrupt frame {self.consecutive_chunks_above}/{SOFT_INTERRUPT_CHUNKS} (RMS: {rms:.4f} > Threshold: {threshold})", level=10) # DEBUG
        else:
            # Drop in energy resets the soft-interrupt counter
            if self.consecutive_chunks_above > 0:
                self.consecutive_chunks_above = 0
                
        # Trigger hard interruption once user exceeds threshold consistently
        if self.consecutive_chunks_above >= SOFT_INTERRUPT_CHUNKS:
            self.consecutive_chunks_above = 0 # Reset after trigger
            log_event("INTERRUPT", f"Speech detected consistently for {SOFT_INTERRUPT_CHUNKS * 30}ms. Triggering interruption.", level=30) # WARNING
            return True, InterruptPriority.MEDIUM
            
        return False, InterruptPriority.LOW
