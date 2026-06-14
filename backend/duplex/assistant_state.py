# ==============================================================================
# J.A.R.V.I.S — STATE MACHINE & TRANSITIONS
# ==============================================================================
# WHAT THIS MODULE DOES:
# Manages the assistant's runtime operational states using a thread-safe
# Finite State Machine (FSM) and monitors session identifiers.
#
# WHY IT EXISTS:
# Voice interfaces are highly asynchronous. Without a state machine:
#   - You might start recording while the system is already playing a response.
#   - Two threads might attempt to update variables concurrently, leading to
#     race conditions.
# Managing states explicitly restricts what actions are valid at any given time.
#
# WHAT ADVANCED CONCEPTS ARE HERE:
#   - Race Condition: When multiple threads access and write to a shared variable
#     simultaneously, the outcome depends on the timing of execution, causing
#     unpredictable behavior.
#   - Locks: We use `threading.Lock()` to serialize state changes, guaranteeing
#     that only one thread can modify the state at a time.
#   - Session IDs: When a user interrupts the assistant, we increment the
#     `session_id`. Any late operations (e.g. STT finishing a transcription of a
#     previous, aborted phrase) are ignored if their session ID does not match.
# ==============================================================================

import time
import uuid
import threading
from enum import Enum
from backend.duplex.logger import log_event

class AssistantState(Enum):
    IDLE = 1          # Sitting quietly waiting for the wake word
    LISTENING = 2     # Actively recording user speech (VAD open)
    THINKING = 3      # STT or LLM thinking in progress
    SPEAKING = 4      # TTS audio playing through speakers
    INTERRUPTED = 5   # Interrupt caught, resetting and clearing queues

class StateTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._state = AssistantState.IDLE
        self._last_transition_time = time.time()
        
        # Session ID management to isolate out-of-order execution packets
        self._current_session_id = str(uuid.uuid4())

    def get_state(self) -> AssistantState:
        """Thread-safe retrieval of current state."""
        with self._lock:
            return self._state

    def get_last_transition_time(self) -> float:
        """Retrieves timestamp of the last state change (used by Watchdog)."""
        with self._lock:
            return self._last_transition_time

    def get_session_id(self) -> str:
        """Retrieves the active conversation session ID."""
        with self._lock:
            return self._current_session_id

    def increment_session(self) -> str:
        """
        Force-cycles the session ID. Used when an interruption is triggered
        to instantly invalidate any ongoing STT or LLM jobs.
        """
        with self._lock:
            self._current_session_id = str(uuid.uuid4())
            log_event("STATE", f"Session Incremented -> {self._current_session_id}")
            return self._current_session_id

    def can_transition(self, old: AssistantState, new: AssistantState) -> bool:
        """
        Defines the valid transition graph of the voice state machine.
        Returns True if the transition is allowed.
        """
        # Graph transition table
        allowed = {
            AssistantState.IDLE: {AssistantState.LISTENING},
            AssistantState.LISTENING: {AssistantState.THINKING, AssistantState.IDLE},
            AssistantState.THINKING: {AssistantState.SPEAKING, AssistantState.INTERRUPTED, AssistantState.IDLE},
            AssistantState.SPEAKING: {AssistantState.IDLE, AssistantState.INTERRUPTED},
            AssistantState.INTERRUPTED: {AssistantState.LISTENING, AssistantState.IDLE}
        }
        return new in allowed.get(old, set())

    def transition_to(self, new_state: AssistantState) -> bool:
        """
        Performs a thread-safe transition to a new state.
        Returns True if successful, False if the transition was illegal.
        """
        with self._lock:
            old_state = self._state
            if old_state == new_state:
                return True # No change needed
            
            if not self.can_transition(old_state, new_state):
                log_event(
                    "STATE",
                    f"Blocked illegal transition: {old_state.name} ➔ {new_state.name}",
                    level=30  # WARNING
                )
                return False

            self._state = new_state
            self._last_transition_time = time.time()
            log_event("STATE", f"{old_state.name} ➔ {new_state.name}")
            return True

    def force_state(self, new_state: AssistantState):
        """
        Bypasses validity rules to force state correction (used for error recovery).
        """
        with self._lock:
            old_state = self._state
            self._state = new_state
            self._last_transition_time = time.time()
            log_event("STATE", f"[FORCE STATE RECOVERY] {old_state.name} ➔ {new_state.name}", level=30)
