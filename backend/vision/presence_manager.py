import time
from enum import Enum
from dataclasses import dataclass
from typing import Callable, Optional

from .face_recognition import RecognitionResult, RecognitionState
from .logger import VisionLogger

log = VisionLogger(__name__)

class PresenceState(Enum):
    ABSENT = "ABSENT"
    VERIFYING = "VERIFYING"
    PRESENT = "PRESENT"
    ERROR = "ERROR"

@dataclass
class PresenceConfig:
    required_confirmations: int = 3
    absence_grace_seconds: float = 5.0

@dataclass
class PresenceEvent:
    previous_state: PresenceState
    current_state: PresenceState
    timestamp: float
    reason: str
    recognition_state: RecognitionState
    activation_requested: bool

class PresenceManager:
    """
    Manages presence state transitions (ABSENT -> VERIFYING -> PRESENT),
    debounces temporary camera failures, enforces grace periods,
    and triggers a one-shot activation callback per presence session.
    """
    def __init__(
        self,
        config: Optional[PresenceConfig] = None,
        activation_callback: Optional[Callable[[PresenceEvent], None]] = None,
        time_provider: Optional[Callable[[], float]] = None
    ):
        self.config = config or PresenceConfig()
        self.activation_callback = activation_callback
        self.time_provider = time_provider or time.time
        
        self.current_state = PresenceState.ABSENT
        self.consecutive_confirmations = 0
        self.last_seen_time = 0.0

    def update(self, recognition_result: RecognitionResult) -> None:
        """
        Feeds a single recognition result into the state machine.
        """
        now = self.time_provider()
        previous_state = self.current_state
        activation_requested = False
        reason = "update"

        if recognition_result.state == RecognitionState.KNOWN_USER:
            self.last_seen_time = now
            
            if self.current_state == PresenceState.ABSENT:
                self.current_state = PresenceState.VERIFYING
                self.consecutive_confirmations = 1
                reason = "known_user_detected"
                
                # Check immediately in case required_confirmations == 1
                if self.consecutive_confirmations >= self.config.required_confirmations:
                    self.current_state = PresenceState.PRESENT
                    activation_requested = True
                    reason = "authorized_presence_confirmed"
                    
            elif self.current_state == PresenceState.VERIFYING:
                self.consecutive_confirmations += 1
                if self.consecutive_confirmations >= self.config.required_confirmations:
                    self.current_state = PresenceState.PRESENT
                    activation_requested = True
                    reason = "authorized_presence_confirmed"
                else:
                    reason = "verifying_presence"
                    
            elif self.current_state == PresenceState.PRESENT:
                # Already active, reset confirmation just in case
                self.consecutive_confirmations = self.config.required_confirmations
                reason = "presence_maintained"
                
            elif self.current_state == PresenceState.ERROR:
                self.current_state = PresenceState.VERIFYING
                self.consecutive_confirmations = 1
                reason = "recovered_from_error"

        else:
            # NO_FACE, UNKNOWN_USER, MULTIPLE_FACES, CAMERA_ERROR, etc.
            self.consecutive_confirmations = 0
            
            if self.current_state == PresenceState.VERIFYING:
                self.current_state = PresenceState.ABSENT
                reason = "verification_failed"
                
            elif self.current_state == PresenceState.PRESENT:
                time_absent = now - self.last_seen_time
                if time_absent > self.config.absence_grace_seconds:
                    self.current_state = PresenceState.ABSENT
                    reason = "grace_period_expired"
                else:
                    reason = f"grace_period_active ({self.config.absence_grace_seconds - time_absent:.1f}s remaining)"
                    
            elif self.current_state == PresenceState.ABSENT:
                reason = "user_absent"

            # If it's a hard error, maybe transition to ERROR if not PRESENT
            if recognition_result.state in (RecognitionState.CAMERA_ERROR, RecognitionState.RECOGNITION_ERROR):
                if self.current_state != PresenceState.PRESENT:
                    self.current_state = PresenceState.ERROR
                    reason = "hard_error"

        # Log state changes
        if previous_state != self.current_state:
            log.info("presence_state_changed", old=previous_state.value, new=self.current_state.value, reason=reason)

        # Trigger activation
        if activation_requested:
            log.info("activation_requested", reason=reason)
            event = PresenceEvent(
                previous_state=previous_state,
                current_state=self.current_state,
                timestamp=now,
                reason=reason,
                recognition_state=recognition_result.state,
                activation_requested=True
            )
            if self.activation_callback:
                self.activation_callback(event)
