import pytest
import time
from backend.vision.face_recognition import RecognitionResult, RecognitionState
from backend.vision.presence_manager import PresenceManager, PresenceConfig, PresenceState

class MockClock:
    def __init__(self, initial_time=1000.0):
        self.current_time = initial_time

    def get_time(self):
        return self.current_time

    def advance(self, seconds: float):
        self.current_time += seconds

def create_mock_result(state: RecognitionState):
    return RecognitionResult(
        state=state,
        recognized=(state == RecognitionState.KNOWN_USER),
        distance=50.0 if state == RecognitionState.KNOWN_USER else 100.0,
        face_count=1 if state in (RecognitionState.KNOWN_USER, RecognitionState.UNKNOWN_USER) else 0,
        processing_time_ms=10.0,
        timestamp=0.0,
        camera_available=True
    )

def test_initial_state():
    manager = PresenceManager()
    assert manager.current_state == PresenceState.ABSENT

def test_consecutive_confirmations_activation():
    clock = MockClock()
    config = PresenceConfig(required_confirmations=3)
    
    callbacks = []
    def on_activation(event):
        callbacks.append(event)
        
    manager = PresenceManager(config=config, time_provider=clock.get_time, activation_callback=on_activation)
    
    known_result = create_mock_result(RecognitionState.KNOWN_USER)
    
    # Frame 1 -> VERIFYING
    manager.update(known_result)
    assert manager.current_state == PresenceState.VERIFYING
    assert len(callbacks) == 0
    
    # Frame 2 -> VERIFYING
    clock.advance(0.5)
    manager.update(known_result)
    assert manager.current_state == PresenceState.VERIFYING
    assert len(callbacks) == 0
    
    # Frame 3 -> PRESENT
    clock.advance(0.5)
    manager.update(known_result)
    assert manager.current_state == PresenceState.PRESENT
    assert len(callbacks) == 1
    
    # Frame 4 -> maintain state, no double activation
    clock.advance(0.5)
    manager.update(known_result)
    assert manager.current_state == PresenceState.PRESENT
    assert len(callbacks) == 1

def test_grace_period_prevents_absence():
    clock = MockClock()
    config = PresenceConfig(required_confirmations=1, absence_grace_seconds=5.0)
    manager = PresenceManager(config=config, time_provider=clock.get_time)
    
    known = create_mock_result(RecognitionState.KNOWN_USER)
    no_face = create_mock_result(RecognitionState.NO_FACE)
    unknown = create_mock_result(RecognitionState.UNKNOWN_USER)
    
    manager.update(known)
    assert manager.current_state == PresenceState.PRESENT
    
    # Disappear (NO_FACE) for 2 seconds
    clock.advance(2.0)
    manager.update(no_face)
    assert manager.current_state == PresenceState.PRESENT
    
    # Unknown user (UNKNOWN_USER) for another 2 seconds
    clock.advance(2.0)
    manager.update(unknown)
    assert manager.current_state == PresenceState.PRESENT # Still PRESENT due to grace
    
    # Another 2 seconds (total 6 > 5)
    clock.advance(2.0)
    manager.update(no_face)
    assert manager.current_state == PresenceState.ABSENT # Grace expired

def test_grace_period_recovery():
    clock = MockClock()
    config = PresenceConfig(required_confirmations=1, absence_grace_seconds=5.0)
    manager = PresenceManager(config=config, time_provider=clock.get_time)
    
    known = create_mock_result(RecognitionState.KNOWN_USER)
    no_face = create_mock_result(RecognitionState.NO_FACE)
    
    manager.update(known)
    assert manager.current_state == PresenceState.PRESENT
    
    clock.advance(2.0)
    manager.update(no_face)
    assert manager.current_state == PresenceState.PRESENT
    
    clock.advance(1.0)
    manager.update(known)
    assert manager.current_state == PresenceState.PRESENT
    
    clock.advance(4.0)
    manager.update(no_face)
    assert manager.current_state == PresenceState.PRESENT
    
    clock.advance(2.0)
    manager.update(no_face)
    assert manager.current_state == PresenceState.ABSENT

def test_unknown_user_never_activates():
    manager = PresenceManager(config=PresenceConfig(required_confirmations=1))
    
    unknown = create_mock_result(RecognitionState.UNKNOWN_USER)
    manager.update(unknown)
    assert manager.current_state == PresenceState.ABSENT
    
def test_multiple_faces_resets_verification():
    manager = PresenceManager(config=PresenceConfig(required_confirmations=2))
    
    known = create_mock_result(RecognitionState.KNOWN_USER)
    multi = create_mock_result(RecognitionState.MULTIPLE_FACES)
    
    manager.update(known)
    assert manager.current_state == PresenceState.VERIFYING
    
    manager.update(multi)
    assert manager.current_state == PresenceState.ABSENT
    
    manager.update(known)
    manager.update(known)
    assert manager.current_state == PresenceState.PRESENT

def test_error_states_never_activate():
    manager = PresenceManager(config=PresenceConfig(required_confirmations=1))
    
    no_enrollment = create_mock_result(RecognitionState.NO_ENROLLMENT)
    camera_error = create_mock_result(RecognitionState.CAMERA_ERROR)
    recog_error = create_mock_result(RecognitionState.RECOGNITION_ERROR)
    
    manager.update(no_enrollment)
    assert manager.current_state == PresenceState.ABSENT
    
    manager.update(camera_error)
    assert manager.current_state == PresenceState.ERROR
    
    manager.update(recog_error)
    assert manager.current_state == PresenceState.ERROR

def test_re_activation_after_absence():
    clock = MockClock()
    config = PresenceConfig(required_confirmations=1, absence_grace_seconds=1.0)
    
    callbacks = []
    manager = PresenceManager(config=config, time_provider=clock.get_time, activation_callback=lambda e: callbacks.append(e))
    
    known = create_mock_result(RecognitionState.KNOWN_USER)
    no_face = create_mock_result(RecognitionState.NO_FACE)
    
    # Activate
    manager.update(known)
    assert manager.current_state == PresenceState.PRESENT
    assert len(callbacks) == 1
    
    # Leave
    clock.advance(2.0)
    manager.update(no_face)
    assert manager.current_state == PresenceState.ABSENT
    
    # Return (new sequence activates again)
    clock.advance(1.0)
    manager.update(known)
    assert manager.current_state == PresenceState.PRESENT
    assert len(callbacks) == 2
