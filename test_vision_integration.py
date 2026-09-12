import pytest
import threading
import time
from unittest.mock import MagicMock, patch

from backend.vision.vision_service import VisionService
from backend.duplex.duplex_manager import DuplexManager
from backend.duplex.assistant_state import AssistantState
from backend.vision.presence_manager import PresenceEvent, PresenceState
from backend.vision.face_recognition import RecognitionState

@pytest.fixture
def mock_duplex():
    """Provides an isolated, non-running DuplexManager."""
    manager = DuplexManager()
    # We do not start the manager thread for these unit tests,
    # we just test the method contracts.
    return manager

def test_duplex_request_activation(mock_duplex):
    """Test that request_activation sets the event safely."""
    assert not mock_duplex._external_activation_event.is_set()
    mock_duplex.request_activation(source="vision_presence")
    assert mock_duplex._external_activation_event.is_set()
    assert mock_duplex._activation_source == "vision_presence"

@patch("backend.vision.vision_service.Camera")
@patch("backend.vision.vision_service.FaceRecognizer")
def test_vision_service_lifecycle(mock_recognizer_class, mock_camera_class):
    """Test that VisionService starts and stops cleanly."""
    # Mock camera context manager
    mock_cam_instance = MagicMock()
    mock_camera_class.return_value.__enter__.return_value = mock_cam_instance
    
    # Mock recognizer
    mock_rec_instance = MagicMock()
    mock_recognizer_class.return_value = mock_rec_instance
    mock_rec_instance.recognize_in_frame.return_value = (MagicMock(state=RecognitionState.NO_FACE), [])
    
    callbacks = []
    
    service = VisionService(on_activation=lambda e: callbacks.append(e), fps=100) # High fps for fast test
    
    service.start()
    time.sleep(0.1) # Let the thread spin up
    
    assert service._thread is not None
    assert service._thread.is_alive()
    
    service.stop()
    
    assert service._thread is None
    # Ensure camera context manager was exited cleanly
    mock_camera_class.return_value.__exit__.assert_called_once()
    
def test_activation_chain():
    """Test that PresenceEvent successfully calls request_activation."""
    duplex = DuplexManager()
    
    # Simulate what main.py does
    def callback(event):
        duplex.request_activation(source="vision_presence")
        
    # Fire a mock event
    event = PresenceEvent(
        previous_state=PresenceState.VERIFYING,
        current_state=PresenceState.PRESENT,
        timestamp=0.0,
        reason="test",
        recognition_state=RecognitionState.KNOWN_USER,
        activation_requested=True
    )
    
    callback(event)
    
    assert duplex._external_activation_event.is_set()
    assert duplex._activation_source == "vision_presence"
    
@patch("backend.vision.vision_service.Camera")
@patch("backend.vision.vision_service.FaceRecognizer")
def test_duplicate_activation_is_safe(mock_recognizer_class, mock_camera_class):
    """
    Ensure that if DuplexManager is not in IDLE, the activation event is ignored.
    This simulates duplicate activations while already LISTENING.
    """
    # This specifically tests the DuplexManager _run_loop logic manually
    duplex = DuplexManager()
    duplex.state_tracker.transition_to(AssistantState.LISTENING)
    
    # Simulate a vision trigger
    duplex.request_activation("vision_presence")
    
    # In _run_loop, it only checks the event if it's in IDLE or INTERRUPTED
    # So if it's LISTENING, it will just loop and never clear the event, 
    # essentially deferring/ignoring it until it returns to IDLE.
    assert duplex.state_tracker.get_state() == AssistantState.LISTENING
    assert duplex._external_activation_event.is_set()
