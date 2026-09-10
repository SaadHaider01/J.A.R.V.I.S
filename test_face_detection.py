import pytest
import numpy as np
import cv2
from backend.vision.face_detection import FaceDetector, Camera, CameraError, PresenceResult

# Mock a black frame
def get_mock_empty_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)

# Mock a frame with a face pattern (just drawing a white circle for simplicity)
def get_mock_face_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # This might not trigger a Haar cascade for faces reliably without a real face image,
    # but we can test the detector API at least. To reliably trigger it, we'd need a real face.
    # We will test the API primarily.
    return frame

class MockVideoCapture:
    def __init__(self, is_opened=True, ret=True, frame=None):
        self._is_opened = is_opened
        self._ret = ret
        self._frame = frame if frame is not None else get_mock_empty_frame()
        self.released = False

    def isOpened(self):
        return self._is_opened

    def read(self):
        return self._ret, self._frame

    def release(self):
        self.released = True

def test_detector_initialization():
    detector = FaceDetector()
    assert detector is not None
    assert not detector.classifier.empty()

def test_camera_unavailable(monkeypatch):
    # Mock cv2.VideoCapture to return an unopened camera
    monkeypatch.setattr(cv2, "VideoCapture", lambda index: MockVideoCapture(is_opened=False))
    
    with pytest.raises(CameraError):
        with Camera(0):
            pass

def test_camera_read_failure(monkeypatch):
    # Mock cv2.VideoCapture to return a failure on read()
    monkeypatch.setattr(cv2, "VideoCapture", lambda index: MockVideoCapture(is_opened=True, ret=False, frame=None))
    
    with Camera(0) as cam:
        with pytest.raises(CameraError):
            cam.read_frame()

def test_successful_frame_acquisition(monkeypatch):
    mock_frame = get_mock_empty_frame()
    monkeypatch.setattr(cv2, "VideoCapture", lambda index: MockVideoCapture(is_opened=True, ret=True, frame=mock_frame))
    
    with Camera(0) as cam:
        frame = cam.read_frame()
        assert frame is not None
        assert frame.shape == (480, 640, 3)

def test_no_face_detected():
    detector = FaceDetector()
    empty_frame = get_mock_empty_frame()
    result, boxes = detector.detect_in_frame(empty_frame)
    
    assert isinstance(result, PresenceResult)
    assert result.face_detected is False
    assert result.face_count == 0
    assert result.error is None
    assert result.camera_available is True
    assert len(boxes) == 0

def test_detect_presence_api(monkeypatch):
    # Tests the wrapper detect_presence() function
    mock_frame = get_mock_empty_frame()
    monkeypatch.setattr(cv2, "VideoCapture", lambda index: MockVideoCapture(is_opened=True, ret=True, frame=mock_frame))
    
    detector = FaceDetector()
    result = detector.detect_presence(0)
    assert isinstance(result, PresenceResult)
    assert result.face_detected is False

def test_camera_cleanup_on_exception(monkeypatch):
    mock_cap = MockVideoCapture(is_opened=True)
    monkeypatch.setattr(cv2, "VideoCapture", lambda index: mock_cap)
    
    try:
        with Camera(0) as cam:
            raise ValueError("Some random error")
    except ValueError:
        pass
    
    # Assert release was called even though an exception occurred inside the context manager
    assert mock_cap.released is True
