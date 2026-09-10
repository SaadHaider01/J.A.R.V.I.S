import pytest
import numpy as np
import os
import cv2
from backend.vision.face_recognition import FaceRecognizer, BiometricStore, RecognitionState

# Mock a black frame
def get_mock_empty_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)

class MockFaceDetector:
    def __init__(self, face_count=1):
        self.face_count = face_count

    def detect_in_frame(self, frame):
        from backend.vision.face_detection import PresenceResult
        boxes = []
        if self.face_count == 1:
            boxes = [(100, 100, 50, 50)]
        elif self.face_count > 1:
            boxes = [(100, 100, 50, 50), (200, 200, 50, 50)]
            
        result = PresenceResult(
            face_detected=(self.face_count > 0),
            face_count=self.face_count,
            processing_time_ms=10.0,
            timestamp=12345.0,
            camera_available=True
        )
        return result, boxes

class MockLBPHModel:
    def __init__(self, distance=40.0):
        self.distance = distance

    def predict(self, face_roi):
        # returns label, distance
        return 1, self.distance
        
    def train(self, src, labels):
        pass

    def save(self, path):
        # Create an empty file to simulate save
        with open(path, 'w') as f:
            f.write("mocked_model")

def test_store_initialization(tmp_path):
    store = BiometricStore(data_dir=str(tmp_path))
    assert not store.is_enrolled()

def test_enrollment_success(monkeypatch, tmp_path):
    recognizer = FaceRecognizer()
    recognizer.store = BiometricStore(data_dir=str(tmp_path))
    recognizer.detector = MockFaceDetector(face_count=1)
    
    # Mock LBPH train to not fail
    monkeypatch.setattr(cv2.face, "LBPHFaceRecognizer_create", lambda: MockLBPHModel())
    
    frame = get_mock_empty_frame()
    success = recognizer.enroll_face(frame)
    
    assert success is True
    assert recognizer.store.is_enrolled() is True

def test_enrollment_failure_no_face(tmp_path):
    recognizer = FaceRecognizer()
    recognizer.store = BiometricStore(data_dir=str(tmp_path))
    recognizer.detector = MockFaceDetector(face_count=0)
    
    frame = get_mock_empty_frame()
    success = recognizer.enroll_face(frame)
    
    assert success is False
    assert recognizer.store.is_enrolled() is False

def test_enrollment_failure_multiple_faces(tmp_path):
    recognizer = FaceRecognizer()
    recognizer.store = BiometricStore(data_dir=str(tmp_path))
    recognizer.detector = MockFaceDetector(face_count=2)
    
    frame = get_mock_empty_frame()
    success = recognizer.enroll_face(frame)
    
    assert success is False
    assert recognizer.store.is_enrolled() is False

def test_recognition_no_enrollment(tmp_path):
    recognizer = FaceRecognizer()
    recognizer.store = BiometricStore(data_dir=str(tmp_path))
    recognizer.model = None # No model loaded
    recognizer.detector = MockFaceDetector(face_count=1)
    
    frame = get_mock_empty_frame()
    result, boxes = recognizer.recognize_in_frame(frame)
    
    assert result.state == RecognitionState.NO_ENROLLMENT
    assert result.recognized is False

def test_recognition_known_user(tmp_path):
    recognizer = FaceRecognizer(threshold=60.0)
    recognizer.store = BiometricStore(data_dir=str(tmp_path))
    recognizer.model = MockLBPHModel(distance=40.0) # Below threshold
    recognizer.detector = MockFaceDetector(face_count=1)
    
    frame = get_mock_empty_frame()
    result, boxes = recognizer.recognize_in_frame(frame)
    
    assert result.state == RecognitionState.KNOWN_USER
    assert result.recognized is True
    assert result.distance == 40.0

def test_recognition_unknown_user(tmp_path):
    recognizer = FaceRecognizer(threshold=60.0)
    recognizer.store = BiometricStore(data_dir=str(tmp_path))
    recognizer.model = MockLBPHModel(distance=80.0) # Above threshold
    recognizer.detector = MockFaceDetector(face_count=1)
    
    frame = get_mock_empty_frame()
    result, boxes = recognizer.recognize_in_frame(frame)
    
    assert result.state == RecognitionState.UNKNOWN_USER
    assert result.recognized is False
    assert result.distance == 80.0

def test_recognition_multiple_faces(tmp_path):
    recognizer = FaceRecognizer()
    recognizer.store = BiometricStore(data_dir=str(tmp_path))
    recognizer.model = MockLBPHModel(distance=40.0)
    recognizer.detector = MockFaceDetector(face_count=2)
    
    frame = get_mock_empty_frame()
    result, boxes = recognizer.recognize_in_frame(frame)
    
    assert result.state == RecognitionState.MULTIPLE_FACES
    assert result.recognized is False
