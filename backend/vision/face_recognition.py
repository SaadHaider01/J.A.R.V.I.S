import os
import time
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np

from .face_detection import FaceDetector, CameraError, PresenceResult
from .logger import VisionLogger

log = VisionLogger(__name__)

class RecognitionState(Enum):
    NO_FACE = "NO_FACE"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    KNOWN_USER = "KNOWN_USER"
    UNKNOWN_USER = "UNKNOWN_USER"
    NO_ENROLLMENT = "NO_ENROLLMENT"
    CAMERA_ERROR = "CAMERA_ERROR"
    RECOGNITION_ERROR = "RECOGNITION_ERROR"

@dataclass
class RecognitionResult:
    """
    Result model for face recognition.
    Does NOT contain the raw image frame to preserve privacy and prevent leakage.
    Distance is the LBPH distance metric (lower is better).
    """
    state: RecognitionState
    recognized: bool
    distance: float
    face_count: int
    processing_time_ms: float
    timestamp: float
    error: Optional[str] = None
    camera_available: bool = True

class BiometricStore:
    """
    Handles secure local storage of the biometric LBPH model.
    Never logs or exposes the underlying histogram array.
    """
    def __init__(self, data_dir: str = "data/biometrics", filename: str = "enrolled_user.yml"):
        self.data_dir = os.path.abspath(data_dir)
        self.file_path = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)

    def is_enrolled(self) -> bool:
        return os.path.isfile(self.file_path)

    def save_model(self, model) -> None:
        try:
            model.save(self.file_path)
            log.info("face_enrollment_complete", path=self.file_path)
        except Exception as e:
            log.error("face_enrollment_failed", error=str(e))
            raise RuntimeError(f"Failed to save biometric model: {e}")

    def load_model(self):
        if not self.is_enrolled():
            log.info("biometric_store_missing", path=self.file_path)
            return None
        
        try:
            model = cv2.face.LBPHFaceRecognizer_create()
            model.read(self.file_path)
            log.info("biometric_store_loaded", path=self.file_path)
            return model
        except Exception as e:
            log.error("biometric_store_load_failed", error=str(e))
            return None

class FaceRecognizer:
    """
    Local Face Recognizer utilizing LBPH.
    Combines FaceDetector with LBPH representation.
    """
    def __init__(self, threshold: float = 60.0):
        """
        Threshold: LBPH distance threshold. 
        Lower distance = better match.
        Default 60.0 is a reasonable starting point for LBPH, but should be tuned.
        """
        self.detector = FaceDetector()
        self.store = BiometricStore()
        self.model = self.store.load_model()
        self.threshold = threshold
        self.target_size = (150, 150) # Standardize size for LBPH

    def reload_model(self):
        """Reloads the model from disk (useful after new enrollment)."""
        self.model = self.store.load_model()

    def enroll_face(self, frame) -> bool:
        """
        Takes a frame, detects exactly one face, and trains the LBPH model.
        Returns True on success, False otherwise.
        """
        result, boxes = self.detector.detect_in_frame(frame)
        if result.face_count != 1:
            log.info("enrollment_failed_multiple_or_no_faces", face_count=result.face_count)
            return False

        (x, y, w, h) = boxes[0]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_roi = gray[y:y+h, x:x+w]
        
        # Resize to standardized size
        face_roi = cv2.resize(face_roi, self.target_size)

        # Train a new LBPH model
        model = cv2.face.LBPHFaceRecognizer_create()
        # LBPH requires labels. We just use label '1' for the authorized user.
        model.train([face_roi], np.array([1]))
        
        self.store.save_model(model)
        self.model = model
        return True

    def recognize_in_frame(self, frame) -> Tuple[RecognitionResult, Any]:
        """
        Recognizes faces in a given frame.
        Returns the RecognitionResult and the list of bounding boxes.
        """
        start_time = time.time()
        
        try:
            # 1. First, check presence
            presence_result, boxes = self.detector.detect_in_frame(frame)
            
            processing_time_ms = presence_result.processing_time_ms
            
            if presence_result.error:
                return RecognitionResult(
                    state=RecognitionState.RECOGNITION_ERROR,
                    recognized=False,
                    distance=float('inf'),
                    face_count=0,
                    processing_time_ms=processing_time_ms,
                    timestamp=time.time(),
                    error=presence_result.error,
                    camera_available=presence_result.camera_available
                ), []

            if presence_result.face_count == 0:
                return RecognitionResult(
                    state=RecognitionState.NO_FACE,
                    recognized=False,
                    distance=float('inf'),
                    face_count=0,
                    processing_time_ms=processing_time_ms,
                    timestamp=time.time(),
                    camera_available=True
                ), []
                
            if presence_result.face_count > 1:
                log.debug("multiple_faces_detected", count=presence_result.face_count)
                return RecognitionResult(
                    state=RecognitionState.MULTIPLE_FACES,
                    recognized=False,
                    distance=float('inf'),
                    face_count=presence_result.face_count,
                    processing_time_ms=processing_time_ms,
                    timestamp=time.time(),
                    camera_available=True
                ), boxes

            # Exactly 1 face. Proceed to recognition.
            if self.model is None:
                return RecognitionResult(
                    state=RecognitionState.NO_ENROLLMENT,
                    recognized=False,
                    distance=float('inf'),
                    face_count=1,
                    processing_time_ms=processing_time_ms,
                    timestamp=time.time(),
                    camera_available=True
                ), boxes

            # Perform recognition
            (x, y, w, h) = boxes[0]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, self.target_size)
            
            # Predict
            label, distance = self.model.predict(face_roi)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            if distance <= self.threshold:
                state = RecognitionState.KNOWN_USER
                recognized = True
            else:
                state = RecognitionState.UNKNOWN_USER
                recognized = False
                log.debug("unknown_face", distance=distance, threshold=self.threshold)
                
            log.debug("face_recognition_result", state=state.value, distance=distance)

            return RecognitionResult(
                state=state,
                recognized=recognized,
                distance=distance,
                face_count=1,
                processing_time_ms=processing_time_ms,
                timestamp=time.time(),
                camera_available=True
            ), boxes

        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            log.error("recognition_failed", error=str(e))
            return RecognitionResult(
                state=RecognitionState.RECOGNITION_ERROR,
                recognized=False,
                distance=float('inf'),
                face_count=0,
                processing_time_ms=processing_time_ms,
                timestamp=time.time(),
                error=str(e),
                camera_available=True
            ), []
