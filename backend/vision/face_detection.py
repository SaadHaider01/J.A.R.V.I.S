import time
import os
import cv2
from dataclasses import dataclass
from typing import Optional, Tuple, Any
from .logger import VisionLogger

log = VisionLogger(__name__)

class CameraError(Exception):
    """Raised when the camera fails to initialize or capture frames."""
    pass

@dataclass
class PresenceResult:
    """
    Result model for presence detection.
    Does NOT contain the raw image frame to preserve privacy and prevent leakage.
    """
    face_detected: bool
    face_count: int
    processing_time_ms: float
    timestamp: float
    error: Optional[str] = None
    camera_available: bool = True

class Camera:
    """
    Context manager for safely handling the camera lifecycle.
    Ensures the camera is always released, even on errors.
    """
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def open(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            log.error("Camera unavailable", reason="Failed to open VideoCapture", camera_index=self.camera_index)
            raise CameraError(f"Failed to open camera index {self.camera_index}")
        log.info("Camera initialized", camera_index=self.camera_index)

    def read_frame(self):
        if not self.cap or not self.cap.isOpened():
            raise CameraError("Camera is not opened")
        
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise CameraError("Failed to read frame from camera")
        return frame

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
            log.info("Camera released", camera_index=self.camera_index)


class FaceDetector:
    """
    Detects faces locally using OpenCV Haar cascades.
    Returns a PresenceResult. Does NOT perform face recognition.
    """
    def __init__(self, cascade_path: Optional[str] = None):
        if cascade_path is None:
            cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            
        self.cascade_path = cascade_path
        self.classifier = cv2.CascadeClassifier(self.cascade_path)
        
        if self.classifier.empty():
            log.error("Face detector initialization failed", cascade_path=self.cascade_path)
            raise RuntimeError(f"Failed to load Haar cascade at {self.cascade_path}")
            
        log.info("Face detector initialized", cascade_path=self.cascade_path)

    def detect_in_frame(self, frame) -> Tuple[PresenceResult, Any]:
        """
        Detects faces in a given frame.
        Returns the PresenceResult and the list of bounding boxes (for debug/demo).
        """
        start_time = time.time()
        
        try:
            # Convert to grayscale for Haar Cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            # Adjust minSize or scaleFactor to optimize performance if needed
            faces = self.classifier.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            processing_time_ms = (time.time() - start_time) * 1000
            face_count = len(faces)
            
            # Logging metadata only
            if face_count > 0:
                log.info("Face detected", face_count=face_count, processing_ms=processing_time_ms)
            
            result = PresenceResult(
                face_detected=(face_count > 0),
                face_count=face_count,
                processing_time_ms=processing_time_ms,
                timestamp=time.time(),
                camera_available=True
            )
            return result, faces
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            log.error("Face detection failed", error=str(e))
            result = PresenceResult(
                face_detected=False,
                face_count=0,
                processing_time_ms=processing_time_ms,
                timestamp=time.time(),
                error=str(e),
                camera_available=True
            )
            return result, []

    def detect_presence(self, camera_index: int = 0) -> PresenceResult:
        """
        Opens the camera, captures one frame, detects presence, and closes it.
        """
        start_time = time.time()
        try:
            with Camera(camera_index) as cam:
                frame = cam.read_frame()
                result, _ = self.detect_in_frame(frame)
                
                # Add camera open time overhead to the overall processing time if desired, 
                # but we just return the result from detect_in_frame.
                return result
                
        except CameraError as e:
            log.error("Camera error during presence detection", error=str(e))
            return PresenceResult(
                face_detected=False,
                face_count=0,
                processing_time_ms=(time.time() - start_time) * 1000,
                timestamp=time.time(),
                error=str(e),
                camera_available=False
            )
