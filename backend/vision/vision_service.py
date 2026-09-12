import threading
import time
import logging
from typing import Callable, Optional
from backend.vision.face_detection import Camera, CameraError
from backend.vision.face_recognition import FaceRecognizer, RecognitionState
from backend.vision.presence_manager import PresenceManager, PresenceEvent

logger = logging.getLogger("ZYTRIX.VisionService")

class VisionService:
    """
    Background service that integrates Camera, FaceRecognizer, and PresenceManager
    to monitor user presence at a consistent framerate (e.g., 10 FPS).
    """
    def __init__(self, on_activation: Callable[[PresenceEvent], None], fps: int = 10):
        self.on_activation = on_activation
        self.interval = 1.0 / fps if fps > 0 else 0.1
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Starts the background vision processing loop."""
        if self._thread and self._thread.is_alive():
            logger.warning("VisionService is already running.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="Vision-Service", daemon=True)
        self._thread.start()
        logger.info("VisionService started.")

    def stop(self):
        """Signals the background loop to stop and waits for termination."""
        if not self._stop_event.is_set():
            logger.info("Stopping VisionService...")
            self._stop_event.set()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            self._thread = None
            logger.info("VisionService stopped.")

    def _run_loop(self):
        try:
            # Initialize internal vision pipeline objects
            recognizer = FaceRecognizer()
            manager = PresenceManager(activation_callback=self.on_activation)
            
            with Camera(0) as cam:
                logger.info("VisionService camera activated.")
                while not self._stop_event.is_set():
                    loop_start = time.time()
                    
                    try:
                        frame = cam.read_frame()
                        result, _ = recognizer.recognize_in_frame(frame)
                        manager.update(result)
                    except CameraError as e:
                        logger.error(f"VisionService camera error: {e}")
                        # If camera drops, we might want to sleep before retry or just continue
                        time.sleep(1.0)
                        
                    # Calculate how long to sleep to maintain target FPS
                    elapsed = time.time() - loop_start
                    sleep_time = max(0, self.interval - elapsed)
                    time.sleep(sleep_time)
                    
        except Exception as e:
            logger.error(f"VisionService encountered an unexpected error: {e}")
            # Ensure safe stop on catastrophic failure
            self._stop_event.set()
        finally:
            logger.info("VisionService background loop terminated.")
