import cv2
import time
from backend.vision.face_detection import FaceDetector, Camera, CameraError

def main():
    print("Initializing Phase 2.2.1 Presence Detection Demo...")
    try:
        detector = FaceDetector()
    except Exception as e:
        print(f"Failed to initialize FaceDetector: {e}")
        return

    print("Opening camera... Press 'Q' to exit.")
    
    try:
        with Camera(0) as cam:
            while True:
                try:
                    frame = cam.read_frame()
                except CameraError as e:
                    print(f"Camera read error: {e}")
                    break

                # Detect faces
                result, faces = detector.detect_in_frame(frame)
                
                # Draw bounding boxes
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Draw HUD
                fps_text = f"Processing: {result.processing_time_ms:.1f}ms"
                status_text = "YES" if result.face_detected else "NO"
                
                color = (0, 255, 0) if result.face_detected else (0, 0, 255)
                cv2.putText(frame, f"Face Detected: {status_text}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"Faces: {result.face_count}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, fps_text, (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display preview
                cv2.imshow("Zytrix Presence Detection Demo", frame)
                
                # Handle exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting demo...")
                    break
                    
    except CameraError as e:
        print(f"Camera Error: {e}")
    finally:
        cv2.destroyAllWindows()
        print("Demo completed.")

if __name__ == "__main__":
    main()
