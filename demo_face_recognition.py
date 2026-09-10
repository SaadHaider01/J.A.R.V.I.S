import cv2
import time
from backend.vision.face_detection import Camera, CameraError
from backend.vision.face_recognition import FaceRecognizer, RecognitionState

def main():
    print("Initializing Phase 2.2.2 Face Recognition Demo...")
    try:
        # Configurable threshold can be passed here
        recognizer = FaceRecognizer(threshold=60.0)
    except Exception as e:
        print(f"Failed to initialize FaceRecognizer: {e}")
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

                # Recognize faces
                result, boxes = recognizer.recognize_in_frame(frame)
                
                # Draw bounding boxes
                for (x, y, w, h) in boxes:
                    if result.state == RecognitionState.KNOWN_USER:
                        color = (0, 255, 0)
                    elif result.state == RecognitionState.UNKNOWN_USER:
                        color = (0, 0, 255)
                    else:
                        color = (0, 255, 255) # Yellow for multi-face or no-enrollment
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw HUD
                fps_text = f"Processing: {result.processing_time_ms:.1f}ms"
                
                if result.state == RecognitionState.KNOWN_USER:
                    identity_text = "AUTHORIZED"
                    color = (0, 255, 0)
                elif result.state == RecognitionState.UNKNOWN_USER:
                    identity_text = "UNKNOWN"
                    color = (0, 0, 255)
                elif result.state == RecognitionState.MULTIPLE_FACES:
                    identity_text = "MULTIPLE FACES"
                    color = (0, 255, 255)
                elif result.state == RecognitionState.NO_ENROLLMENT:
                    identity_text = "NO ENROLLMENT"
                    color = (0, 255, 255)
                elif result.state == RecognitionState.NO_FACE:
                    identity_text = "NO FACE"
                    color = (128, 128, 128)
                else:
                    identity_text = "ERROR"
                    color = (0, 0, 255)
                
                cv2.putText(frame, f"Face: {'YES' if result.face_count > 0 else 'NO'}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Faces: {result.face_count}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Identity: {identity_text}", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Display distance metric if 1 face is detected and enrolled
                if result.face_count == 1 and result.state not in (RecognitionState.NO_ENROLLMENT, RecognitionState.RECOGNITION_ERROR):
                    distance_text = f"Distance: {result.distance:.1f}"
                    cv2.putText(frame, distance_text, (10, 120), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                cv2.putText(frame, fps_text, (10, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display preview
                cv2.imshow("Zytrix Face Recognition Demo", frame)
                
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
