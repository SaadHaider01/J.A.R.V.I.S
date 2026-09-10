import argparse
import sys
from backend.vision.face_detection import Camera, CameraError
from backend.vision.face_recognition import FaceRecognizer, BiometricStore

def main():
    print("Phase 2.2.2: Face Enrollment")
    print("-" * 30)
    
    parser = argparse.ArgumentParser(description="Zytrix Local Face Enrollment")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite existing enrollment")
    args = parser.parse_args()

    store = BiometricStore()
    
    if store.is_enrolled() and not args.force:
        response = input("An enrollment already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Enrollment cancelled.")
            sys.exit(0)
    
    print("Initializing camera...")
    try:
        recognizer = FaceRecognizer()
    except Exception as e:
        print(f"Failed to initialize FaceRecognizer: {e}")
        sys.exit(1)

    print("\nPlease position exactly ONE face in front of the camera.")
    print("Capturing in 3 seconds...")
    import time
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)

    print("Capturing...")
    try:
        with Camera(0) as cam:
            # We take a few frames to let the camera auto-exposure settle
            for _ in range(5):
                cam.read_frame()
                time.sleep(0.1)
                
            frame = cam.read_frame()
            
            success = recognizer.enroll_face(frame)
            if success:
                print("\nEnrollment Successful! Biometric model saved locally.")
                print(f"Location: {store.file_path}")
            else:
                print("\nEnrollment Failed! Ensure exactly ONE face is visible.")
                sys.exit(1)
                
    except CameraError as e:
        print(f"\nCamera Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected Error during enrollment: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
