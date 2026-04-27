import os
import time
import numpy as np
import sounddevice as sd
import librosa
import onnxruntime as ort

# Configuration
MODEL_PATH = os.path.join("models", "hey_jarvis.onnx")
SAMPLE_RATE = 16000
DURATION = 1.5  # seconds
THRESHOLD = 0.5
N_MFCC = 40
EXPECTED_FRAMES = 151

def load_model(model_path):
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return ort.InferenceSession(model_path)

def record_audio(duration, sample_rate):
    print(f"\n[Listening for {duration} seconds... Say 'Hey Jarvis']")
    
    # Record audio using sounddevice (blocking)
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    
    print("[Done recording]")
    return audio.flatten()

def extract_features(audio, sample_rate, n_mfcc, max_frames):
    print(f"[Debug] Audio max: {np.max(audio):.4f}, min: {np.min(audio):.4f}")
    
    # Extract MFCCs with a hop_length of 160 (10ms) and n_fft of 400 (25ms)
    # This ensures 1.5s of audio (24000 samples) results in exactly 151 frames!
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, n_fft=400, hop_length=160)
    print(f"[Debug] New MFCC shape: {mfccs.shape}")
    
    # Pad or trim to exactly max_frames
    if mfccs.shape[1] < max_frames:
        pad_width = max_frames - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_frames]
        
    print(f"[Debug] MFCC Mean: {np.mean(mfccs):.4f}, Std: {np.std(mfccs):.4f}")
    
    # Standard normalization (Z-score scaling)
    # Most custom models expect normalized inputs
    mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-6)
    
    # Reshape to (1, 40, 151, 1)
    features = mfccs.reshape(1, n_mfcc, max_frames, 1)
    return features.astype(np.float32)

def main():
    try:
        session = load_model(MODEL_PATH)
        input_name = session.get_inputs()[0].name
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("\n--- Wake Word Tester Started ---")
    print("Press Ctrl+C to exit.")
    
    try:
        while True:
            # 1. Record audio
            audio_data = record_audio(DURATION, SAMPLE_RATE)
            
            # 2. Extract features
            features = extract_features(audio_data, SAMPLE_RATE, N_MFCC, EXPECTED_FRAMES)
            
            # 3. Run inference
            outputs = session.run(None, {input_name: features})
            score = outputs[0][0][0]  # Assuming output is shape (1, 1)
            
            # 4. Results
            print(f"Raw Score: {score:.4f}")
            if score > THRESHOLD:
                print("🚨 WAKE WORD DETECTED! 🚨")
            else:
                print("... Nothing detected.")
                
            time.sleep(0.5)  # Small pause before next recording
            
    except KeyboardInterrupt:
        print("\nExiting tester...")

if __name__ == "__main__":
    main()
