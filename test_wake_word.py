import os
import time
import numpy as np
import sounddevice as sd
import openwakeword
from openwakeword.model import Model
from config import WAKE_WORD_NAME, WAKE_WORD_SENSITIVITY

SAMPLE_RATE = 16000
CHUNK_SIZE = 1280  # 80ms chunks — what openWakeWord expects

def main():
    print(f"Loading model: {WAKE_WORD_NAME}")
    openwakeword.utils.download_models()
    model = Model(wakeword_models=[WAKE_WORD_NAME], inference_framework="onnx")
    print("✅ Model loaded. Say your wake word!\n")
    print("Press Ctrl+C to exit.\n")

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16',
                            blocksize=CHUNK_SIZE) as stream:
            while True:
                chunk, _ = stream.read(CHUNK_SIZE)
                audio_int16 = chunk.flatten()

                prediction = model.predict(audio_int16)
                if not prediction:
                    continue

                score = max(prediction.values())
                if score > 0.1:  # Print anything above noise floor
                    flag = "🚨 DETECTED!" if score > WAKE_WORD_SENSITIVITY else "..."
                    print(f"Score: {score:.4f}  {flag}")

    except KeyboardInterrupt:
        print("\nExiting tester.")

if __name__ == "__main__":
    main()
