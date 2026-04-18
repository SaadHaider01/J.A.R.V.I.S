import sounddevice as sd
import numpy as np
import queue
import logging
from backend.wake_word.detector import WakeWordDetector
from backend.voice.tts import speak

logger = logging.getLogger("JARVIS.Listener")

class MicrophoneListener:
    def __init__(self):
        self.sample_rate = 16000 # 16kHz is required by Whisper and OpenWakeWord
        self.chunk_size = 1280   # Chunk size (frames). 1280 frames = 80ms of audio
        self.audio_queue = queue.Queue()
        # Initializing the detector inside the listener
        self.detector = WakeWordDetector("hey_mycroft")
        self.is_listening = False

    def audio_callback(self, indata, frames, time, status):
        """
        This callback is automatically called by continuous sounddevice streams.
        We put the raw audio chunk into a queue so our main loop can process it without blocking the microphone.
        """
        if status:
            logger.warning(status)
        # indata is purely whatever the mic caught. We copy and flatten it to a clean 1D Array.
        self.audio_queue.put(indata.copy().flatten())

    def listen_for_wake_word(self):
        """
        Opens the microphone stream and continuously listens in a loop for the wake word.
        """
        logger.info("Starting microphone stream...")
        self.is_listening = True
        
        # Initialize the audio stream
        with sd.InputStream(samplerate=self.sample_rate, 
                            channels=1, 
                            dtype='int16', # Int16 is exactly what openwakeword is trained on
                            blocksize=self.chunk_size, 
                            callback=self.audio_callback):
            
            logger.info("Listening for wake word ('hey mycroft')...")
            
            while self.is_listening:
                # Grab a chunk of audio from the queue (this blocks until a chunk is available)
                audio_chunk = self.audio_queue.get()
                
                # Check if the wake word is in this chunk
                if self.detector.process_audio(audio_chunk):
                    logger.info("Wake word detected!")
                    speak("Yes, how can I help?")
                    
                    # TODO: Trigger the STT (Speech-To-Text) listening phase here.
                    # Currently, we just tell the system we heard it and loop!
                    
                    # Clear the queue to prevent double triggers/echoes
                    while not self.audio_queue.empty():
                        self.audio_queue.get()

    def stop(self):
        self.is_listening = False
