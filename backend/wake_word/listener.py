import sounddevice as sd
import numpy as np
import queue
import logging
from backend.wake_word.detector import WakeWordDetector
from backend.voice.tts import speak
from backend.voice.stt import SpeechToText
from backend.nlp.agent import JarvisAgent
from backend.commands.app_launcher import launch_app

logger = logging.getLogger("JARVIS.Listener")

class MicrophoneListener:
    def __init__(self):
        self.sample_rate = 16000
        self.chunk_size = 1280
        self.audio_queue = queue.Queue()
        
        # Bring all modules online!
        self.detector = WakeWordDetector("hey_mycroft")
        self.stt = SpeechToText()
        self.agent = JarvisAgent()
        self.is_listening = False

    def audio_callback(self, indata, frames, time, status):
        """Pushes microphone feed into a continuous queue."""
        if status:
            pass # ignore minor underrun warnings
        self.audio_queue.put(indata.copy().flatten())

    def listen_for_wake_word(self):
        """
        The continuous loop holding JARVIS together.
        """
        self.is_listening = True
        
        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='int16', blocksize=self.chunk_size, callback=self.audio_callback):
            logger.info("JARVIS is actively listening for 'Hey Mycroft'...")
            
            while self.is_listening:
                audio_chunk = self.audio_queue.get()
                
                # Check if the user said "Hey Mycroft"
                if self.detector.process_audio(audio_chunk):
                    logger.info("Wake word detected!")
                    speak("Yes sir?")
                    
                    # 1. Listen to the command for 5 seconds
                    logger.info("Recording command for 5 seconds...")
                    # We switch to float32 because Whisper STT expects high fidelity arrays!
                    command_audio = sd.rec(int(5 * self.sample_rate), samplerate=self.sample_rate, channels=1, dtype='float32')
                    sd.wait() # Blocks python execution until 5 seconds passes
                    
                    # 2. Transcribe using Whisper
                    logger.info("Transcribing audio...")
                    command_text = self.stt.transcribe(command_audio)
                    logger.info(f"User said: {command_text}")
                    
                    # Process the text if we actually heard something
                    if command_text.strip():
                        # HARDCODED OS OVERRIDES
                        if "open notepad" in command_text.lower():
                            speak("Opening Notepad.")
                            launch_app("notepad")
                        elif "open browser" in command_text.lower():
                            speak("Opening Microsoft Edge.")
                            launch_app("browser")
                        else:
                            # 3. LLM Agent Processing (Normal conversation / Web access)
                            logger.info("Processing via Groq Agent...")
                            reply = self.agent.think(command_text)
                            
                            # 4. Speak the answer
                            speak(reply)
                    
                    # Clear the queue to prevent it double-triggering off its own voice
                    while not self.audio_queue.empty():
                        self.audio_queue.get()
                    
                    logger.info("Resuming wake word detection. Waiting for 'Hey Mycroft'...")

    def stop(self):
        self.is_listening = False
