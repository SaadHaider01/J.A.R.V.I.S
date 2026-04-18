import sounddevice as sd
import queue
import logging
from config import CONVERSATION_TIMEOUT
from backend.wake_word.detector import WakeWordDetector
from backend.voice.tts import speak
from backend.voice.stt import SpeechToText
from backend.nlp.agent import JarvisAgent

logger = logging.getLogger("JARVIS.Listener")

# Words Whisper hallucinates when it hears silence
SILENCE_PHRASES = {
    "", ".", "thank you.", "bye.", "stop.", "never mind.",
    "never mind", "stop", "dismissed", "go to sleep", "goodbye.", "goodbye"
}

class MicrophoneListener:
    def __init__(self):
        self.sample_rate = 16000
        self.chunk_size = 1280
        self.audio_queue = queue.Queue()

        self.detector = WakeWordDetector("hey_mycroft")
        self.stt = SpeechToText()
        self.agent = JarvisAgent()
        self.is_listening = False
        self.ignore_mic = False  # Safety lock: prevents JARVIS from hearing himself

    def audio_callback(self, indata, frames, time, status):
        """Streams raw mic bytes into a queue for the Wake Word detector."""
        if self.ignore_mic:
            return
        self.audio_queue.put(indata.copy().flatten())

    def listen_for_wake_word(self):
        """
        The master event loop. Sits silently until 'Hey Mycroft' is detected,
        then enters a continuous conversation loop until the user goes silent.
        """
        self.is_listening = True

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='int16',
            blocksize=self.chunk_size,
            callback=self.audio_callback
        ):
            logger.info("JARVIS is actively listening for 'Hey Mycroft'...")

            while self.is_listening:
                audio_chunk = self.audio_queue.get()

                if self.detector.process_audio(audio_chunk):
                    logger.info("Wake word detected! Entering conversation mode.")
                    self.ignore_mic = True

                    in_conversation = True
                    while in_conversation:
                        # ── Record the user's spoken command ──────────────
                        logger.info(f"Mic open for {CONVERSATION_TIMEOUT} seconds...")
                        command_audio = sd.rec(
                            int(CONVERSATION_TIMEOUT * self.sample_rate),
                            samplerate=self.sample_rate,
                            channels=1,
                            dtype='float32'
                        )
                        sd.wait()

                        # ── Transcribe audio to text ───────────────────────
                        logger.info("Transcribing...")
                        command_text = self.stt.transcribe(command_audio)
                        logger.info(f"User said: '{command_text}'")

                        clean = command_text.lower().strip()

                        # ── Check for silence or exit command ──────────────
                        if clean in SILENCE_PHRASES:
                            logger.info("Silence or exit command detected. Sleeping.")
                            speak("Standing by.")
                            in_conversation = False
                            continue

                        # ── Hand ALL commands to the Phase 8 Agent ─────────
                        # The Agent now decides EVERYTHING:
                        # whether to open an app, type text, search the web,
                        # or just have a conversation — no more hardcoded rules!
                        logger.info("Routing to Phase 8 Agent Brain...")
                        reply = self.agent.think(command_text)
                        speak(reply)

                        # Flush the queue to avoid stale-audio double triggers
                        while not self.audio_queue.empty():
                            self.audio_queue.get()

                    # ── Return to Wake Word detection ──────────────────────
                    self.ignore_mic = False
                    logger.info("Resuming wake word detection. Waiting for 'Hey Mycroft'...")

    def stop(self):
        self.is_listening = False
