import asyncio
import edge_tts
import os
import tempfile
import ctypes
import threading
from config import TTS_VOICE

def play_audio_windows(file_path):
    """
    Uses native Windows DLLs to play an MP3 silently.
    This completely bypasses the need for bloated libraries like Pygame or PyAudio
    which fail to install on bleeding-edge Python versions (like Python 3.14).
    """
    alias = "jarvis_audio"
    
    # 1. Open the MP3 file natively
    ctypes.windll.winmm.mciSendStringW(f'open "{file_path}" alias {alias}', None, 0, None)
    
    # 2. Play it and wait for it to finish ('wait' ensures Python blocks until the speech stops)
    ctypes.windll.winmm.mciSendStringW(f'play {alias} wait', None, 0, None)
    
    # 3. Close the stream so the file lock is released
    ctypes.windll.winmm.mciSendStringW(f'close {alias}', None, 0, None)

def speak(text: str, voice: str = TTS_VOICE):
    """
    Converts text to speech and plays it over the speakers.
    Edge-TTS is asynchronous, so we use asyncio to run it.
    """
    async def _amain():
        communicate = edge_tts.Communicate(text, voice)
        
        # Create a temporary file to save the audio
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_file.close() # Close immediately so edge-tts can write to it
        
        # Synthesize and save the speech
        await communicate.save(temp_file.name)
        
        # Play the audio using our native Windows bypass
        play_audio_windows(temp_file.name)
        
        # Clean up the file from the OS
        try:
            os.remove(temp_file.name)
        except Exception:
            pass
        
    def _thread_worker():
        asyncio.run(_amain())
        
    t = threading.Thread(target=_thread_worker)
    t.start()
    t.join()

if __name__ == "__main__":
    speak("Hello, my systems are now online.")
