import asyncio
import edge_tts
import os
import tempfile
import pygame
from config import TTS_VOICE

def speak(text: str, voice: str = TTS_VOICE):
    """
    Converts text to speech and plays it over the speakers.
    Edge-TTS is asynchronous, so we use asyncio to run it.
    """
    async def _amain():
        communicate = edge_tts.Communicate(text, voice)
        
        # Create a temporary file to save the audio
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_file.close() # Close so edge-tts can write to it
        
        # Synthesize and save the speech
        await communicate.save(temp_file.name)
        
        # Initialize pygame mixer and play the audio seamlessly
        pygame.mixer.init()
        pygame.mixer.music.load(temp_file.name)
        pygame.mixer.music.play()
        
        # Wait until the audio finishes playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
            
        # Clean up pygame mixer so we can delete the file from OS
        pygame.mixer.quit()
        os.remove(temp_file.name)
        
    asyncio.run(_amain())

if __name__ == "__main__":
    speak("Hello, my systems are now online.")
