# ==============================================================================
# J.A.R.V.I.S — CONCURRENT SPEECH SYNTHESIS (TTS)
# ==============================================================================
# WHAT THIS MODULE DOES:
# Converts LLM textual responses to speech and plays them natively via Windows
# MCI. It supports non-blocking concurrent playback, session tracking, and
# graceful interruption volume fade-outs.
#
# WHY IT EXISTS:
# In a full-duplex system, speech playback must run on its own thread so it does
# not block the microphone thread. Additionally, we must track who owns the
# active audio stream so we can cancel it instantly when a barge-in occurs.
#
# WHAT ADVANCED CONCEPTS ARE HERE:
#   - Playback Token Ownership: Assigning a unique UUID token to each playback request.
#     Only the holder of the active token can stop playback or change speaking states,
#     preventing race conditions from overlapping responses.
#   - Windows Multimedia Control Interface (MCI): A high-level Windows DLL API
#     governing media devices. Sending MCI commands enables us to play and stop
#     sound files asynchronously without Python overhead.
#   - Graceful Fade-Out: Linearly dropping speaker volume over 50-100ms prior to
#     a hard stop to avoid clicking/popping audio artifacts.
# ==============================================================================

import asyncio
import edge_tts
import os
import tempfile
import ctypes
import threading
import time
import uuid
from config import TTS_VOICE
from backend.duplex.logger import log_event
from backend.duplex.assistant_state import AssistantState

# Global lock to synchronize modifications to token tracking variables
_playback_lock = threading.Lock()
current_playback_token = None

def stop_tts(token: str):
    """
    Instantly stops the Windows MCI player associated with the token.
    Fades out the audio volume over 75ms before stopping to avoid speaker pops.
    """
    if not token:
        return
        
    alias = f"zytrix_{token}"
    log_event("TTS", f"Requesting stop for active playback token: {token}")
    
    # 1. Graceful volume fade-out
    # MCI audio volume ranges from 0 (silence) to 1000 (maximum).
    # We step it down in 5 increments with brief sleeps to make the cutoff smooth.
    for vol in [800, 600, 400, 200, 0]:
        ctypes.windll.winmm.mciSendStringW(f"setaudio {alias} volume to {vol}", None, 0, None)
        time.sleep(0.015) # 15ms delay * 5 = 75ms total fade
        
    # 2. Hard stop and close commands sent to Windows winmm DLL
    # Sending 'stop' halts the playback stream.
    # Sending 'close' releases the lock on the temporary MP3 file on disk.
    ctypes.windll.winmm.mciSendStringW(f"stop {alias}", None, 0, None)
    ctypes.windll.winmm.mciSendStringW(f"close {alias}", None, 0, None)
    log_event("TTS", f"[TTS STOPPED] Audio stopped and file released for alias: {alias}")

def play_audio_windows_threaded(file_path: str, token: str, session_id: str, state_tracker, assistant_speaking_event, post_tts_callback=None):
    """
    Worker thread that runs Windows MCI playback.
    Blocks the playback thread, but is safely interrupted if another thread calls stop_tts().
    post_tts_callback: optional callable invoked after natural (non-interrupted) TTS completion
                       to let the DuplexManager set a mic-echo cooldown.
    """
    alias = f"zytrix_{token}"
    
    log_event("TTS", f"Opening MCI audio file under alias: {alias}")
    # 1. Open the MP3 file natively in Windows
    ctypes.windll.winmm.mciSendStringW(f'open "{file_path}" alias {alias}', None, 0, None)
    
    # Verify that the session is still active before we start play
    if state_tracker and state_tracker.get_session_id() != session_id:
        log_event("TTS", "Session ID changed before play started. Discarding audio.")
        ctypes.windll.winmm.mciSendStringW(f"close {alias}", None, 0, None)
        try:
            os.remove(file_path)
        except Exception:
            pass
        return

    # Set maximum volume initially (1000)
    ctypes.windll.winmm.mciSendStringW(f"setaudio {alias} volume to 1000", None, 0, None)
    
    log_event("TTS", "Starting audio playback...")
    # 2. Play and block. The 'wait' parameter suspends this thread until playback ends
    # OR until another thread calls 'stop'/'close' on the alias.
    start_play = time.time()
    ctypes.windll.winmm.mciSendStringW(f'play {alias} wait', None, 0, None)
    play_duration = time.time() - start_play
    log_event("TTS", f"Playback thread unblocked after {play_duration:.2f}s.")
    
    # 3. Clean up the MCI device
    ctypes.windll.winmm.mciSendStringW(f"close {alias}", None, 0, None)

    # 4. State updates and file deletion
    naturally_completed = False
    with _playback_lock:
        global current_playback_token
        # Check if we were the thread that just finished speaking naturally
        if current_playback_token == token:
            naturally_completed = True
            current_playback_token = None
            if assistant_speaking_event:
                assistant_speaking_event.clear()
            # Fire the post-TTS echo cooldown BEFORE transitioning to IDLE.
            # CRITICAL ORDERING: The cooldown_until timestamp must be set before
            # the IDLE state is visible to _run_loop. If we transition first,
            # the run_loop can process wake-word audio in the ~1ms gap before
            # the callback runs, causing immediate false wake word re-triggers.
            if post_tts_callback:
                try:
                    post_tts_callback()
                except Exception:
                    pass
            if state_tracker and state_tracker.get_state() == AssistantState.SPEAKING:
                state_tracker.transition_to(AssistantState.IDLE)

                
    # Clean up the temporary MP3 file from the disk
    try:
        os.remove(file_path)
    except Exception as e:
        log_event("TTS", f"Failed to remove temp audio file {file_path}: {e}", level=30)

def speak(text: str, voice: str = TTS_VOICE, session_id: str = None, state_tracker = None, assistant_speaking_event = None, tts_stop_event = None, post_tts_callback = None):
    """
    Converts text to speech and schedules playback in a non-blocking background thread.
    Can be called synchronously (fallback) if duplex arguments are omitted.
    """
    # 1. Synthesize edge-tts asynchronously
    async def _synthesize(temp_name):
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(temp_name)

    # Generate a temporary file to store audio
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_file.close() # Close handle so edge-tts stream can write to it
    
    try:
        # Measure startup time of synthesis
        start_synth = time.time()
        asyncio.run(_synthesize(temp_file.name))
        synth_duration = time.time() - start_synth
        log_event("TTS", f"Synthesized speech in {synth_duration * 1000.0:.1f}ms.")
    except Exception as e:
        log_event("TTS", f"Synthesis failed: {e}", level=40)
        return

    # Check if this request is still relevant (e.g. user didn't interrupt while we were synthesizing)
    if state_tracker and state_tracker.get_session_id() != session_id:
        log_event("TTS", "Session ID changed during TTS synthesis. Discarding audio.")
        try:
            os.remove(temp_file.name)
        except:
            pass
        return

    # 2. Check if we are running in full duplex mode or fallback
    if state_tracker and assistant_speaking_event and tts_stop_event:
        with _playback_lock:
            # Cancel any active speech before starting the new one
            global current_playback_token
            if current_playback_token:
                stop_tts(current_playback_token)
                
            # Assign a new token to lock ownership
            playback_token = str(uuid.uuid4())
            current_playback_token = playback_token
            
            # Reset event flags
            tts_stop_event.clear()
            assistant_speaking_event.set()
            
            # Transition state machine
            state_tracker.transition_to(AssistantState.SPEAKING)
            
        # Spawn the thread that plays the MP3 file natively
        # We prefix the thread name with 'TTS-Playback-' so the Watchdog can track its status.
        t = threading.Thread(
            target=play_audio_windows_threaded,
            args=(temp_file.name, playback_token, session_id, state_tracker, assistant_speaking_event, post_tts_callback),
            name=f"TTS-Playback-{playback_token}",
            daemon=True
        )
        t.start()
    else:
        # Fallback synchronous playback (for testing scripts or simple mode)
        alias = "zytrix_sync"
        ctypes.windll.winmm.mciSendStringW(f'open "{temp_file.name}" alias {alias}', None, 0, None)
        ctypes.windll.winmm.mciSendStringW(f'play {alias} wait', None, 0, None)
        ctypes.windll.winmm.mciSendStringW(f'close {alias}', None, 0, None)
        try:
            os.remove(temp_file.name)
        except:
            pass

if __name__ == "__main__":
    # Test script run
    speak("Hello, my systems are now online in fallback mode.")
