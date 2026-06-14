# ==============================================================================
# J.A.R.V.I.S — DUPLEX MANAGER (MASTER COORDINATOR)
# ==============================================================================
# WHAT THIS MODULE DOES:
# The core runtime coordinator of the full-duplex conversational engine.
# It runs the background microphone stream thread, processes incoming audio
# chunks through the state machine, manages VAD collection, checks for
# interruptions (barge-in), and routes speech to the STT/LLM/TTS workers.
#
# WHY IT EXISTS:
# To coordinate all concurrent operations (listening, speaking, thinking, and
# interrupting) in a single loop, ensuring low latency and preventing race
# conditions between incoming audio and outgoing speech.
#
# WHAT ADVANCED CONCEPTS ARE HERE:
#   - Continuous Stream Capture: The microphone is opened once and runs continuously
#     rather than opening and closing it repeatedly (which causes audio pop noises
#     and latency).
#   - Concurrency Events: Using `threading.Event` to control thread execution
#     flow, signaling shutdown, and active playback states safely.
#   - Temporal Prepending: Merging the rolling pre-speech ring buffer with VAD
#     speech recordings to prevent word-level clipping on user barge-in.
# ==============================================================================

import time
import queue
import threading
import numpy as np
import sounddevice as sd
from config import WAKE_WORD_NAME, DEMO_MODE, TTS_VOICE
from backend.wake_word.detector import WakeWordDetector
from backend.voice.stt import SpeechToText
from backend.nlp.agent import JarvisAgent
from backend.voice import tts
from backend.duplex.constants import (
    SAMPLE_RATE,
    CHUNK_SAMPLES,
    INTERRUPT_COOLDOWN_S,
    THINKING_INTERRUPT_GUARD_S,
    SILENCE_CHUNKS_TO_STOP,
    MAX_RECORDING_S,
    MIN_SPEECH_DURATION_S,
    PRE_SPEECH_BUFFER_CHUNKS,
    NORMAL_ENERGY_THRESHOLD
)
from backend.duplex.logger import log_event
from backend.duplex.metrics import metrics_tracker
from backend.duplex.assistant_state import AssistantState, StateTracker
from backend.duplex.audio_bus import AudioBus
from backend.duplex.interrupt_handler import InterruptionDetector, InterruptPriority
from backend.duplex.watchdog import SystemWatchdog
from backend.wake_word.listener import _run_demo_sequence

# Silence phrases Whisper hallucinates when it transcribes ambient quiet
SILENCE_PHRASES = {
    "", ".", "thank you.", "bye.", "stop.", "never mind.",
    "never mind", "stop", "dismissed", "go to sleep", "goodbye.", "goodbye"
}

# Demo Mode Toggle phrases
_DEMO_ON_PHRASES  = {"activate demo mode", "enable demo mode", "start demo mode", "demo mode on"}
_DEMO_OFF_PHRASES = {"deactivate demo mode", "disable demo mode", "stop demo mode", "demo mode off"}
DEMO_RETRIGGER_GUARD_S = 8

class DuplexManager:
    def __init__(self):
        log_event("MANAGER", "Initializing Duplex Conversational Manager...")
        
        # Concurrency Events
        self.shutdown_event = threading.Event()
        self.assistant_speaking_event = threading.Event()
        self.tts_stop_event = threading.Event()
        
        # State & Routing cores
        self.state_tracker = StateTracker()
        self.audio_bus = AudioBus()
        self.interrupt_detector = InterruptionDetector()
        
        # Core Models (VAD, STT, Agent)
        self.ww_detector = WakeWordDetector(WAKE_WORD_NAME)
        self.stt = SpeechToText()
        self.agent = JarvisAgent()
        
        # Demo mode tracking
        self.demo_mode = DEMO_MODE
        self._demo_last_run: float = 0.0
        
        # Cooldown guard after interruptions to ignore lingering room echo
        self.cooldown_until = 0.0
        
        # Timestamp of when the system entered THINKING state.
        # Used by the THINKING-state interrupt guard to enforce a 1.5s
        # tail-decay window before monitoring for new user speech.
        self._thinking_enter_time: float = 0.0
        
        # Initialize Watchdog Thread
        self.watchdog = SystemWatchdog(
            state_tracker=self.state_tracker,
            shutdown_event=self.shutdown_event,
            tts_stop_callback=self._stop_tts_playback
        )

        # Thread containers
        self.mic_stream = None
        self.manager_thread = None

    def start(self):
        """Starts the duplex processing loop and resources."""
        self.shutdown_event.clear()
        
        # Start Watchdog
        self.watchdog.start()
        
        # Start Manager processing thread
        self.manager_thread = threading.Thread(target=self._run_loop, name="Duplex-Manager", daemon=True)
        self.manager_thread.start()
        
        # Start Mic Input Stream
        self._start_mic_stream()
        log_event("MANAGER", "Duplex Conversational System is fully online.")

    def stop(self):
        """Terminates all duplex threads and cleans audio channels."""
        log_event("MANAGER", "Initiating graceful system shutdown...")
        self.shutdown_event.set()
        
        # Stop mic stream
        if self.mic_stream:
            try:
                self.mic_stream.stop()
                self.mic_stream.close()
            except Exception as e:
                log_event("MANAGER", f"Error closing audio stream: {e}", level=40)
                
        # Cancel any active TTS playback
        self._stop_tts_playback()
        
        # Flush queue
        self.audio_bus.clear()
        log_event("MANAGER", "Graceful shutdown complete.")

    def _start_mic_stream(self):
        """Starts continuous microphone capture."""
        def _audio_callback(indata, frames, time_info, status):
            if status:
                log_event("MIC", f"Status warning: {status}", level=30)
            if not self.shutdown_event.is_set():
                # Stream arrives as float32 — pass directly to queue
                self.audio_bus.put_chunk(indata.copy().flatten())

        self.mic_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=CHUNK_SAMPLES,
            callback=_audio_callback
        )
        self.mic_stream.start()
        log_event("MIC", "Microphone stream thread started successfully.")

    def _stop_tts_playback(self):
        """Internal callback to stop TTS playback instantly."""
        token = tts.current_playback_token
        if token:
            tts.stop_tts(token)
            self.tts_stop_event.set()
            self.assistant_speaking_event.clear()

    def _run_loop(self):
        """The main duplex event consumer loop."""
        import os
        ww_display = os.path.basename(WAKE_WORD_NAME).split('.')[0].replace('_', ' ')
        log_event("MANAGER", f"Waiting for wake word: '{ww_display}'...")

        # Accumulator for wake word processing
        ww_accumulator = []

        # Variables for VAD collection
        listening_buffer = []
        speech_detected = False
        silent_chunk_count = 0
        # Tracks when LISTENING state began — used to enforce a listening timeout.
        # If the user doesn't complete a speech within 12 seconds, we return to IDLE
        # to prevent the state machine from getting stuck waiting forever.
        listening_enter_time = 0.0
        # Maximum time to wait for speech in LISTENING before giving up (seconds).
        LISTENING_TIMEOUT_S = 12.0

        while not self.shutdown_event.is_set():
            try:
                # Poll chunk from queue (timeout to allow loop to check shutdown_event)
                timestamp, chunk = self.audio_bus.get_chunk(timeout=0.1)
            except queue.Empty:
                continue

            current_state = self.state_tracker.get_state()

            # ── COOLDOWN HANDLING ─────────────────────────────────────────────
            # During an interruption cooldown, we discard incoming audio chunks
            # so room echo doesn't corrupt the incoming voice buffers.
            if time.time() < self.cooldown_until:
                continue

            # ── STATE: IDLE / INTERRUPTED (Listen for Wake Word) ──────────────
            if current_state in (AssistantState.IDLE, AssistantState.INTERRUPTED):
                # openWakeWord expects 1280 sample int16 chunks
                ww_accumulator.append(chunk)
                accumulated_samples = sum(len(c) for c in ww_accumulator)
                
                if accumulated_samples >= 1280:
                    flat_float = np.concatenate(ww_accumulator)
                    # Slice exactly 1280 samples for the wake word detector to ensure consistent 80ms windows
                    to_process = flat_float[:1280]
                    leftover = flat_float[1280:]
                    
                    # Convert float32 -> int16
                    flat_int16 = (to_process * 32767).astype(np.int16)
                    
                    # Feed wake word detector
                    if self.ww_detector.process_audio(flat_int16):
                        # Retrigger guard check for demo mode
                        if self.demo_mode and (time.time() - self._demo_last_run) < DEMO_RETRIGGER_GUARD_S:
                            log_event("DEMO", "Re-trigger suppressed by demo guard timer.")
                            ww_accumulator = []
                            continue
                            
                        log_event("WAKEWORD", "Wake word detected!")
                        
                        # Demo Mode sequence bypass
                        if self.demo_mode:
                            _run_demo_sequence()
                            self._demo_last_run = time.time()
                            self.audio_bus.clear()
                            ww_accumulator = []
                            self.state_tracker.transition_to(AssistantState.IDLE)
                            continue
                            
                        # Standard Conversation Initiation
                        self.state_tracker.increment_session()
                        self.state_tracker.transition_to(AssistantState.LISTENING)
                        
                        # ── VAD Echo Cooldown after Wake Word ─────────────────────────────
                        # PROBLEM: After the wake word fires, the speaker produces TTS audio
                        # (e.g. a confirmation beep or previous TTS echo) and that energy
                        # leaks into the microphone. Without a brief cooldown, the first few
                        # high-energy chunks in the LISTENING state immediately set
                        # `speech_detected = True`, causing the system to think the user is
                        # speaking even before they've said anything.
                        # SOLUTION: Discard VAD input for a short window (0.5s = 16 chunks)
                        # right after wake word triggers, letting the speaker output settle.
                        self.cooldown_until = time.time() + 0.5
                        
                        # Reset VAD parameters
                        listening_buffer = []
                        speech_detected = False
                        silent_chunk_count = 0
                        listening_enter_time = time.time()  # Start listening timeout clock
                        
                    # Maintain sliding accumulator window without discarding leftover frames
                    ww_accumulator = [leftover] if len(leftover) > 0 else []

            # ── STATE: LISTENING (VAD Collection) ─────────────────────────────
            elif current_state == AssistantState.LISTENING:
                # ── LISTENING TIMEOUT GUARD ───────────────────────────────────
                # PROBLEM IT SOLVES: If an interruption fires while Whisper is running,
                # the state jumps to LISTENING to capture the new command. But if the
                # user doesn't say anything (their speech was clipped by the cooldown,
                # or they changed their mind), the VAD resets silently and the system
                # stays stuck in LISTENING forever — wake word is never processed,
                # and the assistant appears completely frozen.
                # FIX: After 12 seconds of being in LISTENING with no successful speech
                # committed to the pipeline, force-return to IDLE so the user can
                # say the wake word again.
                if listening_enter_time > 0 and (time.time() - listening_enter_time) > LISTENING_TIMEOUT_S:
                    log_event("VAD", f"LISTENING timed out after {LISTENING_TIMEOUT_S}s with no speech. Returning to IDLE.", level=30)
                    listening_buffer = []
                    speech_detected = False
                    silent_chunk_count = 0
                    listening_enter_time = 0.0
                    self.state_tracker.transition_to(AssistantState.IDLE)
                    continue
                
                # Check VAD Energy Levels
                rms = self.interrupt_detector.calculate_rms(chunk)
                is_silent = rms < NORMAL_ENERGY_THRESHOLD
                
                if not is_silent:
                    # Only count towards speech_detected if we actually have substantive
                    # above-threshold energy chunks, not just a single transient pop.
                    speech_detected = True
                    silent_chunk_count = 0
                    listening_buffer.append(chunk)
                else:
                    if speech_detected:
                        silent_chunk_count += 1
                        listening_buffer.append(chunk)  # Append trailing silence too (needed for natural trailing words)
                    # If speech hasn't started yet, discard silent pre-speech chunks.
                    # WHY: Without this, long silent waits before the user speaks inflate
                    # the buffer with dead frames, wasting Whisper's inference on empty audio.
                        
                # End user speech conditions:
                # - silence duration exceeded SILENCE_CHUNKS_TO_STOP
                # - recording length reached MAX_RECORDING_S ceiling
                max_chunks = int(MAX_RECORDING_S * 1000 / 30)
                
                # Minimum speech chunks guard (at least 10 chunks = 300ms of real speech detected)
                # WHY: The VAD can be fooled by a single loud pop or one chunk above threshold
                # into setting speech_detected=True and then immediately timing out 1.5s of
                # silence to submit ~2048 samples of garbage to Whisper.
                MIN_SPEECH_CHUNKS = 10  # 10 x 30ms = 300ms minimum detected above threshold
                actual_speech_chunks = len(listening_buffer) - silent_chunk_count
                
                if (speech_detected and silent_chunk_count >= SILENCE_CHUNKS_TO_STOP and actual_speech_chunks >= MIN_SPEECH_CHUNKS) \
                    or (len(listening_buffer) >= max_chunks and speech_detected):
                    
                    log_event("VAD", f"Speech finished. Recorded {len(listening_buffer) * 30 / 1000:.1f}s of audio ({actual_speech_chunks} active chunks).")
                    
                    # Consolidate user speech
                    recorded_audio = np.concatenate(listening_buffer)
                    
                    # Prepend Pre-Speech Ring Buffer to prevent word clipping
                    pre_speech = self.audio_bus.get_pre_speech_audio()
                    full_audio = np.concatenate([pre_speech, recorded_audio]) if pre_speech.size > 0 else recorded_audio
                    
                    # Transition to thinking
                    self.state_tracker.transition_to(AssistantState.THINKING)
                    
                    # Record when we entered THINKING so the interrupt guard
                    # below can enforce the 1.5s tail-decay cooldown window.
                    self._thinking_enter_time = time.time()
                    
                    # Spawn off Speech processing pipeline in separate worker thread
                    session_id = self.state_tracker.get_session_id()
                    t = threading.Thread(
                        target=self._process_speech_pipeline,
                        args=(full_audio, session_id),
                        name=f"Speech-Pipeline-{session_id}",
                        daemon=True
                    )
                    t.start()
                    
                    # Reset VAD trackers
                    listening_buffer = []
                    speech_detected = False
                    silent_chunk_count = 0
                    listening_enter_time = 0.0
                elif speech_detected and silent_chunk_count >= SILENCE_CHUNKS_TO_STOP and actual_speech_chunks < MIN_SPEECH_CHUNKS:
                    # Speech threshold was briefly crossed but not enough for a real utterance.
                    # PROBLEM IT SOLVES: A mic pop, click, or short breath spike sets
                    # speech_detected=True. Then 1.5s of silence causes the system to commit
                    # the recording to Whisper even though the user barely made a sound.
                    # IMPORTANT: We transition back to IDLE here (not just reset VAD).
                    # WHY: If we stay in LISTENING after a failed VAD, the state machine
                    # gets stuck — wake word never runs, and the assistant appears frozen.
                    # The user must say the wake word again to start a fresh conversation.
                    log_event("VAD", f"Discarded short non-speech event ({actual_speech_chunks} active chunks). Returning to IDLE.", level=30)
                    listening_buffer = []
                    speech_detected = False
                    silent_chunk_count = 0
                    listening_enter_time = 0.0
                    self.state_tracker.transition_to(AssistantState.IDLE)

            # ── STATE: THINKING (Delayed Interruption Monitoring) ─────────────
            # WHY THIS BLOCK EXISTS:
            # Previously, THINKING had no handler — all audio chunks were silently
            # discarded while the STT/LLM pipeline ran. If the user spoke a new
            # command during this window, it was completely ignored until TTS began.
            #
            # WHY WE DELAYED MONITORING (THINKING_INTERRUPT_GUARD_S):
            # The user's own voice leaves energy in the room for ~1.0–1.5s after
            # they stop speaking (room reverb, mic release). Without a guard,
            # that tail-decay would immediately exceed INTERRUPT_ENERGY_THRESHOLD
            # and self-cancel the pipeline the instant THINKING starts.
            elif current_state == AssistantState.THINKING:
                # Skip the guard window — let voice tail-decay settle first
                if (time.time() - self._thinking_enter_time) < THINKING_INTERRUPT_GUARD_S:
                    continue

                # After the guard window, use the strict SPEAKING threshold (0.02)
                # to detect clear new speech while rejecting breath and room noise
                interrupted, priority = self.interrupt_detector.detect_interrupt(chunk, is_speaking=True)

                if interrupted and priority == InterruptPriority.MEDIUM:
                    log_event("INTERRUPT", "[THINKING INTERRUPT] New speech detected during pipeline. Aborting pipeline.")

                    # Increment session ID — the pipeline worker will see a mismatch
                    # at its next checkpoint (post-STT, post-agent, or pre-TTS) and
                    # abort cleanly without any additional changes to the worker.
                    self.state_tracker.increment_session()

                    # Transition THINKING → INTERRUPTED → LISTENING
                    self.state_tracker.transition_to(AssistantState.INTERRUPTED)
                    self.cooldown_until = time.time() + INTERRUPT_COOLDOWN_S

                    # Flush queue to remove stale frames accumulated during pipeline
                    self.audio_bus.clear()

                    # Jump straight into LISTENING to capture the new command
                    self.state_tracker.transition_to(AssistantState.LISTENING)

                    # Reset VAD trackers for the fresh recording
                    listening_buffer = []
                    speech_detected = False
                    silent_chunk_count = 0
                    listening_enter_time = time.time()
                    self._thinking_enter_time = 0.0

            # ── STATE: SPEAKING (Interruption Monitoring) ─────────────────────
            elif current_state == AssistantState.SPEAKING:
                # Educational Design Decision:
                # We strictly monitor interruptions only when the assistant is SPEAKING.
                # If we were to monitor interruptions during the THINKING state (while
                # models are transcribing or running NLP reasoning), the tail-decay of
                # the user's voice, ambient room echo, or breathing would exceed the
                # highly sensitive silence threshold (0.01). This would trigger a false
                # barge-in, increment the session ID, and discard the STT/LLM pipeline
                # output before it could even start playing.
                # Disabling thinking-state interruptions prevents self-cancellation loops.
                interrupted, priority = self.interrupt_detector.detect_interrupt(chunk, is_speaking=True)
                
                if interrupted and priority == InterruptPriority.MEDIUM:
                    log_event("INTERRUPT", "[INTERRUPT DETECTED] Triggering barge-in cancellation.")
                    
                    # 1. Stop current speech
                    self._stop_tts_playback()
                    
                    # 2. Increment session ID to invalidate pending pipeline workers
                    session_id = self.state_tracker.increment_session()
                    
                    # 3. Transition to Interrupted State
                    self.state_tracker.transition_to(AssistantState.INTERRUPTED)
                    
                    # 4. Activate Echo Cooldown Window
                    self.cooldown_until = time.time() + INTERRUPT_COOLDOWN_S
                    
                    # Flush stale frames accumulated during playing
                    self.audio_bus.clear()
                    
                    # 5. Instantly jump back to LISTENING to record the user's interruption text
                    self.state_tracker.transition_to(AssistantState.LISTENING)
                    
                    # Reset VAD vars for new input
                    listening_buffer = []
                    # IMPORTANT FIX: Do NOT pre-set speech_detected=True after an interruption.
                    # WHY: We added a 0.5s cooldown window after interruption to let echo settle.
                    # If speech_detected=True, those 0.5s of discarded (cooldown) audio still
                    # count as silence chunks against our SILENCE_CHUNKS_TO_STOP counter.
                    # After ~1.5s, the system sees: speech_detected=True + lots of silence chunks
                    # but ZERO actual speech chunks → silently resets → stays stuck in LISTENING.
                    # The pre-speech ring buffer already captures the beginning of the user's
                    # voice, so we don't need to pre-assume speaking to avoid word clipping.
                    speech_detected = False
                    silent_chunk_count = 0
                    listening_enter_time = time.time()  # Start listening timeout clock

    def _process_speech_pipeline(self, audio_data: np.ndarray, session_id: str):
        """
        Runs Whisper transcription and routes results through the NLP Agent.
        Executes on a background worker thread.
        
        Educational Design Decision:
            Running the STT and LLM reasoning pipeline asynchronously on a worker
            thread prevents blocking the main coordinator loop. However, worker threads
            must be exceptionally resilient; any uncaught exception would leave the
            entire system stuck in a 'THINKING' state. Wrapping the execution in a
            try/except block and implementing immediate fallback state recovery
            avoids catastrophic lockups and bypasses the 30-second watchdog latency.
        """
        try:
            # Step 1: Transcribe Speech
            if self.state_tracker.get_session_id() != session_id:
                log_event("PIPELINE", "Session expired during thread spinup. Aborting.")
                return

            start_stt = time.time()
            log_event("PIPELINE", "Sending audio to Whisper STT...")
            command_text = self.stt.transcribe(audio_data)
            stt_duration = time.time() - start_stt
            metrics_tracker.record_stt_duration(stt_duration)
            
            # Verify Session ID
            if self.state_tracker.get_session_id() != session_id:
                log_event("PIPELINE", "Session expired during transcription. Aborting.")
                return
                
            log_event("PIPELINE", f"Whisper transcribed: '{command_text}'")
            clean = command_text.lower().strip().rstrip(".,!?;:")
            
            # Silence phrase filter
            if clean in SILENCE_PHRASES or not clean:
                log_event("PIPELINE", "User text classified as silence/empty. Returning to IDLE silently.")
                # PROBLEM REMOVED: Previously we played a "Standing by" TTS response here.
                # That TTS audio leaks back into the microphone, gets picked up by the
                # wake word detector (whose accumulated 1280-sample window contains the
                # TTS energy), and immediately re-triggers a new conversation cycle.
                # This creates a feedback loop: empty speech → TTS "Standing by" → fake
                # wake word → empty speech → TTS "Standing by" → loop forever.
                # FIX: Silently transition back to IDLE with no TTS audio output.
                if self.state_tracker.get_state() == AssistantState.THINKING:
                    self.state_tracker.transition_to(AssistantState.IDLE)
                return

            # Demo mode toggling commands
            _demo_on  = (clean in _DEMO_ON_PHRASES  or any(p in clean for p in _DEMO_ON_PHRASES))
            _demo_off = (clean in _DEMO_OFF_PHRASES or any(p in clean for p in _DEMO_OFF_PHRASES))
            
            if _demo_on:
                self.demo_mode = True
                log_event("DEMO", "Demo mode enabled by user.")
                if self.state_tracker.get_state() == AssistantState.THINKING:
                    self.state_tracker.transition_to(AssistantState.IDLE)
                    tts.speak("Demo mode activated.")
                return

            if _demo_off:
                self.demo_mode = False
                log_event("DEMO", "Demo mode disabled by user.")
                if self.state_tracker.get_state() == AssistantState.THINKING:
                    self.state_tracker.transition_to(AssistantState.IDLE)
                    tts.speak("Demo mode deactivated.")
                return

            # Step 2: Route through NLP Agent Brain
            log_event("PIPELINE", "Routing command to Groq Agent...")
            reply = self.agent.think(command_text)
            
            # Final Session ID check prior to playing speech
            if self.state_tracker.get_session_id() != session_id:
                log_event("PIPELINE", "Session expired during Agent thinking. Discarding output.")
                return

            # Step 3: Speak Response (Non-blocking)
            tts.speak(
                text=reply,
                voice=TTS_VOICE,
                session_id=session_id,
                state_tracker=self.state_tracker,
                assistant_speaking_event=self.assistant_speaking_event,
                tts_stop_event=self.tts_stop_event
            )
        except Exception as e:
            log_event("PIPELINE", f"Critical crash inside speech pipeline worker thread: {e}", level=40)
            # Safe FSM recovery logic
            if self.state_tracker.get_session_id() == session_id:
                if self.state_tracker.get_state() == AssistantState.THINKING:
                    log_event("PIPELINE", "Recovering stuck THINKING state back to IDLE post-crash.", level=30)
                    self.state_tracker.transition_to(AssistantState.IDLE)
