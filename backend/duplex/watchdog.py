# ==============================================================================
# J.A.R.V.I.S — SYSTEM WATCHDOG & RECOVERY
# ==============================================================================
# WHAT THIS MODULE DOES:
# Runs a low-overhead background thread to monitor the health of worker threads
# and state machine transitions, executing deadman recovery if lockups occur.
#
# WHY IT EXISTS:
# Real-time systems interact with external network APIs (Groq) and low-level
# hardware streams (Windows Core Audio). If the API hangs or the audio device
# crashes silently, the system can get stuck in `THINKING` or `SPEAKING` forever.
# The watchdog acts as an external monitor to rescue the assistant when stuck.
#
# WHAT ADVANCED CONCEPTS ARE HERE:
#   - Watchdog Thread: A thread that runs in a continuous loop checking health.
#   - Deadman Recovery: Automatically returning variables/states to safe defaults
#     (like IDLE) when a worker thread dies unexpectedly.
#   - Thread Enumeration: Inspecting Python's runtime environment to verify that
#     specific worker threads are alive and active.
# ==============================================================================

import time
import threading
from backend.duplex.logger import log_event
from backend.duplex.assistant_state import StateTracker, AssistantState
from backend.duplex.constants import STATE_TIMEOUTS

class SystemWatchdog:
    def __init__(self, state_tracker: StateTracker, shutdown_event: threading.Event, tts_stop_callback=None):
        self.state_tracker = state_tracker
        self.shutdown_event = shutdown_event
        self.tts_stop_callback = tts_stop_callback
        
        self.thread = None

    def start(self):
        """Spins up the watchdog checking thread."""
        self.thread = threading.Thread(target=self._watch_loop, name="System-Watchdog", daemon=True)
        self.thread.start()
        log_event("WATCHDOG", "System health watchdog thread active.")

    def _watch_loop(self):
        """Continuously polls system parameters every 1 second until shutdown."""
        while not self.shutdown_event.is_set():
            # Sleep 1s (using wait on the shutdown_event is a great, instant-response alternative to sleep)
            self.shutdown_event.wait(timeout=1.0)
            if self.shutdown_event.is_set():
                break
                
            current_state = self.state_tracker.get_state()
            last_transition = self.state_tracker.get_last_transition_time()
            elapsed = time.time() - last_transition

            # 1. State Timeout Protection
            timeout_bound = STATE_TIMEOUTS.get(current_state.name)
            if timeout_bound and elapsed > timeout_bound:
                log_event(
                    "WATCHDOG",
                    f"State '{current_state.name}' timed out after {elapsed:.1f}s (Max: {timeout_bound}s). Triggering recovery.",
                    level=40  # ERROR
                )
                self._trigger_recovery()
                continue

            # 2. Deadman Recovery: State is SPEAKING but no active TTS player thread is running
            if current_state == AssistantState.SPEAKING:
                tts_thread_alive = False
                for t in threading.enumerate():
                    if t.name.startswith("TTS-Playback"):
                        tts_thread_alive = True
                        break
                        
                if not tts_thread_alive:
                    log_event(
                        "WATCHDOG",
                        "Deadman Triggered: State is SPEAKING but TTS Playback thread is dead. Forcing state back to IDLE.",
                        level=30  # WARNING
                    )
                    self._trigger_recovery()

    def _trigger_recovery(self):
        """Forcibly resets system threads and state machines to safe defaults."""
        # 1. Stop any active speaker stream
        if self.tts_stop_callback:
            try:
                self.tts_stop_callback()
            except Exception as e:
                log_event("WATCHDOG", f"Failed to stop TTS during recovery: {e}", level=40)
                
        # 2. Force state machine back to IDLE
        self.state_tracker.force_state(AssistantState.IDLE)
        # 3. Cycle session id to drop queued/stale work
        self.state_tracker.increment_session()
        log_event("WATCHDOG", "Deadman Recovery completed. Assistant is IDLE.")
