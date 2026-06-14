# ==============================================================================
# J.A.R.V.I.S — CONVERSATIONAL METRICS TRACKER
# ==============================================================================
# WHAT THIS MODULE DOES:
# Keeps track of key performance indicators (KPIs) of the real-time system,
# including response delays, queue pressure, dropped packets, and latency targets.
#
# WHY IT EXISTS:
# In real-time speech systems, humans perceive delays above 250-300ms as lagging
# and unnatural. To optimize this, we must measure performance. This module acts
# as a software profiler tracking real execution times against our targets.
#
# WHAT ADVANCED CONCEPTS ARE HERE:
#   - Thread Safety: Since metrics are updated from multiple threads (e.g. Duplex
#     Manager, TTS Worker, Watchdog), we protect all read/write operations using a
#     threading Lock to prevent race conditions.
# ==============================================================================

import threading
from backend.duplex.logger import log_event
from backend.duplex.constants import (
    GOAL_INTERRUPT_DETECTION_MS,
    GOAL_TTS_STOP_MS,
    GOAL_STATE_TRANSITION_MS
)

class MetricsTracker:
    def __init__(self):
        self._lock = threading.Lock()
        
        # Metrics storage
        self.interrupt_latencies = []
        self.stt_durations = []
        self.tts_startup_times = []
        
        self.dropped_chunks = 0
        self.false_interrupts = 0
        
        self.peak_queue_pressure = 0
        self.current_queue_size = 0

    def record_interrupt_latency(self, latency_seconds: float):
        """Records the time it took from speech start until TTS actually stopped."""
        ms = latency_seconds * 1000.0
        with self._lock:
            self.interrupt_latencies.append(ms)
            
        status = "PASSED" if ms <= GOAL_INTERRUPT_DETECTION_MS else "FAILED"
        log_event(
            "METRICS",
            f"Barge-in Latency: {ms:.1f}ms (Budget: {GOAL_INTERRUPT_DETECTION_MS}ms) | Target Status: {status}"
        )

    def record_stt_duration(self, duration_seconds: float):
        """Records how long Whisper STT took to transcribe user input."""
        ms = duration_seconds * 1000.0
        with self._lock:
            self.stt_durations.append(ms)
        log_event("METRICS", f"Whisper STT processing time: {ms:.1f}ms")

    def record_tts_startup(self, startup_seconds: float):
        """Records delay between sending request to edge-tts and sound playing."""
        ms = startup_seconds * 1000.0
        with self._lock:
            self.tts_startup_times.append(ms)
        log_event("METRICS", f"TTS synthesis/startup latency: {ms:.1f}ms")

    def record_dropped_chunk(self):
        """Increments count of dropped audio chunks due to queue saturation."""
        with self._lock:
            self.dropped_chunks += 1
        log_event("METRICS", "Audio chunk dropped due to queue overflow!", level=30)  # WARNING level

    def record_false_interrupt(self):
        """Tracks cases where interruption triggers but STT produces empty transcription."""
        with self._lock:
            self.false_interrupts += 1
        log_event("METRICS", "False interruption detected (no speech transcribed).")

    def update_queue_pressure(self, size: int, maxsize: int):
        """Logs and monitors the size of the central audio queue."""
        pressure_pct = (size / maxsize) * 100.0
        with self._lock:
            self.current_queue_size = size
            if size > self.peak_queue_pressure:
                self.peak_queue_pressure = size
                
        if pressure_pct >= 80.0:
            log_event(
                "METRICS",
                f"Queue Pressure critical: {size}/{maxsize} ({pressure_pct:.1f}%)",
                level=30  # WARNING
            )

    def print_diagnostics_report(self):
        """Prints a summary of system latency and drops for debugging."""
        with self._lock:
            avg_interrupt = (sum(self.interrupt_latencies) / len(self.interrupt_latencies)) if self.interrupt_latencies else 0.0
            avg_stt = (sum(self.stt_durations) / len(self.stt_durations)) if self.stt_durations else 0.0
            avg_tts = (sum(self.tts_startup_times) / len(self.tts_startup_times)) if self.tts_startup_times else 0.0
            
            report = (
                f"\n==== JARVIS PERFORMANCE DIAGNOSTICS REPORT ====\n"
                f" - Avg Barge-In Stop: {avg_interrupt:.1f}ms (Target: <{GOAL_INTERRUPT_DETECTION_MS}ms)\n"
                f" - Avg Whisper STT:   {avg_stt:.1f}ms\n"
                f" - Avg TTS Startup:   {avg_tts:.1f}ms\n"
                f" - Peak Queue Size:   {self.peak_queue_pressure} frames\n"
                f" - Total Drops:       {self.dropped_chunks} chunks\n"
                f" - False Interrupts:  {self.false_interrupts} events\n"
                f"================================================"
            )
            log_event("METRICS", report)

# Singleton global instance
metrics_tracker = MetricsTracker()
