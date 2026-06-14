# ==============================================================================
# J.A.R.V.I.S — CENTRAL AUDIO BUS ROUTER
# ==============================================================================
# WHAT THIS MODULE DOES:
# Manages the pipeline routing for incoming microphone audio. It implements a
# custom queue with overflow protections, queue pressure metrics, and a rolling
# pre-speech ring buffer.
#
# WHY IT EXISTS:
# Microphone callbacks run on high-priority hardware driver threads. If any
# processing (like wake-word detection or VAD) blocks that thread, audio frames
# are lost ("chopped audio"). This router decouples frame reception from processing.
#
# WHAT ADVANCED CONCEPTS ARE HERE:
#   - Producer-Consumer Pattern: The mic stream acts as the "Producer" putting
#     chunks onto the queue, while the Duplex Manager loop acts as the "Consumer"
#     getting and processing them.
#   - Queue Overflow Protection: In real-time systems, keeping latency low is more
#     important than preserving old frames. If processing lags, the queue drops
#     the oldest frames to recover immediately.
#   - Pre-Speech Buffer: A rolling window (`deque`) storing the last 1.0 second
#     of audio. When we detect an interruption, we prepend this buffer to ensure
#     the user's first spoken word (e.g. "Actually...") is not clipped.
# ==============================================================================

import time
import queue
import collections
import numpy as np
from backend.duplex.logger import log_event
from backend.duplex.metrics import metrics_tracker
from backend.duplex.constants import (
    QUEUE_MAXSIZE,
    QUEUE_PRESSURE_WARNING_PCT,
    PRE_SPEECH_BUFFER_CHUNKS
)

class AudioBus:
    def __init__(self):
        # We specify a fixed maxsize for queue overflow prevention
        self.queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
        
        # Ring buffer for pre-speech frames, protected by size constraints.
        # It automatically drops the oldest items when the maxlen is exceeded.
        self.pre_speech_ring = collections.deque(maxlen=PRE_SPEECH_BUFFER_CHUNKS)

    def put_chunk(self, chunk: np.ndarray):
        """
        Pushes a new microphone audio chunk into the bus.
        Each chunk is packed as a tuple: (timestamp, audio_data)
        """
        timestamp = time.time()
        
        # Save chunk to rolling pre-speech ring buffer
        self.pre_speech_ring.append(chunk.copy())
        
        size = self.queue.qsize()
        metrics_tracker.update_queue_pressure(size, QUEUE_MAXSIZE)
        
        try:
            # Attempt to insert frame without blocking
            self.queue.put_nowait((timestamp, chunk))
        except queue.Full:
            # Queue saturated! Drop oldest chunk to make room
            try:
                dropped_item = self.queue.get_nowait()
                metrics_tracker.record_dropped_chunk()
            except queue.Empty:
                pass
            
            # Try inserting the fresh chunk again
            try:
                self.queue.put_nowait((timestamp, chunk))
            except queue.Full:
                log_event("QUEUE", "Failed to insert chunk even after dropping old frame!", level=40) # ERROR

    def get_chunk(self, timeout: float = 0.1) -> tuple[float, np.ndarray]:
        """
        Retrieves a chunk from the queue. Blocks up to `timeout` seconds.
        Raises queue.Empty if timeout expires.
        """
        return self.queue.get(block=True, timeout=timeout)

    def clear(self):
        """Discards all accumulated chunks currently sitting in the queue."""
        log_event("QUEUE", "Clearing audio queue (flushing stale frames).")
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

    def get_pre_speech_audio(self) -> np.ndarray:
        """
        Concatenates all chunks currently in the rolling pre-speech ring buffer.
        Returns a single flat float32 array.
        """
        if not self.pre_speech_ring:
            return np.array([], dtype=np.float32)
        
        # Combine all stored numpy array chunks
        return np.concatenate(list(self.pre_speech_ring))
