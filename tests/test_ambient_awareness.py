import unittest
import queue
import time
import numpy as np
from unittest.mock import MagicMock

from backend.duplex.audio_bus import AudioBus
from backend.audio.ambient_awareness import AmbientAwarenessService, AmbientEvent

class MockClock:
    def __init__(self, start_time=0.0):
        self._time = start_time
    def time(self):
        return self._time
    def advance(self, seconds):
        self._time += seconds

class TestAmbientAwareness(unittest.TestCase):
    def test_audio_bus_fan_out(self):
        """Verify AudioBus pushes chunks to both queues and neither blocks the producer."""
        bus = AudioBus()
        chunk = np.zeros(480, dtype=np.float32)
        
        bus.put_chunk(chunk)
        
        self.assertEqual(bus.queue.qsize(), 1)
        self.assertEqual(bus.ambient_queue.qsize(), 1)
        
        # Test getting from duplex
        t1, c1 = bus.get_chunk(timeout=0)
        self.assertTrue(np.array_equal(c1, chunk))
        self.assertEqual(bus.queue.qsize(), 0)
        self.assertEqual(bus.ambient_queue.qsize(), 1)
        
        # Test getting from ambient
        t2, c2 = bus.get_ambient_chunk(timeout=0)
        self.assertTrue(np.array_equal(c2, chunk))
        self.assertEqual(bus.ambient_queue.qsize(), 0)
        
    def test_audio_bus_ambient_overflow(self):
        """Verify that ambient queue overflowing does not drop duplex frames or crash producer."""
        bus = AudioBus()
        chunk = np.zeros(480, dtype=np.float32)
        
        # Fill both queues to max
        import backend.duplex.constants as constants
        max_size = constants.QUEUE_MAXSIZE
        
        for _ in range(max_size):
            bus.put_chunk(chunk)
            
        self.assertEqual(bus.queue.qsize(), max_size)
        self.assertEqual(bus.ambient_queue.qsize(), max_size)
        
        # Add one more. Duplex will drop oldest. Ambient will drop oldest.
        bus.put_chunk(chunk)
        
        self.assertEqual(bus.queue.qsize(), max_size)
        self.assertEqual(bus.ambient_queue.qsize(), max_size)
        
    def test_classifier_whitelist_and_confidence(self):
        """Verify that only whitelisted classes above threshold emit events."""
        mock_clock = MockClock()
        bus = AudioBus()
        service = AmbientAwarenessService(audio_bus=bus, clock=mock_clock)
        service.classifier_ready = True
        
        # Mock classifier to return specific predictions
        emitted_events = []
        
        # Replace _handle_detection to spy on it
        original_handle = service._handle_detection
        def spy_handle(event_type, conf):
            emitted_events.append((event_type, conf))
            original_handle(event_type, conf)
        service._handle_detection = spy_handle
        
        # 1. High confidence, whitelisted (Telephone)
        service.classifier = MagicMock(return_value=[{'score': 0.95, 'label': 'Telephone'}])
        service._process_audio_window(np.zeros(16000))
        self.assertEqual(len(emitted_events), 1)
        self.assertEqual(emitted_events[-1][0], "PHONE_RING")
        
        # 2. High confidence, NOT whitelisted (Dog barking)
        service.classifier = MagicMock(return_value=[{'score': 0.99, 'label': 'Dog barking'}])
        service._process_audio_window(np.zeros(16000))
        self.assertEqual(len(emitted_events), 1) # Unchanged
        
        # 3. Low confidence, whitelisted (Knock)
        service.classifier = MagicMock(return_value=[{'score': 0.49, 'label': 'Knock'}])
        service._process_audio_window(np.zeros(16000))
        self.assertEqual(len(emitted_events), 1) # Unchanged
        
    def test_debounce_cooldown(self):
        """Verify that consecutive same-type events within cooldown are debounced."""
        mock_clock = MockClock(0.0)
        bus = AudioBus()
        service = AmbientAwarenessService(audio_bus=bus, clock=mock_clock)
        service.classifier_ready = True
        
        emitted_events = []
        # Monkey patch actual event logging/delivery to just append to our test list
        def mock_delivery(event_type, conf):
            # We call the original handle, but we patch the inner emission logic
            current_time = service.clock.time()
            last_time = service.last_event_times.get(event_type, -999.0)
            if (current_time - last_time) >= service.event_cooldown_seconds:
                emitted_events.append(event_type)
                service.last_event_times[event_type] = current_time
        
        service._handle_detection = mock_delivery
        
        # First ring at t=0
        service.classifier = MagicMock(return_value=[{'score': 0.9, 'label': 'Telephone'}])
        service._process_audio_window(np.zeros(1600))
        self.assertEqual(emitted_events, ["PHONE_RING"])
        
        # Second ring at t=2 (debounced)
        mock_clock.advance(2.0)
        service._process_audio_window(np.zeros(1600))
        self.assertEqual(emitted_events, ["PHONE_RING"]) # No new event
        
        # Knock at t=3 (different type, should emit)
        mock_clock.advance(1.0)
        service.classifier = MagicMock(return_value=[{'score': 0.9, 'label': 'Knock'}])
        service._process_audio_window(np.zeros(1600))
        self.assertEqual(emitted_events, ["PHONE_RING", "KNOCK"])
        
        # Third ring at t=11 (past cooldown, should emit)
        mock_clock.advance(8.0) # total time = 11.0
        service.classifier = MagicMock(return_value=[{'score': 0.9, 'label': 'Telephone'}])
        service._process_audio_window(np.zeros(1600))
        self.assertEqual(emitted_events, ["PHONE_RING", "KNOCK", "PHONE_RING"])

if __name__ == '__main__':
    unittest.main()
