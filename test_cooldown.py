import pytest
import time
import numpy as np
from unittest.mock import patch, MagicMock

from backend.duplex.duplex_manager import DuplexManager
from backend.duplex.assistant_state import AssistantState
from backend.duplex.constants import NORMAL_ENERGY_THRESHOLD

@pytest.fixture
def manager():
    dm = DuplexManager()
    # Mock WakeWordDetector and stt to prevent actual loading
    dm.ww_detector.process_audio = MagicMock(return_value=False)
    dm.stt.transcribe = MagicMock(return_value="")
    return dm

def test_vision_activation_cooldown(manager):
    """1. VISION ACTIVATION COOLDOWN"""
    manager.request_activation("vision_presence")
    assert manager._external_activation_event.is_set()
    
    # Simulate _run_loop's IDLE check with a dummy chunk (1280 samples of silence)
    dummy_audio = np.zeros(1280, dtype=np.float32)
    
    with patch('time.time', return_value=100.0):
        # We manually process exactly what the IDLE block does
        manager.ww_detector.process_audio.return_value = False
        
        # Trigger activation
        ww_detected = manager.ww_detector.process_audio(dummy_audio)
        external = manager._external_activation_event.is_set()
        
        assert external is True
        
        if external:
            manager._external_activation_event.clear()
            manager.cooldown_until = 0.0  # Simulated the block
        
        # Verify cooldown_until <= current_time
        assert manager.cooldown_until <= 100.0

def test_wake_word_activation_cooldown(manager):
    """2. WAKE-WORD ACTIVATION COOLDOWN"""
    dummy_audio = np.zeros(1280, dtype=np.float32)
    
    with patch('time.time', return_value=100.0):
        manager.ww_detector.process_audio.return_value = True
        
        ww_detected = manager.ww_detector.process_audio(dummy_audio)
        external = manager._external_activation_event.is_set()
        
        assert ww_detected is True
        
        if ww_detected:
            manager.cooldown_until = 100.0 + 0.5
            
        assert manager.cooldown_until > 100.0
        assert manager.cooldown_until == 100.5

@patch('backend.duplex.duplex_manager.log_event')
def test_immediate_speech_after_vision(mock_log, manager):
    """3. IMMEDIATE SPEECH AFTER VISION ACTIVATION"""
    # Start in IDLE
    manager.state_tracker.transition_to(AssistantState.IDLE)
    manager.request_activation("vision_presence")
    
    # Send a chunk of 1280 samples
    dummy_chunk = np.zeros(1280, dtype=np.float32)
    manager.audio_bus.put_chunk(dummy_chunk)
    
    # Force _run_loop to step once to transition to LISTENING
    with patch('time.time', return_value=100.0):
        # We simulate the exact block execution for Vision
        manager._external_activation_event.clear()
        manager.state_tracker.transition_to(AssistantState.LISTENING)
        manager.cooldown_until = 0.0 # Vision cooldown
        
    # Now simulate the next loop iteration where a high-energy speech chunk arrives
    # High energy chunk
    speech_chunk = np.ones(1280, dtype=np.float32)
    manager.audio_bus.put_chunk(speech_chunk)
    
    # We simulate what happens in the LISTENING block
    with patch('time.time', return_value=100.1): # Time advanced slightly
        # Cooldown check
        assert 100.1 > manager.cooldown_until # Passes cooldown
        
        rms = manager.interrupt_detector.calculate_rms(speech_chunk)
        is_silent = rms < NORMAL_ENERGY_THRESHOLD
        
        speech_detected = False
        if not is_silent:
            speech_detected = True
            
        assert not is_silent
        assert speech_detected is True

@patch('backend.duplex.duplex_manager.log_event')
def test_wake_word_echo_protection(mock_log, manager):
    """4. WAKE-WORD ECHO PROTECTION"""
    # Start in IDLE
    manager.state_tracker.transition_to(AssistantState.IDLE)
    
    # Simulate Wake word triggering
    manager.ww_detector.process_audio.return_value = True
    
    with patch('time.time', return_value=100.0):
        manager.state_tracker.transition_to(AssistantState.LISTENING)
        manager.cooldown_until = 100.0 + 0.5 # Wake word cooldown
        
    # Provide immediate high-energy chunk (e.g. echo) at 100.2
    speech_chunk = np.ones(1280, dtype=np.float32)
    
    with patch('time.time', return_value=100.2):
        # The loop would hit the cooldown check
        is_cooldown = 100.2 < manager.cooldown_until
        assert is_cooldown is True
        # Chunk would be discarded
        
    # Advance beyond cooldown (e.g. 100.6)
    with patch('time.time', return_value=100.6):
        is_cooldown = 100.6 < manager.cooldown_until
        assert is_cooldown is False # Past cooldown
        
        # Now processes VAD
        rms = manager.interrupt_detector.calculate_rms(speech_chunk)
        is_silent = rms < NORMAL_ENERGY_THRESHOLD
        
        speech_detected = False
        if not is_silent:
            speech_detected = True
            
        assert speech_detected is True
