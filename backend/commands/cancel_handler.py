import logging
import threading

logger = logging.getLogger("JARVIS.CancelHandler")

# Global flag protecting long-running macros
_CANCEL_FLAG = False
_lock = threading.Lock()

def set_cancel_flag(value: bool):
    global _CANCEL_FLAG
    with _lock:
        _CANCEL_FLAG = value
    if value:
        logger.info("Global execution cancel flag RAISED!")

def check_cancelled() -> bool:
    """
    Checks if a cancellation was requested. 
    If True, it consumes the flag (resets it) and returns True.
    """
    global _CANCEL_FLAG
    with _lock:
        if _CANCEL_FLAG:
            _CANCEL_FLAG = False  # Reset after triggering
            return True
        return False

def trigger_cancellation():
    """Called asynchronously when the user says 'stop', 'cancel', or 'wait'."""
    set_cancel_flag(True)
