# ==============================================================================
# J.A.R.V.I.S — CENTRALIZED EVENT LOGGER
# ==============================================================================
# WHAT THIS MODULE DOES:
# Implements a centralized logging helper `log_event` to format, filter, and
# track events across the real-time full-duplex pipeline.
#
# WHY IT EXISTS:
# Real-time voice systems are highly concurrent and notoriously hard to debug.
# Scatterings of random `print` statements result in jumbled text, lost context,
# and no timestamp traceability. Standardizing logs allows us to debug timing
# synchronization, state jumps, and race conditions accurately.
#
# WHAT ADVANCED CONCEPTS ARE HERE:
#   - Concurrency Safety: Multiple worker threads might attempt to print to
#     stdout at the same time. The Python `logging` module handles synchronization
#     under the hood (using an internal lock), preventing garbled text collisions.
# ==============================================================================

import logging
from config import DEBUG

# Configure standard format
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# Base logger for duplex operations
logger = logging.getLogger("ZYTRIX.Duplex")

def log_event(component: str, message: str, level: int = logging.INFO):
    """
    Standardized logging output for full duplex tracing.
    
    Args:
        component: The system sub-component triggering the event (e.g. 'STATE', 'VAD', 'INTERRUPT', 'TTS').
        message: The actual details of the event.
        level: Python standard logging level (DEBUG, INFO, WARNING, ERROR).
    
    Educational Note on Component Tracing:
        Standardizing component tags in brackets (e.g. `[STATE]`, `[VAD]`) makes it
        extremely easy to grep/filter terminal outputs when chasing down latency bugs.
    """
    formatted_msg = f"[{component.upper()}] {message}"
    
    if level == logging.DEBUG:
        logger.debug(formatted_msg)
    elif level == logging.INFO:
        logger.info(formatted_msg)
    elif level == logging.WARNING:
        logger.warning(formatted_msg)
    elif level == logging.ERROR:
        logger.error(formatted_msg)
    else:
        logger.info(formatted_msg)
