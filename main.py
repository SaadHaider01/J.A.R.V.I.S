import argparse
import sys
import logging
from config import DEBUG
from backend.wake_word.listener import MicrophoneListener

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("JARVIS")

def main():
    logger.info("Initializing JARVIS system cores...")
    
    # Load the unified listener wrapper (which internally loads STT, TTS, and Agent Brain)
    try:
        jarvis_ears = MicrophoneListener()
    except Exception as e:
        logger.error(f"Failed to boot JARVIS models: {e}")
        sys.exit(1)
        
    logger.info("All systems nominal.")
    
    try:
        # Start the infinite Wake Word detection loop
        jarvis_ears.listen_for_wake_word()
    except KeyboardInterrupt:
        logger.info("Shutting down JARVIS explicitly...")
        jarvis_ears.stop()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Critical System Failure in runtime loop: {e}")

if __name__ == "__main__":
    main()
