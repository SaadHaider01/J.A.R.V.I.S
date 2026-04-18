import argparse
import sys
import logging
from config import DEBUG

# Configure minimal logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("JARVIS")

def main():
    parser = argparse.ArgumentParser(description="JARVIS Core System")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode manually")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    logger.info("Initializing JARVIS system...")
    
    # TODO: Initialize Wake Word Detection, STT, and TTS engines
    # TODO: Load Memory and Settings
    # TODO: Start UI and Background Service Hooks
    
    logger.info("JARVIS is standing by. Press Ctrl+C to exit.")
    
    try:
        # Placeholder for main event loop
        while True:
            pass
    except KeyboardInterrupt:
        logger.info("Shutting down JARVIS...")
        sys.exit(0)

if __name__ == "__main__":
    main()
