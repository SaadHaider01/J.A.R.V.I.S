# ==============================================================================
# J.A.R.V.I.S — MAIN BOOTSTRAP & SIGNAL ENTRY POINT
# ==============================================================================
# WHAT THIS MODULE DOES:
# Initializes the central logging levels, handles OS signal interruptions,
# boots up the Duplex Manager conversational core, and ensures resources are
# safely cleaned up on shutdown.
#
# WHY IT EXISTS:
# Acts as the system entry point. In a multithreaded application, if the main
# thread terminates, background daemon threads might keep running (zombie threads)
# or audio hardware streams might remain locked. This module guarantees a clean
# shutdown and exports performance diagnostics when exiting.
#
# WHAT ADVANCED CONCEPTS ARE HERE:
#   - Signal Handling: Capturing Ctrl+C (KeyboardInterrupt) or termination requests
#     and converting them into a thread-safe shutdown event signal.
#   - Resource Allocation Is Isolation (RAII): Binding the lifecycle of hardware
#     streams and files to a structured startup/shutdown block (`try/finally`).
# ==============================================================================

import argparse
import sys
import logging
import time
from config import DEBUG
from backend.duplex.duplex_manager import DuplexManager
from backend.duplex.metrics import metrics_tracker
from backend.duplex.logger import log_event

# Configure root system logger
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ZYTRIX.Main")

def main():
    log_event("MAIN", "Initializing ZYTRIX system cores...")
    
    # Initialize the master duplex coordinator
    try:
        zytrix = DuplexManager()
    except Exception as e:
        logger.critical(f"Failed to boot ZYTRIX models: {e}")
        sys.exit(1)
        
    log_event("MAIN", "All systems nominal. Booting threads...")
    
    # Start the continuous stream duplex workers
    try:
        zytrix.start()
        
        # Keep the main thread alive while workers process audio in background
        while not zytrix.shutdown_event.is_set():
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        log_event("MAIN", "KeyboardInterrupt detected. Shutting down system...")
    except Exception as e:
        logger.error(f"Critical Runtime Exception: {e}")
    finally:
        # Guarantee safe cleanup of mic streams, threads, and files
        zytrix.stop()
        
        # Print performance diagnostics metrics before exiting
        metrics_tracker.print_diagnostics_report()
        log_event("MAIN", "ZYTRIX core is offline. Goodbye.")
        sys.exit(0)

if __name__ == "__main__":
    main()
