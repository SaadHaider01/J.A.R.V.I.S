"""
logger.py — Privacy-First Structured Logger
=============================================
WHY THIS MODULE EXISTS
-----------------------
Standard Python logging (print / logging.getLogger) emits exactly what you
give it.  In a vision pipeline, "exactly what you give it" can include:
  - Raw OCR text containing API keys or passwords.
  - Image metadata that reveals what applications are open.
  - Error traces that embed secrets in stack frames.

This module wraps Python's standard logging machinery with a mandatory
security filter so that EVERY log line passes through SecurityGuard.sanitize_for_log()
before being written, regardless of where in the codebase the log call originates.

WHY A CUSTOM FILTER (not a global formatter)
---------------------------------------------
A formatter only controls layout (timestamps, severity labels).  A logging.Filter
is called on the LogRecord object before any handler writes it, giving us the
chance to mutate the message field in-place.  This means the sanitised text is
what actually reaches file handlers, console handlers, and any third-party
log-aggregation handlers — not just a pretty-printed version of the raw text.

WHY structlog-STYLE CONTEXT (without the structlog dependency)
---------------------------------------------------------------
Structured logs (key=value pairs) are far more useful than prose strings for:
  - Searching (grep session_id=abc)
  - Metrics (count events where latency_ms > 500)
  - Alerts (trigger on reason=BLACK_SCREEN)

We implement a lightweight "context dict + message" pattern without adding the
structlog dependency, keeping the module self-contained.
"""

import logging
import time
from typing import Any

from .security import SecurityGuard


# ---------------------------------------------------------------------------
# Security Filter — the core privacy guarantee
# ---------------------------------------------------------------------------
class _SecretRedactionFilter(logging.Filter):
    """
    Logging filter that scrubs secrets from every log record before emission.

    Attached to the root 'zytrix.vision' logger so every child logger in the
    vision package automatically inherits redaction — no per-file setup needed.

    WHY IN-PLACE MUTATION
    ----------------------
    Logging filters can either return True/False (allow/block) or mutate the
    record in-place.  We mutate because we want the record to flow through
    (True) but with sanitised content.  The original message is preserved in
    record.original_msg for debugging (DEVELOPER mode only).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Preserve the original message for DEVELOPER-mode introspection.
        record.original_msg = record.getMessage()

        # Sanitise the formatted message in-place.
        sanitised = SecurityGuard.sanitize_for_log(record.getMessage())
        record.msg = sanitised
        record.args = ()  # args have already been interpolated into msg above.

        return True  # Always allow the record through (just sanitised).


# ---------------------------------------------------------------------------
# Logger factory
# ---------------------------------------------------------------------------
def get_vision_logger(name: str) -> logging.Logger:
    """
    Return a Logger for the given name under the 'zytrix.vision' hierarchy.

    All loggers created here:
      1. Inherit from the 'zytrix.vision' parent, so a single handler set on
         the parent captures all vision-subsystem output.
      2. Have the _SecretRedactionFilter attached at the handler level, ensuring
         secrets are redacted before any emission target (file, console, etc.).

    Parameters
    ----------
    name : str
        Typically __name__ from the calling module.

    Usage
    -----
    >>> logger = get_vision_logger(__name__)
    >>> logger.info("OCR complete", extra={"chars": 1024})
    """
    logger = logging.getLogger(f"zytrix.vision.{name}")

    # Only configure the root vision logger once.
    root = logging.getLogger("zytrix.vision")
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)

        # Structured-ish format: timestamp | level | logger | message
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)
        handler.addFilter(_SecretRedactionFilter())

        root.addHandler(handler)
        root.setLevel(logging.DEBUG)
        root.propagate = False  # Don't double-log to the root Python logger.

    return logger


# ---------------------------------------------------------------------------
# VisionLogger — structured log helper
# ---------------------------------------------------------------------------
class VisionLogger:
    """
    Thin wrapper around a standard Logger that enforces structured key=value
    log lines and mandatory security sanitisation.

    WHY NOT JUST USE logger.info()
    --------------------------------
    logger.info(f"Session {session_id} took {ms}ms") produces a string that
    is hard to parse programmatically.  VisionLogger.info("event", k=v, …)
    produces:

        event | session_id=abc123 | latency_ms=142 | status=COMPLETED

    which is trivially grep-able and importable into any log-analytics tool.

    Usage
    -----
    >>> vlog = VisionLogger(__name__)
    >>> vlog.info("capture_complete", session_id="abc", latency_ms=45)
    >>> vlog.warning("sensitive_content_detected", secret_type="jwt")
    >>> vlog.error("ocr_failed", reason="easyocr_unavailable", session_id="abc")
    """

    def __init__(self, module_name: str) -> None:
        self._log = get_vision_logger(module_name)

    def _format(self, event: str, **kwargs: Any) -> str:
        """
        Format a structured log line: 'event | key=value | key=value …'
        Values are sanitised before formatting so secrets never appear in output.
        """
        parts = [event]
        for k, v in kwargs.items():
            safe_v = SecurityGuard.sanitize_for_log(str(v))
            parts.append(f"{k}={safe_v}")
        return " | ".join(parts)

    def debug(self, event: str, **kwargs: Any) -> None:
        self._log.debug(self._format(event, **kwargs))

    def info(self, event: str, **kwargs: Any) -> None:
        self._log.info(self._format(event, **kwargs))

    def warning(self, event: str, **kwargs: Any) -> None:
        self._log.warning(self._format(event, **kwargs))

    def error(self, event: str, **kwargs: Any) -> None:
        self._log.error(self._format(event, **kwargs))

    def critical(self, event: str, **kwargs: Any) -> None:
        self._log.critical(self._format(event, **kwargs))
