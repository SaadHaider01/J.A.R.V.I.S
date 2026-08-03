"""
backend/vision — Screen Understanding Subsystem
================================================
This package gives Zytrix visual awareness: the ability to look at what the
user is seeing and explain it in plain language.

Design principles
-----------------
  - Privacy-first: captures are ephemeral, secrets are redacted, cloud APIs
    are opt-in per PrivacyMode.
  - Separation of concerns: each module owns exactly one responsibility.
  - Fail-safe: every component degrades gracefully; nothing crashes Zytrix.
  - Future-proof: extension hooks are left for Phase 3 (continuous vision,
    object detection, GUI automation) without implementing them now.

Public surface
--------------
Import the types you need directly from this package:

    from backend.vision import VisionManager, CaptureScope, PrivacyMode, SessionStatus

Internal modules should be treated as implementation details and imported
via their full path (e.g. backend.vision.security.SecurityGuard).
"""

from enum import Enum


# ---------------------------------------------------------------------------
# Core enumerations — shared across ALL vision modules
# ---------------------------------------------------------------------------
class CaptureScope(Enum):
    """
    Defines the area of the screen to capture.

    Priority (narrowest → widest) follows the privacy principle:
    always capture the least amount of screen necessary.

        REGION        — A specific bounding box (most precise, most private).
        ACTIVE_WINDOW — The foreground application window.
        MONITOR       — A single physical display.
        FULL_SCREEN   — The combined virtual desktop (all monitors).

    WHY AN ENUM (not string constants)
    -----------------------------------
    Enums are type-safe: passing CaptureScope.REGION to a function that expects
    a CaptureScope is caught at type-check time, not at runtime.  String constants
    like "region" fail silently on typos.
    """
    REGION        = 1
    ACTIVE_WINDOW = 2
    MONITOR       = 3
    FULL_SCREEN   = 4


class SessionStatus(Enum):
    """
    State machine for a CaptureSession.

    Each status maps to a concrete pipeline stage, enabling precise:
      - Debug messages ("session stuck in VALIDATING for 3s")
      - Metrics ("average time spent in RUNNING_OCR")
      - Cancellation ("cancel any session in ANALYZING or earlier")
      - Retry logic ("retry if status is FAILED, not if COMPLETED")

    Valid transitions
    -----------------
    CREATED → CAPTURING → VALIDATING → RUNNING_OCR → ANALYZING → BUILDING_CONTEXT → COMPLETED
                                                                                   ↓
                                                        FAILED / CANCELLED / TIMED_OUT (from any stage)
    """
    CREATED          = "CREATED"
    CAPTURING        = "CAPTURING"
    VALIDATING       = "VALIDATING"
    RUNNING_OCR      = "RUNNING_OCR"
    ANALYZING        = "ANALYZING"
    BUILDING_CONTEXT = "BUILDING_CONTEXT"
    COMPLETED        = "COMPLETED"
    FAILED           = "FAILED"
    CANCELLED        = "CANCELLED"
    TIMED_OUT        = "TIMED_OUT"

    @property
    def is_terminal(self) -> bool:
        """Return True if this status represents a final (non-retryable) state."""
        return self in (
            SessionStatus.COMPLETED,
            SessionStatus.FAILED,
            SessionStatus.CANCELLED,
            SessionStatus.TIMED_OUT,
        )


class PrivacyMode(Enum):
    """
    Controls the privacy policy applied to the entire vision pipeline.

    STRICT
        All processing is local. No cloud APIs. No disk writes. Logs are
        sanitised. Intended for users working with sensitive data (finance,
        healthcare, legal).

    BALANCED (default)
        Local models are preferred. Cloud APIs are allowed after an explicit
        user confirmation and only when the context does not appear sensitive.
        Logs are sanitised.

    DEVELOPER
        All features enabled. Debug screenshots may be written to disk.
        Verbose logging. Cloud APIs allowed without confirmation.
        Intended for Zytrix development and debugging only.
    """
    STRICT    = "STRICT"
    BALANCED  = "BALANCED"
    DEVELOPER = "DEVELOPER"
