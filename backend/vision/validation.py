"""
validation.py — Image Quality & Resource Gate
==============================================
WHY THIS MODULE EXISTS
-----------------------
Before we spend CPU time on OCR or GPU time on a vision model, we need to know
whether the captured image is actually worth processing. A black screen, a tiny
thumbnail, or an image that would exceed our memory budget should be rejected
immediately and cheaply — before any expensive downstream work begins.

WHY A ValidationResult OBJECT (not just raise/no-raise)
---------------------------------------------------------
Raising an exception is binary: pass or crash. A ValidationResult is richer:
- It carries structured metadata (reason code, contrast score, resolution)
  that the logger, metrics, and future analytics layers can consume.
- It lets the caller decide whether to raise, retry, warn, or fall back
  without coupling validation logic to exception-handling policy.
- It enables a "soft reject" path: log it and degrade gracefully, rather than
  failing the whole session.

WHY NOT A SIMPLE BOOLEAN
--------------------------
`is_valid: bool` alone tells you nothing. A structured result tells you WHAT
failed, WHY it failed, and HOW severely — all essential for production systems.
"""

import io
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------
class ScreenAnalysisError(Exception):
    """
    Base exception for all Vision subsystem errors.
    Catching this one type is enough to prevent a Zytrix crash.
    """
    pass


class ValidationError(ScreenAnalysisError):
    """
    Raised by callers who want strict fail-fast behaviour.
    Carry a ValidationResult inside for full context.
    """
    def __init__(self, result: "ValidationResult"):
        self.result = result
        super().__init__(result.reason)


# ---------------------------------------------------------------------------
# Resource Limits
# ---------------------------------------------------------------------------
class VisionLimits:
    """
    Hard resource ceilings for every captured frame.

    WHY CENTRALISE LIMITS HERE
    ---------------------------
    Limits that live in multiple places drift apart over time. One constant
    here propagates to every check automatically when someone changes it.
    """

    # Highest resolution we will process — anything larger is downsampled or rejected.
    MAX_RESOLUTION: tuple[int, int] = (3840, 2160)   # 4K UHD

    # Rough in-memory footprint ceiling for a single image (RGB / RGBA bytes).
    MAX_MEMORY_MB: int = 512

    # Wall-clock time budget for the full vision pipeline.
    MAX_PROCESSING_TIME: float = 5.0  # seconds

    # Vision model context limit — approximate token ceiling.
    MAX_TOKENS: int = 4096

    # Disk / payload ceiling for cloud APIs or disk-backed OCR.
    MAX_FILE_SIZE_BYTES: int = 5 * 1024 * 1024  # 5 MB

    # Thumbnail size used for fast heuristic checks (black screen, contrast).
    # Small enough to be cheap; large enough to be representative.
    CHECK_THUMB_SIZE: tuple[int, int] = (64, 64)

    # Pixel intensity ceiling below which a channel is considered "black".
    BLACK_SCREEN_MAX_INTENSITY: int = 5

    # Minimum acceptable pixel dimensions (width or height).
    MIN_DIMENSION_PX: int = 10


# ---------------------------------------------------------------------------
# ValidationResult — the structured gate decision
# ---------------------------------------------------------------------------
@dataclass
class ValidationResult:
    """
    Returned by Validator.validate_image() for every image it inspects.

    Fields
    ------
    is_valid:   True if the image is safe to process downstream.
    reason:     A short, machine-readable reason code (e.g. "BLACK_SCREEN").
    message:    A human-readable explanation for logs and UI feedback.
    details:    Arbitrary key/value diagnostic data (contrast, resolution, etc.).
                Callers should treat this as informational; never act on specific
                keys without first checking `is_valid`.

    WHY DATACLASS
    -------------
    @dataclass gives us __repr__, __eq__, and type-checked fields for free,
    which helps both automated tests and debug logging.
    """

    is_valid: bool
    reason: str                         # e.g. "OK", "BLACK_SCREEN", "TOO_LARGE"
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    # Convenience constant reason codes — avoids magic strings throughout the codebase.
    REASON_OK                = "OK"
    REASON_NULL_IMAGE        = "NULL_IMAGE"
    REASON_TOO_SMALL         = "TOO_SMALL"
    REASON_RESOLUTION_TOO_LARGE = "RESOLUTION_TOO_LARGE"
    REASON_MEMORY_EXCEEDED   = "MEMORY_EXCEEDED"
    REASON_BLACK_SCREEN      = "BLACK_SCREEN"
    REASON_UNSUPPORTED_MODE  = "UNSUPPORTED_MODE"

    def raise_if_invalid(self) -> None:
        """
        Convenience: convert a failed result into a ValidationError.
        Callers that want strict fail-fast behaviour use this; callers that
        want graceful degradation inspect `is_valid` directly.
        """
        if not self.is_valid:
            raise ValidationError(self)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------
class Validator:
    """
    Stateless image quality gate.

    Design choices
    --------------
    - Every check is O(tiny thumbnail), never O(full image pixels), so this
      runs in microseconds and cannot become a performance bottleneck.
    - Checks are ordered cheapest → most expensive so we short-circuit early.
    - The method returns a ValidationResult rather than raising, giving callers
      full control over error-handling policy.
    """

    @staticmethod
    def validate_image(image: Optional[Image.Image]) -> ValidationResult:
        """
        Run all quality checks on the provided PIL Image.

        Returns a ValidationResult. If is_valid is False, the reason and
        details fields describe exactly what failed.

        Parameters
        ----------
        image : PIL.Image or None
            The captured frame to validate. None is treated as an empty capture.
        """

        # --- Check 1: Not None -------------------------------------------
        if image is None:
            return ValidationResult(
                is_valid=False,
                reason=ValidationResult.REASON_NULL_IMAGE,
                message="Capture returned no image (None). The screen may be unavailable.",
                details={}
            )

        width, height = image.size
        resolution_str = f"{width}x{height}"

        # --- Check 2: Minimum dimensions ---------------------------------
        if width < VisionLimits.MIN_DIMENSION_PX or height < VisionLimits.MIN_DIMENSION_PX:
            return ValidationResult(
                is_valid=False,
                reason=ValidationResult.REASON_TOO_SMALL,
                message=f"Image is too small to contain useful content ({resolution_str}).",
                details={"resolution": resolution_str, "min_required": VisionLimits.MIN_DIMENSION_PX}
            )

        # --- Check 3: Resolution ceiling ---------------------------------
        max_w, max_h = VisionLimits.MAX_RESOLUTION
        if width > max_w or height > max_h:
            return ValidationResult(
                is_valid=False,
                reason=ValidationResult.REASON_RESOLUTION_TOO_LARGE,
                message=f"Resolution {resolution_str} exceeds the maximum allowed {max_w}x{max_h}.",
                details={"resolution": resolution_str, "max_resolution": f"{max_w}x{max_h}"}
            )

        # --- Check 4: Memory ceiling -------------------------------------
        # Approximate raw in-memory size: width × height × bytes-per-pixel.
        mode_bpp = 4 if image.mode == "RGBA" else 3
        memory_mb = (width * height * mode_bpp) / (1024 * 1024)
        if memory_mb > VisionLimits.MAX_MEMORY_MB:
            return ValidationResult(
                is_valid=False,
                reason=ValidationResult.REASON_MEMORY_EXCEEDED,
                message=(
                    f"Image would occupy ~{memory_mb:.1f} MB in memory, "
                    f"exceeding the {VisionLimits.MAX_MEMORY_MB} MB limit."
                ),
                details={"estimated_memory_mb": round(memory_mb, 2), "limit_mb": VisionLimits.MAX_MEMORY_MB}
            )

        # --- Check 5: Supported pixel mode -------------------------------
        if image.mode not in ("RGB", "RGBA", "L"):
            return ValidationResult(
                is_valid=False,
                reason=ValidationResult.REASON_UNSUPPORTED_MODE,
                message=f"Pixel mode '{image.mode}' is not supported. Expected RGB, RGBA, or L.",
                details={"mode": image.mode}
            )

        # --- Check 6: Black screen (thumbnail heuristic) -----------------
        # We copy and shrink the image to a tiny thumbnail so this check is
        # always O(thumb pixels), not O(original pixels).  We never modify
        # the caller's image object.
        thumb = image.copy()
        thumb.thumbnail(VisionLimits.CHECK_THUMB_SIZE)
        extrema = thumb.getextrema()

        # getextrema() returns (min, max) per channel, or a flat (min, max) for L mode.
        black_threshold = VisionLimits.BLACK_SCREEN_MAX_INTENSITY
        if isinstance(extrema[0], tuple):
            # Multi-channel (RGB / RGBA): check the first three channels (R, G, B).
            channel_maxes = [ch_max for (_, ch_max) in extrema[:3]]
        else:
            # Single-channel (L / grayscale).
            channel_maxes = [extrema[1]]

        # Compute a simple contrast score: average of per-channel max intensities,
        # normalised to [0.0, 1.0].  A perfectly black screen scores 0.0.
        contrast_score = round(sum(channel_maxes) / (len(channel_maxes) * 255), 4)

        if all(mx <= black_threshold for mx in channel_maxes):
            return ValidationResult(
                is_valid=False,
                reason=ValidationResult.REASON_BLACK_SCREEN,
                message="Screen appears completely black. The display may be off or the window minimised.",
                details={"contrast_score": contrast_score, "resolution": resolution_str}
            )

        # --- All checks passed -------------------------------------------
        return ValidationResult(
            is_valid=True,
            reason=ValidationResult.REASON_OK,
            message="Image passed all quality checks.",
            details={
                "resolution": resolution_str,
                "mode": image.mode,
                "estimated_memory_mb": round(memory_mb, 2),
                "contrast_score": contrast_score,
            }
        )
