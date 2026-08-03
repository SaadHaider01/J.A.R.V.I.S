"""
screen_capture.py — Screen, Window, and Region Capture
=======================================================
WHY THIS MODULE EXISTS
-----------------------
Every other vision component (OCR, vision models, validators) needs pixels.
This module is the single, authoritative source of those pixels. By isolating
capture here, we guarantee:
  - No other module ever touches display APIs directly.
  - Capture technology can be swapped (mss → D3D, Quartz, X11) without
    touching OCR, security, or any downstream layer.
  - Resource lifecycle (GPU buffers, MSS contexts) is managed in one place.

WHY mss
--------
mss (multi-screenshot) is:
  - Cross-platform (Windows, macOS, Linux).
  - Extremely fast: it reads pixels directly from the OS framebuffer with
    minimal copying (~2–8 ms for a 1080p frame on modern hardware).
  - Dependency-light: no X11/Quartz headers or heavy frameworks needed.

WHY pywin32 FOR WINDOW CAPTURE
--------------------------------
mss only knows about monitors; it has no concept of "windows". On Windows,
pywin32's GetForegroundWindow() + GetWindowRect() give us the exact bounding
box of the active application window, allowing us to grab only what the user
is looking at — preserving privacy, reducing payload size, and cutting API cost.

WHY CAPTURE-AND-DESTROY (no disk files)
-----------------------------------------
Screenshots are PII. Storing them to disk — even temporarily — creates:
  - Audit liability
  - Data breach surface
  - Unexpected persistence if the process crashes before cleanup

By keeping everything as in-memory PIL Image objects we guarantee images are
released the moment no reference holds them, with no OS-level file to orphan.

WHY SHA-256 OF THE THUMBNAIL (not the full image)
---------------------------------------------------
Hashing a 64×64 thumbnail is:
  - ~1000× faster than hashing a 4K frame
  - Good enough for duplicate detection (two visually identical screens
    produce the same thumbnail hash)
  - Privacy-safe: the thumbnail is too low-resolution to reveal sensitive text
"""

import hashlib
import io
import platform
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

from PIL import Image

from . import CaptureScope

# ---------------------------------------------------------------------------
# Optional dependencies — degrade gracefully if unavailable.
# ---------------------------------------------------------------------------
try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False

try:
    import win32gui
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False


# ---------------------------------------------------------------------------
# CaptureMetadata — lightweight fingerprint of a captured frame
# ---------------------------------------------------------------------------
@dataclass
class CaptureMetadata:
    """
    Produced alongside every captured image.

    WHY THIS EXISTS
    ---------------
    Metadata decouples identity from content. The ScreenCache, metrics layer,
    and audit log all need to identify a capture without holding the full
    pixel buffer. CaptureMetadata is cheap to store and share.

    Fields
    ------
    image_hash:   SHA-256 of a 64×64 thumbnail — used for deduplication and caching.
    width:        Full-resolution frame width in pixels.
    height:       Full-resolution frame height in pixels.
    scope:        The CaptureScope that produced this image.
    monitor_index: Index into the mss monitor list (1-based; 0 = all monitors).
    timestamp:    Unix timestamp (float seconds) of capture start.
    capture_ms:   Wall-clock time taken for the capture call, in milliseconds.
    """
    image_hash:    str
    width:         int
    height:        int
    scope:         CaptureScope
    monitor_index: int
    timestamp:     float
    capture_ms:    float


# ---------------------------------------------------------------------------
# CaptureResult — image + metadata bundled together
# ---------------------------------------------------------------------------
@dataclass
class CaptureResult:
    """
    The single object returned by ScreenCapture.capture().

    Bundling the image with its metadata means the caller always has both
    without needing to call a second method.  The image field should be
    treated as short-lived; discard it (let the reference drop) as soon as
    downstream processing finishes.
    """
    image:    Image.Image
    metadata: CaptureMetadata


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _thumbnail_hash(image: Image.Image, size: Tuple[int, int] = (64, 64)) -> str:
    """
    Compute a SHA-256 fingerprint of a tiny thumbnail of the image.

    We copy-and-shrink so the original image is never mutated, then encode
    the thumbnail as raw bytes for hashing.  The result is a 64-character
    hex digest that uniquely identifies visually distinct frames while being
    fast and storage-cheap.
    """
    thumb = image.copy().convert("RGB")  # Normalise mode so hash is consistent
    thumb.thumbnail(size)
    buf = io.BytesIO()
    thumb.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


def _to_rgb(sct_img) -> Image.Image:
    """
    Convert an mss ScreenShot object to a PIL RGB Image.

    mss returns pixels in BGRA order.  We use Pillow's raw decoder with
    the 'BGRX' stride to drop the alpha channel and reorder to RGB in one
    pass — no intermediate array allocation required.
    """
    return Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")


# ---------------------------------------------------------------------------
# ScreenCapture
# ---------------------------------------------------------------------------
class ScreenCapture:
    """
    The single entry-point for all pixel acquisition in the Vision subsystem.

    Usage
    -----
    >>> sc = ScreenCapture()
    >>> result = sc.capture(scope=CaptureScope.ACTIVE_WINDOW)
    >>> print(result.metadata.image_hash)

    Thread safety
    -------------
    mss contexts are NOT thread-safe.  Create one ScreenCapture per thread
    (or use an asyncio executor and ensure serial access).

    Resource cleanup
    ----------------
    Use as a context manager (with ScreenCapture() as sc:) to guarantee the
    mss context is released even if an exception occurs.  If used without a
    context manager, call sc.close() explicitly when done.
    """

    def __init__(self) -> None:
        if not HAS_MSS:
            raise RuntimeError(
                "The 'mss' library is required for screen capture. "
                "Install it with: pip install mss"
            )
        self._sct = mss.mss()

    # -----------------------------------------------------------------------
    # Context manager support — guarantees cleanup of the mss handle
    # -----------------------------------------------------------------------
    def __enter__(self) -> "ScreenCapture":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
        return False  # Do not suppress exceptions

    def close(self) -> None:
        """Release the underlying mss context and any associated OS handles."""
        try:
            self._sct.close()
        except Exception:
            pass  # Suppress errors during cleanup — we're already tearing down.

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def capture(
        self,
        scope:         CaptureScope = CaptureScope.ACTIVE_WINDOW,
        monitor_index: int = 1,
        bbox:          Optional[Tuple[int, int, int, int]] = None,
    ) -> CaptureResult:
        """
        Capture pixels according to the requested scope and return a CaptureResult.

        Scope priority (narrowest-first preserves privacy):
            REGION > ACTIVE_WINDOW > MONITOR > FULL_SCREEN

        Parameters
        ----------
        scope :
            Which portion of the display to capture.
        monitor_index :
            1-based monitor index used when scope is MONITOR or as the
            fallback when an active-window capture is unavailable.
        bbox :
            (left, top, right, bottom) in screen coordinates.
            Only used when scope == CaptureScope.REGION.

        Returns
        -------
        CaptureResult containing the PIL Image and CaptureMetadata.

        Raises
        ------
        RuntimeError if mss is not available (raised at __init__ time).
        """
        t_start = time.monotonic()
        image: Optional[Image.Image] = None

        try:
            if scope == CaptureScope.REGION and bbox is not None:
                image = self._capture_region(bbox)

            elif scope == CaptureScope.ACTIVE_WINDOW:
                image = self._capture_active_window(monitor_index)

            elif scope == CaptureScope.MONITOR:
                image = self._capture_monitor(monitor_index)

            else:
                # FULL_SCREEN or any future unknown scope — safe default.
                image = self._capture_full_screen()

            capture_ms = (time.monotonic() - t_start) * 1000
            width, height = image.size

            metadata = CaptureMetadata(
                image_hash    = _thumbnail_hash(image),
                width         = width,
                height        = height,
                scope         = scope,
                monitor_index = monitor_index,
                timestamp     = time.time(),
                capture_ms    = round(capture_ms, 2),
            )

            return CaptureResult(image=image, metadata=metadata)

        except Exception:
            # Ensure the image buffer is freed if we fail after allocation
            # but before returning.  Downstream cleanup guards (finally blocks)
            # in callers handle their own references.
            if image is not None:
                image.close()
            raise

    # -----------------------------------------------------------------------
    # Private capture methods — each handles one specific scope
    # -----------------------------------------------------------------------
    def _capture_region(self, bbox: Tuple[int, int, int, int]) -> Image.Image:
        """
        Capture a rectangular region in screen coordinates.

        bbox is (left, top, right, bottom) — the same format used by Windows
        GetWindowRect() and most cross-platform windowing APIs.
        """
        left, top, right, bottom = bbox
        monitor = {
            "left":   left,
            "top":    top,
            "width":  right - left,
            "height": bottom - top,
        }
        return _to_rgb(self._sct.grab(monitor))

    def _capture_monitor(self, monitor_index: int) -> Image.Image:
        """
        Capture a single physical monitor.

        mss.monitors[0] is the virtual bounding box of all monitors combined.
        mss.monitors[1] is the primary monitor; 2, 3, … are secondary monitors.
        We clamp the index to the valid range rather than crashing.
        """
        monitors = self._sct.monitors
        if monitor_index < 1 or monitor_index >= len(monitors):
            # Clamp to primary monitor instead of raising.
            monitor_index = 1
        return _to_rgb(self._sct.grab(monitors[monitor_index]))

    def _capture_full_screen(self) -> Image.Image:
        """
        Capture the combined virtual desktop (all monitors stitched together).

        mss.monitors[0] is the bounding rectangle that encloses every
        physical monitor.  This produces a wide image on multi-monitor setups.

        WHY NOT DEFAULT
        ---------------
        Sending the full virtual desktop to a vision model is expensive (more
        tokens, higher API cost, slower inference) and reveals more of the
        user's screen than necessary.  We prefer narrower scopes.
        """
        return _to_rgb(self._sct.grab(self._sct.monitors[0]))

    def _capture_active_window(self, fallback_monitor: int) -> Image.Image:
        """
        Capture only the foreground window (the app the user is interacting with).

        On Windows we use pywin32's GetForegroundWindow() + GetWindowRect() to
        obtain the exact bounding box of the active window, then delegate to
        _capture_region().

        On non-Windows platforms (or when pywin32 is unavailable) we fall back
        to the specified monitor — a reasonable proxy for "what the user sees".

        WHY ACTIVE WINDOW FIRST
        ------------------------
        When a user says "explain this error", they almost certainly mean
        "the window that has my attention right now", not their second monitor
        showing a browser.  Targeting the active window is both more accurate
        and more privacy-friendly.

        EDGE CASES HANDLED
        ------------------
        - Minimised windows: GetWindowRect() may return a rect at (-32000, -32000).
          We detect this and fall back to monitor capture.
        - hwnd == 0: No foreground window (e.g. desktop has focus).
          We fall back to monitor capture.
        - pywin32 unavailable: Fall back to monitor capture.
        - Any unexpected exception: Fall back silently (logged by caller).
        """
        if platform.system() == "Windows" and HAS_WIN32:
            hwnd = win32gui.GetForegroundWindow()
            if hwnd:
                try:
                    left, top, right, bottom = win32gui.GetWindowRect(hwnd)

                    # Windows places minimised windows at large negative coordinates.
                    MINIMISED_SENTINEL = -30000
                    if left < MINIMISED_SENTINEL or top < MINIMISED_SENTINEL:
                        # Window is minimised; fall back to monitor.
                        return self._capture_monitor(fallback_monitor)

                    # Sanity-check: the rect must have positive area.
                    if right <= left or bottom <= top:
                        return self._capture_monitor(fallback_monitor)

                    return self._capture_region((left, top, right, bottom))

                except Exception:
                    pass  # Fall through to monitor fallback.

        return self._capture_monitor(fallback_monitor)
