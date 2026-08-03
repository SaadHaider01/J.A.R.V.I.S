"""
ocr.py — Optical Character Recognition
========================================

WHY THIS MODULE EXISTS
-----------------------
Vision models understand images holistically, but they are expensive, slow, and
require a GPU or a cloud API.  For many queries — "what error am I seeing?",
"read the text in this terminal", "what does this log say?" — the answer is
already sitting plainly in the image as machine-readable text.  Extracting that
text with OCR is:

  - ~10× faster than a vision model call (milliseconds vs. seconds)
  - Free of API cost
  - Local and private
  - Deterministic (same image → same output every time)

OCR is therefore the first intelligence layer in the pipeline.  Vision models
only receive the result as structured context — they do not re-read raw pixels
unless OCR fails or is insufficient.

WHY THIS MODULE MUST NEVER REASON
-----------------------------------
Separation of concerns.  The pipeline contract is:

    OCR      → extracts pixels → text  (this module)
    Context  → structures text         (screen_context.py)
    Vision   → reasons about text      (vision_models.py)

If OCR starts reasoning ("this looks like a Python traceback"), it:
  - Couples extraction to interpretation (hard to test, hard to change).
  - May hallucinate structure that doesn't exist.
  - Prevents screen_context.py from applying its own, better heuristics.

This module returns raw text.  Nothing more.

WHY EasyOCR AS PRIMARY
------------------------
EasyOCR (https://github.com/JaidedAI/EasyOCR):
  - Handles dark themes, monospace fonts, and low-contrast terminal output
    significantly better than Tesseract's classic LSTM model.
  - Runs on CPU without CUDA (slower, but functional everywhere).
  - Returns per-word bounding boxes and confidence scores natively — no
    post-processing needed.
  - Supports 80+ languages out of the box.
  - No system-level binary installation required (pure Python + PyTorch).

WHY pytesseract AS FALLBACK
-----------------------------
Tesseract is the industry-standard OCR engine, battle-tested over 30 years.
It is the most widely deployed OCR tool and the expected fallback when EasyOCR
is unavailable (e.g. on a minimal server without PyTorch).  Its main weakness
is poor performance on dark-theme text, which is why it is the fallback, not
the default.

WHY THREE OCRMode VALUES
--------------------------
A single quality level is always a compromise:
  - Running ACCURATE mode on a 60-FPS terminal is wasteful.
  - Running FAST mode on a scanned legal document is inadequate.

Three modes let callers select the right trade-off:

    FAST      — Low preprocessing, single-pass, best for code/terminal/logs.
    BALANCED  — Light preprocessing (contrast enhancement), best for browser/PDF.
    ACCURATE  — Full preprocessing stack, best for low-quality scans or images.

WHY BOUNDING BOXES FROM THE START
------------------------------------
TextBlock carries (x, y, width, height) even though Phase 2 does not use them.
Future features that WILL use them:
  - "Click on the error" — needs the pixel location.
  - "Highlight this word" — needs the bounding box.
  - GUI understanding — needs spatial layout.
  - Document understanding — needs reading order.

Adding bounding boxes later requires a breaking API change.
Adding them now costs nothing.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from PIL import Image, ImageEnhance, ImageFilter

from .logger import VisionLogger
from .metrics import MetricsCollector

log = VisionLogger(__name__)


# ---------------------------------------------------------------------------
# Optional engine imports — degrade gracefully
# ---------------------------------------------------------------------------
try:
    import easyocr
    _easyocr_reader: Optional["easyocr.Reader"] = None  # Lazy-loaded singleton
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False


# ---------------------------------------------------------------------------
# OCRMode — quality vs. speed trade-off
# ---------------------------------------------------------------------------
class OCRMode(Enum):
    """
    Controls the preprocessing pipeline and OCR parameters applied to the image.

    FAST
        Minimal preprocessing.  No image enhancement.  Best for:
          - Terminals and log viewers (high contrast, monospace)
          - IDE code editors (syntax-highlighted, clear text)
          - Error dialogs (plain text on solid background)
        Target latency: < 200 ms

    BALANCED
        Light preprocessing: contrast boost, mild sharpening.  Best for:
          - Browser pages (mixed fonts, colours, backgrounds)
          - PDF viewers (rendered at screen resolution)
          - Chat applications (variable font sizes)
        Target latency: < 500 ms

    ACCURATE
        Full preprocessing stack: greyscale conversion, contrast enhancement,
        sharpening, noise reduction.  Best for:
          - Scanned documents or photographs of text
          - Low-resolution screenshots
          - Images with complex backgrounds
          - Non-Latin scripts
        Target latency: < 1500 ms (acceptable for document tasks)

    AUTO
        [RESERVED — DO NOT IMPLEMENT YET]

        Future Phase 3 extension.  A lightweight image classifier will inspect
        the screenshot and automatically select FAST, BALANCED, or ACCURATE
        based on detected content type (terminal vs. browser vs. scanned doc).

        Reserving this enum value now means no breaking API change is needed
        when AUTO is implemented.  Callers that pass AUTO today receive
        BALANCED as the safe default until the classifier is built.
    """
    FAST     = "FAST"
    BALANCED = "BALANCED"
    ACCURATE = "ACCURATE"
    AUTO     = "AUTO"   # Reserved — falls back to BALANCED until classifier exists


# ---------------------------------------------------------------------------
# TextBlock — a single detected text region with spatial metadata

# ---------------------------------------------------------------------------
# OCREngineCapabilities — engine self-description
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OCREngineCapabilities:
    """
    Describes what a specific OCR engine version can and cannot do.

    WHY THIS EXISTS
    ---------------
    Without capability metadata, every caller that wants to know "can this
    engine handle Japanese text?" has to hard-code:

        if engine_name == "easyocr":
            ...

    That couples callers to engine identities and breaks every time a new
    engine is added.  OCREngineCapabilities turns engine knowledge into data,
    which callers can query without knowing which engine is active.

    Future use cases
    ----------------
    - OCREngine selection: VisionManager picks the engine with the best
      capabilities for the detected content type (e.g. multilingual support
      for a Japanese IDE).
    - Metrics: log which capability profiles are active in production.
    - Fallback logic: if the primary engine has no GPU, prefer the local
      engine with GPU support.

    Fields
    ------
    engine_name           : Versioned name, e.g. "easyocr-1.7", "tesseract-5".
    supports_gpu          : True if the engine can use CUDA/Metal acceleration.
    supports_multilingual : True if the engine handles non-Latin scripts.
    supports_handwriting  : True if the engine can read handwritten text.
    average_latency_ms    : Typical extraction time for a 1080p screenshot on CPU.
    min_confidence        : Lowest confidence this engine typically reports for
                            valid text (useful for calibrating threshold filters).
    notes                 : Human-readable caveats (e.g. "dark themes require ACCURATE mode").
    """
    engine_name:           str
    supports_gpu:          bool
    supports_multilingual: bool
    supports_handwriting:  bool
    average_latency_ms:    int
    min_confidence:        float = 0.0
    notes:                 str   = ""


# Known capability profiles — used by OCREngine.capabilities property.
_EASYOCR_CAPABILITIES = OCREngineCapabilities(
    engine_name           = "easyocr",
    supports_gpu          = True,   # Uses CUDA if available; gracefully falls back to CPU.
    supports_multilingual = True,   # 80+ languages supported out of the box.
    supports_handwriting  = False,  # EasyOCR's printed-text model; handwriting is a separate model.
    average_latency_ms    = 400,    # Rough CPU average for a 1080p screenshot.
    min_confidence        = 0.05,
    notes                 = "Best for dark-theme terminals and IDEs. Use ACCURATE mode for images.",
)

_TESSERACT_CAPABILITIES = OCREngineCapabilities(
    engine_name           = "tesseract",
    supports_gpu          = False,  # Tesseract is CPU-only.
    supports_multilingual = True,   # Requires language pack installation.
    supports_handwriting  = False,
    average_latency_ms    = 250,    # Faster than EasyOCR on CPU due to classical algorithms.
    min_confidence        = 0.0,
    notes                 = "Best for clean light-background text. Weaker on dark themes.",
)

_NO_ENGINE_CAPABILITIES = OCREngineCapabilities(
    engine_name           = "none",
    supports_gpu          = False,
    supports_multilingual = False,
    supports_handwriting  = False,
    average_latency_ms    = 0,
    notes                 = "No OCR engine is installed. Install easyocr or pytesseract.",
)


# ---------------------------------------------------------------------------
@dataclass
class TextBlock:
    """
    A contiguous block of text detected by the OCR engine, with its location.

    WHY STORE BOUNDING BOXES
    -------------------------
    Bounding boxes are the bridge between text understanding (Phase 2) and
    GUI interaction (Phase 3).  Without them, you know WHAT is on screen but
    not WHERE.  Including them now means Phase 3 can build on this contract
    without a breaking change.

    Coordinates use the top-left origin convention (screen coordinates).
    All values are in pixels, relative to the captured image, not the screen.

    Fields
    ------
    text          : The raw extracted text for this block.
    confidence    : Per-block OCR confidence in [0.0, 1.0].  Use this to
                    filter unreliable detections (e.g. discard < 0.4).
    x, y          : Top-left corner of the bounding box in image pixels.
    width, height : Dimensions of the bounding box in image pixels.
    """
    text:       str
    confidence: float
    x:          int
    y:          int
    width:      int
    height:     int


# ---------------------------------------------------------------------------
# OCRResult — the complete output of one OCR call
# ---------------------------------------------------------------------------
@dataclass
class OCRResult:
    """
    The structured result returned by OCREngine.extract().

    WHY A DATACLASS (not a plain dict)
    ------------------------------------
    Type-safe fields mean mistakes like result["confodence"] are caught by
    a type checker, not silently at runtime.  The dataclass also makes
    downstream code self-documenting: screen_context.py reads result.confidence,
    not result.get("conf", 0).

    Fields
    ------
    text              : The full extracted text (all blocks joined, ordered
                        top-to-bottom, left-to-right).
    confidence        : Average confidence across all detected blocks.
                        0.0 means no text found or all blocks were below threshold.
    blocks            : Individual TextBlock detections with bounding boxes.
    engine            : Which engine produced this result ("easyocr" or "tesseract").
    mode              : The OCRMode used for this extraction.
    processing_time_ms: Wall-clock extraction time in milliseconds.
    word_count        : Convenience count of whitespace-separated words in text.
    image_hash        : SHA-256 thumbnail fingerprint of the source image.
                        Populated from CaptureMetadata.image_hash, allowing the
                        ScreenCache, metrics layer, and audit log to correlate
                        an OCRResult back to its originating capture without
                        holding a reference to the (potentially large) PIL Image.
                        None if the hash was not available at extraction time.
    error             : If extraction failed partially, a human-readable message.
                        None means success.  The result may still contain partial
                        text even when error is set.
    """
    text:               str
    confidence:         float
    blocks:             list[TextBlock]
    engine:             str
    mode:               OCRMode
    processing_time_ms: int
    word_count:         int           = 0
    image_hash:         Optional[str] = None
    error:              Optional[str] = None

    def __post_init__(self):
        # Derive word_count from text so callers never need to compute it.
        self.word_count = len(self.text.split()) if self.text.strip() else 0

    @property
    def is_empty(self) -> bool:
        """True if no text was found (or all text was below the confidence threshold)."""
        return not self.text.strip()

    @property
    def high_confidence_text(self, threshold: float = 0.6) -> str:  # type: ignore[override]
        """
        Return only the text from blocks that exceed the given confidence threshold.

        Useful when the image quality is mixed (e.g. a dark terminal window
        with a bright modal dialog overlaid) and only the reliable parts matter.
        """
        return "\n".join(
            block.text for block in self.blocks
            if block.confidence >= threshold
        )


# ---------------------------------------------------------------------------
# Image preprocessing helpers
# ---------------------------------------------------------------------------
def _resolve_mode(mode: OCRMode) -> OCRMode:
    """
    Resolve AUTO to a concrete mode.

    Until the lightweight image classifier exists, AUTO safely falls back to
    BALANCED.  When the classifier is implemented, this is the single function
    to update — no callers need to change.
    """
    if mode == OCRMode.AUTO:
        return OCRMode.BALANCED  # TODO Phase 3: replace with classifier call
    return mode


def _preprocess(image: Image.Image, mode: OCRMode) -> Image.Image:
    """
    Apply a preprocessing pipeline to improve OCR accuracy.

    WHY PREPROCESSING
    ------------------
    Modern OCR models were trained on clean, high-contrast text.  Real screenshots
    often contain:
      - Dark IDE themes (white text on #1e1e1e background)
      - Anti-aliased fonts (grey fringe pixels around letterforms)
      - Compressed JPEG artefacts (block noise around text)
      - Small font sizes (sub-12px terminal text)

    Preprocessing normalises these inputs towards the training distribution,
    improving both accuracy and confidence scores.

    WHY NOT ALWAYS APPLY FULL PREPROCESSING
    -----------------------------------------
    Preprocessing has a cost:
      - ImageFilter.SHARPEN and ImageEnhance.Contrast run on every pixel.
      - For FAST mode (terminal/IDE text), the input is already clean and
        preprocessing adds latency for zero accuracy gain.

    We apply the minimum processing that serves each mode.

    IMPORTANT: We always work on a copy.  The caller's Image object is never
    mutated — it may still be needed by the security layer or vision model.
    """
    mode = _resolve_mode(mode)   # Collapse AUTO → concrete mode before branching
    img = image.copy()

    if mode == OCRMode.FAST:
        # No preprocessing — terminal and IDE text is already clean.
        return img

    if mode == OCRMode.BALANCED:
        # Light contrast boost: makes browser text pop against busy backgrounds.
        img = ImageEnhance.Contrast(img).enhance(1.4)
        img = img.filter(ImageFilter.SHARPEN)
        return img

    # ACCURATE: full preprocessing stack.
    # 1. Convert to greyscale — reduces noise from colour anti-aliasing.
    img = img.convert("L")
    # 2. Aggressive contrast enhancement — maximises text/background separation.
    img = ImageEnhance.Contrast(img).enhance(2.0)
    # 3. Sharpen — compensates for JPEG compression or low-DPI rendering.
    img = img.filter(ImageFilter.SHARPEN)
    # 4. Mild smoothing — reduces salt-and-pepper noise without blurring text edges.
    img = img.filter(ImageFilter.SMOOTH_MORE)
    return img


# ---------------------------------------------------------------------------
# EasyOCR engine
# ---------------------------------------------------------------------------
def _get_easyocr_reader() -> "easyocr.Reader":
    """
    Return a shared EasyOCR Reader instance, initialising it on first call.

    WHY A SINGLETON / LAZY INIT
    ----------------------------
    EasyOCR's Reader loads a neural network model (~100–400 MB) on creation.
    Doing this at import time would slow Zytrix startup by several seconds even
    when vision is never used.  Lazy initialisation means the model only loads
    the first time the user actually asks for screen analysis.

    WHY SHARED (not one Reader per call)
    --------------------------------------
    Creating a new Reader per OCR call would reload the model from disk every
    time — unacceptable for latency.  A shared instance keeps the model in RAM
    between calls (hot path is effectively free).
    """
    global _easyocr_reader
    if _easyocr_reader is None:
        log.info("easyocr_reader_init", status="loading_model")
        # gpu=False: safe default; EasyOCR auto-detects CUDA if available.
        _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        log.info("easyocr_reader_init", status="ready")
    return _easyocr_reader


def _run_easyocr(image: Image.Image, mode: OCRMode) -> OCRResult:
    """
    Extract text from an image using EasyOCR.

    EasyOCR's readtext() returns a list of:
        ( [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], "text", confidence )

    We normalise these into TextBlock objects using the axis-aligned bounding
    box derived from the four corner points (ignoring rotation for Phase 2).

    WHY IGNORE ROTATION
    --------------------
    Screen text is almost always axis-aligned (horizontal).  Computing the
    rotated bounding box correctly requires more geometry and adds complexity
    for a case that effectively never occurs in desktop screenshots.
    We store the axis-aligned bbox now; rotation support is an extension point.
    """
    t0 = time.monotonic()
    reader = _get_easyocr_reader()
    processed = _preprocess(image, mode)

    # detail=1 returns bounding boxes and confidence. paragraph=False gives
    # word-level results, which produces finer-grained TextBlocks.
    raw_results = reader.readtext(processed, detail=1, paragraph=False)

    ms = (time.monotonic() - t0) * 1000
    blocks: list[TextBlock] = []

    for (bbox_points, text, conf) in raw_results:
        # bbox_points: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] (clockwise from top-left)
        xs = [pt[0] for pt in bbox_points]
        ys = [pt[1] for pt in bbox_points]
        x, y = int(min(xs)), int(min(ys))
        w    = int(max(xs)) - x
        h    = int(max(ys)) - y

        blocks.append(TextBlock(
            text=text, confidence=round(conf, 4),
            x=x, y=y, width=w, height=h,
        ))

    # Join blocks top-to-bottom for the full text field.
    # Sort by y first (reading order) then x (left to right).
    blocks.sort(key=lambda b: (b.y, b.x))
    full_text = "\n".join(b.text for b in blocks)
    avg_conf  = round(sum(b.confidence for b in blocks) / len(blocks), 4) if blocks else 0.0

    return OCRResult(
        text=full_text,
        confidence=avg_conf,
        blocks=blocks,
        engine="easyocr",
        mode=mode,
        processing_time_ms=int(ms),
    )


# ---------------------------------------------------------------------------
# Tesseract fallback engine
# ---------------------------------------------------------------------------
def _run_tesseract(image: Image.Image, mode: OCRMode) -> OCRResult:
    """
    Extract text from an image using pytesseract (Tesseract wrapper).

    WHY pytesseract AS FALLBACK (not co-primary)
    ----------------------------------------------
    Tesseract's main weakness is dark-theme text (white-on-dark), which is
    extremely common in developer tools.  EasyOCR handles this significantly
    better.  Tesseract is kept as a fallback because it is widely available,
    has no PyTorch dependency, and is robust on clean, light-background text
    like PDFs and standard web pages.

    WHY image_to_data (not image_to_string)
    -----------------------------------------
    image_to_data returns per-word confidence and bounding box information.
    image_to_string returns only the full text string — we would lose spatial
    metadata.  We use image_to_data and reconstruct the full text ourselves.
    """
    import pytesseract

    t0 = time.monotonic()
    processed = _preprocess(image, mode)

    # PSM 6: Assume a uniform block of text.  Good default for screenshots.
    config = "--psm 6"
    data = pytesseract.image_to_data(
        processed,
        config=config,
        output_type=pytesseract.Output.DICT,
    )

    ms = (time.monotonic() - t0) * 1000
    blocks: list[TextBlock] = []

    n = len(data["text"])
    for i in range(n):
        word = data["text"][i].strip()
        conf_raw = int(data["conf"][i])

        # Tesseract returns -1 for non-text regions; skip those and empty words.
        if not word or conf_raw < 0:
            continue

        conf_normalised = round(conf_raw / 100.0, 4)  # Tesseract uses 0–100
        blocks.append(TextBlock(
            text=word,
            confidence=conf_normalised,
            x=int(data["left"][i]),
            y=int(data["top"][i]),
            width=int(data["width"][i]),
            height=int(data["height"][i]),
        ))

    blocks.sort(key=lambda b: (b.y, b.x))
    full_text = " ".join(b.text for b in blocks)
    avg_conf  = round(sum(b.confidence for b in blocks) / len(blocks), 4) if blocks else 0.0

    return OCRResult(
        text=full_text,
        confidence=avg_conf,
        blocks=blocks,
        engine="tesseract",
        mode=mode,
        processing_time_ms=int(ms),
    )


# ---------------------------------------------------------------------------
# OCREngine — the public API
# ---------------------------------------------------------------------------
class OCREngine:
    """
    The single entry-point for all text extraction in the Vision subsystem.

    Responsibilities
    ----------------
    1. Accept a PIL Image.
    2. Apply the selected OCRMode preprocessing.
    3. Run EasyOCR (or Tesseract if EasyOCR is unavailable).
    4. Return a structured OCRResult.

    Non-responsibilities
    --------------------
    - This class NEVER interprets the text it extracts.
    - It NEVER classifies errors, detects programming languages, or guesses intent.
    - It NEVER stores images or results to disk.
    - It NEVER communicates with a network.

    Reasoning belongs in vision_models.py and screen_context.py.

    Usage
    -----
    >>> engine = OCREngine(collector=metrics_collector)
    >>> result = engine.extract(image, mode=OCRMode.FAST)
    >>> print(result.text)
    >>> print(result.confidence)
    >>> for block in result.blocks:
    ...     print(block.text, block.x, block.y)
    """

    # Confidence below which we treat OCR as "unreliable" and log a warning.
    LOW_CONFIDENCE_THRESHOLD = 0.35

    def __init__(self, collector: Optional[MetricsCollector] = None) -> None:
        """
        Parameters
        ----------
        collector :
            Optional MetricsCollector for recording latency and confidence.
            If None, metrics are silently skipped (useful in tests).
        """
        self._collector = collector

        if not HAS_EASYOCR and not HAS_TESSERACT:
            log.warning(
                "no_ocr_engine_available",
                hint="Install easyocr (pip install easyocr) or pytesseract.",
            )

    @property
    def capabilities(self) -> OCREngineCapabilities:
        """
        Return the capability profile of the currently active OCR engine.

        Callers use this to make decisions without knowing which engine is
        installed.  For example:

            if engine.capabilities.supports_multilingual:
                # safe to process Japanese text

        The profile reflects the PRIMARY engine (EasyOCR if available,
        Tesseract otherwise, _NO_ENGINE_CAPABILITIES if neither is installed).
        """
        if HAS_EASYOCR:
            return _EASYOCR_CAPABILITIES
        if HAS_TESSERACT:
            return _TESSERACT_CAPABILITIES
        return _NO_ENGINE_CAPABILITIES

    def extract(
        self,
        image:      Image.Image,
        mode:       OCRMode = OCRMode.BALANCED,
        image_hash: Optional[str] = None,
    ) -> OCRResult:
        """
        Extract text from a PIL Image and return a structured OCRResult.

        The pipeline:
          1. Attempt EasyOCR (primary engine).
          2. If EasyOCR is unavailable or raises, attempt Tesseract (fallback).
          3. If both fail, return an empty OCRResult with error set.

        Parameters
        ----------
        image :
            A PIL Image produced by ScreenCapture.  Must not be None.
        mode :
            OCRMode controlling preprocessing intensity.
        image_hash :
            Optional SHA-256 thumbnail hash from CaptureMetadata.  Stored in
            the returned OCRResult so metrics and the ScreenCache can correlate
            this result back to its source capture without holding the image.

        Returns
        -------
        OCRResult — always.  Check result.error and result.is_empty for failures.
        """
        t_start = time.monotonic()
        result: Optional[OCRResult] = None

        # --- Primary: EasyOCR ---
        if HAS_EASYOCR:
            try:
                result = _run_easyocr(image, mode)
                log.info(
                    "ocr_complete",
                    engine="easyocr",
                    mode=mode.value,
                    confidence=result.confidence,
                    words=result.word_count,
                    ms=result.processing_time_ms,
                )
            except Exception as exc:
                log.warning(
                    "easyocr_failed",
                    reason=str(exc),
                    fallback="tesseract",
                )
                result = None

        # --- Fallback: Tesseract ---
        if result is None and HAS_TESSERACT:
            try:
                result = _run_tesseract(image, mode)
                log.info(
                    "ocr_complete",
                    engine="tesseract",
                    mode=mode.value,
                    confidence=result.confidence,
                    words=result.word_count,
                    ms=result.processing_time_ms,
                )
            except Exception as exc:
                log.error(
                    "tesseract_failed",
                    reason=str(exc),
                )
                result = None

        # --- Both failed ---
        if result is None:
            elapsed_ms = int((time.monotonic() - t_start) * 1000)
            error_msg  = "All OCR engines failed or are unavailable."
            log.error("ocr_all_engines_failed", hint="Install easyocr or pytesseract.")

            result = OCRResult(
                text="",
                confidence=0.0,
                blocks=[],
                engine="none",
                mode=mode,
                processing_time_ms=elapsed_ms,
                image_hash=image_hash,
                error=error_msg,
            )

        # --- Low-confidence warning ---
        if result.confidence < self.LOW_CONFIDENCE_THRESHOLD and not result.is_empty:
            log.warning(
                "ocr_low_confidence",
                confidence=result.confidence,
                engine=result.engine,
                hint="Consider using OCRMode.ACCURATE or improving image quality.",
            )

        # --- Record metrics ---
        if self._collector:
            self._collector.record_stage_latency("ocr", result.processing_time_ms)
            self._collector.record_ocr_confidence(result.confidence)

        return result
