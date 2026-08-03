"""
vision_models.py — Vision Backend Plugin System
=================================================

WHY THIS MODULE EXISTS
-----------------------
At this point in the pipeline, we have:
  - A validated, in-memory PIL Image (from screen_capture.py)
  - Extracted text with bounding boxes (from ocr.py)

For queries that need deeper understanding — "explain this error", "what does
this code do?", "describe this chart" — OCR text alone is insufficient.  We
need a model that can reason about the image holistically.

This module is the bridge between raw pixels + text and machine reasoning.

WHY A PLUGIN ARCHITECTURE (not if/elif model strings)
-------------------------------------------------------
The naive approach:

    if model == "llava":
        call_llava(image, query)
    elif model == "gpt4v":
        call_openai(image, query)
    ...

This breaks immediately:
  - Adding a new model requires editing this file AND every caller.
  - Testing requires a real model (no mocking).
  - Health checks, cost tracking, and capability queries are scattered.
  - Cloud and local models have different call signatures, retry policies,
    and error types — all leaking into call sites.

A plugin architecture solves every one of these problems:

    class VisionBackend:          # Abstract interface — the contract
    class OllamaBackend:          # Local models via Ollama
    class GPT4oVisionBackend:     # OpenAI cloud
    class GeminiBackend:          # Google cloud
    class OCROnlyBackend:         # Graceful fallback — no model needed

VisionManager holds a VisionBackend reference.  It calls:
    backend.analyze(request)
and never knows which backend is active.

WHY VisionManager MUST NEVER KNOW WHICH BACKEND IS ACTIVE
-----------------------------------------------------------
This is a strict dependency inversion.  The manager depends on the abstraction
(VisionBackend), not the implementation (OllamaBackend).  This means:
  - Swapping backends requires zero changes to VisionManager.
  - Tests inject a MockBackend without touching VisionManager.
  - A/B testing two backends is a config change, not a code change.

WHY VERSIONED BACKEND NAMES (not generic "OllamaBackend")
----------------------------------------------------------
"OllamaBackend" in 2025 could mean Qwen-VL 7B or LLaVA 1.5 or any future
model.  Versioned names like "Qwen2VL7BBackend" and "Llava16Backend":
  - Make metrics meaningful: {"qwen2vl7b": 62, "llava16": 23} is actionable.
  - Make audit logs debuggable: you know exactly which model version produced
    a given answer, months later.
  - Prevent silent model upgrades from changing behaviour unexpectedly.

WHY BackendHealth AND BackendCapabilities
------------------------------------------
VisionManager needs to do two things automatically:
  1. Route requests to the best available backend (health-based routing).
  2. Warn the user if a backend can't handle their query (capability mismatch).

Without health checks, backend failures surface as timeouts or cryptic errors.
Without capabilities, the manager blindly sends a 4K image to a model with a
512-token context limit.

Health checks and capability metadata turn these runtime surprises into
predictable, handleable conditions.

BACKEND PRIORITY (for VisionManager auto-selection)
-----------------------------------------------------
1. Local GPU (fastest, cheapest, most private)
2. Local CPU (slower, but still private)
3. Cloud (fast, costly, requires consent under BALANCED/STRICT mode)
4. OCR-only (always available — the ultimate fallback)
"""

import asyncio
import base64
import io
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from PIL import Image

from .logger import VisionLogger
from .metrics import MetricsCollector
from . import PrivacyMode

log = VisionLogger(__name__)


# ---------------------------------------------------------------------------
# Optional cloud SDK imports — each is isolated to its own try/except block
# so the absence of one SDK never prevents another from loading.
# ---------------------------------------------------------------------------
try:
    import httpx          # Used by OllamaBackend for HTTP calls to Ollama's API
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False


# ---------------------------------------------------------------------------
# BackendCapabilities — what a backend can do
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BackendCapabilities:
    """
    Describes the strengths and constraints of a specific vision backend.

    WHY THIS EXISTS
    ---------------
    Different models have very different characteristics:
      - Context window size determines how much text + image data they accept.
      - Max image resolution determines whether we need to downsample.
      - is_local determines whether we need PrivacyMode consent before calling.
      - supports_ocr_bypass means the model can read text directly from the
        image without relying on our OCR layer (useful for handwriting or
        very small fonts that OCR misses).

    VisionManager uses these fields to:
      - Decide whether to downsample the image before sending.
      - Enforce PrivacyMode policies (STRICT → must be local).
      - Choose the right backend for the content type.

    Fields
    ------
    backend_id          : Versioned identifier, e.g. "qwen2vl7b", "gpt4o_vision".
    display_name        : Human-readable name for logs and UI.
    is_local            : True if this backend runs on the user's machine.
    supports_gpu        : True if the backend can use GPU acceleration.
    max_image_px        : Maximum resolution (width × height) this backend accepts.
                          Images larger than this should be downsampled before sending.
    max_context_tokens  : Approximate token budget for the combined image + text input.
    supports_ocr_bypass : True if the model can read text directly from image pixels
                          (bypassing our OCR layer for better accuracy on edge cases).
    cost_per_image_usd  : Estimated USD cost per image sent to this backend.
                          0.0 for local models.
    notes               : Human-readable caveats or known limitations.
    """
    backend_id:           str
    display_name:         str
    is_local:             bool
    supports_gpu:         bool
    max_image_px:         int         = 3840 * 2160   # Default: up to 4K
    max_context_tokens:   int         = 4096
    supports_ocr_bypass:  bool        = False
    cost_per_image_usd:   float       = 0.0
    notes:                str         = ""


# ---------------------------------------------------------------------------
# BackendHealth — real-time availability state
# ---------------------------------------------------------------------------
@dataclass
class BackendHealth:
    """
    Real-time health snapshot for a vision backend.

    WHY THIS EXISTS
    ---------------
    VisionManager needs to route requests to the best *available* backend.
    Without health checks, a backend that is offline or overloaded only
    reveals its state when the request times out — several seconds later.

    With health checks, VisionManager can skip unavailable backends instantly
    and route to the next best option.

    Fields
    ------
    online          : True if the backend responded successfully to a health probe.
    latency_ms      : Round-trip time for the health probe, in milliseconds.
                      Use this to prefer the faster of two healthy backends.
    gpu_available   : True if the backend detected a usable GPU.
                      Local backends can report this from torch.cuda.is_available().
    error           : If online is False, a human-readable error description.
    checked_at      : Unix timestamp of when this health check was performed.
                      Use this to decide whether a cached health result is stale.
    """
    online:        bool
    latency_ms:    float          = 0.0
    gpu_available: bool           = False
    error:         Optional[str]  = None
    checked_at:    float          = field(default_factory=time.time)

    @property
    def is_stale(self, max_age_seconds: float = 30.0) -> bool:
        """
        Return True if this health result is older than max_age_seconds.

        Stale health results should trigger a fresh health_check() call.
        30 seconds is a reasonable TTL — frequent enough to detect failures
        quickly, infrequent enough not to spam the backend with probes.
        """
        return (time.time() - self.checked_at) > max_age_seconds


# ---------------------------------------------------------------------------
# VisionRequest — the input to any backend
# ---------------------------------------------------------------------------
@dataclass
class VisionRequest:
    """
    Everything a vision backend needs to produce a response.

    WHY A REQUEST OBJECT (not individual parameters)
    --------------------------------------------------
    As the system grows, backends will need additional context: OCR text,
    session ID, privacy mode, language hints, bounding boxes.  A request
    object absorbs these additions without breaking existing backend
    implementations.  Compare:

    BAD (parameter creep):
        backend.analyze(image, query, ocr_text, session_id, privacy_mode, lang, ...)

    GOOD (stable interface):
        backend.analyze(VisionRequest(image=..., query=..., ...))

    Fields
    ------
    image       : The PIL Image to analyse.  May be downsampled by the backend
                  if it exceeds max_image_px.
    query       : The user's question or instruction in plain English.
    ocr_text    : Pre-extracted text from the OCR layer.  Backends should use
                  this to ground their response rather than re-reading pixels
                  for text that OCR already extracted reliably.
    session_id  : The CaptureSession ID, for logging and metrics correlation.
    privacy_mode: The active PrivacyMode.  Backends MUST check this before
                  making any cloud call.  The SecurityGuard in security.py
                  is the authoritative checker; this field is a hint.
    context_hint: Free-form string describing what is on screen (e.g.
                  "VSCode — main.py").  Used by the security layer for
                  sensitive-content detection.
    timeout_s   : Maximum seconds the backend should spend generating a response.
    """
    image:        Image.Image
    query:        str
    ocr_text:     str                   = ""
    session_id:   str                   = ""
    privacy_mode: PrivacyMode           = PrivacyMode.BALANCED
    context_hint: str                   = ""
    timeout_s:    float                 = 5.0


# ---------------------------------------------------------------------------
# VisionResponse — the output from any backend
# ---------------------------------------------------------------------------
@dataclass
class VisionResponse:
    """
    The structured response returned by any vision backend.

    WHY A RESPONSE OBJECT (not a plain string)
    -------------------------------------------
    Callers need more than just the answer text:
      - Metrics need tokens_used and latency_ms to track costs.
      - The ScreenCache needs image_hash to avoid duplicate calls.
      - screen_context.py needs model_id to know which backend produced this.
      - The UI needs error to show a meaningful message on failure.

    Fields
    ------
    answer          : The model's response text.  Empty string on failure.
    model_id        : Versioned backend identifier, e.g. "qwen2vl7b".
    latency_ms      : Wall-clock inference time in milliseconds.
    tokens_used     : Approximate tokens consumed (0 for local models that
                      don't expose token counts).
    estimated_cost  : Estimated USD cost.  0.0 for local models.
    is_cloud        : True if this response came from a cloud API.
    image_hash      : SHA-256 thumbnail hash of the source image, for cache
                      and audit correlation.
    error           : Human-readable error description if the call failed.
                      None means success.
    """
    answer:         str
    model_id:       str
    latency_ms:     float
    tokens_used:    int           = 0
    estimated_cost: float         = 0.0
    is_cloud:       bool          = False
    image_hash:     Optional[str] = None
    error:          Optional[str] = None

    @property
    def success(self) -> bool:
        """True if the backend produced a non-empty, error-free response."""
        return bool(self.answer.strip()) and self.error is None


# ---------------------------------------------------------------------------
# VisionBackend — the abstract interface (the contract every backend signs)
# ---------------------------------------------------------------------------
class VisionBackend(ABC):
    """
    Abstract base class for all vision model backends.

    DESIGN INTENT
    -------------
    Every backend — local or cloud, GPU or CPU — must implement exactly the
    same two async methods.  VisionManager calls these methods and never
    touches backend-specific internals.

    This is the interface contract that makes hot-swapping possible:

        manager.backend = Qwen2VL7BBackend(...)  # local
        # or
        manager.backend = GPT4oVisionBackend(...) # cloud
        # VisionManager code changes: zero

    WHY ASYNC
    ----------
    Local model inference (Ollama HTTP call) and cloud API calls both block
    for seconds.  async def means these calls don't freeze the Zytrix event
    loop.  VisionManager awaits them in a way that keeps the rest of the app
    responsive.

    WHY TWO METHODS (not one)
    --------------------------
    - analyze()      → the hot path (called on every user request)
    - health_check() → the diagnostic path (called periodically or on startup)

    Separating them means health checks can use a lightweight probe (e.g. a
    tiny 1×1 image) without affecting the analyze() contract.
    """

    @abstractmethod
    async def analyze(self, request: VisionRequest) -> VisionResponse:
        """
        Produce a natural-language response to the user's query about the image.

        Implementations MUST:
        - Return a VisionResponse always — never raise an exception to the caller.
        - Set error on VisionResponse if the call fails.
        - Respect request.timeout_s.
        - Check request.privacy_mode before making any cloud call.
        - Never mutate request.image.

        Parameters
        ----------
        request : VisionRequest
            Contains the image, query, OCR text, and policy settings.

        Returns
        -------
        VisionResponse — always.
        """
        ...

    @abstractmethod
    async def health_check(self) -> BackendHealth:
        """
        Probe the backend and return its current health state.

        Implementations MUST:
        - Return a BackendHealth always — never raise.
        - Complete in < 2 seconds (use a short timeout on the probe).
        - Use a lightweight probe (e.g. a 1×1 dummy image or a /api/version call).

        Returns
        -------
        BackendHealth — always.
        """
        ...

    @property
    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """
        Return the static capability profile for this backend.

        This property should be cheap (return a cached dataclass instance).
        It must not make a network call.
        """
        ...


# ---------------------------------------------------------------------------
# Helper: encode PIL Image to base64 JPEG for API payloads
# ---------------------------------------------------------------------------
def _image_to_base64(image: Image.Image, max_px: int = 1920 * 1080) -> str:
    """
    Encode a PIL Image as a base64 JPEG string suitable for API payloads.

    WHY JPEG (not PNG)
    -------------------
    PNG is lossless — ideal for preserving text quality in our internal
    pipeline.  But for cloud API payloads, PNG files are 3–5× larger than
    JPEG for the same screenshot, increasing transfer time and API cost.
    JPEG at quality=85 is indistinguishable to a vision model while being
    dramatically smaller.

    WHY DOWNSAMPLE
    ---------------
    Cloud vision APIs charge per image and have resolution limits.  A 4K
    screenshot costs ~4× more tokens than a 1080p screenshot.  We downsample
    before encoding to stay within cost and context budgets.

    WHY NOT DOWNSAMPLE IN SCREEN CAPTURE
    --------------------------------------
    Screen capture preserves full resolution so the validation layer can
    check actual pixel dimensions.  Downsampling is a backend concern (each
    backend has its own resolution budget) not a capture concern.
    """
    img = image.copy()

    # Downsample if the image exceeds the pixel budget.
    if img.width * img.height > max_px:
        # Compute the scale factor that brings the image within max_px.
        import math
        scale = math.sqrt(max_px / (img.width * img.height))
        new_w = max(1, int(img.width  * scale))
        new_h = max(1, int(img.height * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)

    # Ensure RGB before JPEG encoding (JPEG does not support RGBA or palette modes).
    if img.mode != "RGB":
        img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# OCROnlyBackend — the zero-dependency fallback
# ---------------------------------------------------------------------------
class OCROnlyBackend(VisionBackend):
    """
    A fallback backend that returns the OCR text as its "answer".

    WHY THIS EXISTS
    ---------------
    No backend should ever leave the user with no response.  When:
      - Ollama is not running
      - No internet connection is available
      - The user is in STRICT privacy mode with no local model

    ...the OCROnlyBackend ensures Zytrix always returns something useful:
    the raw extracted text.  For queries like "read the text on screen" or
    "what error am I seeing?", this is often sufficient without a model.

    It is always healthy (no dependencies) and always available.
    """

    _CAPABILITIES = BackendCapabilities(
        backend_id          = "ocr_only",
        display_name        = "OCR-Only (No Vision Model)",
        is_local            = True,
        supports_gpu        = False,
        max_context_tokens  = 0,     # No model context — just returning text.
        supports_ocr_bypass = False,
        cost_per_image_usd  = 0.0,
        notes               = (
            "Returns OCR-extracted text directly. No image reasoning. "
            "Used as the ultimate fallback when no vision model is available."
        ),
    )

    async def analyze(self, request: VisionRequest) -> VisionResponse:
        """Return the pre-extracted OCR text as the answer."""
        t0 = time.monotonic()
        answer = request.ocr_text.strip() if request.ocr_text else (
            "No text could be extracted from the screen. "
            "The screen may be blank, DRM-protected, or the content may be purely graphical."
        )
        return VisionResponse(
            answer     = answer,
            model_id   = "ocr_only",
            latency_ms = (time.monotonic() - t0) * 1000,
            is_cloud   = False,
        )

    async def health_check(self) -> BackendHealth:
        """OCROnlyBackend is always healthy — it has no external dependencies."""
        return BackendHealth(online=True, latency_ms=0.0, gpu_available=False)

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._CAPABILITIES


# ---------------------------------------------------------------------------
# OllamaBackend (base for Qwen2VL7BBackend and Llava16Backend)
# ---------------------------------------------------------------------------
class _OllamaBackendBase(VisionBackend):
    """
    Shared implementation for all Ollama-hosted vision models.

    WHY OLLAMA
    ----------
    Ollama (https://ollama.com) is the simplest way to run local LLMs and
    multimodal models on a developer's machine:
      - Single binary, no Python dependencies.
      - Supports Qwen-VL, LLaVA, and dozens of other models.
      - Exposes a simple HTTP API compatible with OpenAI's chat format.
      - Manages model download, GPU offloading, and VRAM budgeting automatically.

    WHY NOT LOAD MODELS DIRECTLY (transformers, llama.cpp)
    -------------------------------------------------------
    Loading models directly into the Zytrix process would:
      - Use hundreds of MB–GBs of RAM in the assistant process.
      - Compete with Zytrix's own memory budget.
      - Require per-platform CUDA/Metal setup that Ollama handles for free.
      - Prevent model sharing across applications.

    Ollama runs as a separate daemon, isolating model memory from Zytrix.

    WHY HTTPX (not requests)
    --------------------------
    httpx supports async/await natively.  Using the async httpx client means
    Ollama API calls don't block the Zytrix event loop — the assistant stays
    responsive while waiting for model inference.
    """

    _OLLAMA_BASE_URL = "http://localhost:11434"

    def __init__(self, model_name: str, collector: Optional[MetricsCollector] = None) -> None:
        self._model   = model_name
        self._collector = collector

    async def analyze(self, request: VisionRequest) -> VisionResponse:
        if not HAS_HTTPX:
            return VisionResponse(
                answer   = "",
                model_id = self._model,
                latency_ms = 0.0,
                error    = "httpx is not installed. Run: pip install httpx",
            )

        t0 = time.monotonic()

        # Build a prompt that combines the user query with OCR context.
        # Providing OCR text as additional context reduces hallucination because
        # the model doesn't need to re-read text — it focuses on reasoning.
        prompt_parts = [request.query]
        if request.ocr_text.strip():
            prompt_parts.append(
                f"\n\n[Screen text extracted by OCR — use this as reference]:\n{request.ocr_text}"
            )
        prompt = "\n".join(prompt_parts)

        # Encode the image as base64 JPEG for the Ollama multimodal API.
        try:
            img_b64 = _image_to_base64(
                request.image,
                max_px=self.capabilities.max_image_px,
            )
        except Exception as exc:
            return VisionResponse(
                answer   = "",
                model_id = self._model,
                latency_ms = (time.monotonic() - t0) * 1000,
                error    = f"Image encoding failed: {exc}",
            )

        payload = {
            "model": self._model,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "options": {"num_predict": 512},
        }

        try:
            async with httpx.AsyncClient(timeout=request.timeout_s) as client:
                resp = await client.post(
                    f"{self._OLLAMA_BASE_URL}/api/generate",
                    json=payload,
                )
                resp.raise_for_status()
                data   = resp.json()
                answer = data.get("response", "").strip()

        except asyncio.TimeoutError:
            return VisionResponse(
                answer    = "",
                model_id  = self._model,
                latency_ms = (time.monotonic() - t0) * 1000,
                error     = f"Ollama request timed out after {request.timeout_s}s.",
            )
        except Exception as exc:
            return VisionResponse(
                answer    = "",
                model_id  = self._model,
                latency_ms = (time.monotonic() - t0) * 1000,
                error     = f"Ollama API error: {exc}",
            )

        latency_ms = (time.monotonic() - t0) * 1000

        if self._collector:
            self._collector.record_api_call(
                backend_name = self._model,
                is_cloud     = False,
                cost_usd     = 0.0,
            )

        log.info(
            "ollama_analysis_complete",
            model=self._model,
            latency_ms=round(latency_ms, 1),
            session_id=request.session_id,
        )

        return VisionResponse(
            answer     = answer,
            model_id   = self._model,
            latency_ms = latency_ms,
            is_cloud   = False,
        )

    async def health_check(self) -> BackendHealth:
        """Probe the Ollama daemon's /api/version endpoint."""
        if not HAS_HTTPX:
            return BackendHealth(
                online=False, error="httpx not installed — cannot probe Ollama."
            )
        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{self._OLLAMA_BASE_URL}/api/version")
                resp.raise_for_status()
            return BackendHealth(
                online     = True,
                latency_ms = (time.monotonic() - t0) * 1000,
            )
        except Exception as exc:
            return BackendHealth(
                online     = False,
                latency_ms = (time.monotonic() - t0) * 1000,
                error      = f"Ollama unreachable: {exc}",
            )


# ---------------------------------------------------------------------------
# Versioned Ollama backends
# ---------------------------------------------------------------------------
class Qwen2VL7BBackend(_OllamaBackendBase):
    """
    Qwen2-VL 7B — primary local vision backend.

    WHY QWEN2-VL AS PRIMARY LOCAL MODEL
    ------------------------------------
    Qwen2-VL-7B (Alibaba, 2024) outperforms LLaVA-1.6 on:
      - Code understanding (critical for Zytrix's IDE use cases)
      - OCR-heavy tasks (dense text, tables, diagrams)
      - Small-text recognition (terminal font sizes)
      - Multi-language content

    At 7B parameters it runs comfortably on a modern laptop GPU (16GB VRAM)
    and acceptably on CPU (slower but functional).

    Run with: ollama pull qwen2.5-vl:7b
    """

    _CAPABILITIES = BackendCapabilities(
        backend_id          = "qwen2vl7b",
        display_name        = "Qwen2-VL 7B (Local, Ollama)",
        is_local            = True,
        supports_gpu        = True,
        max_image_px        = 1920 * 1080,
        max_context_tokens  = 7168,
        supports_ocr_bypass = True,   # Qwen2-VL can read text from images directly.
        cost_per_image_usd  = 0.0,
        notes               = "Best local model for code, terminals, and dense text.",
    )

    def __init__(self, collector: Optional[MetricsCollector] = None) -> None:
        super().__init__(model_name="qwen2.5-vl:7b", collector=collector)

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._CAPABILITIES


class Llava16Backend(_OllamaBackendBase):
    """
    LLaVA 1.6 (13B) — secondary local vision backend.

    WHY LLAVA AS SECONDARY
    -----------------------
    LLaVA-1.6 is the most battle-tested open-source multimodal model.  It is:
      - Widely available via Ollama.
      - Good general-purpose visual understanding.
      - Reliable on natural images, charts, and screenshots.

    It is the secondary option when Qwen2-VL is unavailable or too slow on
    the user's hardware.

    Run with: ollama pull llava:13b
    """

    _CAPABILITIES = BackendCapabilities(
        backend_id          = "llava16",
        display_name        = "LLaVA 1.6 13B (Local, Ollama)",
        is_local            = True,
        supports_gpu        = True,
        max_image_px        = 1024 * 1024,
        max_context_tokens  = 4096,
        supports_ocr_bypass = False,
        cost_per_image_usd  = 0.0,
        notes               = "Reliable fallback. Best for general visual understanding.",
    )

    def __init__(self, collector: Optional[MetricsCollector] = None) -> None:
        super().__init__(model_name="llava:13b", collector=collector)

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._CAPABILITIES


# ---------------------------------------------------------------------------
# GPT4oVisionBackend — OpenAI cloud backend
# ---------------------------------------------------------------------------
class GPT4oVisionBackend(VisionBackend):
    """
    GPT-4o with vision (OpenAI cloud API).

    WHY CLOUD (and when to use it)
    -------------------------------
    GPT-4o is significantly more capable than any 7B local model for:
      - Complex reasoning about multi-element UIs.
      - Understanding handwritten notes or unusual fonts.
      - Explaining highly technical error traces in prose.
      - Any content where accuracy matters more than cost.

    Cloud MUST only be used when:
      1. The user's PrivacyMode is BALANCED (with consent) or DEVELOPER.
      2. SecurityGuard.can_use_cloud() returns True.
      3. No sensitive content is detected on screen.

    Consent is enforced in VisionManager, not here.

    WHY NOT ENFORCE PRIVACY HERE
    -----------------------------
    This backend doesn't know the full privacy context (session state,
    prior user consent, sensitive-content detection results).  VisionManager
    does.  Single-responsibility: this backend handles API communication;
    VisionManager handles policy.  We pass privacy_mode through so this
    backend can perform a last-resort check before sending data.
    """

    _CAPABILITIES = BackendCapabilities(
        backend_id          = "gpt4o_vision",
        display_name        = "GPT-4o Vision (OpenAI Cloud)",
        is_local            = False,
        supports_gpu        = False,  # Cloud — GPU is OpenAI's problem.
        max_image_px        = 2048 * 2048,
        max_context_tokens  = 128_000,
        supports_ocr_bypass = True,
        cost_per_image_usd  = 0.0013,  # Approximate; varies by image size.
        notes               = "Highest accuracy. Cloud only. Requires BALANCED or DEVELOPER mode.",
    )

    def __init__(self, api_key: str, collector: Optional[MetricsCollector] = None) -> None:
        self._api_key   = api_key
        self._collector = collector

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._CAPABILITIES

    async def analyze(self, request: VisionRequest) -> VisionResponse:
        # Last-resort privacy check — VisionManager should have checked already.
        if request.privacy_mode == PrivacyMode.STRICT:
            return VisionResponse(
                answer   = "",
                model_id = "gpt4o_vision",
                latency_ms = 0.0,
                error    = "Cloud APIs are disabled in STRICT privacy mode.",
            )

        if not HAS_OPENAI:
            return VisionResponse(
                answer   = "",
                model_id = "gpt4o_vision",
                latency_ms = 0.0,
                error    = "openai package is not installed. Run: pip install openai",
            )

        t0 = time.monotonic()

        try:
            img_b64 = _image_to_base64(request.image, max_px=self.capabilities.max_image_px)
        except Exception as exc:
            return VisionResponse(
                answer   = "",
                model_id = "gpt4o_vision",
                latency_ms = (time.monotonic() - t0) * 1000,
                error    = f"Image encoding failed: {exc}",
            )

        # Include OCR text as grounding context to reduce hallucination.
        user_content: list[dict] = [
            {"type": "text", "text": request.query},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        ]
        if request.ocr_text.strip():
            user_content.insert(1, {
                "type": "text",
                "text": f"[Extracted screen text for reference]:\n{request.ocr_text}"
            })

        try:
            client = openai.AsyncOpenAI(api_key=self._api_key)
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model    = "gpt-4o",
                    messages = [{"role": "user", "content": user_content}],
                    max_tokens = 1024,
                ),
                timeout = request.timeout_s,
            )
            answer      = response.choices[0].message.content or ""
            tokens_used = response.usage.total_tokens if response.usage else 0

        except asyncio.TimeoutError:
            return VisionResponse(
                answer    = "",
                model_id  = "gpt4o_vision",
                latency_ms = (time.monotonic() - t0) * 1000,
                error     = f"GPT-4o request timed out after {request.timeout_s}s.",
            )
        except Exception as exc:
            return VisionResponse(
                answer    = "",
                model_id  = "gpt4o_vision",
                latency_ms = (time.monotonic() - t0) * 1000,
                error     = f"OpenAI API error: {exc}",
            )

        latency_ms   = (time.monotonic() - t0) * 1000
        cost         = self.capabilities.cost_per_image_usd

        if self._collector:
            self._collector.record_api_call(
                backend_name = "gpt4o_vision",
                tokens_used  = tokens_used,
                is_cloud     = True,
                cost_usd     = cost,
            )

        log.info(
            "gpt4o_analysis_complete",
            tokens=tokens_used,
            latency_ms=round(latency_ms, 1),
            session_id=request.session_id,
        )

        return VisionResponse(
            answer          = answer,
            model_id        = "gpt4o_vision",
            latency_ms      = latency_ms,
            tokens_used     = tokens_used,
            estimated_cost  = cost,
            is_cloud        = True,
        )

    async def health_check(self) -> BackendHealth:
        """Probe OpenAI's models endpoint as a lightweight availability check."""
        if not HAS_OPENAI:
            return BackendHealth(online=False, error="openai package not installed.")
        t0 = time.monotonic()
        try:
            client = openai.AsyncOpenAI(api_key=self._api_key)
            await asyncio.wait_for(client.models.list(), timeout=2.0)
            return BackendHealth(online=True, latency_ms=(time.monotonic() - t0) * 1000)
        except Exception as exc:
            return BackendHealth(
                online=False,
                latency_ms=(time.monotonic() - t0) * 1000,
                error=f"OpenAI health check failed: {exc}",
            )


# ---------------------------------------------------------------------------
# GeminiBackend — Google cloud backend
# ---------------------------------------------------------------------------
class GeminiBackend(VisionBackend):
    """
    Gemini 1.5 Pro (Google Cloud API) — alternative cloud backend.

    WHY GEMINI AS AN ALTERNATIVE
    -----------------------------
    Gemini 1.5 Pro has a 1M-token context window — the largest of any
    commercial model — making it suitable for:
      - Very long PDF documents.
      - Multi-page OCR sessions.
      - Any scenario where GPT-4o hits context limits.

    It is offered as an alternative, not a replacement, for GPT-4o.
    Capability differences are reflected in the BackendCapabilities profile.
    """

    _CAPABILITIES = BackendCapabilities(
        backend_id          = "gemini15pro",
        display_name        = "Gemini 1.5 Pro (Google Cloud)",
        is_local            = False,
        supports_gpu        = False,
        max_image_px        = 3072 * 3072,
        max_context_tokens  = 1_000_000,
        supports_ocr_bypass = True,
        cost_per_image_usd  = 0.00025,  # Approximate; check Google's pricing page.
        notes               = "Largest context window. Best for long documents. Cloud only.",
    )

    def __init__(self, api_key: str, collector: Optional[MetricsCollector] = None) -> None:
        self._api_key   = api_key
        self._collector = collector

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._CAPABILITIES

    async def analyze(self, request: VisionRequest) -> VisionResponse:
        if request.privacy_mode == PrivacyMode.STRICT:
            return VisionResponse(
                answer   = "",
                model_id = "gemini15pro",
                latency_ms = 0.0,
                error    = "Cloud APIs are disabled in STRICT privacy mode.",
            )

        if not HAS_GEMINI:
            return VisionResponse(
                answer   = "",
                model_id = "gemini15pro",
                latency_ms = 0.0,
                error    = "google-generativeai package not installed. Run: pip install google-generativeai",
            )

        t0 = time.monotonic()
        genai.configure(api_key=self._api_key)

        try:
            # Gemini accepts PIL Images natively — no base64 encoding needed.
            model = genai.GenerativeModel("gemini-1.5-pro")

            prompt_parts: list[Any] = []
            if request.ocr_text.strip():
                prompt_parts.append(
                    f"[Screen text extracted by OCR]:\n{request.ocr_text}\n\n"
                )
            prompt_parts.append(request.query)
            prompt_parts.append(request.image)

            response = await asyncio.wait_for(
                asyncio.to_thread(model.generate_content, prompt_parts),
                timeout=request.timeout_s,
            )
            answer = response.text or ""

        except asyncio.TimeoutError:
            return VisionResponse(
                answer    = "",
                model_id  = "gemini15pro",
                latency_ms = (time.monotonic() - t0) * 1000,
                error     = f"Gemini request timed out after {request.timeout_s}s.",
            )
        except Exception as exc:
            return VisionResponse(
                answer    = "",
                model_id  = "gemini15pro",
                latency_ms = (time.monotonic() - t0) * 1000,
                error     = f"Gemini API error: {exc}",
            )

        latency_ms = (time.monotonic() - t0) * 1000
        cost       = self.capabilities.cost_per_image_usd

        if self._collector:
            self._collector.record_api_call(
                backend_name = "gemini15pro",
                is_cloud     = True,
                cost_usd     = cost,
            )

        log.info(
            "gemini_analysis_complete",
            latency_ms=round(latency_ms, 1),
            session_id=request.session_id,
        )

        return VisionResponse(
            answer         = answer,
            model_id       = "gemini15pro",
            latency_ms     = latency_ms,
            estimated_cost = cost,
            is_cloud       = True,
        )

    async def health_check(self) -> BackendHealth:
        """List available Gemini models as a lightweight availability probe."""
        if not HAS_GEMINI:
            return BackendHealth(online=False, error="google-generativeai not installed.")
        t0 = time.monotonic()
        try:
            genai.configure(api_key=self._api_key)
            await asyncio.to_thread(list, genai.list_models())
            return BackendHealth(online=True, latency_ms=(time.monotonic() - t0) * 1000)
        except Exception as exc:
            return BackendHealth(
                online=False,
                latency_ms=(time.monotonic() - t0) * 1000,
                error=f"Gemini health check failed: {exc}",
            )


# ---------------------------------------------------------------------------
# BackendRegistry — the hot-swap router
# ---------------------------------------------------------------------------
class BackendRegistry:
    """
    Manages a prioritised list of backends and routes requests to the best one.

    WHY A REGISTRY (not just backend = SomeBackend())
    ---------------------------------------------------
    VisionManager needs to:
      - Automatically fall back when the primary backend fails.
      - Skip cloud backends in STRICT mode.
      - Prefer local GPU over local CPU over cloud.
      - Always guarantee a response via OCROnlyBackend.

    A registry encodes this routing logic in one place.  VisionManager just
    calls registry.analyze(request) and trusts the registry to handle failover.

    Usage
    -----
    >>> registry = BackendRegistry()
    >>> registry.register(Qwen2VL7BBackend(), priority=1)
    >>> registry.register(Llava16Backend(),   priority=2)
    >>> registry.register(GPT4oVisionBackend(api_key=...), priority=3)
    >>> response = await registry.analyze(request)
    """

    def __init__(self) -> None:
        # List of (priority, backend) tuples — sorted lowest priority number first.
        self._backends: list[tuple[int, VisionBackend]] = []
        # Always-available fallback — cannot be removed.
        self._fallback = OCROnlyBackend()

    def register(self, backend: VisionBackend, priority: int = 99) -> None:
        """
        Register a backend with a given priority (lower = preferred).

        Backends with the same priority are tried in registration order.
        """
        self._backends.append((priority, backend))
        self._backends.sort(key=lambda x: x[0])
        log.info(
            "backend_registered",
            backend_id=backend.capabilities.backend_id,
            priority=priority,
        )

    async def analyze(self, request: VisionRequest) -> VisionResponse:
        """
        Try each registered backend in priority order, falling back on failure.

        Skips cloud backends if privacy_mode is STRICT.
        Always returns a VisionResponse — worst case from OCROnlyBackend.
        """
        for _priority, backend in self._backends:
            caps = backend.capabilities

            # Enforce privacy mode: skip cloud backends in STRICT mode.
            if not caps.is_local and request.privacy_mode == PrivacyMode.STRICT:
                log.info(
                    "backend_skipped",
                    backend_id=caps.backend_id,
                    reason="STRICT_privacy_mode",
                )
                continue

            response = await backend.analyze(request)

            if response.success:
                return response

            log.warning(
                "backend_failed",
                backend_id=caps.backend_id,
                error=response.error,
                trying_next=True,
            )

        # All registered backends failed — use the guaranteed fallback.
        log.warning("all_backends_failed", falling_back_to="ocr_only")
        return await self._fallback.analyze(request)

    async def health_check_all(self) -> Dict[str, BackendHealth]:
        """
        Run health checks on all registered backends concurrently.

        Returns a dict mapping backend_id → BackendHealth.
        Uses asyncio.gather so all probes run in parallel — total time is
        bounded by the slowest single probe (not the sum of all probes).
        """
        async def _check(backend: VisionBackend) -> tuple[str, BackendHealth]:
            health = await backend.health_check()
            return backend.capabilities.backend_id, health

        results = await asyncio.gather(
            *[_check(b) for _, b in self._backends],
            return_exceptions=True,
        )

        health_map: Dict[str, BackendHealth] = {}
        for result in results:
            if isinstance(result, Exception):
                continue
            backend_id, health = result
            health_map[backend_id] = health

        return health_map
