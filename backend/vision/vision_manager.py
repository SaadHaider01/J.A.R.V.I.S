"""
vision_manager.py — Pipeline Orchestrator
==========================================

WHY THIS MODULE EXISTS
-----------------------
Every other module in the Vision subsystem is a specialist:
  - screen_capture.py  → takes screenshots
  - validation.py      → checks image quality
  - security.py        → guards against privacy violations
  - ocr.py             → extracts text
  - vision_models.py   → reasons about images
  - screen_context.py  → structures the output

None of those modules know about each other.  VisionManager knows about all
of them and is responsible for one thing: running them in the right order,
handling every failure, and returning a complete VisionAnalysisResult.

WHY IS THIS THE MOST DANGEROUS MODULE
---------------------------------------
Because it touches everything.  A bug here can break the entire pipeline.
The implementation follows three strict rules to contain that risk:

  1. VisionManager is the ONLY code that changes CaptureSession.status.
     No other module should mutate session state — this prevents race conditions
     and makes session state machine transitions auditable.

  2. Retries happen HERE, not inside backends.
     Backends report failure.  VisionManager decides what to do about it.
     This keeps retry policy in one place rather than scattered across backends.

  3. Every code path that starts returns.
     Exceptions are caught at every stage.  Callers receive a VisionAnalysisResult
     always — never an unhandled exception.

WHY CaptureSession (not just a session_id string)
---------------------------------------------------
A CaptureSession object:
  - Owns the state machine (CREATED → CAPTURING → ... → COMPLETED).
  - Is the single source of truth about what is happening and when.
  - Can be retrieved by ID so the caller can check progress.
  - Carries a cancellation event so an in-flight analysis can be aborted.
  - Stores its start time so timeout enforcement is simple.

WHY ScreenCache
----------------
Without a cache, two questions asked 3 seconds apart result in:
  - Two screen captures (100 ms × 2)
  - Two OCR runs    (500 ms × 2)
  - Two API calls   ($0.0013 × 2)

With a 10-second cache keyed on image_hash:
  - Second question reuses the first result if the screen hasn't changed.
  - No duplicate API cost.
  - Near-instant response for follow-up questions.

WHY RETRIES ONLY IN VISION_MANAGER (not inside backends)
----------------------------------------------------------
A backend that retries internally hides its failures from VisionManager.
VisionManager can't decide "try the next backend" if the current backend is
silently retrying for 10 seconds.  The retry policy belongs at the orchestration
level where all backends are visible.

WHY VisionAnalysisResult (not just ScreenContext)
---------------------------------------------------
VisionAnalysisResult is the handoff object to the rest of Zytrix.
It carries:
  - The structured context (what was on screen).
  - The session ID (for follow-up queries, audit logs, cancellation).
  - Performance data (total time, backend used, cache hit).
  - Warnings (partial failures, low confidence, cloud fallback used).

The warnings list is what turns "something went wrong silently" into
"OCR confidence was low; treat extracted text carefully" in the UI.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from . import CaptureScope, PrivacyMode, SessionStatus
from .logger import VisionLogger
from .metrics import MetricsCollector
from .ocr import OCREngine, OCRMode, OCRResult
from .screen_capture import CaptureResult, ScreenCapture
from .screen_context import ScreenContext, build_screen_context
from .security import SecurityGuard
from .validation import ValidationResult, Validator, VisionLimits
from .vision_models import (
    BackendRegistry,
    Llava16Backend,
    OCROnlyBackend,
    Qwen2VL7BBackend,
    VisionRequest,
    VisionResponse,
)

log = VisionLogger(__name__)


# ---------------------------------------------------------------------------
# VisionConfig — runtime configuration for the pipeline
# ---------------------------------------------------------------------------
@dataclass
class VisionConfig:
    """
    Runtime configuration for the Vision pipeline.

    Passed to VisionManager at construction time.  Centralising config here
    means callers don't need to know which individual modules accept which
    parameters — they configure once and pass one object.

    Fields
    ------
    privacy_mode         : STRICT, BALANCED, or DEVELOPER (from PrivacyMode enum).
    default_scope        : Default CaptureScope when the caller doesn't specify one.
    default_ocr_mode     : Default OCRMode for text extraction.
    analysis_timeout_s   : Maximum seconds for the entire pipeline.
    cache_ttl_s          : How long captured screen context stays in cache.
    max_retries          : How many times to try the next backend on failure.
    write_debug_images   : Only True in DEVELOPER mode; writes screenshots to disk.
    """
    privacy_mode:       PrivacyMode  = PrivacyMode.BALANCED
    default_scope:      CaptureScope = CaptureScope.ACTIVE_WINDOW
    default_ocr_mode:   OCRMode      = OCRMode.BALANCED
    analysis_timeout_s: float        = 5.0
    cache_ttl_s:        float        = 10.0
    max_retries:        int          = 2
    write_debug_images: bool         = False

    def __post_init__(self) -> None:
        # STRICT mode must never write images to disk.
        if self.privacy_mode == PrivacyMode.STRICT:
            self.write_debug_images = False


# ---------------------------------------------------------------------------
# CaptureSession — per-request state machine
# ---------------------------------------------------------------------------
@dataclass
class CaptureSession:
    """
    Represents a single screen-analysis request from creation to completion.

    WHY A STATE MACHINE
    --------------------
    Session state transitions are:

        CREATED → CAPTURING → VALIDATING → SECURITY_CHECK →
        RUNNING_OCR → ANALYZING → BUILDING_CONTEXT → COMPLETED

    Terminal failure states (from any stage):
        FAILED | TIMED_OUT | CANCELLED

    WHY ONLY VISIONMANAGER CHANGES STATUS
    ----------------------------------------
    Allowing any module to mutate status would create race conditions and make
    the state machine impossible to reason about.  VisionManager is the single
    writer; every other module is read-only with respect to session state.

    Fields
    ------
    session_id    : Unique identifier (UUID4 hex string).
    query         : The user's natural-language question.
    capture_type  : Requested CaptureScope for this session.
    monitor_id    : Target monitor index (1-based).
    window_id     : Platform-specific window handle (future use).
    is_cloud      : True if the analysis required a cloud API call.
    status        : Current SessionStatus.  Only VisionManager writes this.
    created_at    : Wall-clock timestamp of session creation.
    _cancel_event : asyncio.Event set by cancel_analysis() to abort the pipeline.
    """
    session_id:   str
    query:        str
    capture_type: CaptureScope       = CaptureScope.ACTIVE_WINDOW
    monitor_id:   int                = 1
    window_id:    Optional[int]      = None
    is_cloud:     bool               = False
    status:       SessionStatus      = SessionStatus.CREATED
    created_at:   float              = field(default_factory=time.monotonic)
    _cancel_event: asyncio.Event     = field(default_factory=asyncio.Event, repr=False)

    @property
    def elapsed_s(self) -> float:
        """Seconds since this session was created."""
        return time.monotonic() - self.created_at

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def request_cancellation(self) -> None:
        """Signal this session to abort at the next safe checkpoint."""
        self._cancel_event.set()


# ---------------------------------------------------------------------------
# ScreenCache — deduplication layer
# ---------------------------------------------------------------------------
@dataclass
class CacheEntry:
    """
    A cached analysis result keyed on image_hash.

    WHY HASH-BASED (not time-based)
    ---------------------------------
    Time-based caching ("reuse results from the last N seconds") fails when:
    - The user switches windows quickly (different screen, same time window).
    - The user asks a follow-up 11 seconds later (time expired but screen unchanged).

    Hash-based caching ("reuse results when the screenshot looks the same")
    is more accurate: we only reuse when the screen is actually the same.
    The SHA-256 thumbnail hash from CaptureMetadata is the key.

    Fields
    ------
    image_hash     : SHA-256 thumbnail hash of the captured image.
    screen_context : The ScreenContext produced for this image.
    ocr_result     : The raw OCRResult (needed to rebuild VisionRequest on cache hit).
    vision_response: The raw VisionResponse (carried through for VisionAnalysisResult).
    created_at     : When this entry was cached.
    ttl_seconds    : How many seconds this entry stays valid.
    """
    image_hash:      str
    screen_context:  ScreenContext
    ocr_result:      OCRResult
    vision_response: VisionResponse
    created_at:      float = field(default_factory=time.monotonic)
    ttl_seconds:     float = 10.0

    @property
    def is_expired(self) -> bool:
        return (time.monotonic() - self.created_at) > self.ttl_seconds


class ScreenCache:
    """
    In-memory cache of recent screen analysis results.

    Thread-safety
    -------------
    Cache operations are protected by an asyncio.Lock so concurrent sessions
    don't corrupt the cache.  The lock is async (asyncio.Lock, not threading.Lock)
    because VisionManager is fully async — we never want to block the event loop.

    Cache invalidation
    ------------------
    Entries expire after ttl_seconds (configurable, default 10s).  Expired
    entries are evicted lazily (on the next get() or set() call) so we never
    need a background cleanup task.
    """

    def __init__(self) -> None:
        self._store: Dict[str, CacheEntry] = {}
        self._lock  = asyncio.Lock()

    async def get(self, image_hash: str) -> Optional[CacheEntry]:
        """Return a non-expired CacheEntry for image_hash, or None."""
        async with self._lock:
            entry = self._store.get(image_hash)
            if entry is None:
                return None
            if entry.is_expired:
                del self._store[image_hash]
                return None
            return entry

    async def set(self, entry: CacheEntry) -> None:
        """Store a cache entry, evicting expired entries to prevent memory growth."""
        async with self._lock:
            # Lazy eviction: remove all expired entries on each write.
            expired = [k for k, v in self._store.items() if v.is_expired]
            for k in expired:
                del self._store[k]
            self._store[entry.image_hash] = entry

    async def invalidate(self, image_hash: str) -> None:
        """Remove a specific entry (e.g. if the user forces a re-analysis)."""
        async with self._lock:
            self._store.pop(image_hash, None)

    @property
    def size(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# VisionAnalysisResult — the final handoff to the rest of Zytrix
# ---------------------------------------------------------------------------
@dataclass
class VisionAnalysisResult:
    """
    The complete output of one screen analysis request.

    WHY NOT JUST ScreenContext
    ---------------------------
    ScreenContext is what was ON screen.  VisionAnalysisResult is what
    HAPPENED during the analysis.  The rest of Zytrix needs both:
      - The Prompt Builder uses screen_context.
      - The UI uses warnings and cache_hit to show status to the user.
      - The metrics layer uses total_time_ms and backend_used.
      - The Conversation Engine uses session_id for follow-up queries.

    Fields
    ------
    session_id     : The CaptureSession ID for this analysis.
    screen_context : Structured perception output (what was on screen).
    backend_used   : Versioned ID of the vision model that produced the summary.
    total_time_ms  : Wall-clock time for the entire pipeline, in milliseconds.
    cache_hit      : True if the result came from ScreenCache (no capture/OCR).
    warnings       : Human-readable advisory messages for the UI or logs.
                     Examples: "OCR confidence low", "Cloud fallback used".
    status         : Terminal SessionStatus of the completed session.
    """
    session_id:     str
    screen_context: Optional[ScreenContext]
    backend_used:   str
    total_time_ms:  float
    cache_hit:      bool          = False
    warnings:       List[str]     = field(default_factory=list)
    status:         SessionStatus = SessionStatus.COMPLETED

    @property
    def success(self) -> bool:
        return self.screen_context is not None and self.status == SessionStatus.COMPLETED


# ---------------------------------------------------------------------------
# VisionManager — the orchestrator
# ---------------------------------------------------------------------------
class VisionManager:
    """
    The single entry point for all screen analysis requests.

    Manages the full pipeline:
        analyze_screen()
            ↓ CaptureSession created (CREATED)
            ↓ ScreenCache lookup
            ↓ ScreenCapture.capture()            (CAPTURING)
            ↓ Validator.validate_image()         (VALIDATING)
            ↓ SecurityGuard checks               (SECURITY_CHECK)  [new step]
            ↓ OCREngine.extract()                (RUNNING_OCR)
            ↓ BackendRegistry.analyze()          (ANALYZING)
            ↓ build_screen_context()             (BUILDING_CONTEXT)
            ↓ VisionAnalysisResult returned      (COMPLETED)

    Failure at any stage → FAILED / TIMED_OUT / CANCELLED

    Usage
    -----
    >>> manager = VisionManager(config=VisionConfig(privacy_mode=PrivacyMode.BALANCED))
    >>> result  = await manager.analyze_screen(query="Explain this error.")
    >>> if result.success:
    ...     print(result.screen_context.to_prompt_dict())

    Dependency injection
    --------------------
    All collaborators (ScreenCapture, OCREngine, BackendRegistry, ScreenCache)
    are created internally by default but can be injected for testing:

    >>> manager = VisionManager(
    ...     config   = VisionConfig(),
    ...     registry = mock_registry,
    ...     cache    = ScreenCache(),
    ... )
    """

    def __init__(
        self,
        config:    Optional[VisionConfig]    = None,
        registry:  Optional[BackendRegistry] = None,
        collector: Optional[MetricsCollector] = None,
        cache:     Optional[ScreenCache]     = None,
    ) -> None:
        self._config    = config or VisionConfig()
        self._collector = collector or MetricsCollector()
        self._cache     = cache or ScreenCache()

        # Build the backend registry with local-first priority.
        # Cloud backends require API keys — they are NOT registered by default.
        # Phase 3 / config loading will register cloud backends when keys are present.
        self._registry  = registry or self._build_default_registry()

        # Active sessions, keyed by session_id.  Only VisionManager writes to this.
        self._sessions: Dict[str, CaptureSession] = {}

        # ScreenCapture is created lazily (on first use) because mss
        # initialises a display connection at construction time.
        self._capture: Optional[ScreenCapture] = None

        self._ocr    = OCREngine(collector=self._collector)
        self._validator = Validator()

        log.info(
            "vision_manager_ready",
            privacy_mode=self._config.privacy_mode.value,
            timeout_s=self._config.analysis_timeout_s,
            cache_ttl_s=self._config.cache_ttl_s,
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    async def analyze_screen(
        self,
        query:         str,
        scope:         Optional[CaptureScope] = None,
        monitor_index: int                    = 1,
        ocr_mode:      Optional[OCRMode]      = None,
    ) -> VisionAnalysisResult:
        """
        Run the complete screen understanding pipeline and return a result.

        This is the primary entry point for all screen analysis requests.
        Every other public method is a helper around this one.

        Parameters
        ----------
        query :
            The user's question ("What error am I seeing?" / "Explain this code.").
        scope :
            CaptureScope to use.  Defaults to VisionConfig.default_scope.
        monitor_index :
            Target monitor (1-based).  Used as fallback when active-window
            capture is unavailable.
        ocr_mode :
            OCRMode override.  Defaults to VisionConfig.default_ocr_mode.

        Returns
        -------
        VisionAnalysisResult — always.  Never raises.
        """
        t_pipeline_start = time.monotonic()
        scope    = scope    or self._config.default_scope
        ocr_mode = ocr_mode or self._config.default_ocr_mode

        # Create session — status = CREATED.
        session = self._create_session(query=query, scope=scope, monitor_index=monitor_index)
        self._collector.record_session_started()

        try:
            result = await asyncio.wait_for(
                self._run_pipeline(session, scope, monitor_index, ocr_mode),
                timeout=self._config.analysis_timeout_s,
            )
            return result

        except asyncio.TimeoutError:
            self._transition(session, SessionStatus.TIMED_OUT)
            self._collector.record_session_timed_out()
            total_ms = (time.monotonic() - t_pipeline_start) * 1000
            log.error(
                "pipeline_timed_out",
                session_id=session.session_id,
                timeout_s=self._config.analysis_timeout_s,
            )
            return VisionAnalysisResult(
                session_id    = session.session_id,
                screen_context= None,
                backend_used  = "none",
                total_time_ms = total_ms,
                warnings      = [f"Analysis timed out after {self._config.analysis_timeout_s}s."],
                status        = SessionStatus.TIMED_OUT,
            )

        except Exception as exc:
            self._transition(session, SessionStatus.FAILED)
            self._collector.record_session_failed()
            total_ms = (time.monotonic() - t_pipeline_start) * 1000
            log.error("pipeline_unexpected_error", reason=str(exc), session_id=session.session_id)
            return VisionAnalysisResult(
                session_id    = session.session_id,
                screen_context= None,
                backend_used  = "none",
                total_time_ms = total_ms,
                warnings      = [f"Unexpected error: {exc}"],
                status        = SessionStatus.FAILED,
            )

    async def cancel_analysis(self, session_id: str) -> bool:
        """
        Request cancellation of an in-flight analysis session.

        Sets the session's cancel event, which the pipeline checks at each
        safe checkpoint.  Returns True if the session was found and signalled,
        False if the session does not exist or has already completed.

        WHY SIGNAL (not force-stop)
        ----------------------------
        Forcibly stopping an async task mid-execution can leave resources in
        an inconsistent state (open file handles, incomplete API payloads).
        Signalling via an event lets the pipeline finish its current atomic
        operation and then exit cleanly at the next checkpoint.
        """
        session = self._sessions.get(session_id)
        if session is None or session.status.is_terminal:
            return False
        session.request_cancellation()
        log.info("cancellation_requested", session_id=session_id)
        return True

    async def get_session(self, session_id: str) -> Optional[CaptureSession]:
        """Return the CaptureSession for the given ID, or None if not found."""
        return self._sessions.get(session_id)

    async def cleanup(self) -> None:
        """
        Release all resources held by VisionManager.

        Call this when Zytrix is shutting down or when the Vision subsystem
        is being torn down (e.g. user disables screen understanding in settings).
        """
        if self._capture is not None:
            self._capture.close()
            self._capture = None

        # Clear sessions to release any references to PIL Images or API clients.
        self._sessions.clear()
        log.info("vision_manager_cleanup_complete")

    def metrics_snapshot(self):
        """Return a MetricsSnapshot of accumulated pipeline metrics."""
        return self._collector.snapshot()

    async def health_check_backends(self) -> dict:
        """
        Run health checks on all registered backends and return results.

        Useful for startup diagnostics and admin dashboards.
        """
        return await self._registry.health_check_all()

    # -----------------------------------------------------------------------
    # Private: pipeline stages
    # -----------------------------------------------------------------------
    async def _run_pipeline(
        self,
        session:       CaptureSession,
        scope:         CaptureScope,
        monitor_index: int,
        ocr_mode:      OCRMode,
    ) -> VisionAnalysisResult:
        """
        Execute the full pipeline for one session.

        Each stage checks for cancellation before starting.  This keeps
        cancellation latency at most one stage duration (< 500ms typically).
        """
        t_start  = time.monotonic()
        warnings: List[str] = []

        # ===================================================================
        # STAGE 1: Capture
        # ===================================================================
        self._transition(session, SessionStatus.CAPTURING)
        self._check_cancellation(session)

        t_cap = time.monotonic()
        capture_result = self._get_capture().capture(
            scope=scope, monitor_index=monitor_index
        )
        cap_ms = (time.monotonic() - t_cap) * 1000
        self._collector.record_stage_latency("capture", cap_ms)

        log.info(
            "capture_complete",
            session_id=session.session_id,
            hash=capture_result.metadata.image_hash[:12] + "...",
            ms=round(cap_ms, 1),
            scope=scope.name,
        )

        # ===================================================================
        # STAGE 2: Cache lookup (uses image hash from capture)
        # ===================================================================
        image_hash = capture_result.metadata.image_hash
        cached = await self._cache.get(image_hash)
        if cached:
            log.info("cache_hit", session_id=session.session_id, hash=image_hash[:12])
            self._transition(session, SessionStatus.COMPLETED)
            self._collector.record_session_completed()
            total_ms = (time.monotonic() - t_start) * 1000
            return VisionAnalysisResult(
                session_id    = session.session_id,
                screen_context= cached.screen_context,
                backend_used  = cached.vision_response.model_id,
                total_time_ms = total_ms,
                cache_hit     = True,
                warnings      = warnings,
                status        = SessionStatus.COMPLETED,
            )

        # ===================================================================
        # STAGE 3: Validation
        # ===================================================================
        self._transition(session, SessionStatus.VALIDATING)
        self._check_cancellation(session)

        t_val = time.monotonic()
        val_result = self._validator.validate_image(capture_result.image)
        val_ms = (time.monotonic() - t_val) * 1000
        self._collector.record_stage_latency("validation", val_ms)

        if not val_result.is_valid:
            self._collector.record_validation_failure(val_result.reason)
            log.warning(
                "validation_failed",
                reason=val_result.reason,
                details=str(val_result.details),
                session_id=session.session_id,
            )
            self._transition(session, SessionStatus.FAILED)
            self._collector.record_session_failed()
            return VisionAnalysisResult(
                session_id    = session.session_id,
                screen_context= None,
                backend_used  = "none",
                total_time_ms = (time.monotonic() - t_start) * 1000,
                warnings      = [f"Screen validation failed: {val_result.message}"],
                status        = SessionStatus.FAILED,
            )

        # ===================================================================
        # STAGE 4: Security check
        # ===================================================================
        # (Mapped to VALIDATING for now — SessionStatus.SECURITY_CHECK is a
        #  logical sub-step.  A dedicated status can be added to the enum in
        #  a future sprint without breaking this code.)
        self._check_cancellation(session)

        t_sec = time.monotonic()
        can_use_cloud, cloud_warning = SecurityGuard.can_use_cloud(
            privacy_mode  = self._config.privacy_mode,
            context_hint  = session.query,
        )
        sec_ms = (time.monotonic() - t_sec) * 1000
        self._collector.record_stage_latency("security", sec_ms)

        if cloud_warning:
            warnings.append(cloud_warning)

        # ===================================================================
        # STAGE 5: OCR
        # ===================================================================
        self._transition(session, SessionStatus.RUNNING_OCR)
        self._check_cancellation(session)

        ocr_result = self._ocr.extract(
            image      = capture_result.image,
            mode       = ocr_mode,
            image_hash = image_hash,
        )

        if ocr_result.confidence < 0.4 and not ocr_result.is_empty:
            warnings.append(
                f"OCR confidence is low ({ocr_result.confidence:.2f}). "
                "Text extraction may be inaccurate."
            )
        if ocr_result.error:
            warnings.append(f"OCR partial failure: {ocr_result.error}")

        log.info(
            "ocr_stage_complete",
            session_id=session.session_id,
            words=ocr_result.word_count,
            confidence=ocr_result.confidence,
            engine=ocr_result.engine,
        )

        # ===================================================================
        # STAGE 6: Vision model (with retry policy)
        # ===================================================================
        self._transition(session, SessionStatus.ANALYZING)
        self._check_cancellation(session)

        vision_response = await self._run_vision_with_retry(
            session      = session,
            capture_result = capture_result,
            ocr_result   = ocr_result,
            can_use_cloud= can_use_cloud,
            warnings     = warnings,
        )
        session.is_cloud = vision_response.is_cloud

        if vision_response.error:
            warnings.append(f"Vision model warning: {vision_response.error}")

        if vision_response.is_cloud:
            warnings.append(
                f"Screen image was sent to a cloud API ({vision_response.model_id}). "
                "Ensure no sensitive information was visible."
            )

        # ===================================================================
        # STAGE 7: Build screen context
        # ===================================================================
        self._transition(session, SessionStatus.BUILDING_CONTEXT)
        self._check_cancellation(session)

        resolution = f"{capture_result.metadata.width}x{capture_result.metadata.height}"
        screen_context = build_screen_context(
            ocr_result      = ocr_result,
            vision_response = vision_response,
            monitor_index   = capture_result.metadata.monitor_index,
            resolution      = resolution,
            capture_scope   = scope.name,
        )

        # ===================================================================
        # Cache the result for follow-up queries
        # ===================================================================
        await self._cache.set(CacheEntry(
            image_hash      = image_hash,
            screen_context  = screen_context,
            ocr_result      = ocr_result,
            vision_response = vision_response,
            ttl_seconds     = self._config.cache_ttl_s,
        ))

        # ===================================================================
        # Finalise
        # ===================================================================
        self._transition(session, SessionStatus.COMPLETED)
        self._collector.record_session_completed()

        total_ms = (time.monotonic() - t_start) * 1000
        log.info(
            "pipeline_complete",
            session_id    = session.session_id,
            total_ms      = round(total_ms, 1),
            backend       = vision_response.model_id,
            cache_hit     = False,
            warnings      = len(warnings),
            application   = screen_context.application.value,
            content_type  = screen_context.content_type.value,
        )

        return VisionAnalysisResult(
            session_id    = session.session_id,
            screen_context= screen_context,
            backend_used  = vision_response.model_id,
            total_time_ms = total_ms,
            cache_hit     = False,
            warnings      = warnings,
            status        = SessionStatus.COMPLETED,
        )

    async def _run_vision_with_retry(
        self,
        session:       CaptureSession,
        capture_result: CaptureResult,
        ocr_result:    OCRResult,
        can_use_cloud: bool,
        warnings:      List[str],
    ) -> VisionResponse:
        """
        Try the backend registry up to max_retries times, collecting warnings.

        WHY RETRY HERE (not in BackendRegistry)
        -----------------------------------------
        BackendRegistry already implements single-pass failover (try next backend
        on failure).  This layer adds a configurable retry count on top, giving
        transient network errors a second chance before moving to the next backend.

        The retry window is conservative (max_retries = 2 by default) to avoid
        excessive latency on persistent failures.
        """
        request = VisionRequest(
            image        = capture_result.image,
            query        = session.query,
            ocr_text     = ocr_result.text,
            session_id   = session.session_id,
            privacy_mode = self._config.privacy_mode,
            timeout_s    = self._config.analysis_timeout_s,
        )

        last_response: Optional[VisionResponse] = None

        for attempt in range(max(1, self._config.max_retries)):
            self._check_cancellation(session)

            response = await self._registry.analyze(request)
            if response.success:
                if attempt > 0:
                    warnings.append(f"Vision succeeded on retry attempt {attempt + 1}.")
                return response

            last_response = response
            log.warning(
                "vision_retry",
                attempt=attempt + 1,
                max=self._config.max_retries,
                error=response.error,
                session_id=session.session_id,
            )

        # All retries exhausted — return whatever we have (may be an error response).
        warnings.append(
            f"Vision analysis failed after {self._config.max_retries} attempt(s). "
            "Falling back to OCR-only result."
        )
        return last_response or VisionResponse(
            answer    = ocr_result.text,
            model_id  = "ocr_only",
            latency_ms= 0.0,
            error     = "All vision backends failed.",
        )

    # -----------------------------------------------------------------------
    # Private: helpers
    # -----------------------------------------------------------------------
    def _create_session(
        self,
        query:        str,
        scope:        CaptureScope,
        monitor_index: int,
    ) -> CaptureSession:
        """Create, register, and return a new CaptureSession."""
        session = CaptureSession(
            session_id   = uuid.uuid4().hex,
            query        = query,
            capture_type = scope,
            monitor_id   = monitor_index,
        )
        self._sessions[session.session_id] = session
        log.info(
            "session_created",
            session_id=session.session_id,
            scope=scope.name,
            query=session.query[:60],
        )
        return session

    def _transition(self, session: CaptureSession, new_status: SessionStatus) -> None:
        """
        Transition a session to a new status.

        This is the ONLY place in the entire codebase where session.status
        is written.  Centralising mutations here makes the state machine
        auditable and debuggable from a single location.
        """
        old = session.status
        session.status = new_status
        log.debug(
            "session_transition",
            session_id=session.session_id,
            old=old.value,
            new=new_status.value,
            elapsed_s=round(session.elapsed_s, 3),
        )

    def _check_cancellation(self, session: CaptureSession) -> None:
        """
        Raise asyncio.CancelledError if the session has been cancelled.

        Called at the start of each pipeline stage.  If cancel_analysis()
        was called between stages, the pipeline exits cleanly here.
        """
        if session.is_cancelled:
            self._transition(session, SessionStatus.CANCELLED)
            self._collector.record_session_cancelled()
            raise asyncio.CancelledError(
                f"Session {session.session_id} was cancelled by the user."
            )

    def _get_capture(self) -> ScreenCapture:
        """
        Return the shared ScreenCapture, initialising lazily on first use.

        WHY LAZY
        ---------
        mss initialises a display connection at construction time.  Creating
        ScreenCapture at VisionManager.__init__ would fail in headless test
        environments.  Lazy init means tests can construct VisionManager
        without needing a display.
        """
        if self._capture is None:
            self._capture = ScreenCapture()
        return self._capture

    def _build_default_registry(self) -> BackendRegistry:
        """
        Build the default backend registry with local-first priority.

        Priority order:
          1. Qwen2-VL 7B (local GPU, best accuracy for developer content)
          2. LLaVA 1.6   (local GPU, reliable general fallback)
          3. OCROnlyBackend (always available — guaranteed last resort)

        Cloud backends (GPT-4o, Gemini) are NOT registered by default.
        They must be added by the configuration layer when API keys are present.
        This ensures STRICT mode cannot accidentally use cloud APIs even if
        the user forgets to set privacy_mode explicitly.
        """
        registry = BackendRegistry()
        registry.register(Qwen2VL7BBackend(collector=self._collector), priority=1)
        registry.register(Llava16Backend(collector=self._collector),   priority=2)
        # OCROnlyBackend is baked into BackendRegistry as the unconditional fallback.
        return registry
