"""
metrics.py — Vision Pipeline Observability
==========================================
WHY THIS MODULE EXISTS
-----------------------
You cannot improve what you cannot measure.  As the vision subsystem grows to
support cloud models (GPT-4V, Gemini) and local inference (Qwen-VL, LLaVA),
two categories of metrics become critical:

  PERFORMANCE metrics  — latency per stage, memory usage — tell you WHERE the
                         pipeline is slow so you can optimise the right bottleneck.

  COST metrics         — tokens used, images sent, estimated API spend — tell you
                         HOW MUCH each query costs so the user can make informed
                         decisions about local-vs-cloud trade-offs.

  RELIABILITY metrics  — failure rate, cancellation rate, validation failure
                         distribution — tell you WHERE the pipeline is fragile
                         so you can harden the right subsystem.

WHY A DEDICATED MODULE (not inline counters)
---------------------------------------------
Inline counters (global_capture_count += 1 scattered across files) are
invisible, untestable, and impossible to reset.  A dedicated MetricsCollector:
  - Owns all counters in one place.
  - Can be injected (dependency injection) so tests use a mock collector.
  - Exposes a clean snapshot() method for dashboards and health endpoints.
  - Can be swapped for a real metrics sink (Prometheus, StatsD) in production.

WHY dataclasses FOR SNAPSHOTS
--------------------------------
A snapshot is immutable data — a point-in-time view. @dataclass gives us
__repr__ and JSON-serialisability for free, making it trivial to emit snapshots
to a log line or an API response without manual dict construction.
"""

import time
from dataclasses import asdict, dataclass, field
from threading import Lock
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Stage latency tracking
# ---------------------------------------------------------------------------
@dataclass
class StageLatency:
    """
    Latency statistics for a single pipeline stage.

    We track min, max, total, and count so we can compute:
      - average     = total_ms / count
      - throughput  = count / wall_time
    without storing every individual sample (which would leak memory).
    """
    count:    int   = 0
    total_ms: float = 0.0
    min_ms:   float = float("inf")
    max_ms:   float = 0.0

    def record(self, ms: float) -> None:
        self.count    += 1
        self.total_ms += ms
        self.min_ms    = min(self.min_ms, ms)
        self.max_ms    = max(self.max_ms, ms)

    @property
    def avg_ms(self) -> Optional[float]:
        return round(self.total_ms / self.count, 2) if self.count else None


# ---------------------------------------------------------------------------
# MetricsSnapshot — immutable point-in-time view
# ---------------------------------------------------------------------------
@dataclass
class MetricsSnapshot:
    """
    Immutable snapshot of all collected metrics at a point in time.

    Returned by MetricsCollector.snapshot().  Safe to log, serialise to JSON,
    or include in an API response without worrying about concurrent updates.
    """
    # --- Session counters -------------------------------------------------
    sessions_started:    int = 0
    sessions_completed:  int = 0
    sessions_failed:     int = 0
    sessions_cancelled:  int = 0
    sessions_timed_out:  int = 0

    # --- Stage latencies (avg_ms) -----------------------------------------
    capture_avg_ms:    Optional[float] = None
    validation_avg_ms: Optional[float] = None
    security_avg_ms:   Optional[float] = None
    ocr_avg_ms:        Optional[float] = None
    vision_avg_ms:     Optional[float] = None

    # --- API cost tracking ------------------------------------------------
    # Token counts and cost estimates only matter for cloud backends.
    # Local models set these to 0.
    total_tokens_used:  int   = 0
    total_images_sent:  int   = 0
    estimated_cost_usd: float = 0.0  # Approximation; not billing-accurate.
    cloud_calls:        int   = 0
    local_calls:        int   = 0

    # --- Backend usage distribution ---------------------------------------
    # Maps backend name → number of successful calls.
    # Example: {"qwen2vl7b": 62, "llava16": 23, "gpt4o_vision": 15}
    backend_usage: Dict[str, int] = field(default_factory=dict)

    # --- Validation failure distribution ----------------------------------
    # Maps ValidationResult reason codes → count.
    # Example: {"BLACK_SCREEN": 3, "RESOLUTION_TOO_LARGE": 1}
    validation_failures: Dict[str, int] = field(default_factory=dict)

    # --- Reliability rates ------------------------------------------------
    success_rate:      Optional[float] = None  # completed / started
    failure_rate:      Optional[float] = None  # failed / started
    cancellation_rate: Optional[float] = None  # cancelled / started

    # --- Observability metadata -------------------------------------------
    # Average OCR confidence score across all successful OCR calls.
    avg_ocr_confidence: Optional[float] = None
    # Ratio of local to cloud model calls (0.0 = all cloud, 1.0 = all local).
    local_vs_cloud_ratio: Optional[float] = None

    snapshot_timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON emission or log records."""
        return asdict(self)


# ---------------------------------------------------------------------------
# MetricsCollector — the live, mutable accumulator
# ---------------------------------------------------------------------------
class MetricsCollector:
    """
    Thread-safe accumulator for all vision pipeline metrics.

    Design choices
    --------------
    - A single instance is created by VisionManager and passed (injected) to
      every component that needs to record metrics.  This avoids global state.
    - A threading.Lock guards all mutations so concurrent async tasks can
      record metrics safely.
    - snapshot() is the ONLY read path — it copies the current state under
      the lock and returns an immutable MetricsSnapshot.

    Dependency injection (rather than a singleton) makes testing trivial:
    tests can pass in a fresh MetricsCollector and assert on its snapshot
    without worrying about state leaking between test cases.

    Usage
    -----
    >>> collector = MetricsCollector()
    >>> collector.record_session_started()
    >>> collector.record_stage_latency("capture", 42.5)
    >>> collector.record_session_completed()
    >>> snap = collector.snapshot()
    >>> print(snap.capture_avg_ms)  # 42.5
    """

    def __init__(self) -> None:
        self._lock = Lock()

        # Session counters
        self._sessions_started   = 0
        self._sessions_completed = 0
        self._sessions_failed    = 0
        self._sessions_cancelled = 0
        self._sessions_timed_out = 0

        # Stage latency accumulators — keyed by stage name.
        self._latencies: Dict[str, StageLatency] = {
            "capture":    StageLatency(),
            "validation": StageLatency(),
            "security":   StageLatency(),
            "ocr":        StageLatency(),
            "vision":     StageLatency(),
        }

        # API cost / usage tracking
        self._total_tokens_used  = 0
        self._total_images_sent  = 0
        self._estimated_cost_usd = 0.0
        self._cloud_calls        = 0
        self._local_calls        = 0

        # Backend usage distribution — flexible dict (new backends added at runtime).
        self._backend_usage: Dict[str, int] = {}

        # Validation failure distribution
        self._validation_failures: Dict[str, int] = {}

        # OCR confidence accumulator
        self._ocr_confidence_total = 0.0
        self._ocr_confidence_count = 0

    # -----------------------------------------------------------------------
    # Session lifecycle recording
    # -----------------------------------------------------------------------
    def record_session_started(self) -> None:
        with self._lock:
            self._sessions_started += 1

    def record_session_completed(self) -> None:
        with self._lock:
            self._sessions_completed += 1

    def record_session_failed(self) -> None:
        with self._lock:
            self._sessions_failed += 1

    def record_session_cancelled(self) -> None:
        with self._lock:
            self._sessions_cancelled += 1

    def record_session_timed_out(self) -> None:
        with self._lock:
            self._sessions_timed_out += 1

    # -----------------------------------------------------------------------
    # Stage latency recording
    # -----------------------------------------------------------------------
    def record_stage_latency(self, stage: str, ms: float) -> None:
        """
        Record the wall-clock time (in milliseconds) taken by a pipeline stage.

        stage must be one of: "capture", "validation", "security", "ocr", "vision".
        Unknown stages are added dynamically (extensible for future stages).
        """
        with self._lock:
            if stage not in self._latencies:
                self._latencies[stage] = StageLatency()
            self._latencies[stage].record(ms)

    # -----------------------------------------------------------------------
    # API cost and backend usage recording
    # -----------------------------------------------------------------------
    def record_api_call(
        self,
        backend_name: str,
        tokens_used:  int   = 0,
        is_cloud:     bool  = False,
        cost_usd:     float = 0.0,
    ) -> None:
        """
        Record a successful call to a vision model backend.

        Parameters
        ----------
        backend_name :
            Versioned backend identifier, e.g. "qwen2vl7b", "gpt4o_vision".
        tokens_used :
            Tokens consumed by this call (0 for local models that don't report tokens).
        is_cloud :
            True if this call left the local machine (e.g. OpenAI, Gemini APIs).
        cost_usd :
            Estimated cost in USD for this call.  Pass 0.0 for local models.
        """
        with self._lock:
            self._total_tokens_used  += tokens_used
            self._total_images_sent  += 1
            self._estimated_cost_usd += cost_usd
            if is_cloud:
                self._cloud_calls += 1
            else:
                self._local_calls += 1
            self._backend_usage[backend_name] = self._backend_usage.get(backend_name, 0) + 1

    def record_validation_failure(self, reason: str) -> None:
        """
        Record a ValidationResult failure by its reason code.

        Reason codes match ValidationResult.REASON_* constants, e.g. "BLACK_SCREEN".
        Tracking these distributions tells you which validation checks fire most
        often in production, guiding where to invest in image quality improvements.
        """
        with self._lock:
            self._validation_failures[reason] = self._validation_failures.get(reason, 0) + 1

    def record_ocr_confidence(self, confidence: float) -> None:
        """
        Accumulate an OCR confidence score from an individual OCR call.

        confidence should be in [0.0, 1.0].  The average across all calls is
        reported in the snapshot — useful for detecting degraded OCR quality
        (e.g. blurry screenshots, dark themes, unusual fonts).
        """
        with self._lock:
            self._ocr_confidence_total += confidence
            self._ocr_confidence_count += 1

    # -----------------------------------------------------------------------
    # Snapshot
    # -----------------------------------------------------------------------
    def snapshot(self) -> MetricsSnapshot:
        """
        Return an immutable, thread-safe snapshot of all accumulated metrics.

        Derived metrics (rates, ratios, averages) are computed here rather than
        incrementally to avoid floating-point accumulation errors.
        """
        with self._lock:
            started = self._sessions_started

            def _rate(count: int) -> Optional[float]:
                return round(count / started, 4) if started else None

            total_api = self._cloud_calls + self._local_calls

            avg_ocr = (
                round(self._ocr_confidence_total / self._ocr_confidence_count, 4)
                if self._ocr_confidence_count else None
            )

            local_ratio = (
                round(self._local_calls / total_api, 4)
                if total_api else None
            )

            return MetricsSnapshot(
                sessions_started    = started,
                sessions_completed  = self._sessions_completed,
                sessions_failed     = self._sessions_failed,
                sessions_cancelled  = self._sessions_cancelled,
                sessions_timed_out  = self._sessions_timed_out,

                capture_avg_ms    = self._latencies["capture"].avg_ms,
                validation_avg_ms = self._latencies["validation"].avg_ms,
                security_avg_ms   = self._latencies["security"].avg_ms,
                ocr_avg_ms        = self._latencies["ocr"].avg_ms,
                vision_avg_ms     = self._latencies["vision"].avg_ms,

                total_tokens_used   = self._total_tokens_used,
                total_images_sent   = self._total_images_sent,
                estimated_cost_usd  = round(self._estimated_cost_usd, 6),
                cloud_calls         = self._cloud_calls,
                local_calls         = self._local_calls,

                backend_usage       = dict(self._backend_usage),
                validation_failures = dict(self._validation_failures),

                success_rate        = _rate(self._sessions_completed),
                failure_rate        = _rate(self._sessions_failed),
                cancellation_rate   = _rate(self._sessions_cancelled),

                avg_ocr_confidence  = avg_ocr,
                local_vs_cloud_ratio = local_ratio,
            )

    def reset(self) -> None:
        """
        Reset all counters to zero.

        Intended for testing and for future "rolling window" implementations
        where metrics are periodically flushed to an external sink.
        """
        with self._lock:
            self.__init__()  # Re-initialise in-place — simplest correct reset.
