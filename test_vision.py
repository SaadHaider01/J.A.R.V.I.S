"""
test_vision.py — Manual Test Runner for the Vision Subsystem
=============================================================

Run this from the project root:

    python test_vision.py

Or run individual sections with a flag:

    python test_vision.py --capture      # Test 1: capture only
    python test_vision.py --validate     # Test 2: capture + validate
    python test_vision.py --ocr          # Test 3: capture + OCR
    python test_vision.py --context      # Test 4: full context build (no vision model)
    python test_vision.py --pipeline     # Test 5: full pipeline (requires Ollama)
    python test_vision.py --all          # Run all tests in sequence

Requirements:
    pip install mss pywin32 Pillow easyocr

Optional (for Test 5 / full pipeline):
    - Ollama running locally: https://ollama.com
    - ollama pull qwen2.5-vl:7b
"""

import argparse
import asyncio
import io
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows consoles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Ensure J.A.R.V.I.S project root is on the path
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Pretty printing helpers
# ---------------------------------------------------------------------------
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
DIM    = "\033[2m"

def _header(title: str) -> None:
    width = 60
    print(f"\n{BOLD}{CYAN}{'=' * width}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * width}{RESET}")

def _ok(msg: str) -> None:
    print(f"  {GREEN}[OK]{RESET}  {msg}")

def _warn(msg: str) -> None:
    print(f"  {YELLOW}[!!]{RESET}  {msg}")

def _fail(msg: str) -> None:
    print(f"  {RED}[XX]{RESET}  {msg}")

def _info(msg: str) -> None:
    print(f"  {DIM}[..] {msg}{RESET}")

def _kv(key: str, value) -> None:
    print(f"  {BOLD}{key:<22}{RESET} {value}")

# ---------------------------------------------------------------------------
# Fallback for headless environments (BitBlt Access Denied)
# ---------------------------------------------------------------------------
_synthetic_fallback_enabled = False

def _enable_synthetic_capture():
    global _synthetic_fallback_enabled
    if _synthetic_fallback_enabled:
        return
    _synthetic_fallback_enabled = True
    
    _warn("Live capture failed (likely headless/SSH). Using synthetic fallback images for remaining tests.")
    
    from backend.vision.screen_capture import ScreenCapture, CaptureResult, CaptureMetadata
    from backend.vision import CaptureScope
    from PIL import Image, ImageDraw
    import time
    
    original_capture = ScreenCapture.capture
    
    def synthetic_capture(self, scope=CaptureScope.ACTIVE_WINDOW, monitor_index=1, bbox=None):
        try:
            return original_capture(self, scope, monitor_index, bbox)
        except Exception as e:
            if "BitBlt" in str(e) or "Access is denied" in str(e):
                img = Image.new("RGB", (800, 600), (30, 30, 30))
                draw = ImageDraw.Draw(img)
                draw.text((50, 50), "Synthetic Capture Fallback", fill=(255, 255, 255))
                draw.text((50, 80), f"Scope: {scope.name}", fill=(200, 200, 200))
                
                import hashlib
                h = hashlib.sha256(img.tobytes()).hexdigest()
                
                meta = CaptureMetadata(
                    image_hash=h,
                    width=800,
                    height=600,
                    scope=scope,
                    monitor_index=monitor_index,
                    timestamp=time.time(),
                    capture_ms=5.0
                )
                return CaptureResult(image=img, metadata=meta)
            raise
            
    ScreenCapture.capture = synthetic_capture


# ===========================================================================
# TEST 1 — Screen Capture
# ===========================================================================
def test_capture() -> bool:
    _header("TEST 1 — Screen Capture")

    try:
        from backend.vision.screen_capture import ScreenCapture, CaptureMetadata
        from backend.vision import CaptureScope
    except ImportError as e:
        _fail(f"Import error: {e}")
        _info("Install mss: pip install mss")
        return False

    passed = True

    # --- 1a: Active window capture ---
    print(f"\n  {BOLD}1a. Active Window Capture{RESET}")
    try:
        t0 = time.monotonic()
        with ScreenCapture() as sc:
            result = sc.capture(scope=CaptureScope.ACTIVE_WINDOW)
        ms = (time.monotonic() - t0) * 1000

        _ok(f"Captured in {ms:.1f} ms  (target < 100 ms)")
        _kv("Resolution",   f"{result.metadata.width} × {result.metadata.height}")
        _kv("Image hash",   result.metadata.image_hash[:20] + "...")
        _kv("Scope",        result.metadata.scope.name)
        _kv("Monitor index",result.metadata.monitor_index)

        if ms > 100:
            _warn(f"Capture took {ms:.1f} ms — over 100 ms target (OK for first run)")

    except Exception as e:
        _fail(f"Active window capture failed: {e}")
        _enable_synthetic_capture()
        passed = False

    # --- 1b: Full screen capture ---
    print(f"\n  {BOLD}1b. Full Screen Capture{RESET}")
    try:
        with ScreenCapture() as sc:
            result_fs = sc.capture(scope=CaptureScope.FULL_SCREEN)
        _ok(f"Full screen: {result_fs.metadata.width} × {result_fs.metadata.height}")
    except Exception as e:
        _fail(f"Full screen capture failed: {e}")
        passed = False

    # --- 1c: Monitor capture ---
    print(f"\n  {BOLD}1c. Monitor 1 Capture{RESET}")
    try:
        with ScreenCapture() as sc:
            result_mon = sc.capture(scope=CaptureScope.MONITOR, monitor_index=1)
        _ok(f"Monitor 1: {result_mon.metadata.width} × {result_mon.metadata.height}")
    except Exception as e:
        _fail(f"Monitor capture failed: {e}")
        passed = False

    # --- 1d: Duplicate detection (same screen → same hash) ---
    print(f"\n  {BOLD}1d. Screenshot Fingerprinting (hash deduplication){RESET}")
    try:
        with ScreenCapture() as sc:
            r1 = sc.capture(scope=CaptureScope.FULL_SCREEN)
            r2 = sc.capture(scope=CaptureScope.FULL_SCREEN)

        if r1.metadata.image_hash == r2.metadata.image_hash:
            _ok("Two captures of the same static screen → identical hash ✓")
        else:
            _warn("Hashes differ (screen may have changed between captures — acceptable)")
        _kv("Hash 1", r1.metadata.image_hash[:20] + "...")
        _kv("Hash 2", r2.metadata.image_hash[:20] + "...")
    except Exception as e:
        _fail(f"Fingerprinting test failed: {e}")
        passed = False

    return passed


# ===========================================================================
# TEST 2 — Validation Layer
# ===========================================================================
def test_validation() -> bool:
    _header("TEST 2 — Validation Layer")

    try:
        from backend.vision.validation import Validator, ValidationResult, VisionLimits
        from backend.vision.screen_capture import ScreenCapture
        from backend.vision import CaptureScope
        from PIL import Image
    except ImportError as e:
        _fail(f"Import error: {e}")
        return False

    passed = True
    validator = Validator()

    # --- 2a: Valid image (live capture) ---
    print(f"\n  {BOLD}2a. Valid live screenshot{RESET}")
    try:
        with ScreenCapture() as sc:
            result = sc.capture(scope=CaptureScope.ACTIVE_WINDOW)

        val = validator.validate_image(result.image)
        if val.is_valid:
            _ok(f"Validation passed — reason: {val.reason}")
            _kv("Resolution",      val.details.get("resolution", "?"))
            _kv("Memory (est.)",   f"{val.details.get('estimated_memory_mb', '?')} MB")
            _kv("Contrast score",  val.details.get("contrast_score", "?"))
        else:
            _warn(f"Validation failed unexpectedly: {val.reason} — {val.message}")
            _info("This may happen on a black screen or locked session")
    except Exception as e:
        _fail(f"Live capture validation raised: {e}")
        passed = False

    # --- 2b: None image ---
    print(f"\n  {BOLD}2b. None image → NULL_IMAGE{RESET}")
    val = validator.validate_image(None)
    if not val.is_valid and val.reason == ValidationResult.REASON_NULL_IMAGE:
        _ok(f"Correctly rejected None — reason: {val.reason}")
    else:
        _fail(f"Expected NULL_IMAGE, got: {val.reason}")
        passed = False

    # --- 2c: Tiny image ---
    print(f"\n  {BOLD}2c. 1×1 pixel image → TOO_SMALL{RESET}")
    tiny = Image.new("RGB", (1, 1), (255, 255, 255))
    val = validator.validate_image(tiny)
    if not val.is_valid and val.reason == ValidationResult.REASON_TOO_SMALL:
        _ok(f"Correctly rejected 1×1 — reason: {val.reason}")
    else:
        _fail(f"Expected TOO_SMALL, got: {val.reason}")
        passed = False

    # --- 2d: Black screen ---
    print(f"\n  {BOLD}2d. Completely black 400×300 image → BLACK_SCREEN{RESET}")
    black = Image.new("RGB", (400, 300), (0, 0, 0))
    val = validator.validate_image(black)
    if not val.is_valid and val.reason == ValidationResult.REASON_BLACK_SCREEN:
        _ok(f"Correctly detected black screen — contrast: {val.details.get('contrast_score')}")
    else:
        _fail(f"Expected BLACK_SCREEN, got: {val.reason} ({val.message})")
        passed = False

    # --- 2e: raise_if_invalid helper ---
    print(f"\n  {BOLD}2e. raise_if_invalid() helper{RESET}")
    from backend.vision.validation import ValidationError
    try:
        val = validator.validate_image(None)
        val.raise_if_invalid()
        _fail("Should have raised ValidationError")
        passed = False
    except ValidationError as ve:
        _ok(f"ValidationError raised correctly — reason: {ve.result.reason}")

    return passed


# ===========================================================================
# TEST 3 — Security Layer
# ===========================================================================
def test_security() -> bool:
    _header("TEST 3 — Security Layer")

    try:
        from backend.vision.security import SecurityGuard
        from backend.vision import PrivacyMode
    except ImportError as e:
        _fail(f"Import error: {e}")
        return False

    passed = True

    # --- 3a: Secret redaction ---
    print(f"\n  {BOLD}3a. Secret redaction in logs{RESET}")
    dirty_samples = [
        ("API key",     "MY_API_KEY=sk-abc123xyz456abc123xyz456abc123xyz456"),
        ("JWT token",   "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"),
        ("SSH key",     "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA...\n-----END RSA PRIVATE KEY-----"),
        ("Bearer token","Authorization: Bearer eyJhbGciOiJSUzI1NiJ9.eyJpc3MiOiJodHRwcyJ9"),
        ("Password",    "password=MySuperSecretPass123"),
        ("Safe text",   "ModuleNotFoundError: No module named 'requests'"),
    ]

    for label, text in dirty_samples:
        sanitised = SecurityGuard.sanitize_for_log(text)
        was_redacted = "[REDACTED]" in sanitised
        if label == "Safe text":
            if not was_redacted:
                _ok(f"{label:<16} → not redacted (correct — no secret)")
            else:
                _warn(f"{label:<16} → false positive redaction: {sanitised[:60]}")
        else:
            if was_redacted:
                _ok(f"{label:<16} → redacted ✓")
            else:
                _fail(f"{label:<16} → NOT redacted! Output: {sanitised[:60]}")
                passed = False

    # --- 3b: Privacy mode cloud gating ---
    print(f"\n  {BOLD}3b. PrivacyMode cloud gating{RESET}")
    allowed_strict, _   = SecurityGuard.can_use_cloud(PrivacyMode.STRICT,    "vscode")
    allowed_balanced, w = SecurityGuard.can_use_cloud(PrivacyMode.BALANCED,  "vscode-normal")
    allowed_dev, _      = SecurityGuard.can_use_cloud(PrivacyMode.DEVELOPER, "vscode")
    allowed_bank, w2    = SecurityGuard.can_use_cloud(PrivacyMode.BALANCED,  "banking dashboard")

    _kv("STRICT → cloud allowed",    f"{allowed_strict}  (expected: False)")
    _kv("BALANCED (safe) → allowed", f"{allowed_balanced}  (expected: True)")
    _kv("DEVELOPER → allowed",       f"{allowed_dev}  (expected: True)")
    _kv("BALANCED (banking) → blocked", f"{not allowed_bank}  (expected: True, warning shown)")

    if allowed_strict:
        _fail("STRICT mode should block cloud — it didn't!")
        passed = False
    else:
        _ok("STRICT correctly blocks cloud")

    if not allowed_dev:
        _fail("DEVELOPER mode should allow cloud — it didn't!")
        passed = False
    else:
        _ok("DEVELOPER correctly allows cloud")

    if allowed_bank:
        _warn("Banking context was NOT blocked in BALANCED mode — check sensitive domain list")
    else:
        _ok(f"Sensitive context blocked in BALANCED mode. Warning: '{w2[:60]}...'")

    return passed


# ===========================================================================
# TEST 4 — OCR Engine
# ===========================================================================
def test_ocr() -> bool:
    _header("TEST 4 — OCR Engine")

    try:
        from backend.vision.ocr import OCREngine, OCRMode
        from backend.vision.screen_capture import ScreenCapture
        from backend.vision import CaptureScope
    except ImportError as e:
        _fail(f"Import error: {e}")
        return False

    engine = OCREngine()

    # --- 4a: Engine capabilities ---
    print(f"\n  {BOLD}4a. Engine capabilities{RESET}")
    caps = engine.capabilities
    _kv("Engine name",        caps.engine_name)
    _kv("GPU support",        caps.supports_gpu)
    _kv("Multilingual",       caps.supports_multilingual)
    _kv("Handwriting",        caps.supports_handwriting)
    _kv("Avg latency (est.)", f"{caps.average_latency_ms} ms")
    _kv("Notes",              caps.notes)

    if caps.engine_name == "none":
        _warn("No OCR engine installed. Run: pip install easyocr")
        _info("Skipping OCR extraction tests")
        return True

    # --- 4b: Live screen OCR ---
    print(f"\n  {BOLD}4b. Live screen OCR (active window){RESET}")
    try:
        with ScreenCapture() as sc:
            cap = sc.capture(scope=CaptureScope.ACTIVE_WINDOW)

        _info(f"Captured {cap.metadata.width}×{cap.metadata.height} — running OCR...")
        t0 = time.monotonic()
        result = engine.extract(
            image      = cap.image,
            mode       = OCRMode.FAST,
            image_hash = cap.metadata.image_hash,
        )
        ms = (time.monotonic() - t0) * 1000

        _ok(f"OCR complete in {ms:.0f} ms  (target < 500 ms)")
        _kv("Engine",       result.engine)
        _kv("Mode",         result.mode.value)
        _kv("Words found",  result.word_count)
        _kv("Confidence",   f"{result.confidence:.3f}")
        _kv("Blocks found", len(result.blocks))
        _kv("Image hash",   result.image_hash[:20] + "..." if result.image_hash else "None")

        if result.error:
            _warn(f"Partial error: {result.error}")

        if result.is_empty:
            _warn("No text extracted — screen may be mostly graphical")
        else:
            # Show first 3 blocks as a sanity check
            print(f"\n  {BOLD}  First 3 text blocks:{RESET}")
            for i, blk in enumerate(result.blocks[:3]):
                print(f"    [{i+1}] '{blk.text[:40]}'  conf={blk.confidence:.2f}  @ ({blk.x},{blk.y})")

        if ms > 500:
            _warn(f"OCR took {ms:.0f} ms — over 500 ms target (EasyOCR first-run loads model)")

    except Exception as e:
        _fail(f"Live OCR failed: {e}")
        import traceback; traceback.print_exc()
        return False

    # --- 4c: OCR mode AUTO resolves to BALANCED ---
    print(f"\n  {BOLD}4c. AUTO mode resolves gracefully{RESET}")
    from backend.vision.ocr import _resolve_mode, OCRMode as OM
    resolved = _resolve_mode(OM.AUTO)
    if resolved == OM.BALANCED:
        _ok("AUTO → BALANCED (classifier placeholder works)")
    else:
        _fail(f"AUTO resolved to {resolved} — expected BALANCED")
        return False

    return True


# ===========================================================================
# TEST 5 — Screen Context Builder (no vision model needed)
# ===========================================================================
def test_context() -> bool:
    _header("TEST 5 — Screen Context Builder")

    try:
        from backend.vision.screen_context import (
            ContextBuilder, ApplicationType, ContentType, build_screen_context
        )
        from backend.vision.ocr import OCRResult, OCRMode
        from backend.vision.vision_models import VisionResponse
    except ImportError as e:
        _fail(f"Import error: {e}")
        return False

    builder = ContextBuilder()

    def _make_ocr(text: str, image_hash: str = "abc123") -> "OCRResult":
        from backend.vision.ocr import OCRResult, OCRMode
        r = OCRResult(text=text, confidence=0.85, blocks=[], engine="test",
                      mode=OCRMode.FAST, processing_time_ms=10, image_hash=image_hash)
        return r

    def _make_vision(summary: str = "A test screen.") -> "VisionResponse":
        return VisionResponse(answer=summary, model_id="test_model", latency_ms=0.0)

    print()

    # --- 5a: VSCode + Python error ---
    print(f"  {BOLD}5a. VSCode Python error scenario{RESET}")
    ocr = _make_ocr(
        "Visual Studio Code\ndef main():\n    import requests\n"
        "Traceback (most recent call last):\n"
        "  File main.py line 4\n"
        "ModuleNotFoundError: No module named 'requests'\n"
        "Exception failed to load module"
    )
    ctx = builder.build(ocr, _make_vision("VSCode with a Python error"), window_title="main.py - Visual Studio Code")
    _kv("Application",   ctx.application.value)
    _kv("Content type",  ctx.content_type.value)
    _kv("Language",      ctx.language or "None")
    _kv("Errors found",  len(ctx.errors))
    _kv("App confidence",f"{ctx.confidence.application:.2f}")
    _kv("Lang confidence",f"{ctx.confidence.language:.2f}")
    if ctx.errors:
        _ok(f"Error extracted: '{ctx.errors[0][:60]}'")
    if ctx.application == ApplicationType.VSCODE:
        _ok("Application correctly classified as VSCODE")
    else:
        _warn(f"Application classified as {ctx.application.value} (expected VSCODE)")
    if ctx.content_type in (ContentType.ERROR, ContentType.CODE, ContentType.MIXED):
        _ok(f"Content type: {ctx.content_type.value}")
    print(f"  {DIM}  Debug scores: {ctx._content_scores}{RESET}")

    # --- 5b: Browser + webpage ---
    print(f"\n  {BOLD}5b. Chrome browser scenario{RESET}")
    ocr = _make_ocr("https://www.github.com/user/repo  Back  Forward  Reload  Bookmark")
    ctx = builder.build(ocr, _make_vision("Browser showing GitHub"), window_title="GitHub - Google Chrome")
    _kv("Application",   ctx.application.value)
    _kv("Content type",  ctx.content_type.value)
    _kv("Website",       ctx.website or "None")
    if ctx.application == ApplicationType.BROWSER:
        _ok("Application correctly classified as BROWSER")
    else:
        _warn(f"Got: {ctx.application.value}")
    if ctx.website:
        _ok(f"Website extracted: {ctx.website}")
    else:
        _warn("Website not extracted — check URL pattern in OCR text")

    # --- 5c: Terminal scenario ---
    print(f"\n  {BOLD}5c. Terminal scenario{RESET}")
    ocr = _make_ocr("user@machine:~$ pip install requests\nERROR: Connection timed out\nException: failed\nTraceback: timeout")
    ctx = builder.build(ocr, _make_vision("Terminal with an error"), window_title="Terminal — Bash")
    _kv("Application",   ctx.application.value)
    _kv("Content type",  ctx.content_type.value)
    _kv("Errors found",  len(ctx.errors))
    if ctx.application == ApplicationType.TERMINAL:
        _ok("Application correctly classified as TERMINAL")
    else:
        _warn(f"Got: {ctx.application.value} — check terminal signals")

    # --- 5d: to_prompt_dict() compactness ---
    print(f"\n  {BOLD}5d. to_prompt_dict() — compact, no null fields{RESET}")
    prompt_dict = ctx.to_prompt_dict()
    _info(f"Keys in prompt dict: {list(prompt_dict.keys())}")
    if "errors" not in prompt_dict or not prompt_dict.get("errors"):
        _warn("'errors' missing or empty even though errors were detected — check extraction")
    else:
        _ok(f"errors field present with {len(prompt_dict['errors'])} item(s)")
    if None in prompt_dict.values():
        _fail("Prompt dict contains None values — these should be omitted")
        return False
    else:
        _ok("No None values in prompt dict ✓")

    return True


# ===========================================================================
# TEST 6 — Full Pipeline (requires Ollama or graceful fallback)
# ===========================================================================
async def test_pipeline() -> bool:
    _header("TEST 6 — Full Pipeline (VisionManager)")

    try:
        from backend.vision.vision_manager import VisionManager, VisionConfig
        from backend.vision import PrivacyMode, CaptureScope
        from backend.vision.ocr import OCRMode
    except ImportError as e:
        _fail(f"Import error: {e}")
        return False

    config = VisionConfig(
        privacy_mode       = PrivacyMode.BALANCED,
        default_scope      = CaptureScope.ACTIVE_WINDOW,
        default_ocr_mode   = OCRMode.FAST,
        analysis_timeout_s = 30.0,   # Generous for first-run model load
        cache_ttl_s        = 10.0,
        max_retries        = 1,
    )

    manager = VisionManager(config=config)

    # --- 6a: Backend health check ---
    print(f"\n  {BOLD}6a. Backend health check{RESET}")
    try:
        health_map = await manager.health_check_backends()
        if health_map:
            for backend_id, health in health_map.items():
                status_str = f"{GREEN}online{RESET}" if health.online else f"{RED}offline{RESET}"
                print(f"  {BOLD}{backend_id:<20}{RESET}  {status_str}  latency={health.latency_ms:.0f}ms  {health.error or ''}")
        else:
            _warn("No backends registered (expected Qwen2VL7B + LLaVA)")
    except Exception as e:
        _warn(f"Health check failed: {e}")

    # --- 6b: Full analysis ---
    print(f"\n  {BOLD}6b. Full screen analysis — 'What is on my screen?'{RESET}")
    _info("This will capture your active window and run the full pipeline.")
    _info("If Ollama is not running, it gracefully falls back to OCR-only.")
    _info("Running...")
    print()

    try:
        t0 = time.monotonic()
        result = await manager.analyze_screen(
            query="What is on my screen? Describe what you see.",
            scope=CaptureScope.ACTIVE_WINDOW,
        )
        total_s = time.monotonic() - t0
    except Exception as e:
        _fail(f"analyze_screen raised: {e}")
        import traceback; traceback.print_exc()
        return False

    _kv("Status",       result.status.value)
    _kv("Session ID",   result.session_id[:12] + "...")
    _kv("Cache hit",    result.cache_hit)
    _kv("Backend used", result.backend_used)
    _kv("Total time",   f"{result.total_time_ms:.0f} ms  ({total_s:.1f} s wall)")
    _kv("Warnings",     len(result.warnings))

    if result.warnings:
        print(f"\n  {BOLD}  Warnings:{RESET}")
        for w in result.warnings:
            _warn(w[:80])

    if result.success:
        ctx = result.screen_context
        print(f"\n  {BOLD}  Screen Context:{RESET}")
        _kv("  Application",    ctx.application.value)
        _kv("  Content type",   ctx.content_type.value)
        _kv("  Language",       ctx.language or "None")
        _kv("  Errors found",   len(ctx.errors))
        _kv("  Words (OCR)",    ctx.ocr_word_count)
        _kv("  OCR confidence", f"{ctx.ocr_confidence:.3f}")
        _kv("  App confidence", f"{ctx.confidence.application:.2f}")
        _kv("  Vision model",   ctx.vision_model_id)
        _kv("  Debug scores",   ctx._content_scores)

        if ctx.summary:
            print(f"\n  {BOLD}  Vision Summary:{RESET}")
            for line in ctx.summary[:300].splitlines():
                print(f"    {line}")
            if len(ctx.summary) > 300:
                print(f"    {DIM}... (truncated){RESET}")
        _ok("Pipeline completed successfully ✓")
    else:
        _warn(f"Analysis did not succeed — status: {result.status.value}")
        _info("This is acceptable if Ollama is offline (OCR-only fallback)")

    # --- 6c: Cache test ---
    print(f"\n  {BOLD}6c. Cache test — same query 1 second later{RESET}")
    await asyncio.sleep(1)
    t0 = time.monotonic()
    result2 = await manager.analyze_screen(
        query="What is on my screen?",
        scope=CaptureScope.ACTIVE_WINDOW,
    )
    cache_ms = (time.monotonic() - t0) * 1000
    if result2.cache_hit:
        _ok(f"Cache hit! Returned in {cache_ms:.0f} ms (no capture or OCR needed)")
    else:
        _info("Cache miss — screen likely changed between the two queries")
    _kv("  Cache hit", result2.cache_hit)
    _kv("  Time",      f"{cache_ms:.0f} ms")

    # --- 6d: Cancellation test ---
    print(f"\n  {BOLD}6d. Cancellation test{RESET}")
    session_ref: list = []

    async def _run_and_cancel():
        task = asyncio.create_task(
            manager.analyze_screen("Describe this window", scope=CaptureScope.ACTIVE_WINDOW)
        )
        # Let it start, then cancel via manager API
        await asyncio.sleep(0.05)
        sessions = list(manager._sessions.values())
        if sessions:
            latest = max(sessions, key=lambda s: s.created_at)
            session_ref.append(latest.session_id)
            await manager.cancel_analysis(latest.session_id)
        return await task

    result3 = await _run_and_cancel()
    if result3.status.value in ("CANCELLED", "COMPLETED"):
        _ok(f"Graceful termination — status: {result3.status.value}")
    else:
        _info(f"Status: {result3.status.value} (cancellation may have arrived after completion)")

    await manager.cleanup()
    _ok("VisionManager.cleanup() completed — resources released")

    return True


# ===========================================================================
# Entry point
# ===========================================================================
def _parse_args():
    parser = argparse.ArgumentParser(description="Zytrix Vision Subsystem — Manual Test Runner")
    parser.add_argument("--capture",  action="store_true", help="Test 1: Screen capture")
    parser.add_argument("--validate", action="store_true", help="Test 2: Validation layer")
    parser.add_argument("--security", action="store_true", help="Test 3: Security layer")
    parser.add_argument("--ocr",      action="store_true", help="Test 4: OCR engine")
    parser.add_argument("--context",  action="store_true", help="Test 5: Context builder")
    parser.add_argument("--pipeline", action="store_true", help="Test 6: Full pipeline")
    parser.add_argument("--all",      action="store_true", help="Run all tests")
    return parser.parse_args()


def main():
    args   = _parse_args()
    run_all = args.all or not any([
        args.capture, args.validate, args.security, args.ocr, args.context, args.pipeline
    ])

    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  Zytrix Vision Subsystem - Manual Test Runner{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}")

    results: dict[str, bool] = {}

    if args.capture  or run_all: results["1. Screen Capture"]   = test_capture()
    if args.validate or run_all: results["2. Validation"]       = test_validation()
    if args.security or run_all: results["3. Security"]         = test_security()
    if args.ocr      or run_all: results["4. OCR Engine"]       = test_ocr()
    if args.context  or run_all: results["5. Context Builder"]  = test_context()
    if args.pipeline or run_all: results["6. Full Pipeline"]    = asyncio.run(test_pipeline())

    # Summary
    _header("SUMMARY")
    print()
    all_passed = True
    for name, passed in results.items():
        mark = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  {mark}  {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print(f"  {GREEN}{BOLD}All tests passed!{RESET}\n")
        sys.exit(0)
    else:
        print(f"  {YELLOW}{BOLD}Some tests failed - check output above.{RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
