"""
demo_vision.py — Interactive Zytrix Screen Understanding Demo
==============================================================

Run this script from your terminal:
    python demo_vision.py

It will give you 3 seconds to switch to any window you want Zytrix to look at.
Then it will capture the screen, run it through the entire pipeline, and
print out exactly what Zytrix's "brain" sees.
"""

import asyncio
import sys
import time
from pathlib import Path

# Ensure J.A.R.V.I.S project root is on the path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.vision.vision_manager import VisionManager, VisionConfig
from backend.vision import PrivacyMode, CaptureScope
from backend.vision.ocr import OCRMode

# Colors for terminal output
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
DIM    = "\033[2m"

async def main():
    print(f"\n{BOLD}{CYAN}👁  Zytrix Vision Subsystem Demo{RESET}\n")
    print("Get ready! Switching to your target window in...")
    
    for i in range(3, 0, -1):
        print(f"  {BOLD}{i}...{RESET}")
        await asyncio.sleep(1)
        
    print(f"\n{BOLD}{MAGENTA}📸 SNAP! Capturing active window now...{RESET}\n")
    
    # Initialize Vision Manager
    config = VisionConfig(
        privacy_mode=PrivacyMode.BALANCED,
        default_scope=CaptureScope.ACTIVE_WINDOW,
        default_ocr_mode=OCRMode.FAST,
        analysis_timeout_s=30.0,
    )
    manager = VisionManager(config=config)
    
    t0 = time.monotonic()
    
    # Run the pipeline
    try:
        result = await manager.analyze_screen(query="Analyze the contents of this window.")
    except Exception as e:
        print(f"\n{YELLOW}⚠️  Error running vision pipeline: {e}{RESET}")
        print("Note: If you see 'BitBlt: Access is denied', you must run this script from an interactive desktop terminal, not a background SSH session.")
        return
        
    total_time = time.monotonic() - t0
    
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  ZYTRIX PERCEPTION REPORT{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}\n")
    
    if not result.success:
        print(f"{YELLOW}Analysis Failed (Status: {result.status.value}){RESET}")
        if result.warnings:
            for w in result.warnings:
                print(f"  - {w}")
        return

    ctx = result.screen_context
    
    # Display the structured context Zytrix built
    print(f"{BOLD}1. CLASSIFICATION{RESET}")
    print(f"  Application:   {GREEN}{ctx.application.name}{RESET} (Confidence: {ctx.confidence.application:.2f})")
    print(f"  Content Type:  {GREEN}{ctx.content_type.name}{RESET} (Confidence: {ctx.confidence.content_type:.2f})")
    if ctx.language:
        print(f"  Language:      {GREEN}{ctx.language}{RESET}")
    if ctx.website:
        print(f"  Website:       {GREEN}{ctx.website}{RESET}")
    print()
    
    print(f"{BOLD}2. OCR & TEXT EXTRACTION{RESET}")
    print(f"  Words Found:   {ctx.ocr_word_count}")
    print(f"  Confidence:    {ctx.ocr_confidence:.2f}")
    if ctx.keywords:
        print(f"  Keywords:      {DIM}{', '.join(ctx.keywords[:10])}{RESET}")
    print()
    
    if ctx.errors:
        print(f"{BOLD}3. ERRORS DETECTED ON SCREEN{RESET}")
        for err in ctx.errors:
            print(f"  {YELLOW}⚠ {err.strip()}{RESET}")
        print()
        
    print(f"{BOLD}4. VISION MODEL SUMMARY ({result.backend_used}){RESET}")
    if ctx.summary:
        for line in ctx.summary.split('\n'):
            print(f"  {DIM}{line}{RESET}")
    else:
        print(f"  {DIM}(No summary generated. If Ollama is offline, OCR-only fallback was used.){RESET}")
    print()
    
    print(f"{BOLD}5. METADATA{RESET}")
    print(f"  Processing:    {total_time:.2f} seconds")
    print(f"  Resolution:    {ctx.resolution}")
    if result.warnings:
        print(f"  Warnings:      {len(result.warnings)}")
        for w in result.warnings:
            print(f"    - {YELLOW}{w}{RESET}")
            
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}\n")
    
    await manager.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDemo cancelled.")
