import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("JARVIS.Display")


def set_brightness(level: int) -> bool:
    """
    Sets the screen brightness to a value between 0 and 100.
    Uses the screen-brightness-control library which talks to WMI
    (Windows Management Instrumentation) under the hood.
    """
    level = max(0, min(100, level))
    try:
        import screen_brightness_control as sbc
        sbc.set_brightness(level)
        logger.info(f"Brightness set to {level}%")
        return True
    except Exception as e:
        logger.error(f"Failed to set brightness: {e}")
        return False


def get_brightness() -> int:
    """Returns the current screen brightness level (0-100)."""
    try:
        import screen_brightness_control as sbc
        brightness = sbc.get_brightness()
        # sbc.get_brightness() returns a list (one per monitor), take the first
        current = brightness[0] if isinstance(brightness, list) else brightness
        logger.info(f"Current brightness: {current}%")
        return current
    except Exception as e:
        logger.error(f"Failed to read brightness: {e}")
        return -1


def take_screenshot(filename: str = "") -> str:
    """
    Captures a full screenshot and saves it to the user's Pictures folder.
    Returns the file path of the saved screenshot.
    """
    try:
        import pyautogui

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"

        save_dir = Path.home() / "Pictures" / "JARVIS_Screenshots"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(save_dir / filename)

        screenshot = pyautogui.screenshot()
        screenshot.save(save_path)
        logger.info(f"Screenshot saved: {save_path}")
        return save_path
    except Exception as e:
        logger.error(f"Failed to take screenshot: {e}")
        return ""
