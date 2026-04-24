import logging
import time
import pyautogui

logger = logging.getLogger("JARVIS.Shortcuts")

# Disable pyautogui's failsafe pause for instant execution
pyautogui.PAUSE = 0.1


def lock_screen() -> bool:
    """Locks the PC screen (Win+L)."""
    try:
        pyautogui.hotkey('win', 'l')
        logger.info("Screen locked")
        return True
    except Exception as e:
        logger.error(f"Failed to lock screen: {e}")
        return False


def show_desktop() -> bool:
    """Minimizes all windows and shows the desktop (Win+D)."""
    try:
        pyautogui.hotkey('win', 'd')
        logger.info("Show desktop triggered")
        return True
    except Exception as e:
        logger.error(f"Failed to show desktop: {e}")
        return False


def take_screenshot_snip() -> bool:
    """Opens the Windows Snipping Tool (Win+Shift+S)."""
    try:
        pyautogui.hotkey('win', 'shift', 's')
        logger.info("Snipping tool opened")
        return True
    except Exception as e:
        logger.error(f"Failed to open snipping tool: {e}")
        return False


def open_task_manager() -> bool:
    """Opens Windows Task Manager (Ctrl+Shift+Esc)."""
    try:
        pyautogui.hotkey('ctrl', 'shift', 'escape')
        logger.info("Task Manager opened")
        return True
    except Exception as e:
        logger.error(f"Failed to open Task Manager: {e}")
        return False


def virtual_desktop_new() -> bool:
    """Creates a new virtual desktop (Win+Ctrl+D)."""
    try:
        pyautogui.hotkey('win', 'ctrl', 'd')
        logger.info("New virtual desktop created")
        return True
    except Exception as e:
        logger.error(f"Failed to create virtual desktop: {e}")
        return False


def virtual_desktop_switch(direction: str = "right") -> bool:
    """
    Switches to the next virtual desktop.
    direction: 'left' or 'right' (default: right).
    Uses Win+Ctrl+Left/Right Arrow.
    """
    try:
        arrow = "left" if direction.lower() == "left" else "right"
        pyautogui.hotkey('win', 'ctrl', arrow)
        logger.info(f"Switched virtual desktop: {arrow}")
        return True
    except Exception as e:
        logger.error(f"Failed to switch virtual desktop: {e}")
        return False
