import pygetwindow as gw
import logging
import pyautogui
import time

logger = logging.getLogger("JARVIS.WindowManager")

def _find_window(app_name: str):
    app_name_lower = app_name.lower()
    windows = gw.getAllTitles()
    # Try exact match first
    for title in windows:
        if title and app_name_lower == title.lower():
            return gw.getWindowsWithTitle(title)[0]
    # Fallback to fuzzy match
    for title in windows:
        if title and app_name_lower in title.lower():
            return gw.getWindowsWithTitle(title)[0]
    return None

def focus_window(app_name: str) -> bool:
    """Brings any open app window to the front."""
    logger.info(f"Attempting to focus window: {app_name}")
    win = _find_window(app_name)
    if win:
        try:
            if win.isMinimized:
                win.restore()
            time.sleep(0.1)
            win.activate()
            return True
        except Exception as e:
            logger.error(f"Failed to focus window: {e}")
            return False
    logger.warning(f"Window '{app_name}' not found.")
    return False

def minimize_window(app_name: str) -> bool:
    """Minimizes a specific app."""
    win = _find_window(app_name)
    if win:
        try:
            if not win.isMinimized:
                win.minimize()
            return True
        except Exception as e:
            logger.error(f"Failed to minimize window: {e}")
            return False
    return False

def maximize_window(app_name: str) -> bool:
    """Maximizes a specific app."""
    win = _find_window(app_name)
    if win:
        try:
            if win.isMinimized:
                win.restore()
                time.sleep(0.1)
            if not win.isMaximized:
                win.maximize()
            return True
        except Exception as e:
            logger.error(f"Failed to maximize window: {e}")
            return False
    return False

def close_window(app_name: str) -> bool:
    """Closes any app by name."""
    win = _find_window(app_name)
    if win:
        try:
            win.close()
            return True
        except Exception as e:
            logger.error(f"Failed to close window: {e}")
            return False
    return False

def switch_window() -> bool:
    """Simulates Alt+Tab to cycle between open windows."""
    try:
        pyautogui.hotkey('alt', 'tab')
        time.sleep(0.5)
        return True
    except Exception as e:
        logger.error(f"Failed to Alt+Tab: {e}")
        return False

def list_open_windows() -> list[str]:
    """Returns all currently open applications."""
    try:
        titles = [t for t in gw.getAllTitles() if t.strip()]
        # Filter out random invisible background windows that Windows creates
        allowed_titles = [t for t in list(set(titles)) if t not in ('Desktop', 'Program Manager')]
        return allowed_titles
    except Exception as e:
        logger.error(f"Failed to list open windows: {e}")
        return []
