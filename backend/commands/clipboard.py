import logging

logger = logging.getLogger("JARVIS.Clipboard")


def get_clipboard() -> str:
    """
    Reads the current text content from the Windows clipboard.
    Uses pyperclip which handles the Win32 clipboard API under the hood.
    """
    try:
        import pyperclip
        content = pyperclip.paste()
        logger.info(f"Clipboard read: {len(content)} chars")
        return content if content else "(clipboard is empty)"
    except Exception as e:
        logger.error(f"Failed to read clipboard: {e}")
        return "(failed to read clipboard)"


def set_clipboard(text: str) -> bool:
    """
    Writes text to the Windows clipboard so it's ready to paste anywhere.
    """
    try:
        import pyperclip
        pyperclip.copy(text)
        logger.info(f"Clipboard set: {len(text)} chars")
        return True
    except Exception as e:
        logger.error(f"Failed to set clipboard: {e}")
        return False
