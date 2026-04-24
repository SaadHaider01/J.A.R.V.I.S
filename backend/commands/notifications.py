import logging

logger = logging.getLogger("JARVIS.Notifications")


def send_notification(title: str, message: str, duration: int = 5) -> bool:
    """
    Sends a Windows 10/11 toast notification that appears in the
    bottom-right corner of the screen.

    Uses win10toast which wraps the Windows Shell notification API.
    The notification auto-dismisses after `duration` seconds.
    """
    try:
        from win10toast import ToastNotifier
        toaster = ToastNotifier()
        toaster.show_toast(
            title,
            message,
            duration=duration,
            threaded=True  # Non-blocking so JARVIS doesn't freeze
        )
        logger.info(f"Notification sent: '{title}'")
        return True
    except Exception as e:
        logger.error(f"Failed to send notification: {e}")
        return False
