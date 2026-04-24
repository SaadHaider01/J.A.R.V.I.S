import logging

logger = logging.getLogger("JARVIS.AudioControl")


def _get_volume_interface():
    """
    Gets the Windows Core Audio API volume interface via pycaw.

    Why pycaw? Windows doesn't expose volume control through any simple command.
    We have to go through COM (Component Object Model) — the low-level interface
    that Windows apps use to talk to each other. pycaw wraps this so we don't
    have to write C++ code to change the volume.
    """
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(interface, POINTER(IAudioEndpointVolume))


def set_volume(level: int) -> bool:
    """
    Sets the master system volume to a percentage (0-100).

    The Windows audio API works on a logarithmic decibel scale internally,
    but pycaw's SetMasterVolumeLevelScalar takes a linear 0.0-1.0 float,
    which maps naturally to our 0-100 percentage.
    """
    level = max(0, min(100, level))
    try:
        volume = _get_volume_interface()
        volume.SetMasterVolumeLevelScalar(level / 100.0, None)
        logger.info(f"Volume set to {level}%")
        return True
    except Exception as e:
        logger.error(f"Failed to set volume: {e}")
        return False


def get_volume() -> int:
    """Returns the current master volume level as a percentage (0-100)."""
    try:
        volume = _get_volume_interface()
        current = round(volume.GetMasterVolumeLevelScalar() * 100)
        logger.info(f"Current volume: {current}%")
        return current
    except Exception as e:
        logger.error(f"Failed to read volume: {e}")
        return -1


def mute_volume() -> bool:
    """Toggles the system mute state."""
    try:
        volume = _get_volume_interface()
        current_mute = volume.GetMute()
        volume.SetMute(not current_mute, None)
        state = "muted" if not current_mute else "unmuted"
        logger.info(f"System {state}")
        return True
    except Exception as e:
        logger.error(f"Failed to toggle mute: {e}")
        return False


def get_mute_status() -> bool:
    """Returns True if the system is currently muted."""
    try:
        volume = _get_volume_interface()
        return bool(volume.GetMute())
    except Exception as e:
        logger.error(f"Failed to check mute status: {e}")
        return False
