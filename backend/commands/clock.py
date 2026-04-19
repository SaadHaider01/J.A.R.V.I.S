import os
import time
import logging
import pyautogui

logger = logging.getLogger("JARVIS.Clock")

def set_timer(hours: str, minutes: str, seconds: str, label: str = ""):
    """
    Opens the native Windows Clock app and physically types in a new timer using macros.
    Assumes Windows 11 Clock layout.
    """
    logger.info("Opening Windows Clock to Timer tab...")
    os.system("start ms-clock:timer")
    
    window_found = False
    try:
        start_time = time.time()
        while time.time() - start_time < 8:
            active_window = pyautogui.getActiveWindow()
            # The window title is usually "Clock"
            if active_window and "clock" in active_window.title.lower():
                window_found = True
                break
            time.sleep(0.5)
    except Exception as e:
        logger.warning(f"getActiveWindow failed: {e}")
        
    if window_found:
        logger.info("Clock window detected. Waiting 1s for UI to render...")
        time.sleep(1.0)
    else:
        logger.info("Fallback: waiting 4 seconds for Clock app to load.")
        time.sleep(4)
        
    logger.info(f"Typing timer macro: {hours}h {minutes}m {seconds}s")
    
    # Windows 11 Clock Timer Creation Flow:
    # 1. Ctrl + N to open the 'Add New Timer' popup
    pyautogui.hotkey('ctrl', 'n')
    time.sleep(0.8) # Wait for popup animation
    
    # 2. Type hours, tab, minutes, tab, seconds, tab, name, enter
    if hours and str(hours) != "0":
        pyautogui.write(str(hours))
    pyautogui.press('tab')
    
    if minutes and str(minutes) != "0":
        pyautogui.write(str(minutes))
    pyautogui.press('tab')
    
    if seconds and str(seconds) != "0":
        pyautogui.write(str(seconds))
    pyautogui.press('tab')
    
    if label:
        pyautogui.write(str(label))
        
    time.sleep(0.5)
    # Start the timer! In Windows 11 Clock, jumping out of the name field might require an extra Enter
    pyautogui.press('enter') 
    time.sleep(0.5)
    pyautogui.press('enter') # Hit it again to hit "Save/Start" since the first one often just closes the text input context
    
    return True
