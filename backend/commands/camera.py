import os
import time
import logging
import pyautogui

logger = logging.getLogger("JARVIS.Camera")

def take_selfie():
    """
    Opens the native Windows Camera app and physically clicks the shutter button.
    """
    logger.info("Opening Windows Camera...")
    os.system("start microsoft.windows.camera:")
    
    # Robust Wait: Wait for the Camera app to fully render before pressing Enter
    # We use a try-except fallback in case pygetwindow isn't tracking correctly.
    window_found = False
    try:
        start_time = time.time()
        while time.time() - start_time < 8:
            active_window = pyautogui.getActiveWindow()
            if active_window and "camera" in active_window.title.lower():
                window_found = True
                break
            time.sleep(0.5)
    except Exception as e:
        logger.warning(f"getActiveWindow failed, falling back to static sleep: {e}")
        
    if window_found:
        logger.info("Camera window detected. Letting sensor warm up...")
        time.sleep(1.5) # Give the physical camera sensor an extra 1.5s to warm up
    else:
        logger.info("Fallback: waiting fixed 5 seconds for camera to load.")
        time.sleep(5)
        
    logger.info("Pressing shutter (Enter)...")
    pyautogui.press('enter')
    time.sleep(1.0)
    
    os.system("taskkill /IM WindowsCamera.exe /F 2>nul")
    logger.info("Camera closed after snapping photo.")
    
    # Note: On some Windows versions 'space' or 'enter' works for shutter. 
    # Enter is the primary confirmation key.
    
    return True

def record_video(seconds: int):
    """
    Opens the native Windows Camera app, hits the capture button, waits for the specified
    duration, and hits it again to stop.
    Note: The Windows Camera app must already be in 'Video' mode prior to execution.
    """
    from backend.commands.cancel_handler import check_cancelled
    
    logger.info(f"Opening Windows Camera to record for {seconds} seconds...")
    os.system("start microsoft.windows.camera:")
    
    # Wait for rendering
    window_found = False
    try:
        start_time = time.time()
        while time.time() - start_time < 8:
            if check_cancelled(): return False
            active_window = pyautogui.getActiveWindow()
            if active_window and "camera" in active_window.title.lower():
                window_found = True
                break
            time.sleep(0.5)
    except Exception as e:
        logger.warning(f"getActiveWindow failed: {e}")
        
    if window_found:
        time.sleep(1.5)
    else:
        time.sleep(5)
        
    if check_cancelled(): return False
    
    logger.info("Starting recording...")
    pyautogui.press('enter') # Start video
    
    # Wait the specified duration but allow interrupts
    logger.info(f"Recording for {seconds}s...")
    start_rec = time.time()
    while time.time() - start_rec < seconds:
        if check_cancelled():
            logger.info("Recording cancelled prematurely.")
            break
        time.sleep(0.5)
        
    logger.info("Stopping recording...")
    pyautogui.press('enter') # Stop video
    time.sleep(1.0)
    
    os.system("taskkill /IM WindowsCamera.exe /F 2>nul")
    
    return True
