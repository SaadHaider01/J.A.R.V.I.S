import os
import platform
import logging

logger = logging.getLogger("JARVIS.System")

def shutdown_pc():
    """
    Shuts down the Windows PC
    """
    logger.warning("Initiating system shutdown in 30 seconds...")
    if platform.system() == "Windows":
        os.system("shutdown /s /t 30")

def sleep_pc():
    """
    Forces the Windows PC to sleep using native DLL calls.
    
    Why we built it this way:
    Python cannot natively sleep your computer. We have to essentially "trick" Windows 
    into doing it by calling the built-in powrprof.dll (Power Profile dynamically linked library).
    """
    logger.info("Putting system to sleep...")
    if platform.system() == "Windows":
        os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
