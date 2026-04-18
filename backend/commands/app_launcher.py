import os
import logging
import platform

logger = logging.getLogger("JARVIS.AppLauncher")

# A dictionary mapping spoken app names to their Windows executable commands
app_paths = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "command prompt": "start cmd",
    "browser": "start msedge" # or "start chrome"
}

def launch_app(app_name: str) -> bool:
    """
    Attempts to launch an application based on the spoken name.
    
    Why we built it this way: 
    Instead of complex fuzzy string matching, we use a simple Python Dictionary ('app_paths').
    If the NLP Brain detects the 'open application' intent and the Entity Extractor pulls out 'notepad',
    we just look it up in the dictionary and pass the native command straight to the operating system!
    """
    app_name = app_name.lower().strip()
    
    if app_name in app_paths:
        logger.info(f"Commanding OS to launch {app_name}...")
        # os.system() is the bridge between Python and your PC's Command Line
        os.system(app_paths[app_name])
        return True
        
    logger.warning(f"Could not find application: {app_name} in our dictionary.")
    return False
