import os
import logging
import subprocess
import webbrowser
import urllib.parse

logger = logging.getLogger("JARVIS.AppLauncher")

# Chrome executable paths to try (most common Windows install locations)
CHROME_PATHS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    os.path.expanduser(r"~\AppData\Local\Google\Chrome\Application\chrome.exe"),
]

def _get_chrome_path() -> str | None:
    """Returns the first valid Chrome executable path found on this machine."""
    for path in CHROME_PATHS:
        if os.path.exists(path):
            return path
    return None

# A dictionary mapping spoken app names to their Windows executable commands
app_paths = {
    "notepad": "start notepad",
    "calculator": "calc.exe",
    "command prompt": "start cmd",
    "chrome": None,          # handled dynamically via _get_chrome_path()
    "google chrome": None,   # alias
    "browser": "start msedge",
    "edge": "start msedge",
    "microsoft edge": "start msedge",
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

    if app_name in ("chrome", "google chrome"):
        chrome = _get_chrome_path()
        if chrome:
            logger.info("Launching Google Chrome...")
            subprocess.Popen([chrome])
            return True
        else:
            logger.warning("Chrome not found. Falling back to default browser.")
            webbrowser.open("about:blank")
            return True

    if app_name in app_paths:
        logger.info("Commanding OS to launch %s...", app_name)
        os.system(app_paths[app_name])
        return True
        
    logger.warning("Could not find application: %s in our dictionary.", app_name)
    return False


def open_url_in_browser(url: str, prefer_chrome: bool = True) -> str:
    """
    Opens a URL directly in Chrome (preferred) or the system default browser.
    Used for Google searches and direct URL navigation.
    """
    if prefer_chrome:
        chrome = _get_chrome_path()
        if chrome:
            logger.info("Opening URL in Chrome: %s", url)
            subprocess.Popen([chrome, url])
            return f"Opened in Chrome: {url}"

    logger.info("Opening URL in default browser: %s", url)
    webbrowser.open(url)
    return f"Opened in browser: {url}"


def search_in_chrome(query: str) -> str:
    """
    Performs a Google search by opening Chrome with the search URL.
    This is what the agent calls when the user says 'search X in Chrome'.
    """
    encoded = urllib.parse.quote_plus(query)
    url = f"https://www.google.com/search?q={encoded}"
    return open_url_in_browser(url, prefer_chrome=True)
