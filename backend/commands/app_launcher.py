import os
import logging
import subprocess
import webbrowser
import urllib.parse
import re

logger = logging.getLogger("JARVIS.AppLauncher")

# Brave executable paths to try
BRAVE_PATHS = [
    r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
    r"C:\Program Files (x86)\BraveSoftware\Brave-Browser\Application\brave.exe",
    os.path.expanduser(r"~\AppData\Local\BraveSoftware\Brave-Browser\Application\brave.exe"),
]

def _get_brave_path() -> str | None:
    """Returns the first valid Brave executable path found on this machine."""
    for path in BRAVE_PATHS:
        if os.path.exists(path):
            return path
    return None

# A dictionary mapping spoken app names to their Windows executable commands
app_paths = {
    "notepad": "start notepad",
    "calculator": "calc.exe",
    "command prompt": "start cmd",
    "brave": None,          # handled dynamically via _get_brave_path()
    "browser": None,        # default to Brave
    "edge": "start msedge",
    "microsoft edge": "start msedge",
    "whatsapp": "start whatsapp:",
    "spotify": "start spotify:",
    "discord": "start discord:",
    "zoom": "start zoom:",
}

_dynamic_apps = {}
_dynamic_apps_loaded = False

def _load_dynamic_apps():
    """Uses Powershell to dynamically discover all installed Start Menu and UWP apps."""
    global _dynamic_apps_loaded, _dynamic_apps
    if _dynamic_apps_loaded:
        return

    try:
        logger.info("Scanning Windows for installed apps...")
        # Get-StartApps retrieves all visible apps in the Windows Start menu alongside their AppUserModelID
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "Get-StartApps"], 
            capture_output=True, 
            text=True, 
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        
        lines = result.stdout.splitlines()
        # The output is a formatted table. We skip the first 3 lines (headers and separators).
        for line in lines[3:]:
            line = line.strip()
            if not line:
                continue
            
            # The columns are separated by 2 or more spaces.
            parts = re.split(r'\s{2,}', line)
            if len(parts) >= 2:
                name = parts[0].strip().lower()
                appid = parts[-1].strip()  # The AppID is always the last column
                _dynamic_apps[name] = appid
                
        _dynamic_apps_loaded = True
        logger.info(f"Found {len(_dynamic_apps)} installed apps dynamically.")
    except Exception as e:
        logger.error(f"Failed to scan installed apps dynamically: {e}")

def launch_app(app_name: str) -> bool:
    """
    Attempts to launch an application based on the spoken name.
    1. Checks explicit shortcuts
    2. Dynamically queries Windows Start apps (UWP & Win32)
    3. Performs fuzzy matching
    4. Falls back to generic system calls
    """
    app_name = app_name.lower().strip()

    if app_name in ("brave", "browser"):
        brave = _get_brave_path()
        if brave:
            logger.info("Launching Brave Browser...")
            subprocess.Popen([brave])
            return True
        else:
            logger.warning("Brave not found. Falling back to default browser.")
            webbrowser.open("about:blank")
            return True

    # 0. Intercept common web apps so they don't fail looking for a local download
    WEB_APPS = {
        "youtube": "https://www.youtube.com",
        "gmail": "https://mail.google.com",
        "netflix": "https://www.netflix.com",
        "facebook": "https://www.facebook.com",
        "twitter": "https://www.twitter.com",
        "instagram": "https://www.instagram.com",
        "google": "https://www.google.com",
    }
    
    if app_name in WEB_APPS:
        logger.info(f"Intercepted web app request for {app_name}, opening in browser.")
        open_url_in_browser(WEB_APPS[app_name], prefer_brave=True)
        return True

    # 1. Check explicit mappings
    if app_name in app_paths and app_paths[app_name]:
        logger.info("Commanding OS to launch %s...", app_name)
        os.system(app_paths[app_name])
        return True
        
    # 2. Check dynamic start menu apps
    _load_dynamic_apps()
    if app_name in _dynamic_apps:
        appid = _dynamic_apps[app_name]
        logger.info(f"Dynamically launching {app_name} via AppUserModelID...")
        subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{appid}"])
        return True
        
    # 3. Fuzzy matching (e.g. "adobe reader" matches "Adobe Acrobat Reader")
    for installed_name, appid in _dynamic_apps.items():
        # Check if the spoken name is a substantial part of the real name, or vice versa
        # Requires at least 3 letters to prevent "app" matching everything
        if len(app_name) > 3 and (app_name in installed_name or installed_name in app_name):
            logger.info(f"Fuzzy matched '{app_name}' to installed app '{installed_name}'. Launching!")
            subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{appid}"])
            return True

    # 4. Attempt to generically start it as a fallback
    logger.info("App not explicitly found, attempting to launch generically.")
    res = os.system(f"start {app_name} 2>nul")
    return res == 0

def open_url_in_browser(url: str, prefer_brave: bool = True) -> str:
    """
    Opens a URL directly in Brave (preferred) or the system default browser.
    Used for Google searches and direct URL navigation.
    """
    if prefer_brave:
        brave = _get_brave_path()
        if brave:
            logger.info("Opening URL in Brave: %s", url)
            subprocess.Popen([brave, url])
            return f"Opened in Brave: {url}"

    logger.info("Opening URL in default browser: %s", url)
    webbrowser.open(url)
    return f"Opened in browser: {url}"


def search_in_browser(query: str) -> str:
    """
    Performs a Google search by opening Brave with the search URL.
    This is what the agent calls when the user says 'search X in browser'.
    """
    encoded = urllib.parse.quote_plus(query)
    url = f"https://www.google.com/search?q={encoded}"
    return open_url_in_browser(url, prefer_brave=True)
