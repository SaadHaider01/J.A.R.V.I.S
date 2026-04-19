import time
import logging
from config import PLAYWRIGHT_BRAVE_PATH

# We use sync_playwright since the agent loops synchronously
from playwright.sync_api import sync_playwright, Playwright, Browser, Page

logger = logging.getLogger("JARVIS.Browser")

# Persistent playwright session globals
_playwright: Playwright | None = None
_browser: Browser | None = None
_page: Page | None = None

def _get_page(browser_name: str = "brave") -> Page:
    """Gets the active Playwright page, tearing down/booting up browsers if needed."""
    global _playwright, _browser, _page
    
    if _page is None or _page.is_closed():
        if _playwright is None:
            _playwright = sync_playwright().start()
            
        kwargs = {"headless": False}
        executable_path = None
        
        # Load custom path for Brave if selected
        if browser_name.lower() == "brave":
            executable_path = PLAYWRIGHT_BRAVE_PATH

        if executable_path:
            kwargs["executable_path"] = executable_path
            
        try:
            logger.info(f"Launching Playwright controlled browser: {browser_name.upper()}")
            _browser = _playwright.chromium.launch(**kwargs)
        except Exception as e:
            logger.error(f"Failed to launch browser '{browser_name}'. Falling back to default chromium: {e}")
            _browser = _playwright.chromium.launch(headless=False)
            
        _page = _browser.new_page()
    return _page

def open_browser(url: str, browser_name: str = "brave") -> bool:
    """Launches the specified browser and navigates to the URL."""
    try:
        from backend.commands.cancel_handler import check_cancelled
        page = _get_page(browser_name)
        if check_cancelled(): return False
        
        logger.info(f"Navigating to: {url}")
        page.goto(url)
        return True
    except Exception as e:
        logger.error(f"open_browser failed: {e}")
        return False

def search_web(query: str) -> bool:
    """Performs a Google search using Playwright."""
    return open_browser(f"https://www.google.com/search?q={query}")

def search_youtube(query: str) -> bool:
    """Performs a YouTube search using Playwright."""
    return open_browser(f"https://www.youtube.com/results?search_query={query}")

def click_element(selector: str) -> bool:
    """Plays out a physical click on the specified CSS/XPath selector."""
    page = _get_page()
    try:
        logger.info(f"Clicking element: {selector}")
        page.locator(selector).click(timeout=5000)
        return True
    except Exception as e:
        logger.error(f"Failed to click element '{selector}': {e}")
        return False

def fill_form(selector: str, text: str) -> bool:
    """Submits text into a form or input field."""
    page = _get_page()
    try:
        logger.info(f"Filling form {selector} with data...")
        page.locator(selector).fill(text, timeout=5000)
        return True
    except Exception as e:
        logger.error(f"Failed to fill element '{selector}': {e}")
        return False

def scroll_page(direction: str, amount: int = 500) -> bool:
    """Scrolls the current open web page dynamically."""
    page = _get_page()
    try:
        if direction.lower() == "up":
            amount = -amount
        logger.info(f"Scrolling page {direction} by {amount}px")
        page.mouse.wheel(0, amount)
        return True
    except Exception as e:
        logger.error(f"Failed to scroll: {e}")
        return False

def switch_browser(browser_name: str) -> bool:
    """Closes the current tracking session and spins up a new target browser."""
    global _playwright, _browser, _page
    if _page:
        _page.close()
    if _browser:
        _browser.close()
    
    _page = None
    _browser = None
    
    _get_page(browser_name)
    return True

def stop_browser():
    """Teardown method for the Playwright subsystem."""
    global _playwright, _browser, _page
    try:
        if _page: _page.close()
        if _browser: _browser.close()
        if _playwright: _playwright.stop()
    except:
        pass
    finally:
        _page = None
        _browser = None
        _playwright = None
