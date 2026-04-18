import webbrowser
import urllib.parse
import logging

logger = logging.getLogger("JARVIS.WebSearch")

def google_search(query: str):
    """
    Takes a spoken query and opens the default web browser directly to the Google search results.
    
    Why we built it this way:
    Instead of using heavy libraries like Selenium which actually "click" buttons on a screen invisibly,
    we can use Python's built-in `webbrowser` library. By knowing how the Google Search URL works 
    (https://www.google.com/search?q=YOUR_QUERY), we can dynamically build the URL string and just 
    open it! 
    
    The 'urllib.parse.quote_plus' function ensures spaces in your spoken sentence are safely converted 
    to '+' signs, which is how URLs read spaces.
    """
    logger.info(f"Searching google for: {query}")
    safe_query = urllib.parse.quote_plus(query)
    webbrowser.open(f"https://www.google.com/search?q={safe_query}")

def play_youtube(video_name: str):
    """
    Searches YouTube for the requested video explicitly.
    """
    logger.info(f"Searching YouTube for: {video_name}")
    safe_query = urllib.parse.quote_plus(video_name)
    webbrowser.open(f"https://www.youtube.com/results?search_query={safe_query}")
