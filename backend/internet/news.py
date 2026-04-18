import requests
from config import NEWS_API_KEY
import logging

logger = logging.getLogger("JARVIS.News")

def get_latest_news() -> str:
    """
    Similar to the Weather module, this uses 'requests' to hit the NewsAPI.
    It deliberately asks for only the top 3 headlines (pageSize=3) so JARVIS doesn't talk forever.
    """
    if not NEWS_API_KEY or NEWS_API_KEY == "your_news_api_key_here":
        return "News API key is missing in the environment variables."

    # Construct the API Request URL for US top headlines
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}&pageSize=3"
    
    try:
        # GET news data with a 10-second timeout
        response = requests.get(url, timeout=10).json()
        
        if response.get("status") != "ok":
            return "Sorry, I couldn't fetch the news at this moment."
            
        articles = response.get("articles", [])
        if not articles:
            return "There are no new top headlines right now."
            
        # We loop through the JSON array and pull out just the title of the articles.
        headlines = [article["title"] for article in articles]
        
        # We stitch the titles together into a clean, spoken English string.
        report = "Here are the top headlines right now. " + ". Next: ".join(headlines)
        logger.info("News fetched successfully.")
        return report
        
    except Exception as e:
        logger.error(f"News API error: {e}")
        return "There was an error communicating with the news service."
