import requests
from config import OPENWEATHERMAP_API_KEY
import logging

logger = logging.getLogger("JARVIS.Weather")

def get_current_weather(city: str) -> str:
    """
    Reaches out to the OpenWeatherMap API to get real-time weather data.
    
    Why we built it this way:
    JARVIS cannot magically know the weather. We have to ask a server that does. 
    The 'requests' library acts like a web browser in the background. It goes to the OpenWeatherMap URL,
    hands over your API Key, and the server returns a JSON block (a giant Javascript dictionary).
    We then parse that JSON to pull out exactly the temperature and condition we want, and format it 
    into a nice English sentence for JARVIS to speak out loud via the TTS module!
    """
    if not OPENWEATHERMAP_API_KEY or OPENWEATHERMAP_API_KEY == "your_openweathermap_api_key_here":
        return "Weather API key is missing. Please add it to your environment variables."

    # Construct the API Request URL
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHERMAP_API_KEY}&units=metric"
    
    try:
        # GET data from the URL with a 10-second timeout
        response = requests.get(url, timeout=10).json()
        
        # HTTP Code 200 means SUCCESS
        if response.get("cod") != 200:
            return f"Sorry, I couldn't find the weather for {city}."
            
        weather_description = response["weather"][0]["description"]
        temperature = response["main"]["temp"]
        
        report = f"The current weather in {city} is {temperature} degrees with {weather_description}."
        logger.info(f"Weather fetched: {report}")
        return report
        
    except Exception as e:
        logger.error(f"Weather API error: {e}")
        return "There was an error communicating with the weather service."
