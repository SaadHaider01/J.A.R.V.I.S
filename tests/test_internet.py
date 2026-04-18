import pytest
from unittest.mock import patch, MagicMock
from backend.internet.weather import get_current_weather
from backend.internet.news import get_latest_news

@patch('backend.internet.weather.OPENWEATHERMAP_API_KEY', 'fake_key')
@patch('backend.internet.weather.requests.get')
def test_get_current_weather_success(mock_get):
    # Mock successful response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "cod": 200,
        "weather": [{"description": "clear sky"}],
        "main": {"temp": 25.5}
    }
    mock_get.return_value = mock_response

    result = get_current_weather("London")
    assert "The current weather in London is 25.5 degrees with clear sky." in result

@patch('backend.internet.weather.OPENWEATHERMAP_API_KEY', 'fake_key')
@patch('backend.internet.weather.requests.get')
def test_get_current_weather_not_found(mock_get):
    # Mock city not found
    mock_response = MagicMock()
    mock_response.json.return_value = {"cod": 404}
    mock_get.return_value = mock_response

    result = get_current_weather("InvalidCity")
    assert "Sorry, I couldn't find the weather for InvalidCity." in result

@patch('backend.internet.news.NEWS_API_KEY', 'fake_key')
@patch('backend.internet.news.requests.get')
def test_get_latest_news_success(mock_get):
    # Mock successful news response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "status": "ok",
        "articles": [
            {"title": "Headline 1"},
            {"title": "Headline 2"},
            {"title": "Headline 3"}
        ]
    }
    mock_get.return_value = mock_response

    result = get_latest_news()
    assert "Here are the top headlines right now." in result
    assert "Headline 1" in result
    assert "Headline 2" in result
    assert "Headline 3" in result
