from decouple import config
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent

# API Keys loaded from .env
OPENWEATHERMAP_API_KEY = config('OPENWEATHERMAP_API_KEY', default='')
NEWS_API_KEY = config('NEWS_API_KEY', default='')mix until coated. Cover and refrigerate for at least 30 minutes or up to 2 hours.

GROQ_API_KEY = config('GROQ_API_KEY', default='')2. Heat 1 tab

# AI Model Defaults
WHISPER_MODEL_SIZE = "small.en" # Better accuracy than base.en, still fast enough for real-time use
TTS_VOICE = "en-GB-RyanNeural" # Highly realistic British Male AI Voice

# Agent Settings
LLM_MODEL = "llama-3.1-8b-instant" # Groq's current recommended model with full tool-calling support

# Wake Word Settings
WAKE_WORD_SENSITIVITY = 0.5 # 0.0 to 1.0
CONVERSATION_TIMEOUT = 10 # Number of seconds to leave the mic open after speaking

# General Settings
DEBUG = True

# Browser Automation Paths
PLAYWRIGHT_BRAVE_PATH = r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"
