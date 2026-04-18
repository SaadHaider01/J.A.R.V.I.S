from decouple import config
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent

# API Keys loaded from .env
OPENWEATHERMAP_API_KEY = config('OPENWEATHERMAP_API_KEY', default='')
NEWS_API_KEY = config('NEWS_API_KEY', default='')

# AI Model Defaults
WHISPER_MODEL_SIZE = "small" # options: tiny, base, small, medium, large
TTS_VOICE = "en-US-AriaNeural" # edge-tts default female voice

# Wake Word Settings
WAKE_WORD_SENSITIVITY = 0.5 # 0.0 to 1.0

# General Settings
DEBUG = True
