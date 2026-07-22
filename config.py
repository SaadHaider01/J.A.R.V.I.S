from decouple import RepositoryEnv
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent

# Load API Keys EXCLUSIVELY from .env file (bypasses OS environment variables
# which can override .env and cause stale/incorrect API keys to be used).
_env_file = RepositoryEnv(BASE_DIR / '.env')
def _get(key, default=''):
    try:
        return _env_file[key]
    except Exception:
        return default

OPENWEATHERMAP_API_KEY = _get('OPENWEATHERMAP_API_KEY')
NEWS_API_KEY = _get('NEWS_API_KEY')

GROQ_API_KEY = _get('GROQ_API_KEY')

# AI Model Defaults
WHISPER_MODEL_SIZE = "small.en" # Better accuracy than base.en, still fast enough for real-time use
TTS_VOICE = "en-GB-RyanNeural" # Highly realistic British Male AI Voice

# Agent Settings
LLM_MODEL = "llama-3.3-70b-versatile" # Much smarter model, higher accuracy for one-shot tool calling

WAKE_WORD_NAME = os.path.join(BASE_DIR, "models", "zye_triks.onnx")

# Wake Word Settings
WAKE_WORD_SENSITIVITY = 0.5 # 0.0 to 1.0
CONVERSATION_TIMEOUT = 10 # Number of seconds to leave the mic open after speaking

# General Settings
DEBUG = True

# ── Demo Mode ─────────────────────────────────────────────────────────────────
# Default is False (normal mode). Say "activate demo mode" after the wake word
# to enable at runtime, or "deactivate demo mode" to return to normal.
DEMO_MODE = False

# Browser Automation Paths
PLAYWRIGHT_BRAVE_PATH = r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"
