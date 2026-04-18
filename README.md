# J.A.R.V.I.S 🧠⚡
> **An advanced, locally-hosted AI personal assistant inspired by Tony Stark's JARVIS.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status: Active](https://img.shields.io/badge/Status-Active%20Development-orange.svg)]()

JARVIS is a modular, AI-powered desktop assistant built entirely in Python. Unlike basic voice bots, JARVIS is designed with a deep learning pipeline capable of offline wake-word detection, highly accurate local speech transcription, and natural-sounding voice synthesis.

## ✨ Core Features (In Development)
- **👂 Always-Listening Offline Wake Word:** Zero-latency wake word detection powered by [`openwakeword`](https://github.com/dscripka/openWakeWord). No cloud API keys required, running entirely locally on ONNX runtime.
- **🗣️ Advanced Speech-to-Text (STT):** Powered by OpenAI's `Whisper` model for robust, multi-lingual transcription of user commands.
- **🔊 Natural Voice Synthesis (TTS):** Uses Microsoft's `edge-tts` combined with `pygame` for seamless, natural-sounding audio responses.
- **⚙️ Modular Architecture:** Cleanly separated architecture splitting the Wake Word, STT, TTS, NLP Brain, and System Commands into isolated manageable pipelines.

## 📂 Project Structure
- `backend/wake_word/` - Controls microphone streams and neural inference for the wake word.
- `backend/voice/` - Handles the Whispers STT engine and Edge-TTS voice synthesis.
- `backend/nlp/` - (WIP) The brains of the operation. Intent mapping and entity extraction.
- `backend/commands/` - (WIP) Desktop automation, app launching, and system control.
- `frontend/` - (WIP) PyQt6 powered transparent HUD and waveform visualizations.

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/J.A.R.V.I.S.git
   cd J.A.R.V.I.S
   ```

2. **Install Dependencies**
   It's highly recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**
   Rename or create a `.env` file in the root directory for your third-party API Keys (NewsAPI, OpenWeather, etc.). See `config.py` for variables.

4. **Boot up JARVIS**
   ```bash
   python main.py
   ```

---
*Created as a modular project to bring the Marvel Cinematic Universe to the desktop.*
