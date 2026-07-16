# Zytrix 🧠⚡
> **A real-time, voice-activated AI assistant with total Windows laptop control and persistent long-term memory.**

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status: Operational](https://img.shields.io/badge/Status-Operational-brightgreen.svg)]()
[![LLM](https://img.shields.io/badge/LLM-Llama%203.3%2070B-purple.svg)](https://groq.com)

ZYTRIX is a fully operational, modular AI desktop assistant built in Python. It listens for a custom-trained wake word, understands natural language commands, physically controls your Windows PC using 29+ tools, and naturally remembers user preferences and facts across sessions using a custom local SQLite memory layer.

---

## ✅ What Works Right Now

| Capability | Implementation |
|---|---|
| 🎙️ Custom Wake Word | `openWakeWord` + custom `.onnx` model (97%+ confidence) |
| 🗣️ Speech-to-Text | OpenAI `Whisper` (small.en) with VAD auto-stop |
| 🔊 Voice Synthesis | Microsoft Edge TTS — British male voice |
| 🧠 AI Brain | Groq `llama-3.3-70b-versatile` with full tool calling |
| ⚡ Fast-Track Intents | Local regex layer for instant volume/brightness/folder commands |
| 💾 Persistent Memory | Intelligent SQL-based memory classifier (Facts, Projects, Preferences) & session summarization |
| 🖥️ App Control | Launch, terminate, and type into any Windows application |
| 🔊 Volume & Audio | Set level, mute/unmute via Windows Core Audio API |
| 🌐 Web Search | Real-time DuckDuckGo search results |
| 📁 File Management | Create files/folders, including on Desktop |
| 📋 Clipboard | Read and write system clipboard |
| 💡 Brightness | Screen brightness control via `screen-brightness-control` |
| 🔒 System Actions | Lock screen, show desktop, task manager, virtual desktops |
| 📊 System Info | CPU, RAM, battery, Wi-Fi, IP address, running processes |
| 🔔 Notifications | Windows 10/11 toast notifications |
| ⌨️ Keyboard Shortcuts | Hotkeys and system keyboard shortcuts |

---

## 📂 Project Structure

```
Zytrix/
├── main.py                      # Entry point — boots all systems
├── config.py                    # All settings: model, voice, API keys, wake word path
├── models/
│   └── zye_triks.onnx           # Custom-trained wake word model
├── backend/
│   ├── memory/                  # Long-term memory & conversational context subsystem
│   │   ├── memory_manager.py    # Public API orchestrator (Facade pattern)
│   │   ├── memory_classifier.py # Rule-based knowledge extraction
│   │   ├── memory_repository.py # Core CRUD logic for permanent facts (Repository pattern)
│   │   ├── conversation_store.py# Raw message history storage
│   │   └── summarizer.py        # Condenses raw history into short summaries
│   ├── wake_word/
│   │   ├── detector.py          # openWakeWord inference engine
│   │   └── listener.py          # Mic stream, VAD, conversation loop
│   ├── voice/
│   │   ├── stt.py               # Whisper speech-to-text
│   │   └── tts.py               # Edge-TTS voice synthesis
│   ├── nlp/
│   │   └── agent.py             # Groq LLM brain + 29 tool definitions + fast-track layer
│   └── commands/
│       ├── audio_control.py     # Volume, mute via pycaw
│       ├── display.py           # Brightness control
│       ├── file_manager.py      # File/folder create, delete, move
│       ├── clipboard.py         # Clipboard read/write
│       ├── system_info.py       # CPU, RAM, battery, Wi-Fi
│       ├── network.py           # Wi-Fi and IP control
│       ├── notifications.py     # Windows toast notifications
│       ├── keyboard_shortcuts.py# System hotkeys
│       ├── window_manager.py    # App window focus/close
│       └── system_control.py    # Shutdown, sleep, reboot
├── test_memory_system.py        # Automated test suite for the memory core
├── test_wake_word.py            # Standalone wake word tester
└── requirements.txt
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/Zytrix.git
cd Zytrix
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
OPENWEATHERMAP_API_KEY=your_key_here
NEWS_API_KEY=your_key_here
```

### 4. Add your Wake Word Model
Place your custom `.onnx` wake word model inside the `models/` folder and update `config.py`:
```python
WAKE_WORD_NAME = os.path.join(BASE_DIR, "models", "your_model.onnx")
```

### 5. Boot up ZYTRIX
```bash
python main.py
```

### 6. Run Subsystem Tests
```bash
python test_wake_word.py
python test_memory_system.py
```

---

## ⚙️ Architecture: Hybrid Intent & Memory System

ZYTRIX uses a layered processing architecture to achieve both instantaneous responses for hardware actions and intelligent context-aware planning for complex goals.

```
Voice Input
    │
    ├─► Memory Classifier (Extracts new facts & preferences in background)
    │
    ▼
Fast-Track Layer (Regex — <50ms)
  ├─ "set volume to 40"     → Direct hardware call, instant response
  ├─ "create folder X"      → Direct OS call, instant response
  └─ "close terminal"       → Direct window close, instant response
    │
    ▼ (complex/unknown commands only)
Retrieval-Augmented Generation (RAG)
  └─ Fetches relevant user facts from SQLite to build System Prompt
    │
    ▼
Groq LLM Brain (llama-3.3-70b-versatile)
  └─ Full tool calling with 29 registered tools
```

---

## 🗣️ Example Commands

- *"My favourite IDE is VSCode."* (Will be saved as long-term memory)
- *"Set volume to 60"*
- *"Open Notepad and type hello"*
- *"What's the latest news about AI?"*
- *"Create a folder called Project X on my desktop"*
- *"What's my battery percentage?"*
- *"Lock the screen"*
- *"Kill Spotify"*
- *"Search for the weather in Karachi"*

---

## 🔧 Configuration (`config.py`)

| Setting | Default | Description |
|---|---|---|
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Groq model for tool calling |
| `WAKE_WORD_NAME` | `models/zye_triks.onnx` | Path to custom wake word model |
| `WAKE_WORD_SENSITIVITY` | `0.5` | Detection threshold (0.0–1.0) |
| `WHISPER_MODEL_SIZE` | `small.en` | Whisper model size |
| `TTS_VOICE` | `en-GB-RyanNeural` | Edge-TTS voice |
| `CONVERSATION_TIMEOUT` | `10` | Seconds to wait after speaking |

---

*Built to bring the Marvel Cinematic Universe to the desktop — one tool call at a time.*
