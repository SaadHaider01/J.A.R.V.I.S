import json
import logging
import time
import subprocess
import os
import re
import pyautogui
from groq import Groq
from duckduckgo_search import DDGS
from config import GROQ_API_KEY, LLM_MODEL
from backend.memory.context_manager import ConversationalMemory
from backend.commands.app_launcher import launch_app
from backend.commands.system_control import sleep_pc, shutdown_pc
from backend.commands.camera import take_selfie, record_video
from backend.commands.clock import set_timer
from backend.commands.cancel_handler import trigger_cancellation
from backend.commands.browser import (
    open_browser, search_web, search_youtube, click_element,
    fill_form, scroll_page, switch_browser
)
from backend.commands.screen_pilot import find_and_click, find_and_type
from backend.commands.window_manager import (
    focus_window, minimize_window, close_window,
    switch_window, list_open_windows
)
from backend.commands.file_manager import (
    create_file, delete_file, move_file, copy_file,
    rename_file, search_files, open_file, list_directory,
    create_directory
)
from backend.commands.clipboard import get_clipboard, set_clipboard
from backend.commands.display import set_brightness, get_brightness, take_screenshot
from backend.commands.audio_control import set_volume, get_volume, mute_volume
from backend.commands.network import get_wifi_status, list_wifi_networks, connect_wifi, get_ip_address
from backend.commands.system_info import (
    get_battery_status, get_cpu_usage, get_ram_usage,
    list_processes, kill_process_by_name
)
from backend.commands.notifications import send_notification
from backend.commands.keyboard_shortcuts import (
    lock_screen, show_desktop, take_screenshot_snip,
    open_task_manager, virtual_desktop_new, virtual_desktop_switch
)

logger = logging.getLogger("JARVIS.Agent")

# ─────────────────────────────────────────────────────────────────────────────
# TOOL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "launch_application",
            "description": (
                "Opens or launches an application on the user's Windows computer. "
                "Use this when the user asks to open, start, or launch an app like "
                "Notepad, Calculator, browser, command prompt, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "app_name": {
                        "type": "string",
                        "description": (
                            "The name of the app to open. You can provide any common app name "
                            "(e.g., 'WhatsApp', 'Brave', 'Spotify', 'Word', 'Excel'). "
                            "JARVIS will dynamically search the Windows Start Menu to find it."
                        ),
                    }
                },
                "required": ["app_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": (
                "Physically types text on the user's keyboard into whatever window is "
                "currently focused (e.g., Notepad, a search bar, a text field). "
                "Use this when the user asks to write, type, or enter text somewhere."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The exact text content to type into the active window.",
                    }
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Searches the internet for real-time, live information. Use this for "
                "questions about current events, news, weather, sports scores, stock prices, "
                "or anything that requires up-to-date information beyond your training data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The optimized search query to look up on the web.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sleep_computer",
            "description": "Puts the user's Windows computer to sleep or hibernate mode.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shutdown_computer",
            "description": (
                "Performs a FULL SYSTEM SHUTDOWN of the Windows PC. "
                "ONLY call this when the user explicitly uses the words 'shut down the computer' "
                "or 'turn off the pc'. "
                "Do NOT call this for 'terminate', 'close', 'kill', 'stop', or 'exit'. "
                "NEVER trigger this automatically."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "terminate_process",
            "description": (
                "Closes or kills a running application or terminal window on the user's PC. "
                "Use this when the user says 'terminate', 'close', 'kill', 'stop', or 'exit' "
                "a specific app or terminal. This does NOT shut down the computer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "process_name": {
                        "type": "string",
                        "description": (
                            "The name of the process or window to close. Examples: 'cmd', "
                            "'notepad', 'brave', 'terminal', 'powershell'. "
                            "Use the closest Windows executable name."
                        ),
                    }
                },
                "required": ["process_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "take_selfie",
            "description": (
                "Opens the user's laptop camera and physically takes a picture (selfie). "
                "Use this whenever the user asks you to take a photo, take a picture, or take a selfie."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_video",
            "description": (
                "Opens the laptop camera and records a video for a specified number of seconds. "
                "Use this when the user asks you to record a video, take a video, or film something."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "duration_seconds": {
                        "type": "integer",
                        "description": "How many seconds to record for. Default to 10 if unspecified.",
                    }
                },
                "required": ["duration_seconds"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_action",
            "description": (
                "Instantly interrupts and stops any currently executing physical macro. "
                "Use this specifically when the user says 'stop', 'cancel', 'wait', or 'abort'."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_browser_url",
            "description": "Launches the browser and navigates to a specific URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL to visit.",
                    },
                    "browser_name": {
                        "type": "string",
                        "description": "brave, chrome, edge, or firefox. Defaults to brave.",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_google",
            "description": "Opens the browser and performs a Google search for the query.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_youtube",
            "description": "Opens YouTube and searches for a video.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_click_element",
            "description": (
                "Clicks an HTML element (button, link) in the currently open browser "
                "using a CSS selector."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "CSS selector to click.",
                    }
                },
                "required": ["selector"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_fill_form",
            "description": "Types text into a form field in the browser using a CSS selector.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["selector", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_scroll",
            "description": "Scrolls the currently open web page up or down.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "description": "'down' or 'up'",
                    },
                    "amount": {
                        "type": "integer",
                        "description": "Amount of pixels to scroll. Default 500.",
                    },
                },
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "switch_browser",
            "description": "Switches the active automation session to a different browser.",
            "parameters": {
                "type": "object",
                "properties": {
                    "browser_name": {
                        "type": "string",
                        "description": "brave, chrome, edge, or firefox",
                    }
                },
                "required": ["browser_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click_visual_element",
            "description": (
                "Uses computer vision (OpenCV) to find an image template on the screen "
                "and physically clicks it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_name": {
                        "type": "string",
                        "description": "The filename of the template (e.g. 'settings_icon.png').",
                    }
                },
                "required": ["image_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fill_visual_form",
            "description": "Uses computer vision to find a text field visually and types text into it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_name": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["image_name", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "focus_app_window",
            "description": "Brings an application window to the foreground (e.g., 'notepad', 'chrome').",
            "parameters": {
                "type": "object",
                "properties": {"app_name": {"type": "string"}},
                "required": ["app_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "close_app_window",
            "description": "Closes an active application window.",
            "parameters": {
                "type": "object",
                "properties": {"app_name": {"type": "string"}},
                "required": ["app_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "minimize_app_window",
            "description": "Minimizes an active application window.",
            "parameters": {
                "type": "object",
                "properties": {"app_name": {"type": "string"}},
                "required": ["app_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "switch_active_window",
            "description": "Simulates Alt+Tab to cycle to the next open window.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_open_windows",
            "description": "Retrieves a list of all currently open application windows on the desktop.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_timer_ui",
            "description": (
                "Opens the Windows Clock app and visually sets a timer on the screen. "
                "Use this when the user asks you to set a timer for a specific duration."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "hours": {
                        "type": "string",
                        "description": "The number of hours for the timer (e.g. '1', '0').",
                    },
                    "minutes": {
                        "type": "string",
                        "description": "The number of minutes for the timer (e.g. '15', '0').",
                    },
                    "seconds": {
                        "type": "string",
                        "description": "The number of seconds for the timer (e.g. '30', '0').",
                    },
                    "label": {
                        "type": "string",
                        "description": "Optional name or label for the timer (e.g. 'Pasta').",
                    },
                },
                "required": ["hours", "minutes", "seconds"],
            },
        },
    },
    # ── FILE MANAGEMENT TOOLS ────────────────────────────────────────────────
    {"type": "function", "function": {"name": "create_file", "description": "Creates a new file at a path with optional content.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Full file path to create."}, "content": {"type": "string", "description": "Text content to write. Empty for blank file."}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "delete_file", "description": "Safely deletes a file by sending it to the Recycle Bin.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Full path of the file to delete."}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "move_file", "description": "Moves a file or folder to a new location.", "parameters": {"type": "object", "properties": {"source": {"type": "string"}, "destination": {"type": "string"}}, "required": ["source", "destination"]}}},
    {"type": "function", "function": {"name": "copy_file", "description": "Copies a file or folder to a new location.", "parameters": {"type": "object", "properties": {"source": {"type": "string"}, "destination": {"type": "string"}}, "required": ["source", "destination"]}}},
    {"type": "function", "function": {"name": "rename_file", "description": "Renames a file or folder.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Current full path."}, "new_name": {"type": "string", "description": "New filename (not full path)."}}, "required": ["path", "new_name"]}}},
    {"type": "function", "function": {"name": "search_files", "description": "Searches for files by name in a directory. Returns matching file paths.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Filename or partial name to search for."}, "directory": {"type": "string", "description": "Directory to search in. Defaults to user home."}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "open_file", "description": "Opens a file with its default Windows application.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "list_directory", "description": "Lists all files and folders in a directory.", "parameters": {"type": "object", "properties": {"directory": {"type": "string", "description": "Directory path. Defaults to Desktop."}}, "required": []}}},
    {"type": "function", "function": {"name": "create_directory", "description": "Creates a new folder at the specified path.", "parameters": {"type": "object", "properties": {"dir_path": {"type": "string", "description": "Full path of the folder to create."}}, "required": ["dir_path"]}}},
    # ── CLIPBOARD TOOLS ──────────────────────────────────────────────────────
    {"type": "function", "function": {"name": "get_clipboard", "description": "Reads the current text content from the clipboard.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "set_clipboard", "description": "Copies text to the clipboard so the user can paste it.", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
    # ── DISPLAY TOOLS ────────────────────────────────────────────────────────
    {"type": "function", "function": {"name": "set_brightness", "description": "Sets the screen brightness to a percentage (0-100).", "parameters": {"type": "object", "properties": {"level": {"type": "integer", "description": "Brightness percentage 0-100."}}, "required": ["level"]}}},
    {"type": "function", "function": {"name": "take_screenshot", "description": "Captures a full screenshot and saves it to disk.", "parameters": {"type": "object", "properties": {"filename": {"type": "string", "description": "Optional filename. Auto-generated if empty."}}, "required": []}}},
    # ── AUDIO TOOLS ──────────────────────────────────────────────────────────
    {"type": "function", "function": {"name": "set_volume", "description": "Sets the system master volume to a percentage (0-100).", "parameters": {"type": "object", "properties": {"level": {"type": "integer", "description": "Volume percentage 0-100."}}, "required": ["level"]}}},
    {"type": "function", "function": {"name": "mute_volume", "description": "Toggles the system mute on or off.", "parameters": {"type": "object", "properties": {}}}},
    # ── NETWORK TOOLS ────────────────────────────────────────────────────────
    {"type": "function", "function": {"name": "get_wifi_status", "description": "Returns current WiFi connection status, network name, and signal strength.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "connect_wifi", "description": "Connects to a WiFi network by name.", "parameters": {"type": "object", "properties": {"ssid": {"type": "string", "description": "WiFi network name."}, "password": {"type": "string", "description": "WiFi password. Optional for saved networks."}}, "required": ["ssid"]}}},
    {"type": "function", "function": {"name": "get_ip_address", "description": "Returns the local (LAN) and public (WAN) IP address.", "parameters": {"type": "object", "properties": {}}}},
    # ── SYSTEM INFO TOOLS ────────────────────────────────────────────────────
    {"type": "function", "function": {"name": "get_battery_status", "description": "Returns battery percentage and charging state.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "get_cpu_usage", "description": "Returns the current CPU usage percentage.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "get_ram_usage", "description": "Returns RAM usage: total, used, available, and percentage.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "list_processes", "description": "Returns the top resource-consuming processes on the system.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "kill_process_by_name", "description": "Kills a running process by its name. Use for 'kill chrome', 'end spotify', etc.", "parameters": {"type": "object", "properties": {"name": {"type": "string", "description": "Process name to kill."}}, "required": ["name"]}}},
    # ── NOTIFICATION TOOLS ───────────────────────────────────────────────────
    {"type": "function", "function": {"name": "send_notification", "description": "Shows a Windows toast notification popup.", "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "message": {"type": "string"}}, "required": ["title", "message"]}}},
    # ── KEYBOARD SHORTCUT TOOLS ──────────────────────────────────────────────
    {"type": "function", "function": {"name": "lock_screen", "description": "Locks the PC screen immediately (Win+L).", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "show_desktop", "description": "Minimizes all windows and shows the desktop (Win+D).", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "take_screenshot_snip", "description": "Opens the Windows Snipping Tool for a region screenshot (Win+Shift+S).", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "open_task_manager", "description": "Opens Windows Task Manager.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "virtual_desktop_new", "description": "Creates a new virtual desktop.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "virtual_desktop_switch", "description": "Switches to the next or previous virtual desktop.", "parameters": {"type": "object", "properties": {"direction": {"type": "string", "description": "'left' or 'right'. Defaults to 'right'."}}, "required": []}}},
]

# ─────────────────────────────────────────────────────────────────────────────
# ACTION-ONLY TOOLS
# These tools perform a physical action and need no LLM Round 2 summary.
# We skip the second API call and return a pre-built spoken reply instantly.
# ─────────────────────────────────────────────────────────────────────────────

_ACTION_ONLY_TOOLS = {
    "launch_application",
    "type_text",
    "sleep_computer",
    "shutdown_computer",
    "terminate_process",
    "take_selfie",
    "record_video",
    "set_timer_ui",
    "cancel_action",
    "open_browser_url",
    "search_google",
    "search_youtube",
    "browser_click_element",
    "browser_fill_form",
    "browser_scroll",
    "switch_browser",
    "click_visual_element",
    "fill_visual_form",
    "focus_app_window",
    "close_app_window",
    "minimize_app_window",
    "switch_active_window",
    "get_open_windows",
    # ── New Total Control tools ──
    "create_file",
    "create_directory",
    "delete_file",
    "move_file",
    "copy_file",
    "rename_file",
    "open_file",
    "set_clipboard",
    "set_brightness",
    "take_screenshot",
    "set_volume",
    "mute_volume",
    "connect_wifi",
    "kill_process_by_name",
    "send_notification",
    "lock_screen",
    "show_desktop",
    "take_screenshot_snip",
    "open_task_manager",
    "virtual_desktop_new",
    "virtual_desktop_switch",
}

# ─────────────────────────────────────────────────────────────────────────────
# FORMATTING HINT — injected on retry to break malformed tool call syntax
# ─────────────────────────────────────────────────────────────────────────────

_TOOL_FORMAT_HINT = {
    "role": "system",
    "content": (
        "CRITICAL FORMATTING RULE: You MUST use the standard JSON tool_calls format only. "
        "Do NOT use <function=tool_name{...}> syntax under any circumstances. "
        "Always call tools using the proper OpenAI-compatible tool_calls JSON structure."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# TOOL EXECUTOR
# ─────────────────────────────────────────────────────────────────────────────

def execute_tool(tool_name: str, tool_args: dict) -> tuple[str, str]:
    """
    Executes the physical tool on the user's Windows machine.
    Returns a tuple of:
      (tool_result_for_llm, instant_spoken_reply)
    instant_spoken_reply is non-empty only for action-only tools where we
    can skip Round 2 entirely and reply immediately without a second API call.
    """
    logger.info(f"Executing tool: '{tool_name}' with args: {tool_args}")

    # ── launch_application ────────────────────────────────────────────────────
    if tool_name == "launch_application":
        app = tool_args.get("app_name", "").lower()
        success = launch_app(app)
        if success:
            return f"Successfully launched {app}.", f"Opening {app}."
        else:
            return f"Could not find '{app}'.", f"I could not find an app named {app}."

    # ── type_text ─────────────────────────────────────────────────────────────
    elif tool_name == "type_text":
        text = tool_args.get("text", "")
        time.sleep(0.5)
        pyautogui.write(text, interval=0.04)
        return f"Typed: '{text}'.", "Done. I have typed that for you."

    # ── web_search ────────────────────────────────────────────────────────────
    elif tool_name == "web_search":
        query = tool_args.get("query", "")
        try:
            ddgs = DDGS()
            results = ddgs.text(query, max_results=3)
            if not results:
                return "Web search returned no results.", ""
            combined = " ".join([r["body"] for r in results])
            return f"Live web data for '{query}': {combined}", ""  # Needs LLM Round 2
        except Exception as e:
            return f"Web search failed: {e}", "I had trouble searching the web. Please try again."

    # ── sleep_computer ────────────────────────────────────────────────────────
    elif tool_name == "sleep_computer":
        sleep_pc()
        return "Putting the computer to sleep.", "Putting the computer to sleep. Goodnight."

    # ── shutdown_computer ─────────────────────────────────────────────────────
    elif tool_name == "shutdown_computer":
        shutdown_pc()
        return "Shutting down the computer.", "Shutting down. Goodbye."

    # ── terminate_process ─────────────────────────────────────────────────────
    elif tool_name == "terminate_process":
        process = tool_args.get("process_name", "cmd").lower().strip()
        aliases = {
            "terminal": "cmd.exe",
            "command prompt": "cmd.exe",
            "cmd": "cmd.exe",
            "powershell": "powershell.exe",
            "notepad": "notepad.exe",
            "brave": "brave.exe",
            "browser": "brave.exe",
            "edge": "msedge.exe",
            "chrome": "chrome.exe",
            "firefox": "firefox.exe",
        }
        exe = aliases.get(process, process if process.endswith(".exe") else process + ".exe")
        try:
            result = subprocess.run(
                ["taskkill", "/F", "/IM", exe],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return f"Terminated {exe}.", f"Done. I have closed {process}."
            else:
                return f"No running process named '{exe}'.", f"I could not find a running process named {process}."
        except Exception as e:
            return f"Failed to terminate process: {e}", "I had trouble closing that process."

    # ── open_browser_url ──────────────────────────────────────────────────────
    elif tool_name == "open_browser_url":
        url = tool_args.get("url", "")
        browser_name = tool_args.get("browser_name", "brave")
        open_browser(url, browser_name)
        return f"Opened {browser_name} at {url}.", f"Opening {browser_name}."

    # ── search_google ─────────────────────────────────────────────────────────
    elif tool_name == "search_google":
        query = tool_args.get("query", "")
        search_web(query)
        return f"Googled '{query}'.", f"Searching Google for {query}."

    # ── search_youtube ────────────────────────────────────────────────────────
    elif tool_name == "search_youtube":
        query = tool_args.get("query", "")
        search_youtube(query)
        return f"Searched YouTube for '{query}'.", "Opening YouTube."

    # ── browser_click_element ─────────────────────────────────────────────────
    elif tool_name == "browser_click_element":
        selector = tool_args.get("selector", "")
        success = click_element(selector)
        if success:
            return f"Clicked element '{selector}'.", "Clicking."
        return "Failed to click element.", "I could not click that element."

    # ── browser_fill_form ─────────────────────────────────────────────────────
    elif tool_name == "browser_fill_form":
        selector = tool_args.get("selector", "")
        text = tool_args.get("text", "")
        fill_form(selector, text)
        return f"Filled form field '{selector}' with text.", f"Typing {text}."

    # ── browser_scroll ────────────────────────────────────────────────────────
    elif tool_name == "browser_scroll":
        direction = tool_args.get("direction", "down")
        amount = tool_args.get("amount", 500)
        scroll_page(direction, amount)
        return f"Scrolled {direction} by {amount} pixels.", "Scrolling."

    # ── switch_browser ────────────────────────────────────────────────────────
    elif tool_name == "switch_browser":
        browser_name = tool_args.get("browser_name", "brave")
        switch_browser(browser_name)
        return f"Switched to {browser_name}.", f"Switching to {browser_name}."

    # ── click_visual_element ──────────────────────────────────────────────────
    elif tool_name == "click_visual_element":
        image_name = tool_args.get("image_name", "")
        success = find_and_click(image_name)
        if success:
            return f"Clicked visual element '{image_name}'.", f"Clicking {image_name}."
        return f"Could not find '{image_name}' on screen.", f"I could not find {image_name} on screen."

    # ── fill_visual_form ──────────────────────────────────────────────────────
    elif tool_name == "fill_visual_form":
        image_name = tool_args.get("image_name", "")
        text = tool_args.get("text", "")
        success = find_and_type(image_name, text)
        if success:
            return f"Typed into visual form '{image_name}'.", f"Typing {text}."
        return f"Could not find '{image_name}' on screen.", f"I could not find {image_name}."

    # ── focus_app_window ──────────────────────────────────────────────────────
    elif tool_name == "focus_app_window":
        app = tool_args.get("app_name", "")
        focus_window(app)
        return f"Focused window '{app}'.", f"Bringing {app} to the front."

    # ── close_app_window ──────────────────────────────────────────────────────
    elif tool_name == "close_app_window":
        app = tool_args.get("app_name", "")
        close_window(app)
        return f"Closed window '{app}'.", f"Closing {app}."

    # ── minimize_app_window ───────────────────────────────────────────────────
    elif tool_name == "minimize_app_window":
        app = tool_args.get("app_name", "")
        minimize_window(app)
        return f"Minimized window '{app}'.", f"Minimizing {app}."

    # ── switch_active_window ──────────────────────────────────────────────────
    elif tool_name == "switch_active_window":
        switch_window()
        return "Switched to next window.", "Switching windows."

    # ── get_open_windows ──────────────────────────────────────────────────────
    elif tool_name == "get_open_windows":
        windows = list_open_windows()
        windows_str = ", ".join(windows) if windows else "No windows found."
        return f"Open windows: {windows_str}", ""  # Needs LLM Round 2 to summarize naturally

    # ── take_selfie ───────────────────────────────────────────────────────────
    elif tool_name == "take_selfie":
        take_selfie()
        return "Camera macro triggered successfully.", "Opening camera and taking a picture."

    # ── record_video ──────────────────────────────────────────────────────────
    elif tool_name == "record_video":
        duration = tool_args.get("duration_seconds", 10)
        success = record_video(duration)
        if success:
            return f"Recorded video for {duration} seconds.", f"Recording a video for {duration} seconds."
        return "Video recording was cancelled.", "Video recording cancelled."

    # ── cancel_action ─────────────────────────────────────────────────────────
    elif tool_name == "cancel_action":
        trigger_cancellation()
        return "Cancellation flag raised.", "Cancelling the previous action."

    # ── set_timer_ui ──────────────────────────────────────────────────────────
    elif tool_name == "set_timer_ui":
        h = tool_args.get("hours", "0")
        m = tool_args.get("minutes", "0")
        s = tool_args.get("seconds", "0")
        label = tool_args.get("label", "")
        set_timer(h, m, s, label)
        spoken_parts = []
        if h and h != "0": spoken_parts.append(f"{h} hour{'s' if int(h) > 1 else ''}")
        if m and m != "0": spoken_parts.append(f"{m} minute{'s' if int(m) > 1 else ''}")
        if s and s != "0": spoken_parts.append(f"{s} second{'s' if int(s) > 1 else ''}")
        spoken_duration = " and ".join(spoken_parts) if spoken_parts else "the specified duration"
        return "Timer macro triggered.", f"Setting a timer for {spoken_duration}."

    # ═══════════════════════════════════════════════════════════════════════════
    # NEW TOTAL CONTROL TOOLS
    # ═══════════════════════════════════════════════════════════════════════════

    # ── create_file ───────────────────────────────────────────────────────────
    elif tool_name == "create_file":
        path = tool_args.get("path", "")
        content = tool_args.get("content", "")
        if create_file(path, content):
            return f"Created file at {path}.", f"File created at {path}."
        return f"Failed to create file at {path}.", f"I could not create the file."

    # ── create_directory ──────────────────────────────────────────────────────
    elif tool_name == "create_directory":
        path = tool_args.get("dir_path", "")
        if create_directory(path):
            return f"Created directory at {path}.", f"Folder created at {path}."
        return f"Failed to create directory at {path}.", "I could not create the folder."

    # ── delete_file ───────────────────────────────────────────────────────────
    elif tool_name == "delete_file":
        path = tool_args.get("path", "")
        if delete_file(path):
            return f"Deleted {path}.", f"File deleted."
        return f"Failed to delete {path}.", f"I could not delete that file."

    # ── move_file ─────────────────────────────────────────────────────────────
    elif tool_name == "move_file":
        src = tool_args.get("source", "")
        dst = tool_args.get("destination", "")
        if move_file(src, dst):
            return f"Moved {src} to {dst}.", f"File moved."
        return f"Failed to move {src}.", f"I could not move that file."

    # ── copy_file ─────────────────────────────────────────────────────────────
    elif tool_name == "copy_file":
        src = tool_args.get("source", "")
        dst = tool_args.get("destination", "")
        if copy_file(src, dst):
            return f"Copied {src} to {dst}.", f"File copied."
        return f"Failed to copy {src}.", f"I could not copy that file."

    # ── rename_file ───────────────────────────────────────────────────────────
    elif tool_name == "rename_file":
        path = tool_args.get("path", "")
        new_name = tool_args.get("new_name", "")
        if rename_file(path, new_name):
            return f"Renamed to {new_name}.", f"Renamed to {new_name}."
        return f"Failed to rename {path}.", f"I could not rename that file."

    # ── search_files ──────────────────────────────────────────────────────────
    elif tool_name == "search_files":
        query = tool_args.get("query", "")
        directory = tool_args.get("directory", "")
        results = search_files(query, directory)
        if results:
            files_str = "\n".join(results[:10])
            return f"Found {len(results)} files:\n{files_str}", ""  # Needs Round 2
        return "No files found matching that name.", ""

    # ── open_file ─────────────────────────────────────────────────────────────
    elif tool_name == "open_file":
        path = tool_args.get("path", "")
        if open_file(path):
            return f"Opened {path}.", f"Opening the file."
        return f"File not found: {path}.", f"I could not find that file."

    # ── list_directory ────────────────────────────────────────────────────────
    elif tool_name == "list_directory":
        directory = tool_args.get("directory", "")
        items = list_directory(directory)
        if items:
            listing = "\n".join(items[:20])
            return f"Directory contents:\n{listing}", ""  # Needs Round 2
        return "Directory is empty or not found.", ""

    # ── get_clipboard ─────────────────────────────────────────────────────────
    elif tool_name == "get_clipboard":
        content = get_clipboard()
        return f"Clipboard content: {content}", ""  # Needs Round 2

    # ── set_clipboard ─────────────────────────────────────────────────────────
    elif tool_name == "set_clipboard":
        text = tool_args.get("text", "")
        if set_clipboard(text):
            return "Text copied to clipboard.", "Copied to clipboard."
        return "Failed to set clipboard.", "I could not copy that to the clipboard."

    # ── set_brightness ────────────────────────────────────────────────────────
    elif tool_name == "set_brightness":
        level = tool_args.get("level", 50)
        if set_brightness(level):
            return f"Brightness set to {level}%.", f"Brightness set to {level} percent."
        return "Failed to set brightness.", "I could not change the brightness."

    # ── take_screenshot ───────────────────────────────────────────────────────
    elif tool_name == "take_screenshot":
        filename = tool_args.get("filename", "")
        path = take_screenshot(filename)
        if path:
            return f"Screenshot saved to {path}.", "Screenshot taken and saved."
        return "Failed to take screenshot.", "I could not take the screenshot."

    # ── set_volume ────────────────────────────────────────────────────────────
    elif tool_name == "set_volume":
        level = tool_args.get("level", 50)
        if set_volume(level):
            return f"Volume set to {level}%.", f"Volume set to {level} percent."
        return "Failed to set volume.", "I could not change the volume."

    # ── mute_volume ───────────────────────────────────────────────────────────
    elif tool_name == "mute_volume":
        if mute_volume():
            return "Mute toggled.", "Mute toggled."
        return "Failed to toggle mute.", "I could not toggle mute."

    # ── get_wifi_status ───────────────────────────────────────────────────────
    elif tool_name == "get_wifi_status":
        status = get_wifi_status()
        return f"WiFi status: {json.dumps(status)}", ""  # Needs Round 2

    # ── connect_wifi ──────────────────────────────────────────────────────────
    elif tool_name == "connect_wifi":
        ssid = tool_args.get("ssid", "")
        password = tool_args.get("password", "")
        if connect_wifi(ssid, password):
            return f"Connected to {ssid}.", f"Connected to {ssid}."
        return f"Failed to connect to {ssid}.", f"I could not connect to {ssid}."

    # ── get_ip_address ────────────────────────────────────────────────────────
    elif tool_name == "get_ip_address":
        info = get_ip_address()
        return f"IP addresses: {json.dumps(info)}", ""  # Needs Round 2

    # ── get_battery_status ────────────────────────────────────────────────────
    elif tool_name == "get_battery_status":
        info = get_battery_status()
        return f"Battery: {json.dumps(info)}", ""  # Needs Round 2

    # ── get_cpu_usage ─────────────────────────────────────────────────────────
    elif tool_name == "get_cpu_usage":
        usage = get_cpu_usage()
        return f"CPU usage: {usage}%", ""  # Needs Round 2

    # ── get_ram_usage ─────────────────────────────────────────────────────────
    elif tool_name == "get_ram_usage":
        info = get_ram_usage()
        return f"RAM: {json.dumps(info)}", ""  # Needs Round 2

    # ── list_processes ────────────────────────────────────────────────────────
    elif tool_name == "list_processes":
        procs = list_processes()
        return f"Top processes: {json.dumps(procs)}", ""  # Needs Round 2

    # ── kill_process_by_name ──────────────────────────────────────────────────
    elif tool_name == "kill_process_by_name":
        name = tool_args.get("name", "")
        if kill_process_by_name(name):
            return f"Killed process '{name}'.", f"Done. I have killed {name}."
        return f"No process found named '{name}'.", f"I could not find a process named {name}."

    # ── send_notification ─────────────────────────────────────────────────────
    elif tool_name == "send_notification":
        title = tool_args.get("title", "JARVIS")
        message = tool_args.get("message", "")
        if send_notification(title, message):
            return "Notification sent.", "Notification sent."
        return "Failed to send notification.", "I could not send the notification."

    # ── lock_screen ───────────────────────────────────────────────────────────
    elif tool_name == "lock_screen":
        lock_screen()
        return "Screen locked.", "Locking the screen."

    # ── show_desktop ──────────────────────────────────────────────────────────
    elif tool_name == "show_desktop":
        show_desktop()
        return "Desktop shown.", "Showing the desktop."

    # ── take_screenshot_snip ──────────────────────────────────────────────────
    elif tool_name == "take_screenshot_snip":
        take_screenshot_snip()
        return "Snipping tool opened.", "Opening the snipping tool."

    # ── open_task_manager ─────────────────────────────────────────────────────
    elif tool_name == "open_task_manager":
        open_task_manager()
        return "Task Manager opened.", "Opening Task Manager."

    # ── virtual_desktop_new ───────────────────────────────────────────────────
    elif tool_name == "virtual_desktop_new":
        virtual_desktop_new()
        return "New virtual desktop created.", "New virtual desktop created."

    # ── virtual_desktop_switch ────────────────────────────────────────────────
    elif tool_name == "virtual_desktop_switch":
        direction = tool_args.get("direction", "right")
        virtual_desktop_switch(direction)
        return f"Switched virtual desktop {direction}.", f"Switching virtual desktop {direction}."

    # ── fallback ──────────────────────────────────────────────────────────────
    return f"Unknown tool '{tool_name}'.", "I encountered an unknown command."


# ─────────────────────────────────────────────────────────────────────────────
# THE JARVIS AGENT
# ─────────────────────────────────────────────────────────────────────────────

class JarvisAgent:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = LLM_MODEL
        system_instructions = (
            "You are JARVIS, an advanced AI assistant running directly on the user's Windows PC. "
            f"The current user is '{os.getlogin()}'. Their home directory is '{os.path.expanduser('~')}'. "
            "When creating files or folders on the Desktop, always use the path: "
            f"'{os.path.join(os.path.expanduser('~'), 'Desktop')}'. "
            "You are highly intelligent, concise, and extremely professional. "
            "You have access to tools that let you physically control the user's computer. "
            "ALWAYS prefer using a tool when the user asks you to DO something on their PC. "
            "Only answer conversationally when the user is asking a question or having a discussion. "
            "Do NOT roleplay as the Iron Man movie character. Do not mention suits. "
            "Do NOT use markdown (*, #, _, [, ]) because your replies go directly to Text-to-Speech. "
            "Use plain English only."
        )
        self.memory = ConversationalMemory(system_prompt=system_instructions)
        logger.info("Groq Agent with Tool Calling Online.")

    def _reset_memory(self):
        """Clears conversation history to recover from corrupt context."""
        system_msg = self.memory.history[0]
        self.memory.history = [system_msg]
        logger.warning("Conversation memory fully reset to recover from API error.")

    def _inject_format_hint(self):
        """Inserts the tool format hint after the system prompt to guide the model on retry."""
        # Avoid duplicate hints
        self._remove_format_hint()
        self.memory.history.insert(1, _TOOL_FORMAT_HINT)

    def _remove_format_hint(self):
        """Removes any injected format hints from memory to keep history clean."""
        self.memory.history = [
            m for m in self.memory.history
            if not (
                m.get("role") == "system"
                and "CRITICAL FORMATTING RULE" in m.get("content", "")
            )
        ]

    def think(self, user_text: str) -> str:
        if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
            return "My Groq API key is missing. Please add it to your environment variables."

        # 1. Check Fast-Track Intent Layer (Milliseconds response)
        fast_response = self._fast_track_intent(user_text)
        if fast_response:
            return fast_response

        # 2. Proceed to LLM for complex queries
        self.memory.add_user_message(user_text)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                messages = self.memory.get_context()
                logger.info(f"Sending context to Groq (Attempt {attempt + 1}/{max_retries})...")

                response = self.client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0.1,
                    parallel_tool_calls=False,
                )

                response_message = response.choices[0].message
                tool_calls = response_message.tool_calls

                if tool_calls:
                    # Serialize tool calls for memory storage
                    serialized_tool_calls = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ]
                    messages.append({
                        "role": "assistant",
                        "content": response_message.content,
                        "tool_calls": serialized_tool_calls,
                    })

                    final_reply = None
                    needs_round_2 = False

                    for tool_call in tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        tool_result, instant_reply = execute_tool(tool_name, tool_args)

                        logger.info(f"Tool result: {tool_result}")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result,
                        })

                        if tool_name not in _ACTION_ONLY_TOOLS:
                            needs_round_2 = True
                        elif final_reply is None:
                            final_reply = instant_reply

                    if needs_round_2:
                        logger.info("Round 2: LLM summarizing tool output...")
                        final_response = self.client.chat.completions.create(
                            messages=messages,
                            model=self.model,
                            temperature=0.7,
                        )
                        final_reply = final_response.choices[0].message.content

                    if final_reply is None:
                        final_reply = "Done."

                else:
                    # Pure conversational reply — no tool called
                    final_reply = response_message.content

                # Success — clean up any hints and save to memory
                self._remove_format_hint()
                self.memory.add_assistant_message(final_reply)
                return final_reply

            except Exception as e:
                error_str = str(e)

                # ── Groq model decommissioned ──────────────────────────────────
                if "model_decommissioned" in error_str.lower():
                    logger.error(f"Model decommissioned: {error_str}")
                    self._reset_memory()
                    return (
                        "My language model has been retired by Groq. "
                        "Please update LLM_MODEL in config.py to llama-3.3-70b-versatile."
                    )

                # ── Tool call formatting bug (400 / tool_use_failed) ───────────
                elif "400" in error_str or "tool_use_failed" in error_str.lower() or "tool call validation" in error_str.lower():
                    logger.warning(
                        f"Groq tool format bug on attempt {attempt + 1}: {error_str}"
                    )

                    if attempt < max_retries - 1:
                        # Strip the bad user message so we can re-add it cleanly
                        self.memory.history = [
                            m for m in self.memory.history
                            if not (
                                m.get("role") == "user"
                                and m.get("content") == user_text
                            )
                        ]
                        # Inject formatting hint and retry
                        self._inject_format_hint()
                        self.memory.add_user_message(user_text)
                        logger.info("Injected format hint. Retrying...")
                        continue
                    else:
                        # All retries exhausted
                        self._reset_memory()
                        return (
                            "I am having repeated trouble formatting that command. "
                            "Please try rephrasing it."
                        )

                # ── Unauthorized / bad API key ─────────────────────────────────
                elif "401" in error_str or "invalid_api_key" in error_str.lower():
                    logger.error(f"Invalid Groq API key: {error_str}")
                    return (
                        "My API key is invalid. "
                        "Please check your GROQ_API_KEY in the dot env file."
                    )

                # ── Rate limit ─────────────────────────────────────────────────
                elif "429" in error_str or "rate_limit" in error_str.lower():
                    logger.warning(f"Rate limited by Groq: {error_str}")
                    wait_time = 5 * (attempt + 1)
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue

                # ── Unknown error ──────────────────────────────────────────────
                else:
                    logger.error(f"Groq Agent Error: {error_str}")
                    return "I am having trouble connecting to my neural network right now."

        return "I was unable to process that request after multiple attempts. Please try again."

    def _fast_track_intent(self, text: str) -> str:
        """
        Handles common hardware/OS commands locally using Regex.
        Returns the spoken response if a match is found, else None.
        """
        t = text.lower().strip()
        
        # ── VOLUME CONTROL ──
        vol_match = re.search(r'(?:set|change|put|turn)\s+(?:the\s+)?volume\s+(?:to\s+)?(\d+)', t)
        if not vol_match:
            vol_match = re.search(r'volume\s+(\d+)', t)
        if vol_match:
            try:
                level = int(vol_match.group(1))
                from backend.commands.audio_control import set_volume
                if set_volume(level):
                    return f"Volume set to {level} percent."
            except: pass

        if any(w in t for w in ["mute volume", "silence audio", "toggle mute"]):
            from backend.commands.audio_control import mute_volume
            if mute_volume():
                return "Audio muted."

        # ── BRIGHTNESS ──
        bri_match = re.search(r'(?:set|change|put|turn)\s+(?:the\s+)?brightness\s+(?:to\s+)?(\d+)', t)
        if not bri_match:
            bri_match = re.search(r'brightness\s+(\d+)', t)
        if bri_match:
            try:
                level = int(bri_match.group(1))
                from backend.commands.display import set_brightness
                if set_brightness(level):
                    return f"Brightness set to {level} percent."
            except: pass

        # ── SYSTEM ACTIONS ──
        if any(w in t for w in ["lock my computer", "lock screen", "lock the pc"]):
            from backend.commands.keyboard_shortcuts import lock_screen
            lock_screen()
            return "Locking the screen."
        
        if any(w in t for w in ["show desktop", "minimize everything", "hide all windows"]):
            from backend.commands.keyboard_shortcuts import show_desktop
            show_desktop()
            return "Showing the desktop."

        if any(w in t for w in ["open task manager", "start task manager"]):
            from backend.commands.keyboard_shortcuts import open_task_manager
            open_task_manager()
            return "Opening Task Manager."

        # ── FILE MANAGEMENT (DESKTOP QUICK-CREATE) ──
        # Matches "create a folder named X on desktop", "make a folder X on desktop", etc.
        folder_match = re.search(r'(?:create|make)\s+(?:a\s+)?(?:new\s+)?(?:folder|directory)\s+(?:named\s+|called\s+)?(["\']?[\w\s-]+["\']?)\s+(?:on\s+)?(?:the\s+)?desktop', t)
        if folder_match:
            name = folder_match.group(1).strip("'\" ")
            path = os.path.join(os.path.expanduser("~"), "Desktop", name)
            from backend.commands.file_manager import create_directory
            if create_directory(path):
                return f"Done. I have created the folder {name} on your desktop."

        # ── TERMINAL CONTROL ──
        if any(w in t for w in ["close terminal", "exit terminal", "terminate terminal", "close command prompt"]):
            from backend.commands.window_manager import list_open_windows
            windows = list_open_windows()
            for win in windows:
                if any(term in win.lower() for term in ["terminal", "powershell", "cmd.exe", "command prompt"]):
                    from backend.commands.window_manager import focus_window
                    if focus_window(win):
                        pyautogui.hotkey('alt', 'f4')
                        return "Closing the terminal window."
            return "I couldn't find an active terminal window to close."

        return None