import json
import logging
import os
import time
import pyautogui
from groq import Groq
from duckduckgo_search import DDGS
from config import GROQ_API_KEY, LLM_MODEL
from backend.memory.context_manager import ConversationalMemory
from backend.commands.app_launcher import launch_app, app_paths, search_in_chrome
from backend.commands.system_control import sleep_pc, shutdown_pc

logger = logging.getLogger("JARVIS.Agent")

# ─────────────────────────────────────────────────────────────────────────────
# TOOL DEFINITIONS
# This is the "Toolbox" we hand to the Groq Brain.
# It is a JSON schema that tells the LLM WHAT tools exist, what they do,
# and what arguments they need. The LLM reads this and decides which tool
# (if any) to trigger for each user message.
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
                            f"The name of the app to open. Must be one of: "
                            f"{', '.join(app_paths.keys())}"
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
                "ONLY call this when the user explicitly uses the words 'shut down the computer' or 'turn off the pc'. "
                "Do NOT call this for 'terminate', 'close', 'kill', 'stop', or 'exit' — those are process commands, not system commands. "
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
                "Use this when the user says 'terminate', 'close', 'kill', 'stop', or 'exit' a specific app or terminal. "
                "This does NOT shut down the computer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "process_name": {
                        "type": "string",
                        "description": (
                            "The name of the process or window to close. Examples: 'cmd', 'notepad', 'chrome', "
                            "'terminal', 'powershell'. Use the closest Windows executable name."
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
            "name": "search_in_browser",
            "description": (
                "Opens Google Chrome and performs a Google search for the given query. "
                "Use this when the user says 'search for X', 'look up X in Chrome', 'Google X', "
                "or 'open Chrome and search for X'. This visually opens Chrome with search results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search term or phrase to search for on Google.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# TOOL EXECUTOR
# This python function is the "Hands" that actually runs the tool.
# When the Groq brain decides to call a tool, it returns a JSON blob.
# We read that JSON and map it to the actual Python function below.
# ─────────────────────────────────────────────────────────────────────────────
def execute_tool(tool_name: str, tool_args: dict) -> str:
    """
    Executes the physical tool on the user's Windows machine based on 
    what the LLM decided to call, and returns a human-readable result string
    that is fed back to the LLM so it can formulate a spoken reply.
    """
    logger.info(f"Executing tool: '{tool_name}' with args: {tool_args}")

    if tool_name == "launch_application":
        app = tool_args.get("app_name", "").lower()
        success = launch_app(app)
        if success:
            return f"Successfully launched {app}."
        return f"Could not find '{app}'. Available apps are: {', '.join(app_paths.keys())}."

    elif tool_name == "type_text":
        text = tool_args.get("text", "")
        # Give a short delay to let the user's window stay focused
        time.sleep(0.5)
        pyautogui.write(text, interval=0.04)
        return f"Typed the text: '{text}' into the active window."

    elif tool_name == "web_search":
        query = tool_args.get("query", "")
        try:
            ddgs = DDGS()
            results = ddgs.text(query, max_results=3)
            if not results:
                return "Web search returned no results."
            combined = " ".join([r['body'] for r in results])
            return f"Live web data for '{query}': {combined}"
        except Exception as e:
            return f"Web search failed: {e}"

    elif tool_name == "sleep_computer":
        sleep_pc()
        return "Putting the computer to sleep."

    elif tool_name == "shutdown_computer":
        shutdown_pc()
        return "Shutting down the computer."

    elif tool_name == "terminate_process":
        import subprocess
        process = tool_args.get("process_name", "cmd").lower().strip()
        # Normalise common spoken names to real Windows process names
        aliases = {
            "terminal": "cmd.exe",
            "command prompt": "cmd.exe",
            "cmd": "cmd.exe",
            "powershell": "powershell.exe",
            "notepad": "notepad.exe",
            "chrome": "chrome.exe",
            "edge": "msedge.exe",
        }
        exe = aliases.get(process, process if process.endswith(".exe") else process + ".exe")
        try:
            result = subprocess.run(
                ["taskkill", "/F", "/IM", exe],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return f"Successfully terminated {exe}."
            else:
                return f"Could not find a running process named '{exe}'. It may already be closed."
        except Exception as e:
            return f"Failed to terminate process: {e}"

    elif tool_name == "search_in_browser":
        query = tool_args.get("query", "")
        result = search_in_chrome(query)
        return f"Opened Chrome and searched for '{query}'."

    return f"Unknown tool '{tool_name}'."


# ─────────────────────────────────────────────────────────────────────────────
# THE JARVIS AGENT
# ─────────────────────────────────────────────────────────────────────────────
class JarvisAgent:
    def __init__(self):
        """
        Initializes the Groq LLM client with a fully equipped Tool-Calling brain.
        This replaces ALL the hardcoded if/else logic in listener.py.
        """
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = LLM_MODEL

        system_instructions = (
            "You are JARVIS, an advanced AI assistant running directly on the user's Windows PC. "
            "You are highly intelligent, concise, and extremely professional. "
            "You have access to tools that let you physically control the user's computer. "
            "ALWAYS prefer using a tool when the user asks you to DO something on their PC. "
            "Only answer conversationally when the user is asking a question or having a discussion. "
            "Do NOT roleplay as the Iron Man movie character. Do not mention suits. "
            "Do NOT use markdown (*, #, _, [, ]) because your replies go directly to Text-to-Speech. "
            "Use plain English only. "
            "CRITICAL SAFETY INSTRUCTION: NEVER call the shutdown_computer or sleep_computer tools unless the user EXPLICITLY and CLEARLY commands you to 'shut down' or 'sleep' the computer."
        )
        self.memory = ConversationalMemory(system_prompt=system_instructions)
        logger.info("Groq Agent with Tool Calling Online.")

    def think(self, user_text: str) -> str:
        """
        Phase 8 Core: The fully agentic think loop using Groq Tool Calling.
        
        How it works:
        1. We send the user's message + the Toolbox schema to Groq.
        2. Groq decides: "Is this a task I should use a tool for, or just talk?"
        3. If it picks a tool, it returns a JSON call spec (NOT spoken text).
        4. We execute the real Python function on the user's PC.
        5. We send the result back to Groq so it can say "Done, sir." naturally.
        6. The final spoken reply is returned to the TTS engine.
        """
        if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
            return "My Groq API key is missing. Please add it to your environment variables."

        self.memory.add_user_message(user_text)
        messages = self.memory.get_context()

        logger.info("Sending context to Groq API with Tool Calling enabled...")
        try:
            # ── ROUND 1: Ask Groq what to do ──────────────────────────────
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                tools=TOOLS,
                tool_choice="auto",  # Let the LLM decide when to use tools
                temperature=0.7,
            )

            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            # ── Did the LLM decide to call a tool? ────────────────────────
            if tool_calls:
                # Append the LLM's "I want to call this tool" message to memory
                messages.append(response_message)

                # Execute every tool the LLM requested (usually just one)
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    tool_result = execute_tool(tool_name, tool_args)
                    logger.info(f"Tool result: {tool_result}")

                    # Feed the real-world result back into the conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    })

                # ── ROUND 2: Ask Groq to formulate a spoken reply ─────────
                # Now Groq knows what happened and can say "Done, sir." naturally
                final_response = self.client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    temperature=0.7,
                )
                final_reply = final_response.choices[0].message.content

            else:
                # No tool needed — it's a pure conversational answer
                final_reply = response_message.content

            self.memory.add_assistant_message(final_reply)
            return final_reply

        except Exception as e:
            logger.error(f"Groq Agent Error: {e}")
            return "I am having trouble with my neural network right now."
