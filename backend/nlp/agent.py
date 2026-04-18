import json
import logging
import os
import time
import pyautogui
from groq import Groq
from duckduckgo_search import DDGS
from config import GROQ_API_KEY, LLM_MODEL
from backend.memory.context_manager import ConversationalMemory
from backend.commands.app_launcher import launch_app, app_paths, search_in_browser
from backend.commands.system_control import sleep_pc, shutdown_pc

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
                            "The name of the process or window to close. Examples: 'cmd', 'notepad', 'brave', "
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
                "Opens the Brave browser and performs a Google search for the given query. "
                "Use this when the user says 'search for X', 'look up X in browser', 'Google X', "
                "or 'open Brave and search for X'. This visually opens Brave with search results."
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
# ACTION-ONLY TOOLS
# These tools perform a physical action and need no LLM Round 2 summary.
# We skip the second API call and return a pre-built spoken reply instantly,
# making the response feel much faster.
# ─────────────────────────────────────────────────────────────────────────────
_ACTION_ONLY_TOOLS = {
    "launch_application",
    "type_text",
    "search_in_browser",
    "sleep_computer",
    "shutdown_computer",
    "terminate_process",
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
            tool_result = f"Successfully launched {app}."
            spoken = f"Opening {app}."
        else:
            tool_result = f"Could not find '{app}'."
            spoken = f"I could not find an app named {app}."
        return tool_result, spoken

    # ── type_text ─────────────────────────────────────────────────────────────
    elif tool_name == "type_text":
        text = tool_args.get("text", "")
        time.sleep(0.5)
        pyautogui.write(text, interval=0.04)
        return f"Typed: '{text}'.", f"Done. I've typed that for you."

    # ── web_search ────────────────────────────────────────────────────────────
    elif tool_name == "web_search":
        query = tool_args.get("query", "")
        try:
            ddgs = DDGS()
            results = ddgs.text(query, max_results=3)
            if not results:
                return "Web search returned no results.", ""
            combined = " ".join([r['body'] for r in results])
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
        import subprocess
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

    # ── search_in_browser ─────────────────────────────────────────────────────
    elif tool_name == "search_in_browser":
        query = tool_args.get("query", "")
        search_in_browser(query)
        return f"Opened Brave and searched for '{query}'.", f"Searching for {query} in Brave."

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

    def _reset_memory(self):
        """Clears conversation history to recover from corrupt context (e.g. after a 400 error)."""
        system_msg = self.memory.history[0]
        self.memory.history = [system_msg]
        logger.warning("Conversation memory reset to recover from API error.")

    def think(self, user_text: str) -> str:
        """
        The agentic think loop using Groq Tool Calling.

        Round 1: Ask Groq what to do (tool call or conversation).
        Execute:  Run the real tool on the user's PC immediately.
        Round 2:  Only called for informational tools (web_search) that need
                  the LLM to summarize results. Action-only tools skip this
                  entirely, returning a pre-built reply for zero extra latency.
        """
        if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
            return "My Groq API key is missing. Please add it to your environment variables."

        self.memory.add_user_message(user_text)
        messages = self.memory.get_context()

        logger.info("Sending context to Groq with Tool Calling enabled...")
        try:
            # ── ROUND 1: Ask Groq what to do ──────────────────────────────────
            # temperature=0.1 for tool selection — needs to be deterministic,
            # not creative. This prevents malformed JSON tool arguments.
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.1,
            )

            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            # ── Did the LLM decide to call a tool? ────────────────────────────
            if tool_calls:
                # Serialize the Pydantic object to a plain dict so context_manager's
                # .get() calls in _trim_memory don't crash on a ChatCompletionMessage.
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
                    "content": response_message.content,  # None when tool_calls are present
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

                    # Track the quick reply and whether Round 2 is needed
                    if tool_name not in _ACTION_ONLY_TOOLS:
                        needs_round_2 = True
                    elif final_reply is None:
                        final_reply = instant_reply  # Use pre-built reply for action tools

                # ── ROUND 2: Only for tools that need LLM to summarize data ───
                if needs_round_2:
                    logger.info("Round 2: LLM summarizing tool output...")
                    final_response = self.client.chat.completions.create(
                        messages=messages,
                        model=self.model,
                        temperature=0.7,  # Conversational tone for the spoken reply
                    )
                    final_reply = final_response.choices[0].message.content
                elif final_reply is None:
                    final_reply = "Done."

            else:
                # Pure conversational answer — no tool needed
                final_reply = response_message.content

            self.memory.add_assistant_message(final_reply)
            return final_reply

        except Exception as e:
            error_str = str(e)
            # ── 400 Bad Request: context is corrupt — reset and recover ────────
            if "400" in error_str or "bad_request" in error_str.lower():
                logger.error(f"Groq 400 error — resetting memory to recover. Error: {e}")
                self._reset_memory()
                return "I had a brief memory hiccup. Could you repeat that?"
            logger.error(f"Groq Agent Error: {e}")
            return "I am having trouble with my neural network right now."
