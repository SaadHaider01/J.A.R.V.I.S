import logging

logger = logging.getLogger("JARVIS.Context")


class ConversationalMemory:
    def __init__(self, system_prompt: str):
        """
        Stores the running conversation history to provide context/memory to the LLM.

        The trimming logic is tool-call-pair-aware: it never separates an
        assistant[tool_calls] message from its subsequent tool result message(s).
        Groq strictly requires these to always appear together, and violating
        this ordering causes a 400 Bad Request error.
        """
        self.history = [{"role": "system", "content": system_prompt}]
        # Max user+assistant text turn pairs to keep (tool pairs are kept extra)
        self.max_text_turns = 6  # 6 pairs = 12 messages + system + tool pairs

    def add_user_message(self, text: str):
        """Add what the user said to JARVIS's memory."""
        self.history.append({"role": "user", "content": text})
        self._trim_memory()

    def add_assistant_message(self, text: str):
        """Add what JARVIS said back into memory."""
        self.history.append({"role": "assistant", "content": text})
        self._trim_memory()

    def get_context(self) -> list:
        import datetime
        context = list(self.history)
        
        # Dynamically inject the exact current date and time into the system prompt
        # so the LLM ALWAYS knows what day it is without needing a web search tool.
        now = datetime.datetime.now().strftime("%A, %B %d, %Y %I:%M %p")
        system_msg = context[0].copy()
        system_msg["content"] = f"{system_msg['content']}\nThe current date and time is {now}."
        context[0] = system_msg
        
        return context

    def _trim_memory(self):
        """
        Safely trims old conversation turns from history while ALWAYS keeping
        tool_call/tool result pairs intact.

        Strategy:
        1. Keep the system prompt (index 0) always.
        2. Scan through history[1:] and identify atomic "segments":
           - A text turn:  a single user or plain assistant message
           - A tool turn:  an assistant[tool_calls] message PLUS all following
                           tool result messages (mandatory to keep together)
        3. Drop the oldest text-turn segments first until we're within limit.
           Tool pairs are NEVER split or dropped mid-pair.
        """
        system_msg = self.history[0]
        body = self.history[1:]

        # ── Build atomic segments ─────────────────────────────────────────────
        segments = []
        i = 0
        while i < len(body):
            msg = body[i]
            role = msg.get("role", "")

            # Detect an assistant message that issues tool calls
            if role == "assistant" and msg.get("tool_calls"):
                # Greedily consume all immediately following tool result messages
                j = i + 1
                while j < len(body) and body[j].get("role") == "tool":
                    j += 1
                # This entire block (assistant+tool_calls + tool results) is one atomic segment
                segments.append(("tool_pair", body[i:j]))
                i = j
            else:
                # Regular user or plain assistant text message
                segments.append(("text", [msg]))
                i += 1

        # ── Count text-turn pairs and drop oldest if over limit ───────────────
        # We count user messages as the delimiter for "turns"
        text_turn_count = sum(
            1 for seg_type, msgs in segments
            if seg_type == "text" and msgs[0].get("role") == "user"
        )

        while text_turn_count > self.max_text_turns and segments:
            # Drop from the front (oldest) until we find a text segment to remove
            if segments[0][0] == "text":
                removed_role = segments[0][1][0].get("role", "")
                segments.pop(0)
                if removed_role == "user":
                    text_turn_count -= 1
            else:
                # Skip over tool pairs at the front — find the next text pair
                # to avoid leaving orphaned tool results
                break

        # ── Reconstruct history ───────────────────────────────────────────────
        new_body = []
        for _, msgs in segments:
            new_body.extend(msgs)
        self.history = [system_msg] + new_body
