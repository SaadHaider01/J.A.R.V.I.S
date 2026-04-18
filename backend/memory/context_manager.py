import logging

logger = logging.getLogger("JARVIS.Context")

class ConversationalMemory:
    def __init__(self, system_prompt: str):
        """
        Stores the running conversation history to provide 'context' or 'memory' to the LLM.
        """
        self.history = [{"role": "system", "content": system_prompt}]
        self.max_messages = 15 # Keep the last 15 messages so the context limits aren't exceeded

    def add_user_message(self, text: str):
        """Add what you said to JARVIS's memory."""
        self.history.append({"role": "user", "content": text})
        self._trim_memory()

    def add_assistant_message(self, text: str):
        """Add what JARVIS said back into his memory."""
        self.history.append({"role": "assistant", "content": text})
        self._trim_memory()

    def get_context(self) -> list:
        return self.history

    def _trim_memory(self):
        """Ensures the memory list doesn't grow infinitely forever and crash."""
        if len(self.history) > self.max_messages:
            # We keep the system prompt [0], and just slice off the oldest messages in the middle
            self.history = [self.history[0]] + self.history[-(self.max_messages-1):]
