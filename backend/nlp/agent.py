import logging
from groq import Groq
from duckduckgo_search import DDGS
from config import GROQ_API_KEY, LLM_MODEL
from backend.memory.context_manager import ConversationalMemory

logger = logging.getLogger("JARVIS.Agent")

class JarvisAgent:
    def __init__(self):
        """
        Initializes the super-fast Groq LLM API and the DuckDuckGo search engine.
        This entirely replaces the old Intent Classifier and Entity Extractor!
        """
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = LLM_MODEL
        
        # The System Prompt is JARVIS's core brain and personality
        system_instructions = (
            "You are JARVIS, an advanced AI assistant. "
            "You act exactly like JARVIS from Iron Man. "
            "You are highly intelligent, slightly witty, very concise, and speak naturally. "
            "IMPORTANT: Do NOT use markdown (*, #, _, [, ]) in your responses because your replies "
            "are instantly being converted to Text-to-Speech audio. Just use plain English text."
        )
        self.memory = ConversationalMemory(system_prompt=system_instructions)
        self.search_engine = DDGS()
        logger.info("Groq Agent Online.")

    def think(self, user_text: str) -> str:
        """
        JARVIS's core cognition loop.
        Takes the user's speech transcript, maintains context memory, searches the web if needed, 
        and generates an intelligent reply.
        """
        if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
            return "Sir, my Groq API key is missing. Please add it to your environment variables."

        # Dynamic 'Tool Use' - An invisible web search
        # If the user asks a question needing internet access, JARVIS scrapes DuckDuckGo implicitly!
        if any(keyword in user_text.lower() for keyword in ["search", "who is", "what is", "current", "latest"]):
            logger.info("Dynamic web search triggered by Agent...")
            try:
                # Scrape top 2 results silently
                results = self.search_engine.text(user_text, max_results=2)
                context_str = " ".join([r['body'] for r in results])
                
                # Append the hidden web data so JARVIS can "read" it before answering!
                user_text = f"[Hidden Live Web Data Context: {context_str}] \n\n Answer my query naturally: {user_text}"
            except Exception as e:
                logger.error(f"Web search failed: {e}")

        # Add the text to our running memory
        self.memory.add_user_message(user_text)

        logger.info("Sending context to Groq API...")
        try:
            chat_completion = self.client.chat.completions.create(
                messages=self.memory.get_context(),
                model=self.model,
                temperature=0.7, # 0.7 gives a great balance between facts and creativity
            )
            
            response = chat_completion.choices[0].message.content
            
            # Store the response in memory so he remembers what he said!
            self.memory.add_assistant_message(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Groq API Error: {e}")
            return "I am having trouble connecting to my neural network right now."
