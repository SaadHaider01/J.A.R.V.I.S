"""
=============================================================================
backend/memory/memory_manager.py
=============================================================================

WHAT THIS FILE DOES:
    The MemoryManager is the SINGLE PUBLIC INTERFACE to the entire memory
    subsystem. It is the only class the NLP agent imports and interacts with.

    It ORCHESTRATES work by delegating to specialized modules:
    ─ MemoryClassifier  → decides if a message deserves storage
    ─ MemoryRepository  → stores/retrieves long-term memories
    ─ ConversationStore → stores raw conversation history
    ─ Summarizer        → generates and stores session summaries
    ─ MemoryRetriever   → retrieves relevant memories for prompts

EDUCATIONAL CONCEPT — THE MANAGER PATTERN:
    A Manager coordinates multiple subsystems to achieve a business goal.
    It is NOT responsible for doing the actual work — it delegates.

    GOOD MANAGER:
        def process_user_message(self, text):
            if self.classifier.classify(text).should_store:    # delegates
                self.repository.upsert(...)                     # delegates
            self.conv_store.add_message(...)                    # delegates

    BAD MANAGER (doing too much — "God Object" anti-pattern):
        def process_user_message(self, text):
            if re.search(r"my favourite", text):   # ← classifier logic in here!
                conn.execute("INSERT INTO ...")    # ← SQL in here!
                ...                                # ← 500-line method

    The Manager Pattern keeps MemoryManager thin and readable.
    Each component can be improved, tested, or swapped independently.

EDUCATIONAL CONCEPT — FACADE PATTERN:
    MemoryManager is also a FACADE — it provides a simple interface
    (process_user_message, retrieve_context, end_session) that hides
    the complexity of 5 separate subsystems behind it.

    The NLP agent doesn't know about MemoryRepository, ConversationStore,
    or Summarizer. It just calls MemoryManager. This is exactly what
    the Facade pattern is for: simplifying complex subsystem interactions.

THE STRICT LAYERED ARCHITECTURE:
    User
      │
      ▼
    NLP Agent (agent.py)
      │          calls only: memory_manager.MemoryManager
      ▼
    MemoryManager
      │
      ├─► MemoryClassifier.classify(text)
      │
      ├─► MemoryRetriever.retrieve(query)
      │
      ├─► MemoryRepository.upsert(item) / search(keyword)
      │       │
      │       └─► Database.execute(sql)
      │                 │
      │                 └─► SQLite
      │
      ├─► ConversationStore.add_message(...)
      │       └─► Database.execute(sql)
      │
      └─► Summarizer.summarize_session(...)
              └─► Database.execute(sql)

    RULE: No layer skips. The NLP agent NEVER touches the Repository.
          Retriever NEVER touches ConversationStore. Etc.

=============================================================================
"""

import logging
from typing import List, Optional

from backend.memory.database import Database
from backend.memory.memory_models import MemoryItem, MemoryCategory, MemorySource
from backend.memory.memory_repository import MemoryRepository
from backend.memory.conversation_store import ConversationStore, generate_session_id
from backend.memory.summarizer import Summarizer
from backend.memory.memory_classifier import classify
from backend.memory.retrieval import MemoryRetriever

logger = logging.getLogger("zytrix.memory")


class MemoryManager:
    """
    WHAT: The public API for Zytrix's persistent memory system.

    This is the ONLY class the NLP agent should import from this package.
    All other memory modules (MemoryRepository, ConversationStore, etc.)
    are internal implementation details.

    RESPONSIBILITIES:
        ─ Initialize all memory subsystems on startup
        ─ Process incoming messages (classify + store conversation + maybe store memory)
        ─ Build memory context for injection before LLM inference
        ─ End a session (summarize + clean up raw history)
        ─ Expose convenience methods for direct memory operations

    WHAT THIS CLASS DOES NOT DO:
        ─ Execute SQL (that's Database + Repository)
        ─ Classify messages (that's MemoryClassifier)
        ─ Retrieve memories (that's MemoryRetriever)
        ─ Summarize conversations (that's Summarizer)

    THINK OF IT AS: The conductor of an orchestra. Doesn't play any
    instrument. Coordinates all players to produce a coherent result.

    USAGE IN NLP AGENT:
        # At startup:
        self.memory_manager = MemoryManager()

        # Before LLM call:
        context_block = self.memory_manager.build_context(user_text)
        # Inject context_block into system prompt

        # After each message:
        self.memory_manager.process_message(role="user", text=user_text)
        self.memory_manager.process_message(role="assistant", text=reply)

        # At shutdown:
        self.memory_manager.end_session()
    """

    def __init__(self, db_path=None):
        """
        Initializes the entire memory subsystem.

        DEPENDENCY INJECTION CHAIN:
            MemoryManager creates one Database.
            All repositories receive the SAME Database instance.
            WHY ONE DATABASE? One SQLite file, one connection, one source of truth.
            This also means all operations share transactions if needed.

        PARAMETER `db_path`:
            Optional custom path for testing. Pass `:memory:` for SQLite
            in-memory database in unit tests — no file created, very fast,
            automatically discarded when the test ends.

        SESSION ID:
            A new UUID session_id is generated here. All messages in this
            Zytrix run are tagged with this ID. When end_session() is called,
            this ID links all messages for summarization.
        """
        from pathlib import Path

        # Initialize the database layer
        if db_path is None:
            db = Database()
        else:
            db = Database(Path(db_path))

        # Build the dependency graph — inject the shared database into each layer
        # EDUCATIONAL: This is "manual dependency injection" (no DI framework needed).
        # Each class declares its dependency in __init__; we wire them up here.
        self._db           = db
        self._repo         = MemoryRepository(db)
        self._conv_store   = ConversationStore(db)
        self._summarizer   = Summarizer(db)
        self._retriever    = MemoryRetriever(self._repo)

        # Generate a unique session ID for this Zytrix run
        self._session_id   = generate_session_id()

        logger.info(
            f"[MEMORY] MemoryManager initialized. "
            f"Session: {self._session_id[:8]}..."
        )

    # =========================================================================
    # CORE PUBLIC METHODS
    # =========================================================================

    def process_message(self, role: str, text: str) -> bool:
        """
        WHAT: Called after every user input or assistant reply.

        DOES TWO THINGS:
        1. Always stores the message in conversation history (raw log).
        2. If role="user", runs classifier to check if long-term memory deserves storing.

        RETURNS: True if a long-term memory was stored, False otherwise.

        WHY NOT CLASSIFY ASSISTANT MESSAGES?
            The assistant's replies are ZYTRIX's responses, not the user's
            self-description. We're building a memory of WHO THE USER IS,
            not what ZYTRIX said. Classifying assistant messages would create
            noise (ZYTRIX often repeats back what the user said).

        PARAMETERS:
            role : "user" or "assistant"
            text : The message content
        """
        # Always store in conversation history (raw log for summarization)
        self._conv_store.add_message(self._session_id, role, text)

        if role != "user":
            return False  # Only classify user messages for long-term memory

        # Run the lightweight classifier
        result = classify(text)

        if not result.should_store:
            # Conversational noise — no long-term memory needed
            logger.debug(f"[MEMORY] Memory ignored: '{text[:60]}'")
            return False

        # Build a MemoryItem from the classification result
        item = MemoryItem(
            key        = result.key,
            value      = text,                     # Store the full message as value
            category   = result.category,
            confidence = result.confidence,
            importance = result.importance,
            source     = MemorySource.CONVERSATION,
        )

        # Upsert: update existing memory if key already exists, insert if new
        saved = self._repo.upsert(item)
        logger.info(
            f"[MEMORY] Stored {result.category.value} memory: "
            f"key='{result.key}' | conf={result.confidence:.2f}"
        )
        return True

    def build_context(self, query: str) -> str:
        """
        WHAT: Builds the memory context block to inject into the system prompt.

        CALL THIS BEFORE EVERY LLM INFERENCE.

        RETURNS:
            A formatted string like:
                --- Known User Facts ---
                • [preference] favourite_ide: My favourite IDE is VSCode (confidence: 0.95)
                • [project] current_project: I'm building Zytrix (confidence: 0.85)
                ------------------------

            Returns empty string if no relevant memories found.

        WHY INJECT INTO SYSTEM PROMPT (NOT CONVERSATION HISTORY)?
            ─ System prompt = "meta-instructions for the model" — the right
              place for persistent user facts.
            ─ Conversation history = "what was said in this session" — wrong
              place for facts that pre-date this session.
            ─ Injecting into history would make ZYTRIX seem like it's lying
              about things it "said" in previous turns.

        EDUCATIONAL CONCEPT — RETRIEVAL-AUGMENTED GENERATION (RAG):
            This method is the heart of RAG for this system:
            1. User sends a query.
            2. We retrieve the N most relevant memories (not ALL memories).
            3. We format them and inject into the LLM context.
            4. The LLM now has the right facts without receiving database dumps.

        CONTEXT WINDOW COST ANALYSIS:
            Without RAG: inject all 500 memories = ~15,000 tokens. 💸
            With RAG (limit=7): inject top 7 memories = ~200 tokens. ✅
        """
        memories = self._retriever.retrieve(query, limit=7)

        if not memories:
            return ""

        context_block = self._retriever.format_for_prompt(memories)
        logger.info(
            f"[MEMORY] Retrieved {len(memories)} memories for context injection."
        )
        return context_block

    def end_session(self) -> Optional[str]:
        """
        WHAT: Called when a conversation session ends (shutdown, timeout, goodbye).

        DOES THREE THINGS:
        1. Retrieves all messages from the current session.
        2. Generates and stores a summary via Summarizer.
        3. Deletes the raw messages (they're now distilled into the summary).

        RETURNS: The summary text, or None if session had no messages.

        WHY END SESSIONS EXPLICITLY?
            Without explicit session ending, raw messages accumulate forever.
            With explicit ending, we get the best of both worlds:
            ─ Full context during a session (raw messages for this session)
            ─ Efficient long-term storage (summary after session ends)

        GRACEFUL DEGRADATION:
            If summarization fails (e.g. database error), we log the error
            but don't crash. The raw messages stay in the database as a backup.

            TODO: Add try/except with specific error handling.
        """
        messages = self._conv_store.get_session_messages(self._session_id)

        if not messages:
            logger.info(
                f"[MEMORY] No messages to summarize for "
                f"session={self._session_id[:8]}..."
            )
            return None

        # Step 1: Generate and store the summary
        summary = self._summarizer.summarize_session(self._session_id, messages)

        if summary:
            # Step 2: Delete raw messages — they're now distilled into the summary
            self._conv_store.delete_session(self._session_id)
            logger.info(
                f"[MEMORY] Session ended. {len(messages)} messages summarized "
                f"and cleaned up. Session={self._session_id[:8]}..."
            )
            return summary.summary

        return None

    # =========================================================================
    # CONVENIENCE METHODS — Direct memory operations
    # =========================================================================

    def save_memory(
        self,
        key: str,
        value: str,
        category: MemoryCategory = MemoryCategory.GENERAL,
        confidence: float = 1.0,
        importance: float = 0.8,
        source: MemorySource = MemorySource.MANUAL,
    ) -> MemoryItem:
        """
        Directly saves a memory, bypassing the classifier.

        WHEN TO USE:
            When you want to programmatically insert a known fact
            (e.g., from a profile setup screen or onboarding flow).

        WHY SOURCE=MANUAL?
            Manually inserted memories were not inferred from conversation.
            Tracking the source helps debug "where did ZYTRIX learn this?"
        """
        item = MemoryItem(
            key        = key,
            value      = value,
            category   = category,
            confidence = confidence,
            importance = importance,
            source     = source,
        )
        saved = self._repo.upsert(item)
        logger.info(f"[MEMORY] Manual memory saved: '{key}' = '{value}'")
        return saved

    def retrieve_all(self) -> List[MemoryItem]:
        """Returns all stored long-term memories. Useful for debug/display."""
        return self._repo.get_all()

    def search_memory(self, keyword: str) -> List[MemoryItem]:
        """Searches memories by keyword. Useful for conversational recall."""
        return self._repo.search(keyword)

    def delete_memory(self, memory_id: int) -> bool:
        """Deletes a specific memory by its database ID."""
        return self._repo.delete(memory_id)

    def delete_memory_by_key(self, key: str) -> bool:
        """Deletes a memory by its key string."""
        return self._repo.delete_by_key(key)

    def update_memory(self, item: MemoryItem) -> bool:
        """
        Updates an existing memory.
        The item must have a valid `id` (retrieved from a previous call).
        """
        return self._repo.update(item)

    def get_recent_summaries(self, limit: int = 3) -> List[str]:
        """
        Returns formatted text of the N most recent session summaries.

        WHY RETURN STRINGS (NOT OBJECTS)?
            The NLP agent only needs the text to inject into the prompt.
            Returning MemoryItem or ConversationSummary objects would expose
            internal data structures to the agent — a violation of
            the Facade pattern's "hide complexity" goal.
        """
        summaries = self._summarizer.get_recent_summaries(limit=limit)
        return [s.summary for s in summaries]

    @property
    def session_id(self) -> str:
        """The current session's unique identifier. Read-only."""
        return self._session_id

    def shutdown(self):
        """
        Gracefully closes all resources.

        Call this on application exit (use atexit or signal handlers).
        Not calling this on a clean shutdown is harmless for SQLite
        (WAL mode flushes on connection close), but it's good practice.
        """
        self.end_session()
        self._db.close()
        logger.info("[MEMORY] MemoryManager shut down cleanly.")

    # =========================================================================
    # FUTURE EXTENSION POINTS
    # =========================================================================
    #
    # TODO: `set_user_id(user_id)` — for multi-user support. Set at login.
    #       All repository calls would pass user_id as a filter.
    #
    # TODO: `import_memories(filepath)` — bulk import from JSON/CSV.
    #       source=MemorySource.IMPORTED for all imported items.
    #
    # TODO: `export_memories(filepath)` — backup to JSON for portability.
    #
    # TODO: `run_maintenance()` — periodic background task:
    #       ─ Decay confidence of old unconfirmed memories
    #       ─ Archive memories not accessed in N days
    #       ─ Merge duplicate/contradictory memories
    #       Schedule with APScheduler or call on startup.
    #
    # TODO: `get_knowledge_graph()` — extract (subject, predicate, object)
    #       triples from stored memories for graph visualization.
