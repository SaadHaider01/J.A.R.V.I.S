"""
=============================================================================
backend/memory/retrieval.py
=============================================================================

WHAT THIS FILE DOES:
    Provides the retrieval interface for the memory system.
    Given a user query (or keywords extracted from it), returns the most
    relevant stored memories to inject into the LLM prompt.

WHY A SEPARATE RETRIEVAL MODULE?
    Retrieval is a distinct concern from storage:
    ─ Storage asks: "How do I persist this item?"
    ─ Retrieval asks: "Given this query, what's most relevant to return?"

    These are different algorithms that evolve at different rates:
    ─ Storage logic rarely changes (SQL INSERT/UPDATE is stable).
    ─ Retrieval logic is where most AI research happens (vector search,
      ranking, re-ranking, hybrid retrieval, etc.).

    Having retrieval in its own module means you can upgrade your search
    algorithm without touching any storage code.

THE ABSTRACTION PRINCIPLE — WHY CALLERS NEVER SEE SQL:
    MemoryManager calls `retrieve(query)`.
    It does NOT know whether retrieval uses:
    ─ SQL LIKE (current)
    ─ Embedding cosine similarity (future)
    ─ BM25 full-text search (alternative)
    ─ Hybrid (BM25 + cosine similarity) (production-grade)

    This is the ABSTRACTION principle — hide implementation complexity
    behind a clean, stable interface. When retrieval improves, only
    this file changes. The NLP agent and MemoryManager never know.

EDUCATIONAL CONCEPT — RETRIEVAL STRATEGIES:
    ┌─────────────────────┬──────────────────────────────────────────────┐
    │ Strategy            │ How it works                                 │
    ├─────────────────────┼──────────────────────────────────────────────┤
    │ Exact match         │ WHERE key = 'favourite_ide'                  │
    │ Keyword (LIKE)      │ WHERE value LIKE '%vscode%'                  │
    │ BM25                │ Full-text index, term frequency weighting    │
    │ Semantic/Embedding  │ Cosine similarity of dense vectors           │
    │ Hybrid              │ BM25 score × α + embedding score × (1−α)    │
    └─────────────────────┴──────────────────────────────────────────────┘

    Current: Keyword (SQL LIKE) — instant, no dependencies, good enough
    for a personal assistant with < 1000 memories.

    At ~10,000+ memories, LIKE degrades (O(n) scan). That's when you'd
    introduce BM25 (SQLite FTS5 extension) or vector embeddings.

WHY CONTEXT WINDOW DISCIPLINE MATTERS:
    Every token in an LLM context costs money and latency.
    llama-3.3-70b on Groq has a 128K token limit but a rate limit of
    ~6000 tokens/minute on the free tier.

    Injecting 50 memories × 30 tokens = 1,500 tokens for EVERY request.
    Over a 1-hour session with 100 requests: 150,000 tokens just for memory.

    By retrieving only the TOP 5 RELEVANT memories:
    5 × 30 = 150 tokens per request — a 10x reduction.

    This is called "retrieval-augmented generation" (RAG): retrieve first,
    inject only the relevant subset, not the entire database.

=============================================================================
"""

import logging
import re
from typing import List

from backend.memory.memory_models import MemoryItem
from backend.memory.memory_repository import MemoryRepository

logger = logging.getLogger("zytrix.memory.retrieval")

# ─────────────────────────────────────────────────────────────────────────────
# STOP WORDS — Common English words to exclude from keyword extraction.
# We don't want to search the memory database for "what" or "is" — those
# would match nearly everything and flood retrieval with irrelevant results.
# ─────────────────────────────────────────────────────────────────────────────
_STOP_WORDS = {
    "i", "me", "my", "is", "are", "am", "a", "an", "the", "to", "of",
    "in", "on", "at", "what", "who", "where", "when", "how", "why",
    "do", "does", "did", "can", "could", "should", "would", "will",
    "it", "its", "this", "that", "these", "those", "and", "or", "but",
    "for", "with", "you", "your", "tell", "know", "remember", "about",
    "say", "said", "think", "want", "have", "has", "had", "get", "got",
    "from", "by", "be", "been", "being", "was", "were", "not", "no",
}


class MemoryRetriever:
    """
    WHAT: The retrieval engine for the memory system.

    Converts a natural language query into relevant MemoryItem results,
    using keyword extraction + SQL LIKE queries.

    DESIGN FOR SWAPPABILITY:
        MemoryManager only calls `retrieve(query, limit)`.
        The implementation inside can be replaced with vector search
        without changing a single line in MemoryManager or the NLP agent.

        To add vector search:
        1. Add a `vector_search(embedding, top_k)` method to MemoryRepository.
        2. In `retrieve()`, compute the query embedding and call that method
           (or blend both: `hybrid_retrieve()`).
        3. MemoryManager remains completely untouched.

    EDUCATIONAL CONCEPT — DEPENDENCY INJECTION (DI):
        MemoryRetriever receives MemoryRepository via __init__.
        It does NOT create its own repository.
        WHY: Makes it testable (pass a mock repo in unit tests),
             and follows the Dependency Inversion Principle.
    """

    def __init__(self, repository: MemoryRepository):
        """
        Receives the MemoryRepository via Dependency Injection.
        The repository provides the search capability; retrieval logic lives here.
        """
        self.repository = repository

    def retrieve(self, query: str, limit: int = 7) -> List[MemoryItem]:
        """
        WHAT: Returns the most relevant memories for a given user query.

        THIS IS THE ONLY PUBLIC METHOD callers should use.
        MemoryManager calls this before every LLM inference.

        PARAMETERS:
            query : The user's raw input text.
                    Example: "Open my usual workspace"
            limit : Maximum number of memories to return.
                    WHY A LIMIT? Context window discipline — see module docstring.

        RETURNS:
            List[MemoryItem] sorted by relevance (importance DESC).
            Maximum `limit` items. May be empty if nothing relevant.

        ALGORITHM:
            1. Extract meaningful keywords from the query.
            2. For each keyword, search the repository (SQL LIKE).
            3. Deduplicate results (same memory can match multiple keywords).
            4. Sort by importance (highest first) and return top N.

        EXAMPLE:
            Query: "Open my usual workspace"
            Keywords extracted: ["usual", "workspace"]
            Memory match for "usual": routine_usually_code_at_night
            Memory match for "workspace": project_current_project (value="Zytrix")
            Memory match for "workspace": preference_favourite_ide (value="VSCode")
            Returns: [project, preference, routine] (sorted by importance)
        """
        keywords = self._extract_keywords(query)

        if not keywords:
            logger.debug("[MEMORY] No meaningful keywords to retrieve for.")
            return []

        # Collect results from all keyword searches, deduplicating by memory id
        # WHY A DICT? `dict[id] = item` automatically overwrites duplicates.
        # Using a list would require O(n²) deduplication.
        seen_ids: dict[int, MemoryItem] = {}

        for keyword in keywords:
            results = self.repository.search(keyword, limit=limit)
            for item in results:
                if item.id not in seen_ids:
                    seen_ids[item.id] = item

        # Sort by importance (descending) then by confidence (descending)
        # WHY IMPORTANCE FIRST? More important memories should always appear first.
        # WHY CONFIDENCE SECOND? Among equally important memories, prefer confident ones.
        ranked = sorted(
            seen_ids.values(),
            key=lambda item: (item.importance, item.confidence),
            reverse=True,
        )

        # Apply final limit
        top_results = ranked[:limit]

        logger.info(
            f"[MEMORY] Retrieved {len(top_results)} memories for query: "
            f"'{query[:60]}' | keywords={keywords}"
        )
        return top_results

    def format_for_prompt(self, memories: List[MemoryItem]) -> str:
        """
        WHAT: Formats retrieved memories as a structured text block for injection
              into the LLM system prompt.

        WHY THIS METHOD HERE AND NOT IN MemoryManager?
            The retrieval module knows the shape of MemoryItem best.
            Formatting is a concern of how retrieved data is presented.
            However, either location is architecturally valid — it's a minor
            judgment call. We keep it here to make MemoryManager even thinner.

        EDUCATIONAL CONCEPT — PROMPT ENGINEERING:
            The way you format context for an LLM matters significantly.
            Bullet-pointed, clearly labeled facts outperform raw database
            dumps because:
            ─ LLMs are trained on structured human-written text (markdown,
              bullet lists are familiar patterns).
            ─ Clear labels ("Preferred IDE: VSCode") remove ambiguity.
            ─ Sections help the model understand the TYPE of information.

        WHY NOT INJECT INTO CONVERSATION HISTORY?
            Injecting into conversation history would confuse the LLM —
            it would appear as if a previous message said these facts.
            Instead, we inject into the SYSTEM PROMPT as a facts block.
            The system prompt is the appropriate place for "meta-information
            about the user" that shapes all responses.

        EXAMPLE OUTPUT:
            --- Known User Facts ---
            • [preference] favourite_ide: VSCode (confidence: 0.95)
            • [project] current_project: Zytrix (confidence: 0.85)
            • [routine] usually_codes: at night (confidence: 0.80)
            ------------------------
        """
        if not memories:
            return ""

        lines = ["--- Known User Facts ---"]
        for item in memories:
            lines.append(
                f"• [{item.category.value}] {item.key}: {item.value} "
                f"(confidence: {item.confidence:.2f})"
            )
        lines.append("------------------------")

        return "\n".join(lines)

    # =========================================================================
    # PRIVATE HELPERS
    # =========================================================================

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extracts meaningful keywords from a query for database searching.

        STEPS:
            1. Lowercase and strip punctuation.
            2. Split into tokens (words).
            3. Remove stop words and very short tokens.
            4. Deduplicate while preserving order.

        WHY NOT USE NLTK / spaCy?
            Those are powerful NLP libraries, but they're heavy dependencies.
            For a personal assistant with simple queries, manual stop-word
            removal is sufficient and keeps the system lightweight.

            FUTURE: If Zytrix adds full NLP processing elsewhere (e.g. for
            entity extraction), leverage those capabilities here too.

        EXAMPLE:
            "What is my favourite IDE?" → ["favourite", "ide"]
            "Open my usual workspace" → ["usual", "workspace"]
        """
        # Remove punctuation and lowercase
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = cleaned.split()

        # Filter stop words and short tokens (< 3 chars rarely meaningful)
        keywords = []
        seen = set()
        for token in tokens:
            if token not in _STOP_WORDS and len(token) >= 3 and token not in seen:
                keywords.append(token)
                seen.add(token)

        return keywords

    # =========================================================================
    # FUTURE EXTENSION POINTS
    # =========================================================================
    #
    # TODO: `vector_retrieve(query, top_k)` — compute a query embedding and
    #       use cosine similarity against stored embedding_ids in ChromaDB/FAISS.
    #       The repository would expose `vector_search(embedding, top_k)`.
    #       This method would blend keyword + vector scores for hybrid retrieval.
    #
    # TODO: `hybrid_retrieve(query, alpha=0.5)` — blend keyword score and
    #       vector score. alpha=0.0 is pure keyword; alpha=1.0 is pure vector.
    #       This is the state of the art for RAG systems (2024–2025).
    #
    # TODO: Re-ranking — after retrieval, use a cross-encoder model to
    #       re-score each (query, memory) pair for precision. This is more
    #       accurate than bi-encoder similarity alone.
    #
    # TODO: Query expansion — if keywords return < 3 results, expand the
    #       query with synonyms (e.g. "IDE" → also search "editor", "tool").
    #
    # TODO: Importance-weighted retrieval limit — instead of a hard limit of 7,
    #       set a token budget (e.g. 300 tokens) and fill it greedily from
    #       most to least important memory.
