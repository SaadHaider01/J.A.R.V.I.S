"""
=============================================================================
backend/memory/summarizer.py
=============================================================================

WHAT THIS FILE DOES:
    Owns the complete summary lifecycle for a conversation session:
    1. GENERATE  — Build a concise text summary from raw messages.
    2. STORE     — Persist the summary to the `conversation_summary` table.
    3. RETRIEVE  — Return past summaries for context injection.

WHY IS SUMMARIZER SEPARATE FROM conversation_store.py?
    This is a direct application of the SINGLE RESPONSIBILITY PRINCIPLE (SRP):
    each class/module should have exactly ONE reason to change.

    ─ conversation_store.py changes when the RAW HISTORY schema changes.
    ─ summarizer.py changes when the SUMMARIZATION LOGIC changes.

    If they were combined, adding an LLM-based summarizer (a future TODO)
    would require modifying the same file as the raw history CRUD — mixing
    storage concerns with AI inference concerns. That's technical debt.

    ANALOGY:
        A court reporter (conversation_store) records every word verbatim.
        A lawyer (summarizer) reviews the transcript and writes a case brief.
        They are different people with different skills. Conflating them
        makes both jobs worse.

WHY SUMMARIES AT ALL?
    Context windows in LLMs are finite and expensive:

    ─ gpt-4o:        128,000 tokens  (~$0.03 per conversation at scale)
    ─ llama-3.3-70b: 128,000 tokens  (Groq rate-limits by tokens/minute)
    ─ local models:  4,096–8,192 tokens (very tight)

    A 1-hour conversation might have 100+ exchanges = ~20,000 tokens.
    Injecting that into every subsequent LLM call would:
    ─ Exhaust rate limits quickly
    ─ Slow inference dramatically
    ─ Increase cost massively
    ─ Dilute the useful signal with irrelevant noise

    A 5-bullet summary captures the same KEY DECISIONS in ~200 tokens.
    That's a 100x compression ratio. This is how production AI assistants
    like Claude's Projects and ChatGPT's Memory work under the hood.

SUMMARIZATION STRATEGY (CURRENT — RULE-BASED):
    We use a simple extractive approach: take the first N user messages
    as topic indicators. This requires no LLM call and is instant.

    This is intentionally designed to be REPLACED by an LLM-based approach.
    The Summarizer's public API is stable — callers won't need to change.

FUTURE — LLM-BASED SUMMARIZATION:
    Replace `_generate_summary_text()` with a Groq API call:

        prompt = f"Summarize this conversation in 3-5 bullet points:\\n{transcript}"
        summary = groq_client.chat.completions.create(...)

    The rest of the class remains unchanged. This is the power of
    encapsulation — implementation details are hidden behind a stable interface.

=============================================================================
"""

import logging
from datetime import datetime
from typing import List, Optional

from backend.memory.database import Database
from backend.memory.memory_models import ConversationMessage, ConversationSummary

logger = logging.getLogger("zytrix.memory.summarizer")


class Summarizer:
    """
    WHAT: Manages the full lifecycle of conversation summaries.

    Responsibilities:
    ─ Generate a summary from a list of conversation messages
    ─ Store that summary in `conversation_summary` table
    ─ Retrieve past summaries for context injection

    THIS CLASS DOES NOT:
    ─ Store raw messages (that's ConversationStore)
    ─ Extract long-term memories (that's MemoryClassifier + MemoryRepository)
    ─ Make LLM API calls (currently — see Future section above)

    EDUCATIONAL CONCEPT — ENCAPSULATION:
        The inner workings of `_generate_summary_text()` are private
        (prefixed with `_`). Callers only call `summarize_session()`.
        When we swap to LLM-based summarization, callers change nothing.
        This is the benefit of encapsulation: implementation details are
        hidden behind a stable, well-named interface.
    """

    def __init__(self, db: Database):
        """
        Receives the shared Database connection via Dependency Injection.
        See memory_repository.py for a detailed explanation of DI.
        """
        self.db = db

    # =========================================================================
    # PRIVATE HELPERS
    # =========================================================================

    def _row_to_summary(self, row: "sqlite3.Row") -> ConversationSummary:
        """Converts a SQLite row to a typed ConversationSummary object."""
        return ConversationSummary(
            id         = row["id"],
            session_id = row["session_id"],
            summary    = row["summary"],
            created_at = datetime.fromisoformat(row["created_at"]),
        )

    def _generate_summary_text(self, messages: List[ConversationMessage]) -> str:
        """
        WHAT: Generates a human-readable text summary from a list of messages.

        CURRENT IMPLEMENTATION — EXTRACTIVE / RULE-BASED:
            We extract the user's own messages (not the assistant's replies)
            to identify the TOPICS DISCUSSED. The user's messages are the
            "what was talked about" signal; the assistant's are the responses.

            Then we format them as bullet points.

        WHY EXTRACTIVE vs. ABSTRACTIVE?
            ─ EXTRACTIVE: Pick sentences from the original text. Fast, free.
            ─ ABSTRACTIVE: Generate NEW text that captures the meaning (LLM).

            Extractive is used here because:
            1. It requires no external API call (no latency, no cost, no failure).
            2. For a session that just happened, user messages ARE the summary.
            3. It provides a fallback when the LLM is unavailable.

        FUTURE LLM REPLACEMENT:
            When ready, replace this method body with a Groq completion call:

                transcript = "\\n".join(
                    f"{m.role.upper()}: {m.message}" for m in messages
                )
                prompt = (
                    "Summarize this conversation in 3-5 concise bullet points. "
                    "Focus on: decisions made, facts learned, and tasks completed.\\n\\n"
                    f"{transcript}"
                )
                # Call Groq here — but only if Groq client is injected into Summarizer
                # (avoid importing agent.py from here — that creates a circular dependency!)

            ⚠ DEPENDENCY WARNING:
                If you add LLM-based summarization, inject the Groq client
                via __init__ rather than importing ZytrixAgent. Importing
                agent.py from summarizer.py would create a CIRCULAR IMPORT
                (agent → memory_manager → summarizer → agent). Always
                inject dependencies; never import from a parent module.

        RETURNS: A formatted multi-line summary string.
        """
        if not messages:
            return "No messages recorded for this session."

        # Extract user messages as topic indicators
        user_messages = [
            m for m in messages if m.role == "user"
        ]

        if not user_messages:
            return "Session contained only assistant messages — nothing to summarize."

        # Build bullet points from user messages (truncated for readability)
        bullets = []
        for msg in user_messages[:10]:  # Cap at 10 bullets to keep summary short
            # Truncate long messages to 100 characters
            text = msg.message[:100].strip()
            if len(msg.message) > 100:
                text += "..."
            bullets.append(f"• {text}")

        session_start = messages[0].timestamp.strftime("%Y-%m-%d %H:%M")
        session_end   = messages[-1].timestamp.strftime("%H:%M")
        total_turns   = len(messages)

        summary = (
            f"Session on {session_start} – {session_end} "
            f"({total_turns} turns)\n"
            + "\n".join(bullets)
        )

        return summary

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def summarize_session(
        self,
        session_id: str,
        messages: List[ConversationMessage]
    ) -> Optional[ConversationSummary]:
        """
        Generates and stores a summary for a completed conversation session.

        PARAMETERS:
            session_id : The unique session identifier.
            messages   : All conversation messages from this session
                         (obtained from ConversationStore.get_session_messages).

        RETURNS:
            The stored ConversationSummary, or None if nothing to summarize.

        WHEN IS THIS CALLED?
            By MemoryManager.end_session(), which is called when:
            ─ The user says "stop", "exit", or "goodbye"
            ─ The conversation has been idle for too long (future feature)
            ─ The application is shutting down gracefully

        FLOW:
            1. Generate summary text from messages (local rule-based method)
            2. Insert into `conversation_summary` table
            3. Return the summary object to MemoryManager
               (MemoryManager will then instruct ConversationStore to delete
               the raw messages for this session)

        WHY THE CALLER (MemoryManager) DELETES MESSAGES?
            Separation of concerns. Summarizer knows HOW to summarize.
            MemoryManager knows WHEN to delete (after summary is confirmed).
            If we put delete logic here, Summarizer would have two jobs:
            summarizing AND orchestrating storage lifecycle. That's a SRP violation.
        """
        if not messages:
            logger.info(
                f"[MEMORY] No messages to summarize for session={session_id[:8]}..."
            )
            return None

        # Generate the summary text
        summary_text = self._generate_summary_text(messages)
        now = datetime.now().isoformat()

        # Store in database using UPSERT logic (INSERT OR REPLACE)
        # The UNIQUE constraint on session_id means we can safely call this
        # even if a summary for this session already exists — it will be replaced.
        with self.db.connection:
            cursor = self.db.connection.execute(
                """
                INSERT INTO conversation_summary (session_id, summary, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    summary    = excluded.summary,
                    created_at = excluded.created_at;
                """,
                (session_id, summary_text, now)
            )

        # Retrieve the stored summary to get its database-assigned ID
        summary = self.get_summary(session_id)
        logger.info(
            f"[MEMORY] Conversation summarized for session={session_id[:8]}... "
            f"({len(messages)} messages → {len(summary_text)} chars)"
        )
        return summary

    def get_summary(self, session_id: str) -> Optional[ConversationSummary]:
        """Returns the summary for a specific session, or None if not found."""
        cursor = self.db.connection.execute(
            "SELECT * FROM conversation_summary WHERE session_id = ? LIMIT 1;",
            (session_id,)
        )
        row = cursor.fetchone()
        return self._row_to_summary(row) if row else None

    def get_recent_summaries(self, limit: int = 5) -> List[ConversationSummary]:
        """
        Returns the N most recent conversation summaries.

        WHEN IS THIS USED?
            By MemoryManager.build_memory_context() to inject a brief
            "recent activity" block into the system prompt before LLM
            inference. Gives ZYTRIX a sense of recent conversation history
            without injecting all the raw messages.

        WHY LIMIT?
            Context window discipline. Even summaries can add up if you
            inject all 500 past sessions. We inject only the most recent
            few for recency-weighted relevance.
        """
        cursor = self.db.connection.execute(
            """
            SELECT * FROM conversation_summary
            ORDER BY created_at DESC
            LIMIT ?;
            """,
            (limit,)
        )
        summaries = [self._row_to_summary(row) for row in cursor.fetchall()]
        logger.debug(
            f"[MEMORY] Retrieved {len(summaries)} recent summaries."
        )
        return summaries

    def delete_summary(self, session_id: str) -> bool:
        """Removes a summary for a specific session."""
        with self.db.connection:
            cursor = self.db.connection.execute(
                "DELETE FROM conversation_summary WHERE session_id = ?;",
                (session_id,)
            )
        return cursor.rowcount > 0

    # =========================================================================
    # FUTURE EXTENSION POINTS
    # =========================================================================
    #
    # TODO: LLM-based abstractive summarization (replace _generate_summary_text).
    #       Inject a Groq client (or any LLM adapter) via __init__ to avoid
    #       circular imports. Keep the public API identical.
    #
    # TODO: `summarize_all_sessions(session_ids)` — batch summarization for
    #       a memory consolidation background job.
    #
    # TODO: `merge_summaries(session_ids)` — combine multiple session summaries
    #       into a single "monthly" or "weekly" meta-summary.
    #
    # TODO: Store summary quality score (0.0–1.0) alongside the summary.
    #       LLM-generated summaries get higher scores than extractive ones.
    #       Use score to prioritize which summaries to inject into context.
