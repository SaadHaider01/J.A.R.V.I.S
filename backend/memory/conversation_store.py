"""
=============================================================================
backend/memory/conversation_store.py
=============================================================================

WHAT THIS FILE DOES:
    Handles ONLY the raw conversation history — storing, retrieving, and
    deleting individual message turns (user said X, assistant replied Y).

    This is the Repository for the `conversation_history` table.

WHY SEPARATE FROM memory_repository.py?
    Conversation history and long-term memories are FUNDAMENTALLY DIFFERENT:

    ┌──────────────────────┬────────────────────────────────────────────┐
    │ Conversation History │ Long-Term Memory                           │
    ├──────────────────────┼────────────────────────────────────────────┤
    │ Temporary            │ Permanent                                  │
    │ Per-session          │ Cross-session                              │
    │ High volume          │ Low volume, curated                        │
    │ Queried by session   │ Queried by keyword/category                │
    │ Deleted after summary│ Kept indefinitely                          │
    │ Every message        │ Only meaningful messages                   │
    └──────────────────────┴────────────────────────────────────────────┘

    Mixing these responsibilities would eventually create TECHNICAL DEBT:

    1. SCHEMA CONFLICTS: Conversation rows need `session_id` but memory
       rows don't. You'd end up with nullable columns everywhere — a sign
       of poor normalization.

    2. QUERY COMPLEXITY: "Give me all conversations from session X" is
       completely different from "give me all VSCode preferences". Having
       them in the same module (or worse, the same table) makes both
       queries harder to write and maintain.

    3. RETENTION POLICY CONFLICT: We delete old conversations after
       summarization. We NEVER delete core memories. Having them in the
       same module means you'd need careful guards to avoid accidentally
       deleting memories when pruning history.

    ANALOGY: Your email inbox (temporary, high volume) vs. your contacts
    book (permanent, curated). You don't store them in the same place.

WHY NOT CALLED ConversationRepository?
    It could be. "Store" here means a focused store of a specific entity
    type — similar to how React uses "stores" (Zustand, Redux). Both names
    are valid. We use "store" to distinguish it from the full-lifecycle
    Repository pattern, since conversation history has simpler CRUD needs.

RESPONSIBILITY OF THIS MODULE:
    ─ Store individual messages as they happen
    ─ Retrieve all messages for a given session (for summarization)
    ─ Delete old sessions after they've been summarized
    ─ NOTHING ELSE — no summarization, no memory extraction, no LLM calls

=============================================================================
"""

import logging
import uuid
from datetime import datetime
from typing import List, Optional

from backend.memory.database import Database
from backend.memory.memory_models import ConversationMessage

logger = logging.getLogger("zytrix.memory.conversation_store")


class ConversationStore:
    """
    WHAT: Repository for raw conversation history.

    Translates between Python ConversationMessage objects and the
    `conversation_history` SQLite table. Follows the same Repository
    Pattern as MemoryRepository.

    ABOUT SESSIONS:
        Each time Zytrix starts, a new session_id is generated (a UUID).
        All messages during that run are tagged with this session_id.
        When the session ends, all messages can be retrieved by session_id
        for summarization, then deleted to keep the table lean.
    """

    def __init__(self, db: Database):
        """
        EDUCATIONAL CONCEPT — DEPENDENCY INJECTION (again, for emphasis):
            Like MemoryRepository, we receive the Database rather than
            creating it. Both repositories share the SAME Database instance,
            which means they share the same SQLite connection and file.

            WHY SHARE THE SAME CONNECTION?
                SQLite supports multiple tables in one file. Having both
                repositories talk to the same DB connection means:
                ─ One file to backup (zytrix_memory.db)
                ─ One schema version to track
                ─ Transactions can span both tables if needed in the future
        """
        self.db = db

    # =========================================================================
    # PRIVATE HELPERS
    # =========================================================================

    def _row_to_message(self, row: "sqlite3.Row") -> ConversationMessage:
        """Converts a SQLite row to a typed ConversationMessage object."""
        return ConversationMessage(
            id         = row["id"],
            session_id = row["session_id"],
            role       = row["role"],
            message    = row["message"],
            timestamp  = datetime.fromisoformat(row["timestamp"]),
        )

    # =========================================================================
    # CREATE
    # =========================================================================

    def add_message(self, session_id: str, role: str, message: str) -> int:
        """
        Stores a single conversation turn in the database.

        PARAMETERS:
            session_id : The current session UUID (groups messages together).
            role       : "user" or "assistant".
            message    : The text content of the message.

        RETURNS: The auto-generated ID of the inserted row.

        WHEN IS THIS CALLED?
            Immediately after every user input and every assistant reply.
            The session_id comes from MemoryManager, which generates one
            UUID at startup and passes it down.

        WHY STORE EVERY MESSAGE (not just important ones)?
            The conversation store is temporary context, not curated memory.
            We want the full session so the Summarizer can generate an
            accurate summary. The Summarizer then condenses it; we delete
            the raw messages after.
        """
        now = datetime.now().isoformat()
        with self.db.connection:
            cursor = self.db.connection.execute(
                """
                INSERT INTO conversation_history
                    (session_id, role, message, timestamp)
                VALUES (?, ?, ?, ?);
                """,
                (session_id, role, message, now)
            )
        row_id = cursor.lastrowid
        logger.debug(
            f"[MEMORY] Stored conversation message: session={session_id[:8]}... "
            f"role={role} | '{message[:60]}...'"
        )
        return row_id

    # =========================================================================
    # READ
    # =========================================================================

    def get_session_messages(self, session_id: str) -> List[ConversationMessage]:
        """
        Returns all messages from a specific session, in chronological order.

        WHY ORDER BY timestamp?
            Messages must be in time order for the Summarizer to produce a
            coherent narrative ("first the user asked X, then ZYTRIX replied Y").

        WHEN IS THIS CALLED?
            By the Summarizer at the end of a session, to get all messages
            to condense into a summary.
        """
        cursor = self.db.connection.execute(
            """
            SELECT * FROM conversation_history
            WHERE session_id = ?
            ORDER BY timestamp ASC;
            """,
            (session_id,)
        )
        messages = [self._row_to_message(row) for row in cursor.fetchall()]
        logger.debug(
            f"[MEMORY] Retrieved {len(messages)} messages for "
            f"session={session_id[:8]}..."
        )
        return messages

    def get_recent_messages(self, limit: int = 20) -> List[ConversationMessage]:
        """
        Returns the N most recent messages across ALL sessions.

        WHEN IS THIS USED?
            For debugging, or if you want to show the user a recent history
            without having to know the session ID.
        """
        cursor = self.db.connection.execute(
            """
            SELECT * FROM conversation_history
            ORDER BY timestamp DESC
            LIMIT ?;
            """,
            (limit,)
        )
        # Reverse to get chronological order (we fetched newest-first)
        return list(reversed([self._row_to_message(row) for row in cursor.fetchall()]))

    def count_session_messages(self, session_id: str) -> int:
        """Returns the number of messages stored for a session."""
        cursor = self.db.connection.execute(
            "SELECT COUNT(*) FROM conversation_history WHERE session_id = ?;",
            (session_id,)
        )
        return cursor.fetchone()[0]

    # =========================================================================
    # DELETE
    # =========================================================================

    def delete_session(self, session_id: str) -> int:
        """
        Deletes ALL messages from a specific session.

        RETURNS: Number of rows deleted.

        WHEN IS THIS CALLED?
            After the Summarizer has successfully stored a summary for this
            session. At that point, the raw messages are redundant — we have
            the condensed summary. Deleting them keeps the database lean.

        WHY DELETE AFTER SUMMARIZING?
            If we kept all messages forever, the conversation_history table
            would grow unboundedly. After 1000 sessions, you'd have millions
            of rows of raw chat that you never need again — they've all been
            distilled into summaries.

            This is the same pattern email apps use when archiving old messages:
            keep the summary (folder), delete the details.
        """
        with self.db.connection:
            cursor = self.db.connection.execute(
                "DELETE FROM conversation_history WHERE session_id = ?;",
                (session_id,)
            )
        count = cursor.rowcount
        logger.info(
            f"[MEMORY] Deleted {count} messages from session={session_id[:8]}..."
        )
        return count

    def delete_old_sessions(self, keep_n_sessions: int = 5) -> int:
        """
        Deletes messages from all sessions EXCEPT the N most recent.

        EDUCATIONAL CONCEPT — SUBQUERY:
            We use a subquery to first find the session_ids to keep, then
            delete everything not in that set. This is safer than calculating
            dates externally.

        WHY `keep_n_sessions`?
            You might want to keep a few recent sessions un-summarized
            (e.g., the last session is still fresh). The default of 5
            is a safe buffer; adjust as needed.

        WHEN IS THIS CALLED?
            As a periodic cleanup — not after every session. Could be
            triggered on startup, or by a scheduled background task.

            TODO: Hook this into a startup routine in MemoryManager.
        """
        with self.db.connection:
            cursor = self.db.connection.execute(
                """
                DELETE FROM conversation_history
                WHERE session_id NOT IN (
                    SELECT DISTINCT session_id FROM conversation_history
                    ORDER BY MAX(timestamp) DESC
                    LIMIT ?
                );
                """,
                (keep_n_sessions,)
            )
        count = cursor.rowcount
        if count > 0:
            logger.info(f"[MEMORY] Cleaned up {count} old conversation messages.")
        return count


# =============================================================================
# SESSION ID HELPER
# =============================================================================

def generate_session_id() -> str:
    """
    Generates a unique session identifier for a new conversation.

    WHY UUID?
        UUIDs (Universally Unique Identifiers) are 128-bit random numbers
        formatted as hex strings. The probability of two UUIDs colliding
        is astronomically small — effectively zero for any practical use.

        Alternatives considered:
        ─ Timestamp strings: Risk collision if two sessions start in the same
          millisecond (unlikely, but bad for a "unique" ID).
        ─ Incrementing integers: Requires a persistent counter — adds state.
        ─ UUID is the industry standard for this exact use case.

    EXAMPLE OUTPUT: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
    """
    return str(uuid.uuid4())
