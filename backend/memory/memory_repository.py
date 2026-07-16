"""
=============================================================================
backend/memory/memory_repository.py
=============================================================================

WHAT THIS FILE DOES:
    Implements the Repository Pattern for long-term user memories.
    All SQL for the `user_memory` table lives here and nowhere else.

EDUCATIONAL CONCEPT — THE REPOSITORY PATTERN:
    The Repository Pattern is a software design pattern that abstracts the
    data access layer. Instead of your business logic reaching into the
    database directly, it talks to a "repository" that handles all the
    storage details.

    ANALOGY:
        Imagine a library. You (MemoryManager) don't climb into the stacks
        yourself. You give a request to the librarian (MemoryRepository).
        The librarian knows the filing system (SQLite), fetches the book,
        and returns it to you. You never need to know HOW it's stored.

    WITHOUT REPOSITORY:
        MemoryManager contains SQL strings. If you switch from SQLite to
        PostgreSQL, you rewrite MemoryManager. If you add ChromaDB, you
        rewrite MemoryManager. MemoryManager becomes a 1000-line mess.

    WITH REPOSITORY:
        MemoryManager calls `repo.save(item)`. The repository handles SQL.
        To switch to PostgreSQL, you write a new PostgreSQLMemoryRepository
        that implements the same methods. MemoryManager never changes.

    This is the DEPENDENCY INVERSION PRINCIPLE — high-level modules depend
    on abstractions (method signatures), not on concrete implementations (SQL).

EDUCATIONAL CONCEPT — SEPARATION OF CONCERNS:
    Each module in the memory system has exactly ONE reason to change:
    ─ memory_models.py  → changes if the data structure changes
    ─ database.py       → changes if the database engine changes
    ─ memory_repository.py → changes if the SQL queries need optimization
    ─ memory_manager.py → changes if the business logic changes

    If you mix SQL into MemoryManager, then MemoryManager changes for TWO
    reasons: business logic AND query optimization. This violates SRP
    (Single Responsibility Principle) and makes code harder to reason about.

WHY `memory_repository.py` NOT `memory_store.py`?
    "Store" is a generic term. "Repository" explicitly signals that this
    class implements the Repository Pattern — a named, well-understood
    design pattern that other developers will immediately recognize.

THE DEPENDENCY FLOW (STRICT — NO LAYER MAY BE SKIPPED):
    MemoryManager
        → MemoryRepository
            → Database
                → SQLite

    MemoryRepository ONLY knows about Database and MemoryItem.
    It does NOT know about MemoryClassifier, Retriever, or MemoryManager.

=============================================================================
"""

import logging
from datetime import datetime
from typing import List, Optional

from backend.memory.database import Database
from backend.memory.memory_models import MemoryItem, MemoryCategory, MemorySource

logger = logging.getLogger("zytrix.memory.repository")


class MemoryRepository:
    """
    WHAT: The Repository for long-term user memories.

    Translates between Python MemoryItem objects and SQLite rows.
    Provides the full CRUD (Create, Read, Update, Delete) interface
    plus lifecycle operations (upsert, archive).

    EDUCATIONAL CONCEPT — CRUD:
        CRUD stands for Create, Read, Update, Delete — the four fundamental
        operations on any persistent data store. Almost every database
        interaction in any application maps to one of these four.

        CREATE → INSERT INTO user_memory ...
        READ   → SELECT FROM user_memory ...
        UPDATE → UPDATE user_memory SET ...
        DELETE → DELETE FROM user_memory ...

    WHY NOT EXPOSE `connection` DIRECTLY?
        Callers should not need to write SQL. This class provides
        high-level Python methods. This means if you swap SQLite for
        MongoDB tomorrow, only this file changes — not a single line
        in MemoryManager or the NLP agent.

    EDUCATIONAL CONCEPT — MANAGER PATTERN vs REPOSITORY PATTERN:
        Repository: Handles persistence (CRUD, queries). Knows the DB.
        Manager:    Orchestrates business rules. Doesn't know the DB.

        This class is a Repository, not a Manager.
    """

    def __init__(self, db: Database):
        """
        EDUCATIONAL CONCEPT — DEPENDENCY INJECTION:
            Instead of creating the Database inside __init__, we RECEIVE
            it as a parameter. This is Dependency Injection (DI).

            WHY?
            ─ Testability: Pass a test database (in-memory SQLite) in tests.
            ─ Flexibility: Pass a different DB without changing this class.
            ─ Clarity: The constructor signature tells you exactly what
                       this class needs to operate.

            Compare:
                ❌ Bad:   self.db = Database()  — hardcoded, untestable
                ✅ Good:  self.db = db           — injected, flexible
        """
        self.db = db

    # =========================================================================
    # PRIVATE HELPERS — Row ↔ MemoryItem conversion
    # =========================================================================

    def _row_to_item(self, row: "sqlite3.Row") -> MemoryItem:
        """
        Converts a raw SQLite row into a typed MemoryItem dataclass.

        WHY A SEPARATE METHOD?
            Multiple query methods (get_all, search, get_by_key) all need
            to convert rows to objects. Centralizing this in one private
            method means if MemoryItem gains a new field, you update one
            place, not three.

        WHY CONVERT AT ALL?
            SQLite rows are tuples of raw Python types (str, int, float).
            MemoryItem is a typed, structured Python object with enum fields.
            Working with MemoryItem is safer and more readable throughout
            the rest of the codebase.
        """
        return MemoryItem(
            id           = row["id"],
            key          = row["key"],
            value        = row["value"],
            category     = MemoryCategory(row["category"]),
            confidence   = row["confidence"],
            importance   = row["importance"],
            source       = MemorySource(row["source"]),
            embedding_id = row["embedding_id"],
            created_at   = datetime.fromisoformat(row["created_at"]),
            updated_at   = datetime.fromisoformat(row["updated_at"]),
        )

    # =========================================================================
    # CREATE
    # =========================================================================

    def save(self, item: MemoryItem) -> int:
        """
        Inserts a new memory into the database.

        Returns the auto-generated `id` of the inserted row.

        EDUCATIONAL CONCEPT — `lastrowid`:
            After an INSERT, SQLite assigns a unique integer ID (the PRIMARY
            KEY). `cursor.lastrowid` retrieves that ID, so we can set it
            on the Python object — the caller now has a fully populated item.

        EDUCATIONAL CONCEPT — ISO 8601 DATETIME:
            We store datetimes as ISO 8601 strings ("2025-07-16T20:30:00").
            SQLite has no native DATETIME type, so TEXT is the standard.
            ISO 8601 has the useful property of sorting correctly as a string,
            so `ORDER BY created_at` works naturally.

        NOTE: This method only INSERTs. For "save or update" logic (upsert),
              see `upsert()` below, which is what MemoryManager calls.
        """
        now = datetime.now().isoformat()
        with self.db.connection:
            cursor = self.db.connection.execute(
                """
                INSERT INTO user_memory
                    (key, value, category, confidence, importance, source,
                     embedding_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.key,
                    item.value,
                    item.category.value,
                    item.confidence,
                    item.importance,
                    item.source.value,
                    item.embedding_id,
                    now,
                    now,
                )
            )
        new_id = cursor.lastrowid
        logger.info(
            f"[MEMORY] Stored {item.category.value} memory: "
            f"'{item.key}' = '{item.value}' (id={new_id})"
        )
        return new_id

    # =========================================================================
    # READ
    # =========================================================================

    def get_all(self) -> List[MemoryItem]:
        """
        Returns ALL memories from the database.

        WHY SORTED BY importance DESC?
            When we dump everything (e.g. for debugging or full context),
            the most important memories should appear first.

        CAUTION: Do NOT inject all of these into the LLM prompt. Use
                 `search()` or `retrieval.py` for targeted retrieval.
        """
        cursor = self.db.connection.execute(
            "SELECT * FROM user_memory ORDER BY importance DESC, updated_at DESC;"
        )
        rows = cursor.fetchall()
        return [self._row_to_item(row) for row in rows]

    def get_by_key(self, key: str) -> Optional[MemoryItem]:
        """
        Returns the memory with an exact key match, or None if not found.

        WHY: Used by `upsert()` to check if a key already exists before
             deciding to INSERT or UPDATE.
        """
        cursor = self.db.connection.execute(
            "SELECT * FROM user_memory WHERE key = ? LIMIT 1;",
            (key,)
        )
        row = cursor.fetchone()
        return self._row_to_item(row) if row else None

    def get_by_category(self, category: MemoryCategory) -> List[MemoryItem]:
        """
        Returns all memories of a specific category.

        EXAMPLE USE CASE: Retrieve all REMINDER memories to check for due tasks.
        """
        cursor = self.db.connection.execute(
            "SELECT * FROM user_memory WHERE category = ? ORDER BY importance DESC;",
            (category.value,)
        )
        return [self._row_to_item(row) for row in cursor.fetchall()]

    def search(self, keyword: str, limit: int = 10) -> List[MemoryItem]:
        """
        Returns memories whose key OR value contains the keyword.

        EDUCATIONAL CONCEPT — SQL LIKE:
            LIKE performs a case-insensitive pattern match.
            `%keyword%` means "anything before and after keyword".
            This is a simple but effective full-text search for small datasets.

        WHY `%?%` NOT f-string?
            We still use parameterized queries. The `?` placeholder is for
            the parameter, and we wrap the keyword in `%` Python-side before
            passing it. This keeps the benefits of SQL injection protection.

        DESIGN FOR SWAPPABILITY:
            This method is also called by `retrieval.py`. The retrieval module
            provides the high-level interface; this is the SQL implementation.
            To add vector search, you'd add a `vector_search()` method here
            and have retrieval.py call it instead — no other changes needed.
        """
        pattern = f"%{keyword}%"
        cursor = self.db.connection.execute(
            """
            SELECT * FROM user_memory
            WHERE key LIKE ? OR value LIKE ?
            ORDER BY importance DESC, updated_at DESC
            LIMIT ?;
            """,
            (pattern, pattern, limit)
        )
        results = [self._row_to_item(row) for row in cursor.fetchall()]
        logger.debug(
            f"[MEMORY] Search '{keyword}' returned {len(results)} memories."
        )
        return results

    # =========================================================================
    # UPDATE
    # =========================================================================

    def update(self, item: MemoryItem) -> bool:
        """
        Updates an existing memory in the database by its `id`.

        RETURNS: True if a row was updated, False if the id didn't exist.

        WHY UPDATE RATHER THAN DELETE+INSERT?
            Updating preserves `created_at` — we know WHEN this fact was
            first learned. That timestamp is valuable for future confidence
            decay ("this preference was set 2 years ago, maybe reconsider").
            A DELETE+INSERT would lose that historical information.

        WHY `rowcount`?
            `cursor.rowcount` tells us how many rows were affected by the
            UPDATE. If 0, the id didn't exist. This lets the caller handle
            "update failed because item doesn't exist" gracefully.
        """
        now = datetime.now().isoformat()
        with self.db.connection:
            cursor = self.db.connection.execute(
                """
                UPDATE user_memory
                SET value        = ?,
                    category     = ?,
                    confidence   = ?,
                    importance   = ?,
                    source       = ?,
                    embedding_id = ?,
                    updated_at   = ?
                WHERE id = ?;
                """,
                (
                    item.value,
                    item.category.value,
                    item.confidence,
                    item.importance,
                    item.source.value,
                    item.embedding_id,
                    now,
                    item.id,
                )
            )
        success = cursor.rowcount > 0
        if success:
            logger.info(
                f"[MEMORY] Updated memory id={item.id}: "
                f"'{item.key}' = '{item.value}' (confidence={item.confidence:.2f})"
            )
        else:
            logger.warning(f"[MEMORY] Update failed: no memory with id={item.id}.")
        return success

    # =========================================================================
    # UPSERT — The key lifecycle method
    # =========================================================================

    def upsert(self, item: MemoryItem) -> MemoryItem:
        """
        WHAT: "Upsert" = Update if exists, Insert if not.

        This is the PRIMARY method that MemoryManager uses to save memories.
        It prevents duplicate entries when the user repeats the same info.

        EXAMPLE:
            1. "My favourite IDE is VSCode."
               → `get_by_key("favourite_ide")` returns None.
               → INSERT new memory with confidence=0.8.

            2. "My favourite IDE is Cursor." (later)
               → `get_by_key("favourite_ide")` returns existing item.
               → UPDATE: value="Cursor", confidence bumped (repeated mention).

            3. "Actually, my favourite IDE is still Cursor." (confirmed)
               → Same key exists, same value.
               → UPDATE: confidence bumped to max(existing + 0.05, 1.0).
               → This represents "user confirmed this fact again".

        WHY INCREASE CONFIDENCE ON REPEAT?
            If the user mentions the same fact multiple times, it's more
            likely to be accurate. This is a simple Bayesian-inspired
            approach to memory reliability.

        EDUCATIONAL CONCEPT — UPSERT PATTERNS:
            SQLite supports `INSERT OR REPLACE` and `ON CONFLICT` clauses,
            but they always DELETE + INSERT (losing created_at).
            Our manual GET → branch(UPDATE or INSERT) pattern preserves
            all fields and gives us full control over the merge logic.
            This is called "application-level upsert" and is the right
            choice when the merge logic is complex.
        """
        existing = self.get_by_key(item.key)

        if existing is None:
            # No existing memory with this key — create a new one
            new_id = self.save(item)
            item.id = new_id
            return item

        # Existing memory found — decide how to merge
        logger.info(
            f"[MEMORY] Existing memory found for key='{item.key}': "
            f"'{existing.value}' → updating to '{item.value}'"
        )

        # Always update the value to the latest information
        existing.value = item.value
        existing.source = item.source

        # If the value is the same as before, the user is CONFIRMING it
        # → bump confidence. If it changed, reset confidence to the new item's.
        if item.value == existing.value:
            # Confirmation: increase confidence slightly, capped at 1.0
            existing.confidence = min(existing.confidence + 0.05, 1.0)
        else:
            # New value: adopt classifier's confidence for the new fact
            existing.confidence = item.confidence

        # Always take the higher importance score (user re-stating = important)
        existing.importance = max(existing.importance, item.importance)

        # Preserve embedding_id if we have one (don't overwrite with None)
        if item.embedding_id is not None:
            existing.embedding_id = item.embedding_id

        # Update in database, preserving created_at
        self.update(existing)
        return existing

    # =========================================================================
    # DELETE
    # =========================================================================

    def delete(self, memory_id: int) -> bool:
        """
        Permanently removes a memory by ID.

        WHY PERMANENT DELETE (vs. soft-delete)?
            For now, we permanently delete. A soft-delete would add an
            `is_deleted BOOLEAN` column and filter it in every SELECT.
            That's more complex for marginal benefit at this scale.

            TODO: Implement soft-delete as an "archived" state for
                  memories that should be forgotten but may need auditing.
        """
        with self.db.connection:
            cursor = self.db.connection.execute(
                "DELETE FROM user_memory WHERE id = ?;",
                (memory_id,)
            )
        success = cursor.rowcount > 0
        if success:
            logger.info(f"[MEMORY] Deleted memory id={memory_id}.")
        else:
            logger.warning(f"[MEMORY] Delete failed: no memory with id={memory_id}.")
        return success

    def delete_by_key(self, key: str) -> bool:
        """Removes a memory by its key string."""
        with self.db.connection:
            cursor = self.db.connection.execute(
                "DELETE FROM user_memory WHERE key = ?;",
                (key,)
            )
        success = cursor.rowcount > 0
        if success:
            logger.info(f"[MEMORY] Deleted memory with key='{key}'.")
        return success

    # =========================================================================
    # FUTURE EXTENSION POINTS
    # =========================================================================
    #
    # TODO: `archive(memory_id)` — mark memory as inactive without deletion.
    #       Adds `is_archived BOOLEAN DEFAULT FALSE` column (schema v2).
    #
    # TODO: `decay_confidence()` — reduce confidence of memories not
    #       confirmed in N days. Run as a background scheduled task.
    #       SELECT * WHERE updated_at < NOW() - INTERVAL '30 days'
    #       UPDATE confidence = confidence * 0.9
    #
    # TODO: `get_similar(key, threshold=0.8)` — fuzzy key matching for
    #       detecting near-duplicate memories ("fav_ide" vs "favourite_ide").
    #       Current implementation uses exact key match only.
    #
    # TODO: `vector_search(embedding, top_k=5)` — once ChromaDB/FAISS is
    #       integrated, add this method and have retrieval.py call it.
    #       The public API of MemoryRepository grows; callers don't change.
    #
    # TODO: Multi-user support — add `user_id TEXT NOT NULL` column (schema v2).
    #       All queries get a `WHERE user_id = ?` clause.
    #       MemoryManager receives a user_id at construction time.
