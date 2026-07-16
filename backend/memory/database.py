"""
=============================================================================
backend/memory/database.py
=============================================================================

WHAT THIS FILE DOES:
    Manages the raw SQLite database connection, creates the schema (tables +
    indexes), and handles schema versioning for future migrations.
    Nothing else.

WHY THIS FILE EXISTS (THE SINGLE RESPONSIBILITY PRINCIPLE):
    This module's ONLY job is "talk to SQLite". It does not:
    ─ Filter or classify memories (that's memory_classifier.py)
    ─ Decide what to retrieve (that's retrieval.py)
    ─ Orchestrate business logic (that's memory_manager.py)

    Having one file per concern makes bugs easier to find. If the tables
    are wrong, you look here. Nowhere else.

WHY SQLITE?
    ─ Zero configuration: no separate server process, no installation.
    ─ Single file database: easy to backup, inspect, and version control.
    ─ ACID compliant: transactions guarantee data integrity even on crash.
    ─ Built into Python's stdlib as `sqlite3` — no extra dependencies.
    ─ Handles millions of rows comfortably for a personal assistant.

    FUTURE: If Zytrix ever needs to serve multiple users simultaneously
    from a server, swap SQLite for PostgreSQL. The Repository layer
    (memory_repository.py) shields the rest of the app from that change.

WHY STORE THE DB IN data/?
    Keeping runtime-generated files (databases, logs, cache) separate from
    source code is standard engineering practice.

    Benefits:
    ─ `.gitignore data/` prevents accidentally committing your personal data.
    ─ Deployment is cleaner — the data directory is created fresh on first run.
    ─ Makes it obvious what is "code" vs. "state".

EDUCATIONAL CONCEPT — SQLITE CONNECTIONS:
    sqlite3.connect(path) opens a connection to the database file. Think of
    it like opening a book — you must open it before reading/writing, and
    close it when done (or use `with` context manager which does it auto).

    `check_same_thread=False` allows the connection to be used across
    threads. Python's sqlite3 module is thread-safe at the C level, but
    the Python wrapper adds an extra check by default. Since Zytrix's
    memory writes come from a single agent thread, this is safe.

EDUCATIONAL CONCEPT — TRANSACTIONS:
    A transaction groups multiple SQL operations into an atomic unit.
    Either ALL operations succeed, or NONE of them do. This protects
    against partial writes (e.g. database crash mid-operation).

    With sqlite3 in Python:
    ─ Use `conn.commit()` to save changes permanently.
    ─ Use `conn.rollback()` to undo changes if something fails.
    ─ Or use a `with conn:` block which auto-commits on success and
      auto-rolls-back on exception.

EDUCATIONAL CONCEPT — SCHEMA VERSIONING:
    Software evolves. Someday you'll need to add a column, rename a table,
    or split a table. Without schema versioning, you'd have to manually
    inspect the database to know which version it is.

    The `metadata` table stores a key-value pair: ("schema_version", "1").
    When Zytrix starts, it checks the version and can run migration code
    if the stored version is older than the current code expects.

=============================================================================
"""

import sqlite3
import logging
from pathlib import Path

# =============================================================================
# LOGGING SETUP
# =============================================================================
#
# EDUCATIONAL CONCEPT — LOGGING HIERARCHY:
#     Python's `logging` module organizes loggers in a tree by dot-notation.
#
#         "zytrix"
#             └── "zytrix.memory"
#                     └── "zytrix.memory.database"
#
#     If you configure the "zytrix" logger's level to WARNING, all children
#     inherit that unless they override it. This lets you silence all memory
#     logs with one line in config.
#
# WHY NOT `print()`?
#     - print() cannot be filtered by severity (INFO vs WARNING vs ERROR).
#     - print() has no timestamps or module names.
#     - print() cannot be redirected to a file, Syslog, or monitoring service.
#     - In production, print() pollutes stdout and is stripped by log aggregators.
#     Logging is the professional standard. Always use it.

logger = logging.getLogger("zytrix.memory.database")

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
#
# WHY pathlib.Path INSTEAD OF os.path?
#     pathlib gives you an object-oriented API with operator overloading.
#     Path("data") / "zytrix_memory.db" reads more naturally than
#     os.path.join("data", "zytrix_memory.db") and works on all platforms.

# Resolve the project root relative to this file's location.
# __file__ → "d:/J.A.R.V.I.S/backend/memory/database.py"
# .parent  → "d:/J.A.R.V.I.S/backend/memory/"
# .parent  → "d:/J.A.R.V.I.S/backend/"
# .parent  → "d:/J.A.R.V.I.S/"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# The data/ directory lives at the project root, not inside source.
_DATA_DIR = _PROJECT_ROOT / "data"

# The SQLite database file.
DB_PATH = _DATA_DIR / "zytrix_memory.db"

# The current schema version. Increment this whenever you change the schema.
# WHY AN INT? Easy to compare: version 3 > version 2 means migration needed.
SCHEMA_VERSION = 1


# =============================================================================
# SCHEMA SQL DEFINITIONS
# =============================================================================
#
# EDUCATIONAL CONCEPT — NORMALIZATION:
#     Database normalization is the process of structuring tables to reduce
#     redundancy and improve data integrity. The basic principle:
#     each piece of information should live in exactly ONE place.
#
#     WHY SEPARATE TABLES?
#     ─ user_memory has completely different columns than conversation_history.
#       Mixing them would mean many NULL columns for every row.
#     ─ Each table can be queried, indexed, and backed up independently.
#     ─ You can delete ALL conversation history without touching memories.
#
# EDUCATIONAL CONCEPT — PRIMARY KEY:
#     `id INTEGER PRIMARY KEY` in SQLite is special — it auto-increments.
#     Every INSERT automatically gets a unique ID. This ID is used as a
#     stable reference to that row forever (foreign keys, logs, etc.).
#
# EDUCATIONAL CONCEPT — NOT NULL vs NULLABLE:
#     `NOT NULL` means the column must always have a value — the database
#     enforces this, not your Python code. Use it for required fields.
#     Omitting NOT NULL (or using DEFAULT NULL) makes a field optional.
#     `embedding_id` is nullable because vector search isn't implemented yet.

_CREATE_METADATA_TABLE = """
CREATE TABLE IF NOT EXISTS metadata (
    -- Schema version tracking. One row per key.
    -- WHY: Enables safe database migrations in the future.
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_CREATE_USER_MEMORY_TABLE = """
CREATE TABLE IF NOT EXISTS user_memory (
    -- WHAT: Stores long-term facts, preferences, routines, and projects.
    -- WHY SEPARATE: Different query pattern, retention, and schema from history.

    id           INTEGER PRIMARY KEY AUTOINCREMENT,

    -- The "label" for this memory (e.g. "favourite_ide", "current_project").
    -- WHY: Consistent keys enable upsert logic (update if exists vs. insert).
    key          TEXT NOT NULL,

    -- The memory value (e.g. "VSCode", "Zytrix").
    value        TEXT NOT NULL,

    -- Category enum value (e.g. "preference", "fact", "project").
    -- WHY: Enables category-based filtering and future category-level decay.
    category     TEXT NOT NULL,

    -- Confidence score (0.0 to 1.0). How certain are we this is true?
    -- WHY: Allows future "confidence decay" — old unconfirmed memories fade.
    confidence   REAL NOT NULL DEFAULT 0.8,

    -- Importance score (0.0 to 1.0). How significant is this memory?
    -- WHY: Higher importance = prioritized during retrieval ranking.
    -- SEPARATE FROM confidence because you can be 100% confident about
    -- something unimportant (e.g. "the user's cat is named Whiskers").
    importance   REAL NOT NULL DEFAULT 0.5,

    -- Source of this memory (e.g. "conversation", "manual", "system").
    -- WHY: Traceability — crucial for debugging and auditing.
    source       TEXT NOT NULL DEFAULT 'conversation',

    -- Placeholder for future vector embedding integration.
    -- WHY NOW: Prevents a painful schema migration later. Costs nothing NULL.
    -- FUTURE: Store ChromaDB/FAISS vector ID here for semantic search.
    embedding_id TEXT,

    created_at   TEXT NOT NULL,  -- ISO 8601 datetime string. WHY TEXT: SQLite
    updated_at   TEXT NOT NULL   -- has no native DATETIME type; TEXT is universal.
);
"""

_CREATE_CONVERSATION_HISTORY_TABLE = """
CREATE TABLE IF NOT EXISTS conversation_history (
    -- WHAT: Stores raw conversation turns (user said X, assistant replied Y).
    -- WHY: Provides recent context for the LLM within a session.
    -- DIFFERENT FROM user_memory: History is temporary; memories are permanent.

    id         INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Groups messages into a single conversation session.
    -- WHY: Lets us retrieve a full session for summarization when it ends.
    session_id TEXT NOT NULL,

    role       TEXT NOT NULL,  -- "user" or "assistant"
    message    TEXT NOT NULL,
    timestamp  TEXT NOT NULL
);
"""

_CREATE_CONVERSATION_SUMMARY_TABLE = """
CREATE TABLE IF NOT EXISTS conversation_summary (
    -- WHAT: Stores a condensed summary of a past conversation session.
    -- WHY NOT KEEP FULL HISTORY?
    --   Context windows are finite. 200 messages = ~40,000 tokens.
    --   A 5-bullet summary captures the same useful info in ~200 tokens.
    --   This is called "lossy compression with intent preservation".

    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL UNIQUE,  -- One summary per session (UNIQUE enforces this)
    summary    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

# =============================================================================
# INDEXES
# =============================================================================
#
# EDUCATIONAL CONCEPT — INDEXES:
#     Without an index, a SELECT WHERE query scans EVERY row in the table
#     to find matches. This is O(n) — slow for large tables.
#
#     An INDEX is a separate data structure (usually a B-tree) that maps
#     column values to row IDs. With it, lookups become O(log n) — extremely
#     fast even for millions of rows.
#
#     ANALOGY: A book's index at the back vs. reading every page to find
#     a topic. The index lets you jump directly to the right page.
#
#     TRADE-OFF: Indexes consume disk space and slow down INSERTs slightly
#     (the index must be updated too). Always index columns you frequently
#     search/filter on, not every column.
#
# WHY THESE SPECIFIC INDEXES?
#     ─ `key`: Used in upsert detection ("does this key already exist?")
#     ─ `category`: Used in category-filtered retrieval queries
#     ─ `updated_at`: Used to sort by recency ("most recently updated first")
#     ─ `session_id`: Used to retrieve all messages for a given session

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_user_memory_key        ON user_memory(key);",
    "CREATE INDEX IF NOT EXISTS idx_user_memory_category   ON user_memory(category);",
    "CREATE INDEX IF NOT EXISTS idx_user_memory_updated_at ON user_memory(updated_at);",
    "CREATE INDEX IF NOT EXISTS idx_conv_history_session   ON conversation_history(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_conv_summary_session   ON conversation_summary(session_id);",
]


# =============================================================================
# DATABASE CLASS
# =============================================================================

class Database:
    """
    WHAT: Manages the SQLite connection lifecycle and schema initialization.

    WHY A CLASS (NOT JUST FUNCTIONS)?
        A class can hold state (the `connection` object). Functions would need
        to accept the connection as a parameter every time, which is verbose.
        The class also gives us a natural `__init__` to run setup code.

    WHY NOT A SINGLETON?
        For now, ZytrixAgent creates one MemoryManager which creates one
        Database. That's effectively a singleton by construction. We avoid
        the Singleton pattern because it makes testing harder (you can't
        swap in a test database easily). See `MemoryManager` for how to
        pass a custom path during testing.

    THREAD SAFETY NOTE:
        SQLite supports one writer at a time. For a personal assistant that
        processes one voice command at a time, this is perfectly fine.
        If Zytrix ever becomes multi-threaded (parallel speech + background
        tasks writing simultaneously), consider using a connection pool
        or switching to PostgreSQL.
    """

    def __init__(self, db_path: Path = DB_PATH):
        """
        Opens the SQLite connection and initializes the schema on first run.

        WHY `check_same_thread=False`?
            Python's sqlite3 adds a thread-safety guard that raises an error
            if the same connection is used from a different thread than the
            one that created it. We disable this because:
            1. SQLite's C library IS thread-safe at the write level.
            2. Our architecture ensures only MemoryManager calls the database,
               and MemoryManager is always called from the agent thread.
            If you add background threads, protect writes with threading.Lock().

        WHY `timeout=30`?
            If another process has the DB open (e.g. DB Browser for SQLite),
            SQLite will wait up to 30 seconds for the lock to clear before
            raising OperationalError. Better than an instant crash.
        """
        self._db_path = db_path
        self._ensure_data_dir()
        self.connection = sqlite3.connect(
            str(db_path),
            check_same_thread=False,
            timeout=30,
        )

        # Make rows return as `sqlite3.Row` objects (dict-like access by name).
        # Without this, rows are plain tuples — you'd access fields by index:
        #     row[0], row[1], row[2]  ← fragile, breaks if column order changes
        # With Row factory:
        #     row["key"], row["value"]  ← clear and resilient
        self.connection.row_factory = sqlite3.Row

        # Enable WAL (Write-Ahead Logging) mode for better concurrent read
        # performance. In WAL mode, readers don't block writers and vice versa.
        # WHY: Even if Zytrix is single-threaded, DB Browser for SQLite can
        # read the database without blocking Zytrix writes.
        self.connection.execute("PRAGMA journal_mode=WAL;")

        self._initialize_schema()
        logger.info(f"[MEMORY] Database initialized at: {self._db_path}")

    def _ensure_data_dir(self):
        """
        Creates the data/ directory if it doesn't exist yet.

        WHY `exist_ok=True`?
            Without it, `mkdir` raises FileExistsError if the directory
            already exists. `exist_ok=True` makes it idempotent — safe
            to call on every startup.

        WHY `parents=True`?
            If `data/` has a parent that doesn't exist (unlikely here but
            good practice), create the whole path, not just the final dir.
        """
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"[MEMORY] Data directory ensured: {self._db_path.parent}")

    def _initialize_schema(self):
        """
        Creates all tables and indexes on first run, then checks schema version.

        WHY `CREATE TABLE IF NOT EXISTS`?
            Makes this method IDEMPOTENT — safe to call every time Zytrix
            starts. It skips creation if the table already exists.
            Without IF NOT EXISTS, you'd get an error on the second run.

        EDUCATIONAL CONCEPT — TRANSACTIONS:
            We wrap all table creation in a single transaction using `with conn:`.
            If ANY statement fails (e.g. disk full), SQLite automatically
            rolls back all changes. This prevents a half-initialized database.
        """
        with self.connection:
            # Create tables in dependency order (metadata first, as it's standalone)
            self.connection.execute(_CREATE_METADATA_TABLE)
            self.connection.execute(_CREATE_USER_MEMORY_TABLE)
            self.connection.execute(_CREATE_CONVERSATION_HISTORY_TABLE)
            self.connection.execute(_CREATE_CONVERSATION_SUMMARY_TABLE)

            # Create performance indexes
            for index_sql in _CREATE_INDEXES:
                self.connection.execute(index_sql)

        # Check and set schema version (outside the table-creation transaction)
        self._ensure_schema_version()

    def _ensure_schema_version(self):
        """
        Reads the stored schema version and handles migrations.

        HOW SCHEMA VERSIONING WORKS:
            ─ On first run: no version exists → we insert SCHEMA_VERSION.
            ─ On subsequent runs: compare stored version to SCHEMA_VERSION.
            ─ If stored < current: run migration code (future feature).
            ─ If stored == current: nothing to do.

        WHY IS THIS IMPORTANT?
            Imagine you add a new `tags` column to user_memory in v2.
            Without versioning, Zytrix would crash trying to INSERT into
            a column that doesn't exist in the user's v1 database.
            With versioning, you detect "this is v1, run ALTER TABLE to add tags".
        """
        cursor = self.connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version';"
        )
        row = cursor.fetchone()

        if row is None:
            # First run — record the current schema version
            with self.connection:
                self.connection.execute(
                    "INSERT INTO metadata (key, value) VALUES ('schema_version', ?);",
                    (str(SCHEMA_VERSION),)
                )
            logger.info(f"[MEMORY] Schema initialized at version {SCHEMA_VERSION}.")
        else:
            stored_version = int(row["value"])
            if stored_version < SCHEMA_VERSION:
                # TODO: Run migration scripts here in a future version.
                # Example pattern:
                #   if stored_version < 2:
                #       self._migrate_v1_to_v2()
                #   if stored_version < 3:
                #       self._migrate_v2_to_v3()
                logger.warning(
                    f"[MEMORY] Schema version mismatch: DB={stored_version}, "
                    f"Code={SCHEMA_VERSION}. Migration needed (TODO)."
                )
            else:
                logger.debug(f"[MEMORY] Schema version {stored_version} is current.")

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """
        Executes a single SQL statement and returns the cursor.

        WHY A WRAPPER METHOD?
            Direct access to `self.connection.execute()` would work, but
            centralizing all SQL through this method lets us:
            ─ Add logging for every query (useful for debugging)
            ─ Add query timing / performance monitoring
            ─ Add query validation or sanitization in one place

        EDUCATIONAL CONCEPT — PARAMETERIZED QUERIES:
            ALWAYS use `?` placeholders, NEVER string formatting:
                ❌ WRONG:  f"SELECT * FROM user_memory WHERE key = '{key}'"
                ✅ RIGHT:  "SELECT * FROM user_memory WHERE key = ?", (key,)

            String formatting creates SQL INJECTION vulnerabilities.
            With placeholders, SQLite handles escaping automatically.
        """
        logger.debug(f"[MEMORY] SQL: {sql.strip()[:80]} | params={params}")
        return self.connection.execute(sql, params)

    def commit(self):
        """Persists all pending changes to disk."""
        self.connection.commit()

    def close(self):
        """
        Closes the database connection cleanly.

        WHY CLOSE EXPLICITLY?
            Python's garbage collector will eventually close it, but
            "eventually" isn't good enough when you have uncommitted data.
            Explicit close ensures all WAL pages are flushed to the main DB.
        """
        if self.connection:
            self.connection.close()
            logger.info("[MEMORY] Database connection closed.")

    def __enter__(self):
        """Enables `with Database() as db:` context manager syntax."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Closes the connection when exiting the `with` block."""
        self.close()
        return False  # Don't suppress exceptions
