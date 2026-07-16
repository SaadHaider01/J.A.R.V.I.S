"""
=============================================================================
backend/memory/__init__.py
=============================================================================

WHAT THIS FILE DOES:
    Declares `backend.memory` as a Python package and exposes the public API.

EDUCATIONAL CONCEPT — PYTHON PACKAGES:
    A directory becomes a Python "package" (importable module) when it
    contains an `__init__.py` file. Without it, `import backend.memory`
    would fail with ModuleNotFoundError.

WHY CONTROL EXPORTS HERE?
    By listing what to import in __init__.py, we define the PUBLIC API
    of this package. External code (agent.py) imports from the package:

        from backend.memory import MemoryManager

    NOT from the internal module path:

        from backend.memory.memory_manager import MemoryManager  ← implementation detail

    Benefits:
    ─ If we rename memory_manager.py to orchestrator.py, agent.py doesn't break.
    ─ It's immediately clear what this package provides vs. internal details.
    ─ Consistent with Python library conventions (__init__.py = public interface).

PUBLIC API:
    Only MemoryManager is exported. All other classes are internal.
    The NLP agent only needs MemoryManager.

=============================================================================
"""

from backend.memory.memory_manager import MemoryManager

__all__ = ["MemoryManager"]
