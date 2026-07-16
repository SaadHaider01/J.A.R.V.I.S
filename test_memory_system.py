"""
Zytrix Memory System - Full Verification Test
Run from project root: python test_memory_system.py
"""
import sys
sys.path.insert(0, '.')

from backend.memory.memory_manager import MemoryManager

PASS = "PASS"
FAIL = "FAIL"


print("=" * 60)
print("ZYTRIX MEMORY SYSTEM - FULL VERIFICATION TEST")
print("=" * 60)

# Use in-memory SQLite so no file is created during testing
mm = MemoryManager(db_path=":memory:")

# ── TEST 1: Store a preference ──────────────────────────────────────────────
print("\n[TEST 1] Preference: 'My favourite IDE is VSCode.'")
stored = mm.process_message("user", "My favourite IDE is VSCode.")
memories = mm.retrieve_all()
print(f"  Stored as long-term memory: {stored}")
print(f"  Total memories in DB: {len(memories)}")
if memories:
    m = memories[0]
    print(f"  category={m.category.value} | key={m.key} | value={m.value[:60]}")
result1 = stored and len(memories) >= 1 and "VSCode" in memories[0].value
print(f"  {PASS if result1 else FAIL}")

# ── TEST 2: Retrieve via context (simulates query after restart) ────────────
print("\n[TEST 2] Retrieval: 'What is my favourite IDE?'")
context = mm.build_context("What is my favourite IDE?")
print(f"  Context block snippet: {repr(context[:200])}")
result2 = "VSCode" in context
print(f"  {PASS + ': VSCode found in context.' if result2 else FAIL + ': VSCode NOT found.'}")

# ── TEST 3: Update existing memory — no duplicate created ───────────────────
print("\n[TEST 3] Update: 'My favourite IDE is Cursor.' (should UPDATE, not insert)")
mm.process_message("user", "My favourite IDE is Cursor.")
memories_after = mm.retrieve_all()
print(f"  Total memories after update: {len(memories_after)} (expected: 1)")
result3 = len(memories_after) == 1
print(f"  {PASS + ': No duplicate.' if result3 else FAIL + ': Duplicate created!'}")

# ── TEST 4: Noise rejection ─────────────────────────────────────────────────
print("\n[TEST 4] Noise: 'The weather is nice.' (should NOT be stored)")
count_before = len(mm.retrieve_all())
stored_noise = mm.process_message("user", "The weather is nice.")
count_after = len(mm.retrieve_all())
result4 = (not stored_noise) and (count_after == count_before)
print(f"  Stored: {stored_noise} | Memory count changed: {count_after != count_before}")
print(f"  {PASS + ': Noise rejected.' if result4 else FAIL + ': Noise was stored!'}")

# ── TEST 5: Project memory ───────────────────────────────────────────────────
print("\n[TEST 5] Project: 'I am building Zytrix.'")
mm.process_message("user", "I am building Zytrix.")
project_ctx = mm.build_context("What project am I working on?")
print(f"  Context: {repr(project_ctx[:200])}")
result5 = "Zytrix" in project_ctx
print(f"  {PASS + ': Zytrix found in context.' if result5 else FAIL + ': Zytrix NOT found.'}")

# ── TEST 6: Session summarization ───────────────────────────────────────────
print("\n[TEST 6] Session summarization on end_session()")
mm.process_message("assistant", "You prefer Cursor and you are building Zytrix.")
summary = mm.end_session()
result6 = bool(summary)
print(f"  Summary generated: {result6}")
if summary:
    print(f"  Summary text: {summary[:300]}")
print(f"  {PASS + ': Summary stored.' if result6 else FAIL + ': No summary generated.'}")

# ── RESULTS ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
all_passed = all([result1, result2, result3, result4, result5, result6])
if all_passed:
    print("ALL 6 TESTS PASSED. Memory system is working correctly.")
else:
    failures = [i+1 for i, r in enumerate([result1,result2,result3,result4,result5,result6]) if not r]
    print(f"FAILED TESTS: {failures}")
print("=" * 60)
