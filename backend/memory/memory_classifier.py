"""
=============================================================================
backend/memory/memory_classifier.py
=============================================================================

WHAT THIS FILE DOES:
    A lightweight, rule-based text classifier that answers two questions:
    1. "Should this message be stored as a long-term memory?"
    2. "If yes, what category, confidence, and importance should it have?"

WHY NOT STORE EVERYTHING?
    Storing every message would be catastrophically bad for multiple reasons:

    1. SIGNAL-TO-NOISE RATIO:
       Most conversation turns carry no persistent information worth remembering.
       "The weather is nice" is conversational fluff. "My name is Saad" is
       a permanent fact. Treating them the same degrades retrieval quality —
       ZYTRIX would return irrelevant noise when you ask "what do you know
       about me?".

    2. CONTEXT WINDOW POLLUTION:
       Retrieved memories are injected into the LLM prompt. If the database
       contains 10,000 trivial messages, retrieval becomes useless — you'd
       retrieve 10,000 irrelevant entries for any keyword.

    3. STORAGE GROWTH:
       A personal assistant might handle 10,000 voice commands per year.
       If you stored every word, you'd have a bloated database with almost
       no useful information.

    4. LLM ANALOGY:
       Humans don't remember every sentence they've ever heard. They
       selectively encode meaningful experiences into long-term memory.
       The classifier mimics this biological memory selectivity.

DESIGN PHILOSOPHY — START WITH RULES, EVOLVE TO ML:
    Rule-based classifiers are:
    ─ Transparent: you can read exactly WHY a decision was made.
    ─ Deterministic: same input always produces same output.
    ─ Zero-latency: no API call needed.
    ─ Debuggable: add `print(rule_triggered)` and you immediately know why.

    Machine learning classifiers (BERT, fine-tuned transformers) are:
    ─ More accurate for edge cases.
    ─ Black-box: hard to debug.
    ─ Require training data, computational resources, and maintenance.

    FOR A PERSONAL ASSISTANT, rule-based is the RIGHT starting point.
    The rules cover 95%+ of real-world cases. When they fail, you add
    a new rule — much faster than retraining a model.

REPLACEABLE DESIGN:
    This module exposes a single function: `classify(text)`.
    MemoryManager only calls `classify()`.
    Tomorrow you could replace this entire file with an LLM-based classifier
    that calls Groq with a short prompt — MemoryManager never changes.
    This is the OPEN/CLOSED PRINCIPLE: open for extension (swap classifier),
    closed for modification (MemoryManager doesn't need to change).

=============================================================================
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

from backend.memory.memory_models import MemoryCategory

logger = logging.getLogger("zytrix.memory.classifier")


# =============================================================================
# RESULT DATA STRUCTURE
# =============================================================================

@dataclass
class ClassificationResult:
    """
    WHAT: The output of the classifier.

    WHY A DATACLASS INSTEAD OF A TUPLE?
        Returning `(True, MemoryCategory.PREFERENCE, 0.9, 0.7)` is fragile —
        callers must remember the exact order. A dataclass gives named fields:
            result.should_store
            result.category
        This is more readable and immune to ordering bugs.
    """
    should_store : bool
    category     : Optional[MemoryCategory] = None
    key          : Optional[str] = None       # Suggested database key (e.g. "favourite_ide")
    confidence   : float = 0.8
    importance   : float = 0.5


# =============================================================================
# CLASSIFICATION RULES
# =============================================================================
#
# Each rule is a dict with:
#   patterns    : List of regex/string patterns to match against the input text.
#   category    : The MemoryCategory to assign if the rule fires.
#   confidence  : How confident we are this is an accurate memory (0.0 to 1.0).
#   importance  : How important this memory is to preserve (0.0 to 1.0).
#   key_extract : Optional regex pattern to extract the memory KEY from the text.
#                 e.g. "favourite (.+)" → matches "favourite IDE", "favourite editor"
#
# WHY EXPLICIT CONFIDENCE AND IMPORTANCE PER RULE?
#     Different phrases carry different levels of certainty and significance.
#     "My name is Saad" → very important, very confident (direct declaration).
#     "I kind of prefer dark mode" → lower confidence (hedged language).
#
# EDUCATIONAL CONCEPT — NAMED TUPLES vs DICT vs DATACLASS FOR RULES:
#     We use plain dicts here for readability. In a larger system you'd
#     define a `ClassificationRule` dataclass for type safety.

_RULES = [
    # ─────────────────────────────────────────────────────────────────────────
    # NAME & IDENTITY — Very high importance, very high confidence
    # These are the most fundamental facts about the user.
    # ─────────────────────────────────────────────────────────────────────────
    {
        "patterns"   : [
            r"\bmy name is\b",
            r"\bi('m| am) called\b",
            r"\bpeople call me\b",
        ],
        "category"   : MemoryCategory.FACT,
        "confidence" : 1.0,
        "importance" : 1.0,
        "key"        : "user_name",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # PREFERENCES — High confidence when explicitly stated
    # ─────────────────────────────────────────────────────────────────────────
    {
        "patterns"   : [
            r"\bmy (favourite|favorite|preferred|go-to)\b",
            r"\bi (prefer|like|love|use|enjoy)\b.{0,30}\b(editor|ide|tool|browser|language|app|framework|os|music|food)\b",
            r"\bi always use\b",
            r"\bi('m| am) a (vim|vscode|cursor|emacs|nvim|neovim)\b",
        ],
        "category"   : MemoryCategory.PREFERENCE,
        "confidence" : 0.9,
        "importance" : 0.7,
        "key"        : None,  # Key will be extracted dynamically
        # key_extract: A regex that captures the SUBJECT of the preference.
        # WHY SUBJECT, NOT VALUE?
        #   'My favourite IDE is VSCode' and 'My favourite IDE is Cursor'
        #   are UPDATES to the SAME preference (favourite_ide).
        #   If we include the value in the key, they become DIFFERENT keys
        #   and a duplicate is created instead of an update.
        #   We capture the subject ('favourite ide', 'preferred language') as the key.
        "key_extract" : r"\bmy\s+(favourite|favorite|preferred|go-to)\s+(\w+(?:\s+\w+)?)",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # FACTS — Stated facts about the user's life
    # ─────────────────────────────────────────────────────────────────────────
    {
        "patterns"   : [
            r"\bi (graduate|graduated|will graduate|finish|finished)\b",
            r"\bi('m| am) (studying|a student|in (my|the) (first|second|third|fourth|final) year)\b",
            r"\bi work (at|for|in|as)\b",
            r"\bmy (job|role|position|title) is\b",
            r"\bi live in\b",
            r"\bi('m| am) from\b",
            r"\bmy (age|birthday|birth date) is\b",
            r"\bi was born\b",
        ],
        "category"   : MemoryCategory.FACT,
        "confidence" : 0.9,
        "importance" : 0.8,
        "key"        : None,
    },

    # ─────────────────────────────────────────────────────────────────────────
    # PROJECTS — Things the user is building or working on
    # ─────────────────────────────────────────────────────────────────────────
    {
        "patterns"   : [
            r"\bi('m| am) (building|working on|developing|creating|making)\b",
            r"\bmy (project|app|application|startup|side project|hobby project|thesis)\b",
            r"\bi('m| am) (coding|programming|hacking on)\b.{0,30}(project|app|tool|bot|assistant|system)\b",
        ],
        "category"   : MemoryCategory.PROJECT,
        "confidence" : 0.85,
        "importance" : 0.8,
        "key"        : "current_project",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # ROUTINES — Habitual patterns
    # ─────────────────────────────────────────────────────────────────────────
    {
        "patterns"   : [
            r"\bi (usually|normally|typically|always|often|regularly)\b",
            r"\bevery (morning|evening|night|day|week)\b",
            r"\bmy (routine|habit|schedule|workflow)\b",
            r"\bi (wake up|sleep|code|work|study|workout|run|exercise)\b.{0,20}(at|around|in the)\b",
        ],
        "category"   : MemoryCategory.ROUTINE,
        "confidence" : 0.8,
        "importance" : 0.5,
        "key"        : None,
    },

    # ─────────────────────────────────────────────────────────────────────────
    # REMINDERS — Explicit requests to remember
    # ─────────────────────────────────────────────────────────────────────────
    {
        "patterns"   : [
            r"\bremember (that|this|to)\b",
            r"\bdon'?t forget\b",
            r"\bremind me\b",
            r"\bkeep in mind\b",
            r"\bnote that\b",
            r"\bmake a note\b",
        ],
        "category"   : MemoryCategory.REMINDER,
        "confidence" : 1.0,   # User explicitly asked to remember — maximum confidence
        "importance" : 0.9,   # Explicit requests are always important
        "key"        : None,
    },

    # ─────────────────────────────────────────────────────────────────────────
    # GENERAL IMPORTANT — Catch-all for direct declarations
    # ─────────────────────────────────────────────────────────────────────────
    {
        "patterns"   : [
            r"\bmy (setup|config|configuration|environment|stack) is\b",
            r"\bi use\b.{0,15}(for|to|when)\b",
        ],
        "category"   : MemoryCategory.GENERAL,
        "confidence" : 0.75,
        "importance" : 0.5,
        "key"        : None,
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# NOISE PATTERNS — Messages that are NEVER worth storing
# ─────────────────────────────────────────────────────────────────────────────
#
# These patterns fire BEFORE the storage rules. If a message matches any of
# these, it's immediately rejected as conversational noise.
#
# WHY EXPLICIT REJECTION RULES?
#     Without them, "The weather is nice" could technically match a broad
#     "general" rule. Rejection rules create a hard stop for obvious noise.

_NOISE_PATTERNS = [
    # Questions about the world (not about the user)
    r"^(what|who|where|when|how|why|is|are|can|could|should|would|do|does)\b",
    # Greetings and social phrases
    r"^(hi|hello|hey|good morning|good evening|good night|bye|goodbye|thanks|thank you|please|okay|ok|sure|alright|yep|nope|yes|no)\b",
    # Short messages (under 15 characters) — rarely contain meaningful info
    # (checked in code, not regex)
    # Commands directed at ZYTRIX (not user self-description)
    r"^(open|launch|close|play|stop|pause|set|get|search|find|show|tell|check|turn)\b",
    # Weather, news, general world facts
    r"\b(weather|temperature|news|sports|stock|price|forecast)\b",
]

# Compile all patterns once at module load time for performance.
# WHY COMPILE? `re.compile()` parses the regex pattern into an internal
# representation. Calling `re.search(pattern, text)` every time re-parses
# the pattern. With compile, parsing happens once; matching is much faster
# especially if this function is called hundreds of times.
_compiled_rules = [
    {
        **rule,
        "compiled": [re.compile(p, re.IGNORECASE) for p in rule["patterns"]],
        # Also compile the key_extract pattern if present
        "key_extract_compiled": (
            re.compile(rule["key_extract"], re.IGNORECASE)
            if rule.get("key_extract")
            else None
        ),
    }
    for rule in _RULES
]

_compiled_noise = [re.compile(p, re.IGNORECASE) for p in _NOISE_PATTERNS]


# =============================================================================
# PUBLIC API
# =============================================================================

def classify(text: str) -> ClassificationResult:
    """
    WHAT: Determines if a message should be stored as a long-term memory.

    RETURNS:
        ClassificationResult with:
        ─ should_store = True/False
        ─ category, confidence, importance (if should_store is True)
        ─ key: a suggested database key (may be None, caller generates one)

    ARCHITECTURE NOTE:
        This is the ONLY public function in this module.
        MemoryManager calls `classify(text)`. The rule engine, noise filters,
        and scoring logic are all private implementation details.

    THE DECISION PIPELINE:
        1. Reject if text is too short (< 10 chars) — too little info
        2. Reject if any NOISE pattern matches — obvious non-memory content
        3. Match against STORAGE rules in order — first match wins
        4. If no rule matches, reject (most messages should be rejected)

    WHY "FIRST MATCH WINS"?
        Rules are ordered from most specific to most general. "My name is"
        is more specific than "I usually..." We want the most precise
        category label. If multiple rules matched and were combined, you'd
        get ambiguous categories.
    """
    text = text.strip()

    # ── Step 1: Minimum length check ──────────────────────────────────────────
    if len(text) < 10:
        logger.debug(f"[MEMORY] Ignored (too short): '{text}'")
        return ClassificationResult(should_store=False)

    text_lower = text.lower()

    # ── Step 2: Noise filter — reject obvious conversational fluff ────────────
    for noise_re in _compiled_noise:
        if noise_re.search(text_lower):
            logger.debug(
                f"[MEMORY] Ignored (noise pattern '{noise_re.pattern[:30]}'): "
                f"'{text[:60]}'"
            )
            return ClassificationResult(should_store=False)

    # ── Step 3: Match against storage rules (first match wins) ────────────────
    for rule in _compiled_rules:
        for pattern_re in rule["compiled"]:
            if pattern_re.search(text_lower):
                # A storage rule fired!
                suggested_key = rule.get("key")

                # If the rule has a key_extract pattern, use it to derive a
                # stable key from the SUBJECT of the statement, not the VALUE.
                #
                # EDUCATIONAL CONCEPT — WHY SUBJECT-BASED KEYS?
                #   "My favourite IDE is VSCode" → key = "preference_favourite_ide"
                #   "My favourite IDE is Cursor" → key = "preference_favourite_ide" (SAME!)
                # This allows the repository to UPDATE (upsert) the existing memory
                # instead of creating a duplicate with a different key.
                #
                # Without this, keys include the value:
                #   key = "preference_favourite_ide_vscode"
                #   key = "preference_favourite_ide_cursor"
                # These are treated as two DIFFERENT memories — a bug.

                key_extract_re = rule.get("key_extract_compiled")
                if suggested_key is None and key_extract_re:
                    match = key_extract_re.search(text_lower)
                    if match:
                        # Join all non-None capture groups into the key subject
                        groups = [g for g in match.groups() if g]
                        subject = "_".join(groups).replace(" ", "_")
                        suggested_key = f"{rule['category'].value}_{subject[:40]}"

                # Fallback: generate a key from the category + message words
                if suggested_key is None:
                    suggested_key = _generate_key(text, rule["category"])

                # Normalize the key: lowercase, replace spaces, strip trailing underscores
                suggested_key = re.sub(r'[^\w]', '_', suggested_key.lower()).strip('_')

                result = ClassificationResult(
                    should_store = True,
                    category     = rule["category"],
                    key          = suggested_key,
                    confidence   = rule["confidence"],
                    importance   = rule["importance"],
                )
                logger.info(
                    f"[MEMORY] Classified as {rule['category'].value} "
                    f"(confidence={rule['confidence']}, importance={rule['importance']}): "
                    f"'{text[:60]}' -> key='{suggested_key}'"
                )
                return result

    # ── Step 4: No rule matched — this is just conversation ───────────────────
    logger.debug(f"[MEMORY] Ignored (no rule match): '{text[:60]}'")
    return ClassificationResult(should_store=False)


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

def _generate_key(text: str, category: MemoryCategory) -> str:
    """
    Generates a normalized database key from the message text.

    WHY NORMALIZE KEYS?
        Consistent keys are essential for upsert detection. If "My favourite
        IDE is VSCode" generates key "favourite_ide" and later "My favourite
        IDE is Cursor" generates "my_favourite_ide", they become DIFFERENT
        database entries. We normalize to avoid this.

    NORMALIZATION STEPS:
        1. Lowercase
        2. Remove common "noise" words (I, my, is, the...)
        3. Replace spaces/special chars with underscores
        4. Truncate to a reasonable length

    EXAMPLE:
        "I'm building Zytrix" → category=PROJECT → "project_zytrix"
        "My favourite IDE is VSCode" → category=PREFERENCE → "preference_ide"
    """
    # Remove very common words that don't add meaning to the key
    _STOP_WORDS = {
        "i", "my", "me", "is", "are", "am", "the", "a", "an", "to",
        "it", "its", "that", "this", "in", "on", "at", "of", "and",
        "or", "for", "with", "very", "really", "usually", "often"
    }

    words = re.sub(r"[^\w\s]", "", text.lower()).split()
    meaningful_words = [w for w in words if w not in _STOP_WORDS]

    # Use up to 3 meaningful words for the key
    key_words = meaningful_words[:3] if meaningful_words else ["memory"]

    # Prefix with category for namespace clarity
    key = f"{category.value}_{'_'.join(key_words)}"

    # Truncate to 50 characters max (SQLite TEXT is unlimited, but keep keys short)
    return key[:50]


# =============================================================================
# FUTURE EXTENSION POINTS
# =============================================================================
#
# TODO: Confidence modifiers for hedged language:
#       "I think I prefer..." → reduce confidence by 0.1
#       "I definitely prefer..." → keep or increase confidence
#       Implement as a post-processing step after rule matching.
#
# TODO: Entity extraction for automatic key generation:
#       "My favourite IDE is VSCode" → key="favourite_ide", value="VSCode"
#       Currently the full message is stored as the value. Entity extraction
#       would separate label from value automatically.
#
# TODO: LLM-based classification:
#       Replace this entire file with a Groq call:
#           prompt = f"Should this be stored as a memory? If yes, what category?
#                      Reply with JSON: {{should_store, category, key, confidence}}\n{text}"
#       The public `classify(text)` signature stays identical.
#
# TODO: Contradiction detection:
#       If the new message contradicts a stored memory
#       (e.g. "I don't use VSCode anymore"), lower the confidence of the
#       existing memory rather than blindly overwriting it.
#       Requires querying the repository during classification — inject
#       MemoryRepository into the classifier as an optional dependency.
#
# TODO: Negative preference handling:
#       "I hate Jira" → store as PREFERENCE with value="dislikes Jira"
#       Currently ignored because "I hate" doesn't match any rule.
