"""
screen_context.py — Perception to Understanding
================================================

WHY THIS MODULE EXISTS
-----------------------
After the pipeline runs, we have two raw outputs:
  - OCRResult:      extracted text + bounding boxes (from ocr.py)
  - VisionResponse: a natural-language description (from vision_models.py)

These are useful, but they are not yet actionable.  The Prompt Builder and
the LLM need structured, labelled context:

    "The user is in VSCode, editing a Python file, and there is a
     ModuleNotFoundError on screen."

Not:

    "The screen contains the text 'ModuleNotFoundError: No module named
     requests', line 4, and several import statements."

screen_context.py bridges that gap.  It is the LAST module in the pipeline
that is allowed to structure information.  After this module, the structured
ScreenContext object is handed to the Prompt Builder, which formats it for
the LLM.  The LLM then reasons about it.

WHY THIS MODULE MUST NEVER REASON
-----------------------------------
The pipeline contract:

    screen_context.py  → classifies and structures   (THIS module)
    Prompt Builder     → formats for the LLM          (prompt_builder.py)
    LLM                → reasons and generates text   (AI model)

If screen_context.py reasons ("this is definitely a React bug caused by a
missing dependency"), it:
  - Encodes assumptions that may be wrong.
  - Prevents the LLM from forming its own interpretation.
  - Adds hallucination risk.
  - Breaks the single-responsibility contract.

This module classifies.  The LLM reasons.  That boundary is inviolable.

WHY CONSERVATIVE INFERENCE
----------------------------
A false positive (claiming the app is VSCode when it's not) poisons the LLM's
context.  A false negative (ApplicationType.UNKNOWN) is recoverable — the LLM
can still reason from the raw text.

Our rule: only assert a classification if we have positive evidence for it.
Evidence absence → UNKNOWN.  Evidence presence → classify with a confidence
score that honestly reflects the strength of evidence.

WHY ContextConfidence (per-field confidence scores)
----------------------------------------------------
A single top-level confidence score is misleading:
  - We might be 95% sure the app is VSCode (window title says "VS Code")
  - But only 60% sure the content is Python (no explicit file extension found)
  - And 0% sure of the detected website (not a browser)

Separate per-field confidence scores let the Prompt Builder and LLM decide
how much weight to give each piece of context.

WHY InteractionTarget IS RESERVED BUT EMPTY
--------------------------------------------
Phase 3 GUI automation requires knowing WHERE clickable targets are on screen
(buttons, links, input fields).  The spatial data (bounding boxes from
TextBlock) already flows through the pipeline.  By defining InteractionTarget
now and including interaction_candidates in ScreenContext, we:
  - Reserve the field name and type before Phase 3 starts.
  - Allow Phase 3 to populate it without any breaking changes to ScreenContext.
  - Make the architectural intent explicit to future contributors.

NOT implementing the detection logic now is intentional.  The hook exists.
The detection engine comes later.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .logger import VisionLogger
from .ocr import OCRResult
from .vision_models import VisionResponse

log = VisionLogger(__name__)


# ---------------------------------------------------------------------------
# ApplicationType — what application is open?
# ---------------------------------------------------------------------------
class ApplicationType(Enum):
    """
    Classifies the primary application visible on screen.

    WHY CLASSIFICATION (not free-form text)
    ----------------------------------------
    An enum lets the Prompt Builder, metrics layer, and future routing logic
    branch on application type without parsing strings.  Compare:

    BAD:  if context.application == "VS Code" or "vscode" or "VSCode":
    GOOD: if context.application == ApplicationType.VSCODE:

    WHY THESE SPECIFIC VALUES
    --------------------------
    These are the applications most likely to appear in a developer assistant
    context.  Phase 3 can extend the enum as new applications are detected.
    UNKNOWN is always the default — never guess.

    Classification signals used per type
    -------------------------------------
    VSCODE       : "code", ".py/.ts/.js/.rs", "explorer", "terminal" in text
    TERMINAL     : shell prompts ($, >, #, %, PS>), command patterns
    BROWSER      : URL patterns (http://, www., .com/), navigation UI text
    PDF_VIEWER   : page number patterns, "Adobe", "Preview", PDF header text
    FILE_EXPLORER: path patterns, file listing structure, size units (KB/MB)
    IMAGE_VIEWER : minimal text, image format keywords (JPEG, PNG, EXIF)
    CHAT_APP     : message bubbles pattern, usernames with timestamps
    OFFICE       : Microsoft Office / LibreOffice document indicators
    UNKNOWN      : Insufficient evidence for any of the above
    """
    VSCODE        = "VSCODE"
    TERMINAL      = "TERMINAL"
    BROWSER       = "BROWSER"
    PDF_VIEWER    = "PDF_VIEWER"
    FILE_EXPLORER = "FILE_EXPLORER"
    IMAGE_VIEWER  = "IMAGE_VIEWER"
    CHAT_APP      = "CHAT_APP"
    OFFICE        = "OFFICE"
    UNKNOWN       = "UNKNOWN"


# ---------------------------------------------------------------------------
# ContentType — what kind of content is on screen?
# ---------------------------------------------------------------------------
class ContentType(Enum):
    """
    Classifies the primary content type visible on screen, independently of
    the application that is displaying it.

    WHY SEPARATE FROM ApplicationType
    ------------------------------------
    Application and content are orthogonal:
      - Chrome can display CODE (GitHub), ERROR (500 page), DOCUMENT (a blog post)
      - VSCode can display CODE, ERROR (stacktrace), TABLE (a CSV preview)
      - Terminal can display CODE (a script listing), ERROR (an exception)

    Separating them gives the Prompt Builder two independent axes to work with
    when choosing how to frame the LLM's context prompt.

    Classification signals
    ----------------------
    CODE     : Programming keywords, indentation, {}, =>, def, class, function
    ERROR    : "Error", "Exception", "Traceback", "stack trace", "failed", "fatal"
    DOCUMENT : Long prose, paragraph structure, headers, numbered sections
    WEBPAGE  : URL text, HTML/CSS patterns, "cookies", navigation menus
    IMAGE    : Minimal text, image format indicators, canvas/viewport references
    CHAT     : Message timestamps, "@" mentions, "sent", "delivered", "typing"
    TABLE    : Tabular alignment, column headers, repeated delimiters (|, \t)
    MIXED    : Multiple content types detected with similar evidence strength
    UNKNOWN  : Insufficient evidence
    """
    CODE     = "CODE"
    ERROR    = "ERROR"
    DOCUMENT = "DOCUMENT"
    WEBPAGE  = "WEBPAGE"
    IMAGE    = "IMAGE"
    CHAT     = "CHAT"
    TABLE    = "TABLE"
    MIXED    = "MIXED"
    UNKNOWN  = "UNKNOWN"


# ---------------------------------------------------------------------------
# InteractionTarget — RESERVED for Phase 3 GUI automation
# ---------------------------------------------------------------------------
@dataclass
class InteractionTarget:
    """
    [RESERVED — DO NOT POPULATE IN PHASE 2]

    Describes a UI element that could be clicked, typed into, or otherwise
    interacted with in a future Phase 3 GUI automation pipeline.

    WHY DEFINE NOW
    --------------
    Phase 3 automation ("click the Run button", "press Submit") depends on
    knowing WHERE interactive elements are on screen.  The spatial data
    (TextBlock bounding boxes) already flows through the pipeline.

    By defining this dataclass now and reserving the field in ScreenContext,
    Phase 3 can populate interaction_candidates without any breaking change
    to the data contract.

    Fields
    ------
    label      : Human-readable description of the element (e.g. "Run button").
    x, y       : Top-left pixel coordinates of the element in screen space.
    width      : Element width in pixels.
    height     : Element height in pixels.
    confidence : Confidence that this element is correctly identified [0.0, 1.0].
    element_type: Optional hint about the element type (e.g. "button", "input", "link").
    """
    label:        str
    x:            int
    y:            int
    width:        int
    height:       int
    confidence:   float
    element_type: Optional[str] = None


# ---------------------------------------------------------------------------
# ContextConfidence — per-field certainty scores
# ---------------------------------------------------------------------------
@dataclass
class ContextConfidence:
    """
    Granular confidence scores for each classification in ScreenContext.

    WHY PER-FIELD (not one global score)
    -------------------------------------
    Different fields have different evidence quality.  The window title may
    say "VS Code" explicitly (application confidence = 0.99), while the
    programming language may only be inferred from a few keywords with no
    file extension visible (language confidence = 0.55).

    A single top-level confidence score would obscure this and cause the LLM
    to either over-trust or under-trust all fields equally.

    Score interpretation
    --------------------
    0.9–1.0 : Strong evidence (explicit keyword, window title match)
    0.7–0.9 : Good evidence (multiple consistent signals)
    0.5–0.7 : Moderate evidence (one or two signals, no contradictions)
    0.3–0.5 : Weak evidence (suggestive but ambiguous)
    0.0–0.3 : Very low confidence — treat as UNKNOWN
    """
    application:  float = 0.0
    content_type: float = 0.0
    language:     float = 0.0
    errors:       float = 0.0
    website:      float = 0.0

    @property
    def overall(self) -> float:
        """
        Weighted average of all non-zero confidence scores.
        Gives a single number suitable for logging and quick threshold checks.
        """
        scores = [v for v in [
            self.application, self.content_type, self.language, self.errors
        ] if v > 0.0]
        return round(sum(scores) / len(scores), 4) if scores else 0.0


# ---------------------------------------------------------------------------
# ScreenContext — the structured perception output
# ---------------------------------------------------------------------------
@dataclass
class ScreenContext:
    """
    The structured output of the screen understanding pipeline.

    This is what the Prompt Builder receives.  It represents WHAT is on
    screen, structured and labelled — not WHY it matters or WHAT to do about it.
    That reasoning belongs to the LLM.

    Fields
    ------
    application          : Detected application type (enum, never free-form text).
    content_type         : Detected content category (enum).
    language             : Detected programming language, if applicable. None otherwise.
    website              : Detected website or domain, if content is a webpage. None otherwise.
    window_title         : Detected window title from OCR or vision response. None if not found.
    summary              : Brief natural-language description from the vision model.
                           This is the VisionResponse.answer, passed through unchanged.
                           screen_context.py does NOT generate summaries.
    errors               : List of detected error/exception strings from the screen.
                           These are verbatim extracts — no interpretation.
    keywords             : Important terms extracted from the OCR text.
                           Used by the Prompt Builder to emphasise salient content.
    confidence           : Per-field certainty scores (ContextConfidence).
    image_hash           : SHA-256 thumbnail hash of the source capture.
                           Enables cache lookups and audit correlation.
    monitor_index        : Physical monitor index the capture came from.
    resolution           : Capture resolution as "WIDTHxHEIGHT".
    capture_scope        : The CaptureScope used (e.g. "ACTIVE_WINDOW").
    interaction_candidates: [RESERVED] Empty in Phase 2. Populated in Phase 3.
    ocr_word_count       : Total word count from OCR, useful for "is there content?" checks.
    ocr_confidence       : Average OCR confidence from the source OCRResult.
    vision_model_id      : Which vision backend produced the summary.
    """
    application:           ApplicationType
    content_type:          ContentType
    language:              Optional[str]
    website:               Optional[str]
    window_title:          Optional[str]
    summary:               str
    errors:                list[str]
    keywords:              list[str]
    confidence:            ContextConfidence
    image_hash:            Optional[str]

    # Capture provenance — useful for debugging and Phase 3 routing
    monitor_index:         int   = 0
    resolution:            str   = ""
    capture_scope:         str   = ""

    # Phase 3 extension hook — DO NOT POPULATE IN PHASE 2
    interaction_candidates: list[InteractionTarget] = field(default_factory=list)

    # Observability fields
    ocr_word_count:        int   = 0
    ocr_confidence:        float = 0.0
    vision_model_id:       str   = ""

    # Internal debug field — NOT sent to the LLM.
    # Stores the raw content classification scores so metrics and debugging
    # dashboards can see the full distribution, not just the winning label.
    # Example: {"CODE": 0.76, "DOCUMENT": 0.71, "ERROR": 0.12}
    # WHY NOT IN to_prompt_dict()
    # ----------------------------
    # These are internal classification scores, not facts about the screen.
    # Sending them to the LLM would clutter the prompt with meta-information
    # the model cannot act on.  Metrics and logging use them; the LLM does not.
    _content_scores: dict = field(default_factory=dict, repr=False)

    def to_prompt_dict(self) -> dict:
        """
        Serialise to a plain dict for insertion into the Prompt Builder.

        Only includes fields with non-default, non-empty values so the
        prompt stays concise.  The LLM doesn't need to read "errors: []"
        when there are no errors.
        """
        d: dict = {
            "application":  self.application.value,
            "content_type": self.content_type.value,
            "confidence":   round(self.confidence.overall, 3),
            "summary":      self.summary,
        }
        if self.language:
            d["language"] = self.language
        if self.website:
            d["website"] = self.website
        if self.window_title:
            d["window_title"] = self.window_title
        if self.errors:
            d["errors"] = self.errors
        if self.keywords:
            d["keywords"] = self.keywords[:10]  # Cap to avoid prompt bloat
        if self.resolution:
            d["resolution"] = self.resolution
        return d


# ---------------------------------------------------------------------------
# Detection helpers — pure functions, no side effects
# ---------------------------------------------------------------------------
# WHY PURE FUNCTIONS
# -------------------
# Detection logic that lives in standalone functions is easy to unit-test
# in isolation.  Each function takes text as input and returns a score.
# No object state, no side effects.

# --- Application detection signals ---

_VSCODE_SIGNALS = re.compile(
    r"(explorer|source control|extensions|go to file|command palette"
    r"|debug console|problems|output|terminal|vscode|visual studio code"
    r"|\.py|\.ts|\.js|\.rs|\.go|\.java|\.cpp|\.cs|intellisense|git)",
    re.IGNORECASE,
)
_TERMINAL_SIGNALS = re.compile(
    r"(\$\s|\bPS\s*>|[a-z]+@[a-z]+:|^\s*#\s|sudo\s|apt\s|brew\s|pip\s"
    r"|npm\s|cargo\s|python3?\s|bash\s|zsh\s|cmd\.exe|powershell|error:\s)",
    re.IGNORECASE | re.MULTILINE,
)
_BROWSER_SIGNALS = re.compile(
    r"(https?://|www\.|\.com|\.org|\.net|\.io|back|forward|reload"
    r"|bookmark|tab|address bar|incognito|cookie)",
    re.IGNORECASE,
)
_PDF_SIGNALS = re.compile(
    r"(page \d+ of \d+|adobe|acrobat|preview|\.pdf|table of contents"
    r"|abstract|references\s*\[|\bpdf\b)",
    re.IGNORECASE,
)
_FILE_EXPLORER_SIGNALS = re.compile(
    r"(\b\d+\.?\d*\s*(KB|MB|GB)\b|modified|created|size|type|name"
    r"|c:\\|/home/|/usr/|documents|downloads|desktop|my computer|this pc)",
    re.IGNORECASE,
)
_CHAT_SIGNALS = re.compile(
    r"(@\w+|just now|\d{1,2}:\d{2}\s*(AM|PM)?|sent|delivered|seen"
    r"|typing\.\.\.|message|reaction|reply|thread)",
    re.IGNORECASE,
)

# --- Content detection signals ---

_CODE_SIGNALS = re.compile(
    r"(def\s+\w+|class\s+\w+|import\s+\w+|from\s+\w+\s+import"
    r"|function\s+\w+|const\s+\w+|let\s+\w+|var\s+\w+|\bif\s*\("
    r"|for\s*\(|while\s*\(|\breturn\b|\bpublic\s+\w+|\bprivate\s+\w+"
    r"|\{\s*$|\}\s*$|=>|::\s*\w+|#include|package\s+\w+)",
    re.MULTILINE,
)
_ERROR_SIGNALS = re.compile(
    r"(error|exception|traceback|stacktrace|stack trace"
    r"|failed|fatal|abort|panic|segfault|undefined|null pointer"
    r"|cannot|could not|refused|denied|timeout|timed out"
    r"|SyntaxError|TypeError|ValueError|NameError|ImportError"
    r"|AttributeError|KeyError|IndexError|RuntimeError"
    r"|NullPointerException|ClassNotFoundException"
    r"|ENOENT|EACCES|ECONNREFUSED|404|500|503)",
    re.IGNORECASE,
)
_TABLE_SIGNALS = re.compile(
    r"(\|.+\|.+\||\t.+\t.+\t|^\s*[-+]{3,}|\bcolumn\b|\brow\b|\bheader\b)",
    re.IGNORECASE | re.MULTILINE,
)
_DOCUMENT_SIGNALS = re.compile(
    r"(abstract|introduction|conclusion|chapter|section\s+\d"
    r"|\bfigure\s+\d+\b|\btable\s+\d+\b|references|bibliography"
    r"|paragraph|footnote|appendix)",
    re.IGNORECASE,
)

# --- Language detection signals ---

_LANGUAGE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("Python",     re.compile(r"(def |import |from .+ import|print\(|\.py\b|__init__|self\.)", re.IGNORECASE)),
    ("JavaScript", re.compile(r"(const |let |var |function |=>|console\.log|require\(|\.js\b|\.jsx\b)", re.IGNORECASE)),
    ("TypeScript", re.compile(r"(interface |type |:\s*(string|number|boolean|void)|\.ts\b|\.tsx\b)", re.IGNORECASE)),
    ("Rust",       re.compile(r"(fn |let mut |impl |use |pub |mod |\.rs\b|println!|cargo)", re.IGNORECASE)),
    ("Go",         re.compile(r"(func |package |import |go\s+func|var |:=|\.go\b|fmt\.Print)", re.IGNORECASE)),
    ("Java",       re.compile(r"(public class |static void main|System\.out|import java\.|\.java\b|throws )", re.IGNORECASE)),
    ("C/C++",      re.compile(r"(#include|int main\(|printf\(|std::|nullptr|\.cpp\b|\.h\b|->)", re.IGNORECASE)),
    ("C#",         re.compile(r"(using System|namespace |class |\.cs\b|Console\.Write|async Task|await )", re.IGNORECASE)),
    ("HTML",       re.compile(r"(<html|<div|<span|<head|<body|<!DOCTYPE|</\w+>|\.html\b)", re.IGNORECASE)),
    ("CSS",        re.compile(r"(\{.+:.*;|@media|\.class|#id|px;|em;|rem;|\.css\b)", re.IGNORECASE)),
    ("SQL",        re.compile(r"(SELECT |FROM |WHERE |JOIN |INSERT INTO|UPDATE |DELETE FROM|CREATE TABLE)", re.IGNORECASE)),
    ("Bash",       re.compile(r"(#!/bin/bash|#!/bin/sh|\$\(|grep |sed |awk |chmod |export )", re.IGNORECASE)),
]

# --- URL / website extraction ---

_URL_PATTERN = re.compile(
    r"(https?://(?:www\.)?([a-zA-Z0-9\-]+\.[a-zA-Z]{2,})(?:/[^\s]*)?)",
    re.IGNORECASE,
)

# --- Error extraction ---

_ERROR_LINE_PATTERN = re.compile(
    r"^.*(error|exception|traceback|failed|fatal|panic|undefined|cannot|timeout).*$",
    re.IGNORECASE | re.MULTILINE,
)


# ---------------------------------------------------------------------------
# ContextBuilder — assembles ScreenContext from pipeline outputs
# ---------------------------------------------------------------------------
class ContextBuilder:
    """
    Stateless factory that builds a ScreenContext from OCRResult + VisionResponse.

    WHY STATELESS
    --------------
    Each call to build() is independent.  State between calls would create
    subtle bugs where one session's context bleeds into the next.  The builder
    holds no per-call data — it is purely a collection of classification logic.

    WHY NOT INSIDE ScreenContext (as a classmethod)
    -----------------------------------------------
    ContextBuilder is the complex, logic-heavy layer.  ScreenContext is the
    clean data class.  Mixing them would violate the single-responsibility
    principle and make unit-testing harder (you'd need a full OCRResult to
    construct any ScreenContext, even in tests).

    Usage
    -----
    >>> builder = ContextBuilder()
    >>> ctx = builder.build(ocr_result=ocr, vision_response=vision)
    >>> prompt_data = ctx.to_prompt_dict()
    """

    def build(
        self,
        ocr_result:       OCRResult,
        vision_response:  VisionResponse,
        window_title:     Optional[str] = None,
        monitor_index:    int           = 0,
        resolution:       str           = "",
        capture_scope:    str           = "",
    ) -> ScreenContext:
        """
        Build a ScreenContext from OCR and vision pipeline outputs.

        Classification order
        --------------------
        1. Collect all evidence (OCR text + vision response text).
        2. Classify application type (what app is open?).
        3. Classify content type (what kind of content?).
        4. Detect programming language (if applicable).
        5. Extract errors (verbatim — no interpretation).
        6. Extract website (if content is a browser page).
        7. Extract keywords (for Prompt Builder emphasis).
        8. Compute per-field confidence scores.

        Returns
        -------
        ScreenContext with all fields populated conservatively.
        """
        # Combine OCR text and vision description into a single evidence string.
        # We search BOTH because:
        #   - OCR text has the exact words visible on screen.
        #   - Vision description may mention app context not in the text
        #     (e.g. "This appears to be the VSCode interface").
        evidence = f"{ocr_result.text}\n{vision_response.answer}"
        evidence_lower = evidence.lower()

        log.debug(
            "context_build_start",
            ocr_words=ocr_result.word_count,
            vision_model=vision_response.model_id,
            window_title=window_title or "",
        )

        # --- Application type ---
        app_type, app_conf = self._classify_application(evidence, window_title)

        # --- Content type ---
        content_type, content_conf, content_scores = self._classify_content(evidence)

        # --- Language ---
        language, lang_conf = self._detect_language(ocr_result.text)

        # --- Errors ---
        errors, error_conf = self._extract_errors(ocr_result.text)

        # --- Website ---
        website, web_conf = self._extract_website(evidence, app_type)

        # --- Keywords ---
        keywords = self._extract_keywords(ocr_result.text)

        confidence = ContextConfidence(
            application  = app_conf,
            content_type = content_conf,
            language     = lang_conf,
            errors       = error_conf,
            website      = web_conf,
        )

        ctx = ScreenContext(
            application            = app_type,
            content_type           = content_type,
            language               = language,
            website                = website,
            window_title           = window_title,
            summary                = vision_response.answer.strip(),
            errors                 = errors,
            keywords               = keywords,
            confidence             = confidence,
            image_hash             = ocr_result.image_hash or vision_response.image_hash,
            monitor_index          = monitor_index,
            resolution             = resolution,
            capture_scope          = capture_scope,
            interaction_candidates = [],   # Phase 3 extension — empty by design.
            ocr_word_count         = ocr_result.word_count,
            ocr_confidence         = ocr_result.confidence,
            vision_model_id        = vision_response.model_id,
            _content_scores        = content_scores,
        )

        log.info(
            "context_built",
            application  = app_type.value,
            content_type = content_type.value,
            language     = language or "none",
            error_count  = len(errors),
            confidence   = confidence.overall,
        )

        return ctx

    # -----------------------------------------------------------------------
    # Private classification methods
    # -----------------------------------------------------------------------

    def _classify_application(
        self,
        text:         str,
        window_title: Optional[str],
    ) -> tuple[ApplicationType, float]:
        """
        Classify the visible application using keyword signals.

        Strategy
        --------
        1. If a window title is available, it is the strongest signal.
           Window titles like "main.py - Visual Studio Code" are unambiguous.
        2. Fall back to OCR + vision text signal counting.
        3. Return UNKNOWN if no signal reaches the minimum threshold.

        Conservative rule: require at least 2 signal matches for medium
        confidence, and at least 1 strong match (window title) for high
        confidence.
        """
        combined = f"{window_title or ''}\n{text}"

        scores: dict[ApplicationType, float] = {}

        # Window title gives a strong prior — check it first.
        title_lower = (window_title or "").lower()
        if any(kw in title_lower for kw in ("visual studio code", "vscode", "code -")):
            scores[ApplicationType.VSCODE] = 0.97
        if any(kw in title_lower for kw in ("terminal", "bash", "zsh", "powershell", "cmd", "konsole", "iterm")):
            scores[ApplicationType.TERMINAL] = 0.95
        if any(kw in title_lower for kw in ("chrome", "firefox", "safari", "edge", "brave", "opera", "browser")):
            scores[ApplicationType.BROWSER] = 0.95
        if any(kw in title_lower for kw in ("adobe", "preview", ".pdf", "acrobat", "okular", "evince")):
            scores[ApplicationType.PDF_VIEWER] = 0.95

        # Text-based signals (add to or create a score for each type).
        def _score(pattern: re.Pattern, app: ApplicationType, weight: float = 0.20) -> None:
            matches = len(pattern.findall(combined))
            if matches:
                current = scores.get(app, 0.0)
                # Diminishing returns: each additional match adds less.
                added = weight * (1.0 - current) * min(matches, 5) / 5
                scores[app] = round(min(current + added, 0.92), 4)

        _score(_VSCODE_SIGNALS,      ApplicationType.VSCODE)
        _score(_TERMINAL_SIGNALS,    ApplicationType.TERMINAL)
        _score(_BROWSER_SIGNALS,     ApplicationType.BROWSER)
        _score(_PDF_SIGNALS,         ApplicationType.PDF_VIEWER)
        _score(_FILE_EXPLORER_SIGNALS, ApplicationType.FILE_EXPLORER)
        _score(_CHAT_SIGNALS,        ApplicationType.CHAT_APP)

        if not scores:
            return ApplicationType.UNKNOWN, 0.0

        best_app  = max(scores, key=lambda k: scores[k])
        best_conf = scores[best_app]

        # Conservative threshold: if best confidence is below 0.4, report UNKNOWN.
        if best_conf < 0.4:
            return ApplicationType.UNKNOWN, best_conf

        return best_app, best_conf

    def _classify_content(self, text: str) -> tuple[ContentType, float, dict]:
        """
        Classify the content type using signal counting.

        Each pattern contributes to a content type score.  The top score wins,
        unless two types are very close in score (within 0.1 of each other) —
        in that case we return MIXED to honestly reflect ambiguity.

        Returns a 3-tuple: (ContentType, confidence, full_scores_dict).
        The full_scores_dict is stored in ScreenContext._content_scores for
        metrics and debugging — it is never sent to the LLM.
        """
        scores: dict[ContentType, float] = {}

        def _score(pattern: re.Pattern, ct: ContentType, weight: float = 0.25) -> None:
            matches = len(pattern.findall(text))
            if matches:
                current = scores.get(ct, 0.0)
                added   = weight * (1.0 - current) * min(matches, 4) / 4
                scores[ct] = round(min(current + added, 0.93), 4)

        _score(_CODE_SIGNALS,     ContentType.CODE,     weight=0.3)
        _score(_ERROR_SIGNALS,    ContentType.ERROR,    weight=0.35)
        _score(_TABLE_SIGNALS,    ContentType.TABLE,    weight=0.25)
        _score(_DOCUMENT_SIGNALS, ContentType.DOCUMENT, weight=0.2)

        # WEBPAGE: browser signal presence is a proxy.
        if _BROWSER_SIGNALS.search(text):
            scores[ContentType.WEBPAGE] = scores.get(ContentType.WEBPAGE, 0.0) + 0.3

        # Build the debug scores dict — top 3 sorted by score, with string keys.
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        debug_scores  = {ct.value: conf for ct, conf in sorted_scores[:3]}

        if not scores:
            return ContentType.UNKNOWN, 0.0, {}

        top_type, top_conf = sorted_scores[0]

        # Check for MIXED: if two types are very close, content is ambiguous.
        if len(sorted_scores) >= 2:
            second_type, second_conf = sorted_scores[1]
            if top_conf - second_conf < 0.1 and top_conf >= 0.3:
                return ContentType.MIXED, round((top_conf + second_conf) / 2, 4), debug_scores

        if top_conf < 0.3:
            return ContentType.UNKNOWN, top_conf, debug_scores

        return top_type, top_conf, debug_scores


    def _detect_language(self, text: str) -> tuple[Optional[str], float]:
        """
        Detect the programming language from OCR text.

        Strategy: count matches for each language's pattern.  The language
        with the most matches wins.  If no language clears the minimum signal
        threshold (2 matches), return None.

        WHY COUNT MATCHES (not just search)
        ------------------------------------
        A single `import` statement is weak evidence — Python, Java, Go, and
        JavaScript all use import.  Five different Python-specific signals
        (def, self, __init__, .py, print()) is strong evidence.
        """
        match_counts: dict[str, int] = {}
        for lang, pattern in _LANGUAGE_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                match_counts[lang] = len(matches)

        if not match_counts:
            return None, 0.0

        best_lang  = max(match_counts, key=lambda k: match_counts[k])
        best_count = match_counts[best_lang]

        if best_count < 2:
            # Only one weak signal — not confident enough to assert a language.
            return None, 0.0

        # Confidence scales with signal count, capped at 0.95.
        confidence = round(min(0.4 + (best_count * 0.08), 0.95), 4)
        return best_lang, confidence

    def _extract_errors(self, text: str) -> tuple[list[str], float]:
        """
        Extract verbatim error strings from the OCR text.

        WHY VERBATIM
        -------------
        We extract lines that CONTAIN error keywords, unmodified.  We do NOT:
          - Paraphrase ("there seems to be an import error")
          - Classify ("this is a dependency error")
          - Summarise ("multiple errors found")

        The LLM receives the exact error text and reasons about it.

        Deduplication: we strip and deduplicate to avoid repeating the same
        error line from duplicate OCR detections.
        """
        raw_matches = _ERROR_LINE_PATTERN.findall(text)

        # Deduplicate while preserving first occurrence (ordered set pattern).
        seen: set[str] = set()
        errors: list[str] = []
        for line in raw_matches:
            clean = line.strip()
            if clean and clean not in seen and len(clean) < 500:  # Sanity length cap
                seen.add(clean)
                errors.append(clean)

        if not errors:
            return [], 0.0

        # Confidence: more error lines = higher confidence there IS an error.
        conf = round(min(0.5 + len(errors) * 0.1, 0.90), 4)
        return errors[:10], conf  # Cap at 10 error lines to avoid prompt bloat

    def _extract_website(
        self,
        text:     str,
        app_type: ApplicationType,
    ) -> tuple[Optional[str], float]:
        """
        Extract the most prominent URL or domain from the text.

        Only meaningful when app_type is BROWSER.  For other applications,
        we still attempt extraction (DevTools, VS Code browser preview) but
        apply a lower confidence score.
        """
        matches = _URL_PATTERN.findall(text)  # Returns list of (full_url, domain)
        if not matches:
            return None, 0.0

        # Take the first match — it's most likely the address bar URL.
        _, domain = matches[0]
        base_conf = 0.85 if app_type == ApplicationType.BROWSER else 0.55
        return domain, base_conf

    def _extract_keywords(self, text: str, max_keywords: int = 20) -> list[str]:
        """
        Extract salient keywords from the OCR text for Prompt Builder emphasis.

        Strategy: tokenise on whitespace, filter short/common words, deduplicate,
        return the most frequent terms.  This is intentionally simple — we are
        not doing TF-IDF or NLP here.  The goal is to surface the most-mentioned
        terms so the Prompt Builder can highlight them in the LLM context.

        WHY NO NLP LIBRARY
        -------------------
        Adding spaCy or NLTK for keyword extraction would:
          - Add hundreds of MB of dependency weight.
          - Require model downloads (defeating fast startup).
          - Be overkill for extracting ~20 salient terms from a screenshot.

        A frequency-count over filtered tokens gives 80% of the value at 0%
        of the cost.
        """
        # Common words that add no signal — a minimal, hard-coded stop list.
        STOP_WORDS = frozenset({
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "can",
            "this", "that", "these", "those", "it", "its", "they", "their",
            "there", "here", "where", "what", "which", "who", "how", "when",
            "and", "or", "but", "if", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "into", "through", "during",
            "not", "no", "nor", "so", "yet", "both", "either", "neither",
            "each", "few", "more", "most", "other", "some", "such", "than",
            "too", "very", "just", "about", "up", "down", "out", "off",
            "over", "under", "again", "then", "once",
        })

        # Tokenise: split on whitespace and punctuation, lowercase everything.
        tokens = re.findall(r"\b[a-zA-Z_]\w{2,}\b", text)  # Min 3 chars
        freq: dict[str, int] = {}
        for tok in tokens:
            lower = tok.lower()
            if lower not in STOP_WORDS:
                freq[lower] = freq.get(lower, 0) + 1

        # Sort by frequency (descending), take top N.
        sorted_kw = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [kw for kw, _ in sorted_kw[:max_keywords]]


# ---------------------------------------------------------------------------
# Module-level convenience factory
# ---------------------------------------------------------------------------
def build_screen_context(
    ocr_result:      OCRResult,
    vision_response: VisionResponse,
    window_title:    Optional[str] = None,
    monitor_index:   int           = 0,
    resolution:      str           = "",
    capture_scope:   str           = "",
) -> ScreenContext:
    """
    Module-level shortcut for ContextBuilder().build().

    Callers that don't need to configure the builder (no dependency injection)
    can use this function directly:

        ctx = build_screen_context(ocr_result=ocr, vision_response=vision)

    This keeps the common case simple while leaving ContextBuilder injectable
    for tests that need to mock or subclass it.
    """
    return ContextBuilder().build(
        ocr_result      = ocr_result,
        vision_response = vision_response,
        window_title    = window_title,
        monitor_index   = monitor_index,
        resolution      = resolution,
        capture_scope   = capture_scope,
    )
