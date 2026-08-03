"""
security.py — Privacy Guard and Secrets Redaction
===================================================
WHY THIS MODULE EXISTS
-----------------------
Screen captures can contain anything — API keys in a terminal, session cookies
in browser DevTools, SSH private keys in a text editor, passwords in a login
form.  If any of that text were to reach a cloud API or an unfiltered log line,
the consequences could be catastrophic: credential theft, account takeover,
data breach.

This module is the mandatory gate that EVERY piece of text and every outbound
API call must pass through before leaving the local process.

WHY SECURITY BEFORE LOGGING (not after)
-----------------------------------------
Logging happens throughout the pipeline.  If security filtering ran after
logging, a single "debug — raw OCR text:" log line could silently exfiltrate
a private key.  By running SecurityGuard.sanitize_for_log() before every log
call we ensure the logger never sees raw secrets, regardless of log level.

WHY REGEX-BASED DETECTION (not ML)
------------------------------------
ML-based secret detection is more accurate but:
  - Requires a model loaded in memory (defeats "low RAM" principle).
  - Has a latency cost incompatible with sub-5-second pipeline targets.
  - Can produce false negatives on novel secret formats.

Regex patterns are:
  - Deterministic and auditable.
  - Zero-latency (compiled once at import time).
  - Easily extended — add a pattern, done.
  - Industry-standard (GitHub's secret scanning uses the same approach).

Limitation: regex cannot detect secrets that look like ordinary prose.
That is an acceptable trade-off for a local-first, user-requested system.

WHY THREE PRIVACY MODES
------------------------
Different users have radically different threat models:
  - STRICT: medical, legal, or financial data on screen. Zero cloud risk.
  - BALANCED: everyday use. Ask before uploading; local by default.
  - DEVELOPER: debugging Zytrix itself. Allow temporary disk writes, verbose logs.

A single hardcoded policy would be wrong for at least two of these three users.
"""

import re
from dataclasses import dataclass
from typing import Optional

from . import PrivacyMode


# ---------------------------------------------------------------------------
# Compiled secret-detection patterns
# ---------------------------------------------------------------------------
# Each pattern targets a specific credential format. We compile once at module
# load to avoid recompilation overhead on every OCR call.
#
# Patterns are intentionally conservative (may produce false positives) because
# redacting a non-secret is far less harmful than leaking a real one.
_SECRET_PATTERNS: list[tuple[str, re.Pattern]] = [
    # GitHub / GitLab personal access tokens
    ("github_token",        re.compile(r"gh[ps]_[A-Za-z0-9]{36,}", re.I)),
    # AWS access keys
    ("aws_access_key",      re.compile(r"AKIA[0-9A-Z]{16}", re.I)),
    # Generic API key / secret patterns (key=value, secret=value, token=value)
    ("generic_api_key",     re.compile(r"(?:api[-_]?key|secret|token)\s*[:=]\s*['\"]?([A-Za-z0-9\-_.]{16,})", re.I)),
    # JSON Web Tokens (header.payload.signature)
    ("jwt",                 re.compile(r"eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")),
    # SSH private key blocks
    ("ssh_private_key",     re.compile(r"-----BEGIN (?:RSA|DSA|EC|OPENSSH) PRIVATE KEY-----")),
    # PEM certificates
    ("pem_cert",            re.compile(r"-----BEGIN CERTIFICATE-----")),
    # .env file variable assignments (VAR=value)
    ("dotenv_var",          re.compile(r"^[A-Z_]{2,}=.{8,}", re.M)),
    # OAuth access / refresh tokens (Bearer scheme)
    ("bearer_token",        re.compile(r"Bearer\s+[A-Za-z0-9\-_.~+/]+=*", re.I)),
    # Session cookies (common frameworks: session=, PHPSESSID=, JSESSIONID=, etc.)
    ("session_cookie",      re.compile(r"(?:session|PHPSESSID|JSESSIONID|_session)\s*[=:]\s*[A-Za-z0-9\-_.%+]{16,}", re.I)),
    # OAuth tokens (oauth_token, access_token, refresh_token)
    ("oauth_token",         re.compile(r"(?:oauth_token|access_token|refresh_token)\s*[:=]\s*['\"]?([A-Za-z0-9\-_.]{16,})", re.I)),
    # Recovery / backup codes (hyphenated blocks, e.g. 1234-5678-9abc-def0)
    ("recovery_code",       re.compile(r"\b(?:[A-Za-z0-9]{4,6}-){3,}[A-Za-z0-9]{4,6}\b")),
    # Password fields (password=value or "password": "value" in JSON)
    ("password_field",      re.compile(r"(?:password|passwd|pwd)\s*[:=]\s*['\"]?.{4,}", re.I)),
    # Private key IDs / fingerprints (SHA256:xxx)
    ("key_fingerprint",     re.compile(r"SHA256:[A-Za-z0-9+/]{43}")),
]

# Domains considered "high-risk" for cloud capture — we warn before sending
# images from these contexts to any cloud API.
_SENSITIVE_DOMAINS: frozenset[str] = frozenset([
    "bank", "banking", "paypal", "stripe",
    "healthcare", "hospital", "clinic", "medical",
    "tax", "irs", "hmrc", "revenue",
    "password", "vault", "lastpass", "1password", "bitwarden",
    "login", "signin", "auth",
])

_REDACTED_PLACEHOLDER = "[REDACTED]"


# ---------------------------------------------------------------------------
# SecurityGuard
# ---------------------------------------------------------------------------
@dataclass
class SecurityViolation:
    """
    Describes a single detected security concern.

    secret_type:   Machine-readable category (e.g. "jwt", "aws_access_key").
    description:   Human-readable explanation suitable for a warning message.
    can_proceed:   If False, the pipeline should abort rather than continue.
    """
    secret_type:  str
    description:  str
    can_proceed:  bool = True


class SecurityGuard:
    """
    Stateless privacy and secrets guard.

    All methods are classmethods so SecurityGuard needs no instantiation —
    it functions as a policy engine, not an object with state.

    Usage
    -----
    # Before sending text to a logger:
    safe_text = SecurityGuard.sanitize_for_log(raw_ocr_text)

    # Before calling a cloud API:
    can_call, warning = SecurityGuard.can_use_cloud(privacy_mode, context_hint)
    if not can_call:
        raise SecurityError(warning)

    # After detecting context:
    violations = SecurityGuard.detect_sensitive_content(ocr_text)
    """

    @classmethod
    def sanitize_for_log(cls, text: str) -> str:
        """
        Replace all detected secret patterns with [REDACTED].

        This is the primary defence against accidental credential logging.
        Call this on ANY string before passing it to a logger, regardless of
        log level — debug logs are often redirected to files or monitoring
        services that outlive a single session.

        Returns the sanitised string.  The original string is never mutated.
        """
        sanitised = text
        for _name, pattern in _SECRET_PATTERNS:
            sanitised = pattern.sub(_REDACTED_PLACEHOLDER, sanitised)
        return sanitised

    @classmethod
    def redact_secrets(cls, text: str) -> tuple[str, list[str]]:
        """
        Replace secrets and return both the redacted text and a list of the
        categories that were redacted.

        Unlike sanitize_for_log (which is fire-and-forget), this method gives
        the caller visibility into what was found, which feeds into metrics
        and audit records.

        Returns
        -------
        (redacted_text, [list_of_secret_type_names])
        """
        result = text
        found_types: list[str] = []
        for name, pattern in _SECRET_PATTERNS:
            if pattern.search(result):
                found_types.append(name)
                result = pattern.sub(_REDACTED_PLACEHOLDER, result)
        return result, found_types

    @classmethod
    def detect_sensitive_content(cls, ocr_text: str) -> list[SecurityViolation]:
        """
        Scan OCR text for known secret patterns and return a list of violations.

        The caller decides what to do with violations.  Typical responses:
          - Warn the user before sending to a cloud API.
          - Abort the pipeline if can_proceed is False on any violation.
          - Record violation types in metrics for audit reporting.
        """
        violations: list[SecurityViolation] = []
        for name, pattern in _SECRET_PATTERNS:
            if pattern.search(ocr_text):
                violations.append(SecurityViolation(
                    secret_type  = name,
                    description  = f"Possible {name.replace('_', ' ')} detected in screen content.",
                    can_proceed  = True,  # Warn, but do not hard-block by default.
                ))
        return violations

    @classmethod
    def is_sensitive_context(cls, context_hint: str) -> bool:
        """
        Heuristically determine whether the screen context is high-risk.

        context_hint should be a lowercased string containing application name,
        window title, or detected URL — anything that identifies what the user
        is looking at.

        Returns True if the context matches a known sensitive domain keyword,
        prompting the caller to warn the user or switch to local-only mode.
        """
        lower = context_hint.lower()
        return any(domain in lower for domain in _SENSITIVE_DOMAINS)

    @classmethod
    def can_use_cloud(
        cls,
        privacy_mode: PrivacyMode,
        context_hint: str = "",
    ) -> tuple[bool, Optional[str]]:
        """
        Determine whether the current privacy policy permits a cloud API call.

        Returns (allowed: bool, warning_message: Optional[str]).
        - If allowed is False, the caller MUST use a local model.
        - If warning_message is not None, surface it to the user before proceeding.

        Policy table
        ------------
        STRICT   → never allowed; no warning needed (user already knows).
        BALANCED → allowed unless the context looks sensitive; warn first.
        DEVELOPER → always allowed; no warning.
        """
        if privacy_mode == PrivacyMode.STRICT:
            return False, None  # Silent block — STRICT users expect local-only.

        if privacy_mode == PrivacyMode.DEVELOPER:
            return True, None   # Developer mode: trust the user.

        # BALANCED mode: check the context.
        if cls.is_sensitive_context(context_hint):
            warning = (
                "⚠️  The screen appears to show sensitive content "
                f"({context_hint[:60]}). "
                "Sending this image to a cloud API may expose private data. "
                "Switch to a local model or continue?"
            )
            return False, warning  # Block until user explicitly overrides.

        # BALANCED, non-sensitive: allow with a mild reminder.
        warning = (
            "This analysis will send a screenshot to a cloud API. "
            "Ensure no sensitive information is visible."
        )
        return True, warning

    @classmethod
    def validate_privacy_mode(cls, mode: PrivacyMode) -> None:
        """
        Raise ValueError if an unrecognised PrivacyMode is passed.

        Defensive check — prevents silent misconfigurations when new modes
        are added to the enum but callers aren't updated.
        """
        if not isinstance(mode, PrivacyMode):
            raise ValueError(
                f"Expected a PrivacyMode enum member, got {type(mode).__name__}: {mode!r}. "
                f"Valid modes: {[m.value for m in PrivacyMode]}"
            )
