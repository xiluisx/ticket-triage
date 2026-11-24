"""Regex PII redaction (spec 2026-11-24-inference-design.md §3).

Order matters: each pattern sees the output of the previous ones. Every match becomes XXXX,
the CFPB dataset mask, so the classic model sees the same distribution it was trained on.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

MASK = "XXXX"

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_SSN_BARE = re.compile(r"(?i)\b(?:ssn|social security(?: number| no\.?)?)\D{0,20}?(\d{9})\b")
_CARD = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")
_ACCOUNT = re.compile(
    r"(?i)\b(?:account|acct|routing|loan (?:no\.|number|#)|member (?:no\.|number|#))"
    r"[^\d\n]{0,20}?(\d{6,17})\b"
)
_PHONE = re.compile(r"(?<![\d\w])(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}(?!\d)")


@dataclass(frozen=True)
class Redaction:
    text: str
    counts: dict[str, int] = field(default_factory=dict)


def luhn_ok(value: str) -> bool:
    digits = [int(c) for c in value if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _sub(
    pattern: re.Pattern,
    text: str,
    kind: str,
    counts: dict[str, int],
    group: int = 0,
    valid: Callable[[str], bool] | None = None,
) -> str:
    def repl(m: re.Match) -> str:
        if valid is not None and not valid(m.group(group)):
            return m.group(0)
        counts[kind] = counts.get(kind, 0) + 1
        if group == 0:
            return MASK
        start, end = m.start(group) - m.start(), m.end(group) - m.start()
        whole = m.group(0)
        return whole[:start] + MASK + whole[end:]

    return pattern.sub(repl, text)


def redact(text: str) -> Redaction:
    counts: dict[str, int] = {}
    text = _sub(_EMAIL, text, "email", counts)
    text = _sub(_SSN, text, "ssn", counts)
    text = _sub(_SSN_BARE, text, "ssn", counts, group=1)
    text = _sub(_CARD, text, "card", counts, valid=luhn_ok)
    text = _sub(_ACCOUNT, text, "account", counts, group=1)
    text = _sub(_PHONE, text, "phone", counts)
    return Redaction(text, counts)


def merge_counts(*parts: dict[str, int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for part in parts:
        for k, v in part.items():
            out[k] = out.get(k, 0) + v
    return out
