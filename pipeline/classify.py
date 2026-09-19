"""Stage 1 -- email classification.

Rule-first: the subjects in this inbox are coded (department/route/carrier
tokens), so keyword + regex rules resolve almost everything cheaply and
deterministically. Order matters -- checked most-distinctive-first so one
category's boilerplate can't shadow another's.
"""
import re

SPAM_PATTERNS = [
    r"\bwon\b.*\bgift card\b", r"\bclaim now\b", r"\bunpaid customs fee\b",
    r"\bverify your account\b", r"\bstorage is full\b", r"\b% off\b",
    r"\bbank details\b", r"\bbusiness proposal\b", r"\bhot singles\b",
    r"\bbitcoin\b", r"\bguaranteed\b.*\breturns\b", r"\bfree iphone\b",
    r"\bclaim your\b", r"\bundelivered messages\b", r"\bpay \$\d",
]

SI_REQUEST_PATTERNS = [
    r"^\s*(re_\s*)?si\s*-\s*\S+\s*-\s*direct\(",
    r"\bcust si\b", r"\brequest si\b", r"\bsi needed",  # "SI NEEDED_..." (no space before _)
]

BL_COMPARISON_PATTERNS = [
    r"\bto confirm docs\b", r"\brequest bl draft\b", r"\bdraft bl\b.*\bamend\b",
    r"\bamend bl\b",
    # dept - POD (may itself contain spaces, e.g. "NEW YORK_US") - CARRIER(BL#) - ...
    r"^\s*(re_\s*)?[a-z]+\s*-\s*.+?\s*-\s*[a-z]+\([a-z0-9]+\)\s*-",
]

INVOICE_QUERY_PATTERNS = [
    r"\bmissing gr\b", r"\bcancel invoice\b", r"\blocal charges\b",
    r"\bd\s*&\s*d charges\b", r"\btotal freight\b", r"\btelex release charges\b",
    r"\brak billing\b",
]

GENERAL_PATTERNS = [
    r"\bupdate summary\b", r"\bberthing report\b", r"\breminder\b",
    r"\b_rpa_\b", r"\boutstanding bl\b", r"\bpending bl release\b",
    r"\bnew year\b", r"\bapproval required\b", r"\btime off\b",
    r"\bmiss connection\b", r"\bdelivery planning\b",
]

RULES = [
    ("SPAM", SPAM_PATTERNS),
    ("SI_REQUEST", SI_REQUEST_PATTERNS),
    ("INVOICE_QUERY", INVOICE_QUERY_PATTERNS),
    ("BL_COMPARISON", BL_COMPARISON_PATTERNS),
    ("GENERAL", GENERAL_PATTERNS),
]

_COMPILED = [(cat, [re.compile(p, re.IGNORECASE) for p in pats]) for cat, pats in RULES]


def classify_trace(email):
    """Full detail behind a classification decision: which rule (if any)
    fired, on which pattern, against what text span."""
    subject = email.get("subject", "") or ""
    body = email.get("body", "") or ""
    haystack = subject + "\n" + body[:400]  # body header is enough signal

    for category, patterns in _COMPILED:
        for pat in patterns:
            m = pat.search(haystack)
            if m:
                return {
                    "category": category,
                    "decided_by": "rule",
                    "matched_pattern": pat.pattern,
                    "matched_text": m.group(0),
                }

    attachments = email.get("attachments") or []
    if any("_BL." in a or "_SI." in a for a in attachments):
        return {"category": "BL_COMPARISON", "decided_by": "fallback",
                "matched_pattern": None, "matched_text": None,
                "reason": "no subject rule matched, but attachments look like an SI/BL pair"}

    return {"category": "GENERAL", "decided_by": "fallback",
            "matched_pattern": None, "matched_text": None,
            "reason": "no subject rule matched and no SI/BL-style attachments"}


def classify(email):
    """Return (category, decided_by). decided_by is 'rule' or 'fallback'."""
    trace = classify_trace(email)
    return trace["category"], trace["decided_by"]
