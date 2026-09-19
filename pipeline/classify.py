"""Stage 1 -- email classification.

A weighted signal scorer, not a literal template matcher: each category has
a set of *concept* phrases (what a request like this actually says --
"please confirm the docs", "invoice is missing the GR", "claim your prize")
rather than this dataset's exact coded subject-line grammar. Every email
gets a score per category from whichever signals fire; the highest score
wins. This is deliberately looser than matching the generator's literal
subject templates so it has a chance of transferring to real inbox phrasing
it has never seen, at the cost of a little precision on this synthetic set.

Confidence is exposed (not just the winning label) so a genuinely
ambiguous email -- one where no signal fires, or two categories score
within a hair of each other -- can be routed to a human instead of forcing
a guess. See classify_trace()'s `confidence` field.
"""
import re

# (pattern, weight). Weight reflects how uniquely that phrase identifies the
# category -- a generic word like "invoice" is weak on its own; a phrase
# like "cancel invoice" is strong. Patterns match against subject + body.
SIGNALS = {
    "SPAM": [
        (r"\bclaim (?:now|your)\b", 3), (r"\bgift card\b", 3),
        (r"\bverify your account\b", 3), (r"\bunpaid customs fee\b", 3),
        (r"\bwon\b.{0,25}\b(prize|iphone|gift|draw)\b", 3),
        (r"\bbank details\b", 3), (r"\bbitcoin\b", 3),
        (r"\bguaranteed\b.{0,20}\breturns?\b", 3), (r"\bhot singles\b", 3),
        (r"\bstorage (?:is )?full\b", 2), (r"\bundelivered messages?\b", 2),
        (r"\burgent\b.{0,25}\bbusiness proposal\b", 3),
        (r"\b\d{1,3}\s*%\s*off\b", 2), (r"\bclick here\b", 1),
        (r"\bpay\s*\$\d", 2), (r"\baccount will be (?:suspended|deactivated)\b", 2),
    ],
    "SI_REQUEST": [
        (r"\bshipping instructions?\b", 2), (r"\brequest(?:ing)?\s+(?:the\s+)?si\b", 3),
        (r"\bsi\s+needed\b", 3), (r"\bcust\s+si\b", 2), (r"\blatest\s+si\b", 2),
        (r"\bplease find (?:the\s+)?shipping instruction\b", 3),
        (r"\bsi\s*-\s*\S+\s*-\s*direct\(", 3),
        (r"\brevert with (?:the\s+)?draft bl once\b", 1),
        (r"\bdocuments? required\b", 1), (r"\bh\.?s\.?\s*code\b.{0,40}\bgross wt\b", 1),
    ],
    "BL_COMPARISON": [
        (r"\bto confirm docs?\b", 3), (r"\bconfirm(?:ing)?\s+(?:the\s+)?docs?\b", 2),
        (r"\brequest(?:ing)?\s+.{0,15}bl\s+draft\b", 3), (r"\bdraft\s+b/?l\b", 2),
        (r"\bamend(?:ed|ing|ment)?\s+b/?l\b", 2),
        (r"\bcompare\b.{0,40}\b(si|shipping instruction)\b.{0,40}\b(bl|bill of lading)\b", 3),
        (r"\bcheck(?:ing)?\s+the\s+draft\s+bl\s+against\s+the\s+si\b", 3),
        (r"\bverify\b.{0,30}\bbl\b.{0,15}\bmatches?\b.{0,15}\bsi\b", 3),
        (r"\bdraft bill of lading\b", 1), (r"\brevert with any discrepancy\b", 2),
        (r"[a-z]{2,6}\([a-z0-9]{5,}\)", 1),  # carrier-code(reference#) token
    ],
    "INVOICE_QUERY": [
        (r"\bmissing gr\b", 3), (r"\bcancel invoice\b", 3), (r"\blocal charges?\b", 2),
        (r"\bd\s*&\s*d charges?\b", 3), (r"\bdetention\b.{0,20}\bcharges?\b", 2),
        (r"\btotal freight\b", 2), (r"\btelex release\b", 2),
        (r"\breverse the pgi\b", 2), (r"\bthc\b.{0,20}\bcharge\b", 1),
        (r"\bgr is still missing\b", 3), (r"\binvoice\b.{0,30}\b(query|cancel|breakdown)\b", 2),
        (r"\binvoice\b", 1),
    ],
    "GENERAL": [
        (r"\bberthing report\b", 3), (r"\bupdate summary\b", 2), (r"\boutstanding bl\b", 2),
        (r"\breminder\b", 1), (r"\b_?rpa_?\b", 2), (r"\bapproval required\b", 2),
        (r"\bnew year\b", 1), (r"\bpending bl release\b", 2), (r"\btime off\b", 1),
        (r"\bautomated notification\b", 2), (r"\bno action required\b", 1),
        (r"\bmiss connection\b", 1), (r"\bloading completed\b", 1),
        (r"\bberthed on schedule\b", 1), (r"\bsubmit si\s*&\s*aed\b", 1),
    ],
}

_COMPILED = {cat: [(re.compile(p, re.IGNORECASE), w) for p, w in sigs]
             for cat, sigs in SIGNALS.items()}

CONFIDENT_MARGIN = 2  # top score must beat runner-up by this much to call it "rule"


def _score(haystack):
    scores = {cat: 0 for cat in SIGNALS}
    matches = {cat: [] for cat in SIGNALS}
    for cat, patterns in _COMPILED.items():
        for pat, weight in patterns:
            m = pat.search(haystack)
            if m:
                scores[cat] += weight
                matches[cat].append((pat.pattern, weight, m.group(0)))
    return scores, matches


def classify_trace(email):
    """Full detail behind a classification decision: per-category scores,
    which signals fired, and a confidence label so ambiguous cases can be
    routed to a human instead of forcing a guess."""
    subject = email.get("subject", "") or ""
    body = email.get("body", "") or ""
    haystack = f"{subject}\n{body}"

    scores, matches = _score(haystack)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_cat, top_score = ranked[0]
    runner_score = ranked[1][1] if len(ranked) > 1 else 0

    if top_score == 0:
        attachments = email.get("attachments") or []
        if any("_BL." in a or "_SI." in a for a in attachments):
            return {"category": "BL_COMPARISON", "decided_by": "fallback", "confidence": "low",
                    "scores": scores, "matched_signals": [],
                    "reason": "no keyword signal fired, but attachments look like an SI/BL pair"}
        return {"category": "GENERAL", "decided_by": "fallback", "confidence": "low",
                "scores": scores, "matched_signals": [],
                "reason": "no keyword signal fired for any category"}

    confidence = "high" if (top_score - runner_score) >= CONFIDENT_MARGIN else "low"
    top_matches = sorted(matches[top_cat], key=lambda t: -t[1])
    return {
        "category": top_cat,
        "decided_by": "rule",
        "confidence": confidence,
        "scores": scores,
        "matched_signals": [{"pattern": p, "weight": w, "text": t} for p, w, t in top_matches],
        "matched_pattern": top_matches[0][0] if top_matches else None,
        "matched_text": top_matches[0][2] if top_matches else None,
    }


def classify(email):
    """Return (category, decided_by). decided_by is 'rule' or 'fallback'."""
    trace = classify_trace(email)
    return trace["category"], trace["decided_by"]
