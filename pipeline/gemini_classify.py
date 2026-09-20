"""Gemini-backed email classification (SDOC_CLASSIFIER=gemini).

Called from classify.py's classify_trace() dispatcher, never imported by
run.py/app.py directly for a single classification -- they go through the
dispatcher so the rule-based fallback always applies. app.py does call
warm_cache() directly, as a concurrency pre-pass before its normal
one-email-at-a-time loop.

Config (env vars, all optional):
    GEMINI_API_KEY              required to actually call the API
    GEMINI_MODEL                default "gemini-flash-latest" (Google's
                                 rolling fast/cheap alias -- won't silently
                                 404 when a dated model is deprecated;
                                 set to a dated model, e.g.
                                 "gemini-2.5-flash-lite", to pin one)
    SDOC_GEMINI_MAX_WORKERS     default 6 -- keep conservative, free-tier
                                 rate limits are low and change without
                                 notice; the retry/backoff below is the
                                 real safety net, not the pool size
    SDOC_GEMINI_MAX_RETRIES     default 4
    SDOC_GEMINI_CONFIDENT_MARGIN  default 20 -- Gemini's 0-100 score scale
                                 isn't comparable to the rule scorer's small
                                 integer weights, so it gets its own margin
    SDOC_GEMINI_CACHE_DIR       default pipeline/.cache/gemini_classify/
"""
import hashlib
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from classify import _confidence_label

_SERVER_DIR = Path(__file__).resolve().parent.parent / "sdoc-hackathon-docker" / "server"
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))
from scoring import CATEGORIES  # noqa: E402

PROMPT_VERSION = "v1"  # bump whenever the prompt/schema below changes, to invalidate stale cache entries

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
MAX_WORKERS = int(os.environ.get("SDOC_GEMINI_MAX_WORKERS", "6"))
MAX_RETRIES = int(os.environ.get("SDOC_GEMINI_MAX_RETRIES", "4"))
CONFIDENT_MARGIN = float(os.environ.get("SDOC_GEMINI_CONFIDENT_MARGIN", "20"))
CACHE_DIR = Path(os.environ.get("SDOC_GEMINI_CACHE_DIR",
                                 Path(__file__).parent / ".cache" / "gemini_classify"))

RETRYABLE_CODES = {429, 500, 502, 503, 504}

SYSTEM_INSTRUCTION = f"""You classify emails from a shipping-logistics operations inbox into
exactly one of these 5 categories:

- BL_COMPARISON: the sender wants a draft Bill of Lading checked, confirmed,
  verified against a Shipping Instruction, or amended before it's issued.
- SI_REQUEST: the sender is asking for a Shipping Instruction to be sent,
  prepared, or finalized -- requesting/providing the SI itself, not
  comparing it against anything yet.
- INVOICE_QUERY: a billing/invoicing issue -- disputing, cancelling, or
  querying charges, freight costs, detention/demurrage fees, or a missing
  goods-receipt tied to an invoice.
- GENERAL: routine operational correspondence that isn't a document
  request or billing issue -- status updates, reminders, internal/HR
  notices, berthing or loading reports, automated system notifications.
- SPAM: unsolicited, fraudulent, or irrelevant -- prize/lottery scams,
  phishing asking to verify an account or share bank details, fake
  unpaid-fee threats, marketing spam.

Judge the email's real intent, not just keyword overlap -- ignore forwarded
thread boilerplate, signatures, and "external sender" warning banners.
You'll also be given a simple keyword-based heuristic's guess as a hint;
it is frequently right but is tuned to one dataset's phrasing and can be
fooled by a coincidental keyword, so treat it as a weak signal, not ground
truth -- override it when the email's actual content disagrees.

Respond with a score from 0-100 for EACH of the 5 categories (they need
not sum to 100 -- score your confidence in each independently), a
`category` field equal to whichever category you scored highest, and a
one-sentence `reasoning` explaining the decision.

Categories: {", ".join(CATEGORIES)}"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": CATEGORIES},
        "scores": {
            "type": "object",
            "properties": {c: {"type": "number"} for c in CATEGORIES},
            "required": CATEGORIES,
        },
        "reasoning": {"type": "string"},
    },
    "required": ["category", "scores", "reasoning"],
}


class GeminiClassifyError(Exception):
    pass


# --------------------------------------------------------------------------
# client
# --------------------------------------------------------------------------
_client = None


def _get_client():
    global _client
    if _client is None:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise GeminiClassifyError("GEMINI_API_KEY is not set")
        from google import genai
        _client = genai.Client(api_key=key)
    return _client


# --------------------------------------------------------------------------
# prompt
# --------------------------------------------------------------------------
def _user_content(email, rule_trace):
    attachments = email.get("attachments") or []
    looks_like_si_bl = any("_SI." in a or "_BL." in a for a in attachments)
    top_signals = sorted(rule_trace.get("matched_signals", []), key=lambda s: -s["weight"])[:5]

    return f"""EMAIL
From: {email.get('from', '')}
Subject: {email.get('subject', '')}
Body:
{email.get('body', '')}
Attachments: {', '.join(attachments) if attachments else 'none'}
Attachment hint: {"looks like it includes an SI and/or BL-style attachment" if looks_like_si_bl else "no SI/BL-style attachment filenames detected"}

HEURISTIC HINT (from a keyword-based rule scorer; may be wrong or overfit to boilerplate phrasing -- treat as a weak signal, not ground truth):
Rule scorer's guess: {rule_trace['category']} (rule confidence: {rule_trace.get('confidence')})
Rule scores per category: {json.dumps(rule_trace.get('scores', {}))}
Top matched phrases: {[s['text'] for s in top_signals]}"""


# --------------------------------------------------------------------------
# cache -- one file per key (not one shared JSON blob like review_store.py)
# because up to MAX_WORKERS threads write concurrently; a shared
# read-modify-write-whole-file pattern would race and drop entries.
# --------------------------------------------------------------------------
def _cache_key(email, rule_trace):
    material = json.dumps({
        "prompt_version": PROMPT_VERSION,
        "model": GEMINI_MODEL,
        "email_id": email.get("email_id"),
        "subject": email.get("subject"),
        "body": email.get("body"),
        "attachments": sorted(email.get("attachments") or []),
        "rule_trace": {"category": rule_trace.get("category"), "scores": rule_trace.get("scores")},
    }, sort_keys=True)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _cache_path(key):
    return CACHE_DIR / f"{key}.json"


def _load_cached(key):
    p = _cache_path(key)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None  # corrupt/partial write -- treat as a miss, don't crash


def _save_cached(key, value):
    p = _cache_path(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(p)  # atomic rename -- safe even if the process is killed mid-write


# --------------------------------------------------------------------------
# API call with retry
# --------------------------------------------------------------------------
def _call_gemini_api_once(email, rule_trace):
    client = _get_client()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=_user_content(email, rule_trace),
        config={
            "system_instruction": SYSTEM_INSTRUCTION,
            "response_mime_type": "application/json",
            "response_json_schema": RESPONSE_SCHEMA,
            "temperature": 0.1,
        },
    )
    parsed = json.loads(response.text)
    category, scores = parsed.get("category"), parsed.get("scores") or {}
    if category not in CATEGORIES or set(scores) != set(CATEGORIES):
        raise GeminiClassifyError(f"malformed response: {parsed!r}")
    return {"category": category, "scores": {c: float(scores[c]) for c in CATEGORIES},
            "reasoning": parsed.get("reasoning", "")}


def _call_with_retry(email, rule_trace):
    from google.genai import errors

    _get_client()  # config errors (e.g. missing API key) fail immediately, no wasted retries

    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            return _call_gemini_api_once(email, rule_trace)
        except errors.APIError as e:
            code = getattr(e, "code", None)
            if code not in RETRYABLE_CODES or attempt == MAX_RETRIES - 1:
                raise
            last_exc = e
        except (json.JSONDecodeError, KeyError, ValueError, GeminiClassifyError) as e:
            if attempt == MAX_RETRIES - 1:
                raise
            last_exc = e
        time.sleep(min(30, 1.0 * (2 ** attempt)) + random.uniform(0, 0.5))
    raise last_exc


# --------------------------------------------------------------------------
# public entry points
# --------------------------------------------------------------------------
def classify_trace_gemini(email, rule_trace):
    """The entry point classify.py's dispatcher calls for one email."""
    key = _cache_key(email, rule_trace)
    cached = _load_cached(key)
    if cached is None:
        cached = _call_with_retry(email, rule_trace)
        _save_cached(key, cached)

    scores = cached["scores"]
    return {
        "category": cached["category"],
        "decided_by": "gemini",
        "confidence": _confidence_label(scores, CONFIDENT_MARGIN),
        "scores": scores,
        "reasoning": cached.get("reasoning", ""),
        "matched_signals": rule_trace.get("matched_signals", []),
        "rule_hint": {"category": rule_trace.get("category"), "scores": rule_trace.get("scores")},
        "llm_error": None,
    }


def warm_cache(emails):
    """Concurrently pre-fill the cache for a batch of emails so the
    subsequent one-at-a-time classify_trace() loop in run.py/app.py mostly
    hits cache. Each email's failure is isolated -- one bad call can't
    abort the batch. Returns a list of {"email_id", "error"} for anything
    that never resolved (those will simply retry, with fallback, when
    classify_trace() is eventually called on them for real)."""
    from classify import _classify_trace_rules

    to_fetch = []
    for email in emails:
        rule_trace = _classify_trace_rules(email)
        if _load_cached(_cache_key(email, rule_trace)) is None:
            to_fetch.append((email, rule_trace))

    failures = []
    if not to_fetch:
        return failures

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(classify_trace_gemini, email, rule_trace): email
                   for email, rule_trace in to_fetch}
        for future in as_completed(futures):
            email = futures[future]
            try:
                future.result()
            except Exception as e:
                failures.append({"email_id": email.get("email_id"), "error": f"{type(e).__name__}: {e}"})
    return failures
