"""Gemini-backed email classification.

Called from classify.py's classify_trace() dispatcher whenever
classify.gemini_enabled() is true (i.e. GEMINI_API_KEY is configured) --
there's no separate "rules-only" mode to opt into, Gemini is simply always
attempted when a key is present, informed by the rule scorer's output as a
prompt hint. app.py also calls warm_cache() directly, as a concurrency
pre-pass before its normal one-email-at-a-time loop, and as a periodic
poll to retry anything still pending.

On an ordinary API failure (rate limit, network error, malformed response
after retries), classify_trace_gemini() does NOT raise -- it records the
email in a small persisted "pending" store and returns the rule-based
result as a provisional value (decided_by="gemini_pending"). The next time
warm_cache() runs over that email (the next pipeline run, or app.py's
periodic poll), it's retried automatically; nothing needs to notice or
click anything for that to happen. See pipeline/tools/reconcile_pending.py
for the CLI-side equivalent.

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
    SDOC_GEMINI_POLL_INTERVAL  default 60 -- a pending email won't be
                                 retried again until this many seconds have
                                 passed since its last attempt, so a burst
                                 of page reloads/navigations can't hammer a
                                 scarce daily quota re-hitting the same
                                 still-broken item over and over
"""
import hashlib
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
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
PENDING_STORE_PATH = CACHE_DIR.parent / "gemini_pending.json"
POLL_INTERVAL_SECONDS = float(os.environ.get("SDOC_GEMINI_POLL_INTERVAL", "60"))

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


def _build_result(cached, rule_trace):
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


# --------------------------------------------------------------------------
# pending store -- bookkeeping only (not the classification result itself),
# so a simple in-process lock around a single shared JSON file is fine: all
# writers are threads in one process (warm_cache's ThreadPoolExecutor), not
# separate OS processes. Cross-process safety (e.g. the app and
# reconcile_pending.py running at the exact same moment) isn't covered by
# this lock, which is an acceptable gap for bookkeeping data.
# --------------------------------------------------------------------------
_PENDING_LOCK = threading.Lock()


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _load_pending_unlocked():
    if not PENDING_STORE_PATH.exists():
        return {}
    try:
        return json.loads(PENDING_STORE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_pending_unlocked(data):
    PENDING_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PENDING_STORE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(PENDING_STORE_PATH)


def _mark_pending(email_id, error):
    with _PENDING_LOCK:
        data = _load_pending_unlocked()
        entry = data.get(email_id, {"first_queued_at": _now_iso(), "attempts": 0})
        entry["attempts"] += 1
        entry["last_error"] = error
        entry["last_attempt_at"] = _now_iso()
        data[email_id] = entry
        _save_pending_unlocked(data)


def _clear_pending(email_id):
    with _PENDING_LOCK:
        data = _load_pending_unlocked()
        if email_id in data:
            del data[email_id]
            _save_pending_unlocked(data)


def _pending_entry(email_id):
    with _PENDING_LOCK:
        return _load_pending_unlocked().get(email_id)


def _seconds_since(iso_timestamp):
    return (datetime.now(timezone.utc) - datetime.fromisoformat(iso_timestamp)).total_seconds()


def _too_soon_to_retry(entry):
    """True when this email failed recently enough that trying again right
    now would just burn quota re-hitting the same outage. Applies to every
    retry path (the per-email classify_trace call as much as the app's
    periodic poll), so one failing email costs at most one attempt per
    poll interval no matter how many code paths ask about it."""
    if not entry:
        return False
    last = entry.get("last_attempt_at")
    return bool(last) and _seconds_since(last) < POLL_INTERVAL_SECONDS


def list_pending(due_only=False):
    """Every email currently queued for retry, as
    [{"email_id", "first_queued_at", "attempts", "last_error", "last_attempt_at"}, ...].
    With due_only=True, only entries whose last attempt was long enough ago
    (SDOC_GEMINI_POLL_INTERVAL) to be worth retrying again -- this is what
    keeps a periodic poll cheap and quota-friendly in the steady state."""
    with _PENDING_LOCK:
        data = _load_pending_unlocked()
    items = [{"email_id": eid, **entry} for eid, entry in data.items()]
    if not due_only:
        return items
    return [item for item in items if not _too_soon_to_retry(item)]


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


def _call_with_retry(email, rule_trace, max_retries=None):
    from google.genai import errors

    _get_client()  # config errors (e.g. missing API key) fail immediately, no wasted retries
    retries = MAX_RETRIES if max_retries is None else max_retries

    last_exc = None
    for attempt in range(retries):
        try:
            return _call_gemini_api_once(email, rule_trace)
        except errors.APIError as e:
            code = getattr(e, "code", None)
            if code not in RETRYABLE_CODES or attempt == retries - 1:
                raise
            last_exc = e
        except (json.JSONDecodeError, KeyError, ValueError, GeminiClassifyError) as e:
            if attempt == retries - 1:
                raise
            last_exc = e
        time.sleep(min(30, 1.0 * (2 ** attempt)) + random.uniform(0, 0.5))
    raise last_exc


# --------------------------------------------------------------------------
# public entry points
# --------------------------------------------------------------------------
def classify_trace_gemini(email, rule_trace, max_retries=None):
    """The entry point classify.py's dispatcher calls for one email. Never
    raises for an ordinary API failure -- queues the email for retry and
    returns the rule-based result as a provisional value instead."""
    email_id = email.get("email_id")
    key = _cache_key(email, rule_trace)
    cached = _load_cached(key)
    if cached is not None:
        _clear_pending(email_id)
        return _build_result(cached, rule_trace)

    # Already failed within the current poll interval -- don't spend another
    # call on it right now. Without this, a single run would attempt each
    # failing email twice (once in warm_cache's batch pre-pass, again in the
    # caller's per-email loop, which also sees a cache miss), doubling the
    # quota burn during exactly the outage where quota is most precious.
    entry = _pending_entry(email_id)
    if _too_soon_to_retry(entry):
        provisional = dict(rule_trace)
        provisional.update(decided_by="gemini_pending", llm_error=entry.get("last_error"))
        return provisional

    try:
        result = _call_with_retry(email, rule_trace, max_retries=max_retries)
    except Exception as e:
        _mark_pending(email_id, f"{type(e).__name__}: {e}")
        provisional = dict(rule_trace)
        provisional.update(decided_by="gemini_pending", llm_error=f"{type(e).__name__}: {e}")
        return provisional

    _save_cached(key, result)
    _clear_pending(email_id)
    return _build_result(result, rule_trace)


def warm_cache(emails, max_retries=None):
    """Concurrently (re)attempt Gemini for every email not already cached
    -- covers both "brand new, never attempted" and "previously pending,
    worth retrying" in one pass, since both look identical from here (no
    cache entry yet). Pass max_retries=1 for a cheap single-shot poll
    instead of the full backoff chain (see app.py's periodic retry, which
    must not block a page interaction for multiple retries on every click).

    Returns [{"email_id", "decided_by", "category"}, ...] for every email
    attempted -- decided_by is "gemini" on success, "gemini_pending" if it
    failed and got queued (classify_trace_gemini never raises for an
    ordinary failure, so this list is reliable for figuring out what
    resolved just now, e.g. after a retry)."""
    from classify import _classify_trace_rules

    with _PENDING_LOCK:
        already_pending = set(_load_pending_unlocked())  # one read, not one per email

    to_fetch = []
    for email in emails:
        rule_trace = _classify_trace_rules(email)
        if _load_cached(_cache_key(email, rule_trace)) is not None:
            # Cached means resolved -- drop any stale pending entry so the
            # queue self-heals (e.g. an entry left over from a previous
            # prompt version whose email is now cached under a new key).
            if email.get("email_id") in already_pending:
                _clear_pending(email["email_id"])
            continue
        to_fetch.append((email, rule_trace))

    if not to_fetch:
        return []

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(classify_trace_gemini, email, rule_trace, max_retries): email
                   for email, rule_trace in to_fetch}
        for future in as_completed(futures):
            email = futures[future]
            try:
                trace = future.result()
            except Exception as e:
                # defensive backstop only -- classify_trace_gemini shouldn't
                # raise for ordinary failures anymore, so this should be rare
                trace = {"decided_by": "error", "category": None, "llm_error": f"{type(e).__name__}: {e}"}
            results.append({"email_id": email.get("email_id"),
                             "decided_by": trace.get("decided_by"),
                             "category": trace.get("category")})
    return results
