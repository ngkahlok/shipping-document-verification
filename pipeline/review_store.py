"""Persistent human-review overrides.

When the pipeline escalates a case (NEEDS_REVIEW), can't classify it
confidently, or hits an unexpected processing error, a human should be able
to look at the evidence, confirm or correct the result, and have that
correction actually update the report -- not just get shown and discarded.
This is a small JSON-backed store for exactly that: {email_id: override}.

An override always wins over the pipeline's own guess when building the
final submission/report (see app.py's `effective_result`).
"""
import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_STORE = Path(__file__).parent.parent / "review_overrides.json"


def load_overrides(path=None):
    p = Path(path or DEFAULT_STORE)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def save_override(email_id, override, reviewer_note="", action="corrected", path=None):
    """action is 'confirmed' (accepted the pipeline's own result as-is) or
    'corrected' (a human changed the outcome)."""
    p = Path(path or DEFAULT_STORE)
    data = load_overrides(p)
    record = dict(override)
    record["reviewer_note"] = reviewer_note
    record["action"] = action
    record["reviewed_at"] = datetime.now(timezone.utc).isoformat()
    data[email_id] = record
    p.write_text(json.dumps(data, indent=2))
    return data


def clear_override(email_id, path=None):
    p = Path(path or DEFAULT_STORE)
    data = load_overrides(p)
    data.pop(email_id, None)
    p.write_text(json.dumps(data, indent=2))
    return data
