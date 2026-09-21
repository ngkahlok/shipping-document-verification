#!/usr/bin/env python3
"""Build a submission.json for the SDOC hackathon inbox.

    python3 run.py <bundle_dir_or_url> <output.json>

Deliberately built against the participant bundle (no ground truth in
scope) -- score the result afterwards with score_cli.py.
"""
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

load_dotenv(HERE.parent / ".env")  # gitignored; fills GEMINI_API_KEY etc. if present, never overrides a real env var

import classify
from classify import classify as classify_email
from compare import compare_email


def _load_inbox(source):
    # loader.py lives alongside the bundle/dataset, not in this package.
    bundle_dir = Path(source)
    if bundle_dir.exists():
        sys.path.insert(0, str(bundle_dir))
    import loader
    return loader.Inbox(source)


def classify_and_compare(email, inbox):
    """One email's full submission entry -- shared by build_submission's
    main loop and pipeline/tools/reconcile_pending.py, which recomputes
    this for whatever comes back resolved after a retry."""
    category, decided_by = classify_email(email)
    if category == "BL_COMPARISON":
        result = compare_email(email, inbox)
    else:
        result = {"status": "OK", "has_defect": False,
                  "defect_fields": [], "review_reason": None}
    return {
        "category": category,
        "status": result["status"],
        "review_reason": result["review_reason"],
        "has_defect": result["has_defect"],
        "defect_fields": result["defect_fields"],
        "decided_by": decided_by,
    }


def build_submission(source):
    inbox = _load_inbox(source)
    emails = list(inbox)

    if classify.gemini_enabled():
        import gemini_classify
        gemini_classify.warm_cache(emails)  # concurrent pre-warm so the loop below mostly hits cache

    return {email["email_id"]: classify_and_compare(email, inbox) for email in emails}


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    source, out_path = sys.argv[1], sys.argv[2]
    submission = build_submission(source)
    Path(out_path).write_text(json.dumps(submission, indent=2))
    print(f"wrote {len(submission)} predictions to {out_path}")


if __name__ == "__main__":
    main()
