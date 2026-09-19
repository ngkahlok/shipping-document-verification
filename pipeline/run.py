#!/usr/bin/env python3
"""Build a submission.json for the SDOC hackathon inbox.

    python3 run.py <bundle_dir_or_url> <output.json>

Deliberately built against the participant bundle (no ground truth in
scope) -- score the result afterwards with score_cli.py.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from classify import classify
from compare import compare_email


def _load_inbox(source):
    # loader.py lives alongside the bundle/dataset, not in this package.
    bundle_dir = Path(source)
    if bundle_dir.exists():
        sys.path.insert(0, str(bundle_dir))
    import loader
    return loader.Inbox(source)


def build_submission(source):
    inbox = _load_inbox(source)
    submission = {}
    for email in inbox:
        eid = email["email_id"]
        category, decided_by = classify(email)
        if category == "BL_COMPARISON":
            result = compare_email(email, inbox)
        else:
            result = {"status": "OK", "has_defect": False,
                      "defect_fields": [], "review_reason": None}
        submission[eid] = {
            "category": category,
            "status": result["status"],
            "review_reason": result["review_reason"],
            "has_defect": result["has_defect"],
            "defect_fields": result["defect_fields"],
            "decided_by": decided_by,
        }
    return submission


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
