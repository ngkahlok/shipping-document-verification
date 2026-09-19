"""Stage 2/3 -- SI-vs-BL extraction, reliability checks, and field diffing."""
from fields import COMPARE_FIELDS, is_blank_value
from normalize import values_match, normalized_value
from extract import extract_fields, Unreadable

CLEAN_RESULT = {"status": "OK", "has_defect": False, "defect_fields": [], "review_reason": None}


def _needs_review(reason):
    return {"status": "NEEDS_REVIEW", "has_defect": False, "defect_fields": [], "review_reason": reason}


def _pick_si_bl(attachments):
    si = next((a for a in attachments if "_SI." in a), None)
    bl = next((a for a in attachments if "_BL." in a), None)
    if si is None and bl is None and len(attachments) >= 2:
        si, bl = attachments[0], attachments[1]
    return si, bl


def compare_email_trace(email, inbox):
    """Run the full BL_COMPARISON pipeline, recording every intermediate
    decision so a UI can show *why* the final result came out the way it did.

    Returns a dict with:
      result       -- the submission-shape dict (status/has_defect/...)
      steps        -- ordered list of {step, outcome, detail} checkpoints
      si_path/bl_path, si_fields/bl_fields, field_rows (per-field diff table)
    """
    steps = []
    attachments = email.get("attachments") or []
    body_lower = (email.get("body") or "").lower()

    steps.append({"step": "attachment_count", "detail": f"{len(attachments)} attachment(s): {attachments}"})

    if len(attachments) == 0:
        if "compare" in body_lower:
            steps.append({"step": "missing_attachment", "outcome": "NEEDS_REVIEW",
                          "detail": "0 attachments but body explicitly asks to compare -> escalate"})
            return {"result": _needs_review("missing_attachment"), "steps": steps,
                    "si_path": None, "bl_path": None, "si_fields": {}, "bl_fields": {}, "field_rows": []}
        steps.append({"step": "missing_attachment", "outcome": "OK",
                      "detail": "0 attachments and body is just a request (no 'compare') -> nothing to compare, trivially OK"})
        return {"result": dict(CLEAN_RESULT), "steps": steps,
                "si_path": None, "bl_path": None, "si_fields": {}, "bl_fields": {}, "field_rows": []}

    if len(attachments) == 1:
        steps.append({"step": "missing_attachment", "outcome": "NEEDS_REVIEW",
                      "detail": "only 1 of the expected 2 attachments present -> escalate"})
        return {"result": _needs_review("missing_attachment"), "steps": steps,
                "si_path": attachments[0], "bl_path": None, "si_fields": {}, "bl_fields": {}, "field_rows": []}

    si_path, bl_path = _pick_si_bl(attachments)
    if si_path is None or bl_path is None:
        steps.append({"step": "missing_attachment", "outcome": "NEEDS_REVIEW",
                      "detail": "couldn't identify a distinct SI and BL attachment -> escalate"})
        return {"result": _needs_review("missing_attachment"), "steps": steps,
                "si_path": si_path, "bl_path": bl_path, "si_fields": {}, "bl_fields": {}, "field_rows": []}

    steps.append({"step": "identify_attachments", "detail": f"SI={si_path}  BL={bl_path}"})

    try:
        si_fields, si_pairs = extract_fields(inbox.read_bytes(si_path), si_path)
    except Unreadable as e:
        steps.append({"step": "extract_si", "outcome": "NEEDS_REVIEW", "detail": f"SI unreadable: {e}"})
        return {"result": _needs_review("unreadable"), "steps": steps,
                "si_path": si_path, "bl_path": bl_path, "si_fields": {}, "bl_fields": {}, "field_rows": []}
    steps.append({"step": "extract_si", "detail": f"{len(si_fields)}/{len(COMPARE_FIELDS)} fields resolved"})

    try:
        bl_fields, bl_pairs = extract_fields(inbox.read_bytes(bl_path), bl_path)
    except Unreadable as e:
        steps.append({"step": "extract_bl", "outcome": "NEEDS_REVIEW", "detail": f"BL unreadable: {e}"})
        return {"result": _needs_review("unreadable"), "steps": steps,
                "si_path": si_path, "bl_path": bl_path, "si_fields": si_fields, "bl_fields": {}, "field_rows": []}
    steps.append({"step": "extract_bl", "detail": f"{len(bl_fields)}/{len(COMPARE_FIELDS)} fields resolved"})

    si_hits = sum(1 for f in COMPARE_FIELDS if f in si_fields)
    bl_hits = sum(1 for f in COMPARE_FIELDS if f in bl_fields)
    if si_hits < 3 or bl_hits < 3:
        steps.append({"step": "wrong_doc_type_check", "outcome": "NEEDS_REVIEW",
                      "detail": f"too few of the 7 fields resolved (SI={si_hits}, BL={bl_hits}) -- "
                                f"likely not a real SI/BL pair -> escalate"})
        return {"result": _needs_review("wrong_doc_type"), "steps": steps,
                "si_path": si_path, "bl_path": bl_path, "si_fields": si_fields, "bl_fields": bl_fields, "field_rows": []}
    steps.append({"step": "wrong_doc_type_check", "outcome": "pass",
                  "detail": f"SI={si_hits}/7, BL={bl_hits}/7 fields resolved -- looks like a real pair"})

    blank_field = None
    for f in COMPARE_FIELDS:
        if is_blank_value(si_fields.get(f, "")) or is_blank_value(bl_fields.get(f, "")):
            blank_field = f
            break
    if blank_field:
        steps.append({"step": "missing_value_check", "outcome": "NEEDS_REVIEW",
                      "detail": f"field '{blank_field}' is blank/placeholder on one side -> escalate"})
        return {"result": _needs_review("missing_value"), "steps": steps,
                "si_path": si_path, "bl_path": bl_path, "si_fields": si_fields, "bl_fields": bl_fields, "field_rows": []}
    steps.append({"step": "missing_value_check", "outcome": "pass", "detail": "no blank/placeholder values found"})

    def _display(value):
        """Render a normalized value for the UI (e.g. the (count, size)
        tuple container_count normalizes to) as a plain string."""
        if isinstance(value, tuple):
            return " x ".join(str(v) for v in value if v)
        return value

    field_rows = []
    mismatched = []
    for f in COMPARE_FIELDS:
        si_raw, bl_raw = si_fields.get(f), bl_fields.get(f)
        match = values_match(f, si_raw, bl_raw)
        if not match:
            mismatched.append(f)
        field_rows.append({
            "field": f, "si_raw": str(si_raw) if si_raw is not None else None,
            "bl_raw": str(bl_raw) if bl_raw is not None else None,
            "si_normalized": _display(normalized_value(f, si_raw)),
            "bl_normalized": _display(normalized_value(f, bl_raw)),
            "match": match,
        })

    if mismatched:
        steps.append({"step": "field_diff", "outcome": "MISMATCH", "detail": f"mismatched fields: {mismatched}"})
        result = {"status": "MISMATCH", "has_defect": True, "defect_fields": mismatched, "review_reason": None}
    else:
        steps.append({"step": "field_diff", "outcome": "OK", "detail": "all 7 fields match"})
        result = dict(CLEAN_RESULT)

    return {"result": result, "steps": steps, "si_path": si_path, "bl_path": bl_path,
            "si_fields": si_fields, "bl_fields": bl_fields, "field_rows": field_rows}


def compare_email(email, inbox):
    return compare_email_trace(email, inbox)["result"]
