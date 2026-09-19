#!/usr/bin/env python3
"""SDOC hackathon pipeline -- Streamlit dashboard + per-email inspector +
human review queue.

    ./.venv/bin/streamlit run app.py
"""
import json
import sys
import traceback
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

HERE = Path(__file__).parent
PIPELINE_DIR = HERE / "pipeline"
DEFAULT_BUNDLE = HERE / "sdoc-hackathon-bundle"
DEFAULT_GT = HERE / "sdoc-hackathon-docker" / "data_v2" / "ground_truth.json"
SCORING_PATH = HERE / "sdoc-hackathon-docker" / "server"

sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(SCORING_PATH))

from classify import classify_trace  # noqa: E402
from compare import compare_email_trace  # noqa: E402
from fields import COMPARE_FIELDS  # noqa: E402
import review_store  # noqa: E402
import scoring  # noqa: E402

st.set_page_config(page_title="SDOC Pipeline", layout="wide", page_icon="📦")

STATUS_COLOR = {"OK": "#2e7d32", "MISMATCH": "#c62828", "NEEDS_REVIEW": "#f9a825",
                "PROCESSING_ERROR": "#6d4c41"}
CATEGORY_COLOR = {
    "BL_COMPARISON": "#1565c0", "SI_REQUEST": "#6a1b9a", "INVOICE_QUERY": "#ef6c00",
    "GENERAL": "#546e7a", "SPAM": "#b71c1c", "UNKNOWN": "#757575",
}
CLEAN_RESULT = {"status": "OK", "has_defect": False, "defect_fields": [], "review_reason": None}
ERROR_RESULT = {"status": "PROCESSING_ERROR", "has_defect": False, "defect_fields": [], "review_reason": "processing_error"}


# --------------------------------------------------------------------------
# data loading (cached)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_inbox(source: str):
    import loader
    return loader.Inbox(source)


def process_email(email, inbox):
    """Run classify + (if applicable) compare for one email, catching any
    unexpected failure so it surfaces as a visible PROCESSING_ERROR instead
    of taking the whole run down. This is what the "retry" button re-calls."""
    try:
        ctrace = classify_trace(email)
    except Exception:
        return {
            "email_id": email["email_id"], "email": email,
            "classify_trace": {"category": "UNKNOWN", "decided_by": "error", "confidence": "low", "scores": {}},
            "compare_trace": None, "category": "UNKNOWN", **ERROR_RESULT,
            "decided_by": "error", "processing_error": f"classify_trace failed:\n{traceback.format_exc()}",
        }

    category = ctrace["category"]
    ptrace, processing_error = None, None
    if category == "BL_COMPARISON":
        try:
            ptrace = compare_email_trace(email, inbox)
            result = ptrace["result"]
        except Exception:
            processing_error = f"compare_email_trace failed:\n{traceback.format_exc()}"
            result = dict(ERROR_RESULT)
    else:
        result = dict(CLEAN_RESULT)

    return {
        "email_id": email["email_id"], "email": email, "classify_trace": ctrace,
        "compare_trace": ptrace, "category": category,
        "status": result["status"], "has_defect": result["has_defect"],
        "defect_fields": result["defect_fields"], "review_reason": result["review_reason"],
        "decided_by": ctrace["decided_by"], "processing_error": processing_error,
    }


@st.cache_data(show_spinner="Running the pipeline over the inbox...")
def run_pipeline(source: str):
    inbox = get_inbox(source)
    return [process_email(email, inbox) for email in inbox]


@st.cache_data(show_spinner=False)
def load_ground_truth(path: str):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def needs_human_eyes(r):
    """A case worth a human's attention: the pipeline escalated it, choked
    on it, or wasn't confident about the category even though it resolved."""
    return (r["status"] in ("NEEDS_REVIEW", "PROCESSING_ERROR")
            or r["classify_trace"].get("confidence") == "low")


def effective_row(r, overrides):
    """The row's result after a human override is applied, if one exists.
    This -- not the raw pipeline output -- is what "the report" means."""
    ov = overrides.get(r["email_id"])
    if ov is None:
        return r, False
    merged = dict(r)
    merged.update({
        "category": ov["category"], "status": ov["status"],
        "has_defect": ov["has_defect"], "defect_fields": ov["defect_fields"],
        "review_reason": ov["review_reason"],
    })
    return merged, True


def build_submission_dict(effective_rows):
    return {
        r["email_id"]: {
            "category": r["category"], "status": r["status"],
            "review_reason": r["review_reason"], "has_defect": r["has_defect"],
            "defect_fields": r["defect_fields"],
        }
        for r in effective_rows
    }


def badge(text, color):
    return (f'<span style="background:{color};color:white;padding:2px 10px;'
            f'border-radius:12px;font-size:0.85em;font-weight:600">{text}</span>')


# --------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------
st.sidebar.title("📦 SDOC Pipeline")
page = st.sidebar.radio("View", ["📊 Dashboard", "🔍 Email Inspector", "🧑‍⚖️ Review Queue"])

bundle_path = st.sidebar.text_input("Inbox source", str(DEFAULT_BUNDLE))
use_gt = st.sidebar.checkbox("Score against ground truth", value=DEFAULT_GT.exists())
gt_path = st.sidebar.text_input("Ground truth path", str(DEFAULT_GT), disabled=not use_gt)
apply_overrides = st.sidebar.checkbox(
    "Apply human corrections to report", value=True,
    help="When on, the dashboard/score reflect human-reviewed results, not just the raw pipeline guess.")

inbox = get_inbox(bundle_path)
rows = run_pipeline(bundle_path)
gt = load_ground_truth(gt_path) if use_gt else None
by_id = {r["email_id"]: r for r in rows}
overrides = review_store.load_overrides()

eff_rows = []
for r in rows:
    er, was_overridden = effective_row(r, overrides if apply_overrides else {})
    er = dict(er)
    er["was_overridden"] = was_overridden
    eff_rows.append(er)

pending_count = sum(1 for r in rows if needs_human_eyes(r) and r["email_id"] not in overrides)
st.sidebar.caption(f"{len(rows)} emails loaded from `{bundle_path}`")
st.sidebar.metric("Pending human review", pending_count)
if use_gt and gt is None:
    st.sidebar.warning("Ground truth file not found at that path.")


# --------------------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------------------
def classification_confidence(rows):
    """Per-email classification confidence: the winning category's score,
    the margin over the runner-up (a continuous confidence proxy), and the
    high/low label classify_trace already assigns from that margin."""
    out = []
    for r in rows:
        ct = r["classify_trace"]
        scores = ct.get("scores") or {}
        ranked = sorted(scores.values(), reverse=True)
        top = ranked[0] if ranked else 0
        runner = ranked[1] if len(ranked) > 1 else 0
        out.append({
            "email_id": r["email_id"], "category": r["category"],
            "confidence": ct.get("confidence", "high"),
            "top_score": top, "runner_up_score": runner, "margin": top - runner,
        })
    return out


def dashboard():
    st.title("Pipeline results dashboard")

    score = None
    if gt:
        score = scoring.score_all(gt, build_submission_dict(eff_rows))

    conf_stats = classification_confidence(rows)
    low_conf = [c for c in conf_stats if c["confidence"] == "low"]
    avg_margin = sum(c["margin"] for c in conf_stats) / len(conf_stats) if conf_stats else 0.0

    if score:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Final score", f"{score['final_score']:.3f}")
        c2.metric("Stage-1 macro-F1", f"{score['stage1']['macro_f1']:.3f}")
        c3.metric("Stage-3 defect-F1", f"{score['stage3']['defect_f1']:.3f}")
        c4.metric("End-to-end rate", f"{score['end_to_end']['rate']:.3f}",
                  help=f"{score['end_to_end']['success']}/{score['end_to_end']['total']} defect emails caught")
        c5.metric("Escalation F1", f"{score['reliability']['escalation_f1']:.3f}")
        c6.metric("Pending review", pending_count,
                  help="NEEDS_REVIEW, PROCESSING_ERROR, or low-confidence classifications with no human override yet")
        st.caption("Report reflects human corrections." if apply_overrides
                   else "Report reflects raw pipeline output only (human corrections off).")
        st.divider()

    st.subheader("Classification confidence")
    st.caption("Confidence = margin between the winning category's signal score and the runner-up's "
               "(see pipeline/classify.py). A low margin means the classifier isn't sure -- those cases "
               "also show up in the Review Queue.")
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Low-confidence emails", len(low_conf), help="confidence == 'low' (margin < 2)")
    cc2.metric("Avg. confidence margin", f"{avg_margin:.2f}")
    cc3.metric("High-confidence emails", len(conf_stats) - len(low_conf))

    conf_col1, conf_col2 = st.columns(2)
    with conf_col1:
        df = pd.DataFrame(conf_stats)
        by_cat = df.groupby(["category", "confidence"]).size().reset_index(name="count")
        fig = px.bar(by_cat, x="category", y="count", color="confidence",
                     color_discrete_map={"high": "#2e7d32", "low": "#f9a825"}, barmode="stack")
        fig.update_layout(height=300, title="Confidence by category")
        st.plotly_chart(fig, width="stretch")
    with conf_col2:
        fig = px.histogram(pd.DataFrame(conf_stats), x="margin", nbins=20)
        fig.update_layout(height=300, title="Margin distribution")
        st.plotly_chart(fig, width="stretch")

    if low_conf:
        with st.expander(f"Low-confidence emails ({len(low_conf)})"):
            st.dataframe(pd.DataFrame(low_conf), width="stretch", height=min(300, 40 + 35 * len(low_conf)))

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Category distribution")
        cat_counts = Counter(r["category"] for r in eff_rows)
        df = pd.DataFrame({"category": list(cat_counts), "count": list(cat_counts.values())})
        fig = px.bar(df, x="category", y="count", color="category",
                     color_discrete_map=CATEGORY_COLOR, text="count")
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("Status distribution (BL_COMPARISON)")
        doc_rows = [r for r in eff_rows if r["category"] == "BL_COMPARISON"]
        status_counts = Counter(r["status"] for r in doc_rows)
        df = pd.DataFrame({"status": list(status_counts), "count": list(status_counts.values())})
        fig = px.pie(df, names="status", values="count", color="status",
                     color_discrete_map=STATUS_COLOR, hole=0.45)
        fig.update_layout(height=350)
        st.plotly_chart(fig, width="stretch")

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Defect fields (which field mismatches most)")
        field_counts = Counter()
        for r in doc_rows:
            field_counts.update(r["defect_fields"])
        if field_counts:
            df = pd.DataFrame({"field": list(field_counts), "count": list(field_counts.values())})
            df = df.sort_values("count", ascending=True)
            fig = px.bar(df, x="count", y="field", orientation="h")
            fig.update_layout(height=350)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No mismatches found.")

    with col4:
        st.subheader("NEEDS_REVIEW reasons")
        reason_counts = Counter(r["review_reason"] for r in doc_rows if r["review_reason"])
        if reason_counts:
            df = pd.DataFrame({"reason": list(reason_counts), "count": list(reason_counts.values())})
            fig = px.bar(df, x="reason", y="count", color="reason", text="count")
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No NEEDS_REVIEW cases predicted.")

    if score:
        st.divider()
        st.subheader("Stage-1 confusion matrix (actual vs predicted)")
        cats = scoring.CATEGORIES
        conf = score["stage1"]["confusion"]
        z = [[conf.get(a, {}).get(p, 0) for p in cats] for a in cats]
        fig = go.Figure(go.Heatmap(z=z, x=cats, y=cats, colorscale="Blues", text=z,
                                    texttemplate="%{text}", showscale=False))
        fig.update_layout(xaxis_title="predicted", yaxis_title="actual", height=400)
        st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("All emails")
    filter_cols = st.columns(3)
    cat_filter = filter_cols[0].multiselect("Category", scoring.CATEGORIES)
    status_filter = filter_cols[1].multiselect("Status", ["OK", "MISMATCH", "NEEDS_REVIEW", "PROCESSING_ERROR"])
    only_wrong = filter_cols[2].checkbox("Only show mismatches vs ground truth", disabled=not gt)

    table_rows = []
    for r in eff_rows:
        g = gt.get(r["email_id"]) if gt else None
        correct = None
        if g:
            correct = (g["category"] == r["category"] and g["status"] == r["status"]
                       and set(g.get("defect_fields", [])) == set(r["defect_fields"]))
        if cat_filter and r["category"] not in cat_filter:
            continue
        if status_filter and r["status"] not in status_filter:
            continue
        if only_wrong and correct is not False:
            continue
        table_rows.append({
            "email_id": r["email_id"], "subject": r["email"]["subject"][:60],
            "category": r["category"], "status": r["status"],
            "defect_fields": ", ".join(r["defect_fields"]), "review_reason": r["review_reason"] or "",
            "human_reviewed": r["was_overridden"],
            **({"gold_category": g["category"], "gold_status": g["status"], "correct": correct} if g else {}),
        })
    st.dataframe(pd.DataFrame(table_rows), width="stretch", height=350)


# --------------------------------------------------------------------------
# Email Inspector
# --------------------------------------------------------------------------
def step_icon(outcome):
    return {"OK": "✅", "pass": "➡️", "MISMATCH": "🛑", "NEEDS_REVIEW": "⚠️"}.get(outcome, "•")


def inspector():
    st.title("Email pipeline inspector")

    ids = [r["email_id"] for r in rows]
    eid = st.selectbox("Email", ids, index=0,
                        format_func=lambda e: f"{e} — {by_id[e]['email']['subject'][:70]}")
    r = by_id[eid]
    email = r["email"]
    g = gt.get(eid) if gt else None
    override = overrides.get(eid)

    if override:
        st.info(f"👤 Human {override['action']} this on {override['reviewed_at'][:19]}"
                + (f" — \"{override['reviewer_note']}\"" if override.get("reviewer_note") else ""))

    st.markdown(f"**From:** {email['from']}  \n**Subject:** {email['subject']}")
    with st.expander("Email body"):
        st.text(email["body"])

    if r["processing_error"]:
        st.error("⚠️ Processing failed for this email")
        st.code(r["processing_error"])
        st.caption("Head to the Review Queue to retry or manually resolve this one.")
        return

    st.divider()
    st.subheader("Stage 1 — classification")
    ct = r["classify_trace"]
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(badge(ct["category"], CATEGORY_COLOR.get(ct["category"], "#555")), unsafe_allow_html=True)
        conf = ct.get("confidence")
        conf_icon = "🟢" if conf == "high" else "🟡"
        st.caption(f"decided by: **{ct['decided_by']}** &nbsp; {conf_icon} confidence: **{conf}**")
    with c2:
        scores = ct.get("scores") or {}
        if scores and any(scores.values()):
            df = pd.DataFrame({"category": list(scores), "signal score": list(scores.values())})
            fig = px.bar(df, x="category", y="signal score", color="category",
                         color_discrete_map=CATEGORY_COLOR)
            fig.update_layout(showlegend=False, height=200, margin=dict(t=10, b=10))
            st.plotly_chart(fig, width="stretch")
        if ct.get("matched_signals"):
            st.caption("Signals that fired for the winning category:")
            st.dataframe(pd.DataFrame(ct["matched_signals"]), width="stretch", height=140)
        elif ct.get("reason"):
            st.caption(ct["reason"])
    if g:
        ok = g["category"] == ct["category"]
        st.markdown("✅ matches ground truth" if ok else f"❌ ground truth: **{g['category']}**")

    if r["category"] != "BL_COMPARISON":
        st.info("Not a BL_COMPARISON email — Stage 2/3 (SI-vs-BL comparison) doesn't apply.")
        return

    st.divider()
    st.subheader("Stage 2/3 — SI vs BL comparison")
    pt = r["compare_trace"]

    st.markdown("**Pipeline trace:**")
    for step in pt["steps"]:
        icon = step_icon(step.get("outcome", ""))
        st.markdown(f"{icon} `{step['step']}` — {step['detail']}")

    result = pt["result"]
    st.markdown("**Result:** " + badge(result["status"], STATUS_COLOR.get(result["status"], "#555")),
                unsafe_allow_html=True)
    if result["review_reason"]:
        st.caption(f"review_reason: `{result['review_reason']}`")
    if result["defect_fields"]:
        st.caption(f"defect_fields: `{result['defect_fields']}`")

    if g:
        correct = (g["status"] == result["status"]
                   and set(g.get("defect_fields", [])) == set(result["defect_fields"]))
        if correct:
            st.success(f"Matches ground truth ({g['status']})")
        else:
            st.error(f"Ground truth: **{g['status']}**, reason={g.get('review_reason')}, "
                     f"defect_fields={g.get('defect_fields')}")

    if pt["field_rows"]:
        st.markdown("**Field-by-field diff** (SI vs BL, normalized)")
        df = pd.DataFrame(pt["field_rows"])
        df = df.rename(columns={
            "field": "Field", "si_raw": "SI (raw)", "bl_raw": "BL (raw)",
            "si_normalized": "SI (normalized)", "bl_normalized": "BL (normalized)", "match": "Match",
        })

        def highlight(row):
            color = "" if row["Match"] else "background-color: #ffebee"
            return [color] * len(row)

        st.dataframe(df.style.apply(highlight, axis=1), width="stretch")

    with st.expander("Raw extracted (label, value) pairs"):
        col_si, col_bl = st.columns(2)
        col_si.markdown(f"**SI** — `{pt['si_path']}`")
        col_si.json(pt["si_fields"])
        col_bl.markdown(f"**BL** — `{pt['bl_path']}`")
        col_bl.json(pt["bl_fields"])


# --------------------------------------------------------------------------
# Review Queue
# --------------------------------------------------------------------------
STATUS_OPTIONS = ["OK", "MISMATCH", "NEEDS_REVIEW"]


def _review_form(r, key_prefix):
    """Evidence + a confirm/correct/retry form for one email. Returns True
    if the queue should be re-rendered (an action was taken)."""
    email = r["email"]
    st.markdown(f"**From:** {email['from']}  \n**Subject:** {email['subject']}")
    st.text(email["body"][:500] + ("..." if len(email["body"]) > 500 else ""))

    if r["processing_error"]:
        st.error("Processing failed:")
        st.code(r["processing_error"])
    else:
        ct = r["classify_trace"]
        st.caption(f"Auto result: category=**{r['category']}** (confidence: {ct.get('confidence')}), "
                   f"status=**{r['status']}**"
                   + (f", reason=`{r['review_reason']}`" if r["review_reason"] else "")
                   + (f", defect_fields=`{r['defect_fields']}`" if r["defect_fields"] else ""))
        pt = r["compare_trace"]
        if pt:
            for step in pt["steps"]:
                st.caption(f"{step_icon(step.get('outcome', ''))} `{step['step']}` — {step['detail']}")
            if pt["field_rows"]:
                st.dataframe(pd.DataFrame(pt["field_rows"]), width="stretch", height=180)
            if pt["si_fields"] or pt["bl_fields"]:
                with st.expander("Raw extracted fields"):
                    c1, c2 = st.columns(2)
                    c1.json(pt["si_fields"])
                    c2.json(pt["bl_fields"])

    acted = False
    col_confirm, col_retry = st.columns([1, 1])
    if not r["processing_error"] and col_confirm.button("✅ Confirm automatic result", key=f"{key_prefix}_confirm"):
        review_store.save_override(
            r["email_id"],
            {"category": r["category"], "status": r["status"], "has_defect": r["has_defect"],
             "defect_fields": r["defect_fields"], "review_reason": r["review_reason"]},
            reviewer_note="", action="confirmed",
        )
        st.success("Confirmed.")
        acted = True

    if col_retry.button("🔁 Retry processing", key=f"{key_prefix}_retry"):
        fresh = process_email(email, inbox)
        st.session_state[f"retry_result_{r['email_id']}"] = fresh
        st.rerun()

    retried = st.session_state.get(f"retry_result_{r['email_id']}")
    if retried:
        st.markdown("**Retry result:**")
        if retried["processing_error"]:
            st.error("Still failing:")
            st.code(retried["processing_error"])
        else:
            st.success(f"category={retried['category']}, status={retried['status']}, "
                       f"defect_fields={retried['defect_fields']}")
            if st.button("Accept retried result", key=f"{key_prefix}_accept_retry"):
                review_store.save_override(
                    r["email_id"],
                    {"category": retried["category"], "status": retried["status"],
                     "has_defect": retried["has_defect"], "defect_fields": retried["defect_fields"],
                     "review_reason": retried["review_reason"]},
                    reviewer_note="resolved via retry", action="corrected",
                )
                del st.session_state[f"retry_result_{r['email_id']}"]
                st.success("Saved.")
                acted = True

    st.markdown("**Or correct it:**")
    default_cat = r["category"] if r["category"] in scoring.CATEGORIES else scoring.CATEGORIES[0]
    corrected_cat = st.selectbox("Category", scoring.CATEGORIES,
                                  index=scoring.CATEGORIES.index(default_cat), key=f"{key_prefix}_cat")
    corrected_status, corrected_fields, corrected_reason = "OK", [], None
    if corrected_cat == "BL_COMPARISON":
        default_status = r["status"] if r["status"] in STATUS_OPTIONS else "OK"
        corrected_status = st.selectbox("Status", STATUS_OPTIONS,
                                         index=STATUS_OPTIONS.index(default_status), key=f"{key_prefix}_status")
        if corrected_status == "MISMATCH":
            corrected_fields = st.multiselect("Defect fields", COMPARE_FIELDS,
                                               default=[f for f in r["defect_fields"] if f in COMPARE_FIELDS],
                                               key=f"{key_prefix}_fields")
        if corrected_status == "NEEDS_REVIEW":
            reasons = scoring.REVIEW_REASONS
            default_reason = r["review_reason"] if r["review_reason"] in reasons else reasons[0]
            corrected_reason = st.selectbox("Review reason", reasons,
                                             index=reasons.index(default_reason), key=f"{key_prefix}_reason")
    note = st.text_input("Reviewer note", key=f"{key_prefix}_note")
    if st.button("✏️ Save correction", key=f"{key_prefix}_save"):
        review_store.save_override(
            r["email_id"],
            {"category": corrected_cat, "status": corrected_status,
             "has_defect": corrected_status == "MISMATCH", "defect_fields": corrected_fields,
             "review_reason": corrected_reason},
            reviewer_note=note, action="corrected",
        )
        st.success("Saved.")
        acted = True
    return acted


def review_queue():
    st.title("Human review queue")
    st.caption("Cases the pipeline escalated, choked on, or wasn't confident about. "
               "Confirm, correct, or retry -- the result feeds straight into the report above.")

    pending = [r for r in rows if needs_human_eyes(r) and r["email_id"] not in overrides]
    reviewed = [(r, overrides[r["email_id"]]) for r in rows if r["email_id"] in overrides]

    c1, c2 = st.columns(2)
    c1.metric("Pending", len(pending))
    c2.metric("Reviewed", len(reviewed))

    st.subheader(f"Pending ({len(pending)})")
    if not pending:
        st.success("Nothing waiting on a human right now.")
    for r in pending:
        reason = r["review_reason"] or ("low-confidence classification" if r["classify_trace"].get("confidence") == "low" else r["status"])
        with st.expander(f"{r['email_id']} — {r['email']['subject'][:70]}  ·  {reason}"):
            if _review_form(r, key_prefix=f"pend_{r['email_id']}"):
                st.rerun()

    st.divider()
    st.subheader(f"Reviewed ({len(reviewed)})")
    for r, ov in reviewed:
        with st.expander(f"{r['email_id']} — {ov['action']} — {r['email']['subject'][:60]}"):
            st.json(ov)
            if st.button("↩️ Undo (send back to pending)", key=f"undo_{r['email_id']}"):
                review_store.clear_override(r["email_id"])
                st.rerun()


if page == "📊 Dashboard":
    dashboard()
elif page == "🔍 Email Inspector":
    inspector()
else:
    review_queue()
