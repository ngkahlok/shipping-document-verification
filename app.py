#!/usr/bin/env python3
"""SDOC hackathon pipeline -- Streamlit dashboard + per-email inspector.

    ./.venv/bin/streamlit run app.py
"""
import json
import sys
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
import scoring  # noqa: E402

st.set_page_config(page_title="SDOC Pipeline", layout="wide", page_icon="📦")

STATUS_COLOR = {"OK": "#2e7d32", "MISMATCH": "#c62828", "NEEDS_REVIEW": "#f9a825"}
CATEGORY_COLOR = {
    "BL_COMPARISON": "#1565c0", "SI_REQUEST": "#6a1b9a", "INVOICE_QUERY": "#ef6c00",
    "GENERAL": "#546e7a", "SPAM": "#b71c1c",
}


# --------------------------------------------------------------------------
# data loading (cached)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_inbox(source: str):
    import loader
    return loader.Inbox(source)


@st.cache_data(show_spinner="Running the pipeline over the inbox...")
def run_pipeline(source: str):
    inbox = get_inbox(source)
    rows = []
    for email in inbox:
        eid = email["email_id"]
        ctrace = classify_trace(email)
        category = ctrace["category"]
        if category == "BL_COMPARISON":
            ptrace = compare_email_trace(email, inbox)
            result = ptrace["result"]
        else:
            ptrace = None
            result = {"status": "OK", "has_defect": False, "defect_fields": [], "review_reason": None}
        rows.append({
            "email_id": eid, "email": email, "classify_trace": ctrace,
            "compare_trace": ptrace, "category": category,
            "status": result["status"], "has_defect": result["has_defect"],
            "defect_fields": result["defect_fields"], "review_reason": result["review_reason"],
            "decided_by": ctrace["decided_by"],
        })
    return rows


@st.cache_data(show_spinner=False)
def load_ground_truth(path: str):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def build_submission_dict(rows):
    return {
        r["email_id"]: {
            "category": r["category"], "status": r["status"],
            "review_reason": r["review_reason"], "has_defect": r["has_defect"],
            "defect_fields": r["defect_fields"],
        }
        for r in rows
    }


def badge(text, color):
    return (f'<span style="background:{color};color:white;padding:2px 10px;'
            f'border-radius:12px;font-size:0.85em;font-weight:600">{text}</span>')


# --------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------
st.sidebar.title("📦 SDOC Pipeline")
page = st.sidebar.radio("View", ["📊 Dashboard", "🔍 Email Inspector"])

bundle_path = st.sidebar.text_input("Inbox source", str(DEFAULT_BUNDLE))
use_gt = st.sidebar.checkbox("Score against ground truth", value=DEFAULT_GT.exists())
gt_path = st.sidebar.text_input("Ground truth path", str(DEFAULT_GT), disabled=not use_gt)

rows = run_pipeline(bundle_path)
gt = load_ground_truth(gt_path) if use_gt else None
by_id = {r["email_id"]: r for r in rows}

st.sidebar.caption(f"{len(rows)} emails loaded from `{bundle_path}`")
if use_gt and gt is None:
    st.sidebar.warning("Ground truth file not found at that path.")


# --------------------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------------------
def dashboard():
    st.title("Pipeline results dashboard")

    score = None
    if gt:
        score = scoring.score_all(gt, build_submission_dict(rows))

    if score:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Final score", f"{score['final_score']:.3f}")
        c2.metric("Stage-1 macro-F1", f"{score['stage1']['macro_f1']:.3f}")
        c3.metric("Stage-3 defect-F1", f"{score['stage3']['defect_f1']:.3f}")
        c4.metric("End-to-end rate", f"{score['end_to_end']['rate']:.3f}",
                  help=f"{score['end_to_end']['success']}/{score['end_to_end']['total']} defect emails caught")
        c5.metric("Escalation F1", f"{score['reliability']['escalation_f1']:.3f}")
        st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Category distribution")
        cat_counts = Counter(r["category"] for r in rows)
        df = pd.DataFrame({"category": list(cat_counts), "count": list(cat_counts.values())})
        fig = px.bar(df, x="category", y="count", color="category",
                     color_discrete_map=CATEGORY_COLOR, text="count")
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("Status distribution (BL_COMPARISON)")
        doc_rows = [r for r in rows if r["category"] == "BL_COMPARISON"]
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
    status_filter = filter_cols[1].multiselect("Status", ["OK", "MISMATCH", "NEEDS_REVIEW"])
    only_wrong = filter_cols[2].checkbox("Only show mismatches vs ground truth", disabled=not gt)

    table_rows = []
    for r in rows:
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
    default_idx = 0
    eid = st.selectbox("Email", ids, index=default_idx,
                        format_func=lambda e: f"{e} — {by_id[e]['email']['subject'][:70]}")
    r = by_id[eid]
    email = r["email"]
    g = gt.get(eid) if gt else None

    st.markdown(
        f"**From:** {email['from']}  \n**Subject:** {email['subject']}",
    )
    with st.expander("Email body"):
        st.text(email["body"])

    st.divider()
    st.subheader("Stage 1 — classification")
    ct = r["classify_trace"]
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(badge(ct["category"], CATEGORY_COLOR.get(ct["category"], "#555")), unsafe_allow_html=True)
        st.caption(f"decided by: **{ct['decided_by']}**")
    with c2:
        if ct.get("matched_pattern"):
            st.code(f"pattern: {ct['matched_pattern']}\nmatched: {ct['matched_text']!r}", language="text")
        else:
            st.caption(ct.get("reason", ""))
    if g:
        ok = g["category"] == ct["category"]
        st.markdown(("✅ matches ground truth" if ok else f"❌ ground truth: **{g['category']}**"))

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
    st.markdown("**Result:** " + badge(result["status"], STATUS_COLOR[result["status"]]),
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


if page == "📊 Dashboard":
    dashboard()
else:
    inspector()
