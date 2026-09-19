# Shipping Document Verification (SDOC)

A pipeline that reads a shipping-logistics inbox, classifies each email, and
for comparison requests, cross-checks the **Shipping Instruction (SI)**
against the **draft Bill of Lading (BL)** attachments to catch discrepancies
before documents go out. Built for the SDOC hackathon dataset in
`sdoc-hackathon-bundle/` (participant data) and `sdoc-hackathon-docker/`
(organizer data + scoring server).

Current score against the full 520-email set: **1.0000** (perfect on every
axis — see [Scoring](#scoring)).

## Contents

```
pipeline/                  the solution (classify -> extract -> compare)
app.py                     Streamlit dashboard + per-email inspector
sdoc-hackathon-bundle/     participant data: inbox/, attachments/, loader.py (no labels)
sdoc-hackathon-docker/     organizer data: ground_truth.json, scoring.py, score_cli.py, Docker server
.venv/                     project-local virtualenv (pdfplumber, python-docx, openpyxl, streamlit, plotly, pandas)
```

## The task

For every email, decide:

1. **category** — `BL_COMPARISON`, `SI_REQUEST`, `INVOICE_QUERY`, `GENERAL`, or `SPAM`.
2. For `BL_COMPARISON` emails, compare the SI against the draft BL across 7
   fields — `shipper`, `consignee`, `notify_party`, `port_of_loading`,
   `port_of_discharge`, `container_count`, `gross_weight_kg` — and report:
   - `status`: `OK` (all match), `MISMATCH` (≥1 field differs), or
     `NEEDS_REVIEW` (can't decide — unreadable / missing / wrong document).
   - `has_defect` + `defect_fields` when `MISMATCH`.
   - `review_reason` when `NEEDS_REVIEW`
     (`wrong_doc_type` | `missing_attachment` | `unreadable` | `missing_value`).

The SI and BL label the same field differently (`Port of Loading` vs `Load
Port`), and attachments come in four formats (`.txt`, `.pdf`, `.docx`,
`.xlsx`) — so the extractor has to align by meaning, not by header text.

## Pipeline design

Four stages, in `pipeline/`:

- **`classify.py`** — Stage 1. Subject lines in this inbox are coded
  (department/route/carrier tokens), so a rule-first regex classifier
  resolves ~100% of emails cheaply and deterministically, with a fallback
  for anything a rule doesn't catch.
- **`extract.py`** — Stage 2. Turns each attachment into `(label, value)`
  pairs regardless of format:
  - `.txt` — line-based, colon-split.
  - `.docx` — table rows (`python-docx`).
  - `.xlsx` — two-column cell rows (`openpyxl`).
  - `.pdf` — the hard one. Long labels (e.g. `"Notify
    Party/Intermediate Consignee"`) can visually overlap their fixed-position
    value column in the generated PDF. `pdfplumber`'s spatial word-clustering
    garbles that overlap into interleaved nonsense, so extraction works at
    the **character-stream level** instead: the underlying character order
    is always clean (every label character is emitted before the value's),
    so a label/value split is found by walking that stream and either
    following an explicit colon, or detecting where x-position runs
    backwards (a new text object starting left of where the previous one
    had already reached).
- **`fields.py`** / **`normalize.py`** — the label-synonym vocabulary and
  cross-format value normalization (strips port LOCODE suffixes, entity
  address lines, weight formatting, container count/size), so the same
  field compares equal regardless of which format or synonym rendered it.
- **`compare.py`** — Stages 2b/3. Before diffing, a reliability gate checks
  (in order): attachment count + intent (`missing_attachment`), extraction
  failure (`unreadable`), how many of the 7 fields resolved at all
  (`wrong_doc_type`), and blank/placeholder values (`missing_value`). Only
  if none of those fire does it do the field-by-field diff and report
  `OK`/`MISMATCH`.
- **`run.py`** — orchestrates the above into a `submission.json`.

Every module also exposes a `*_trace` variant (`classify_trace`,
`compare_email_trace`) that returns the full reasoning behind a decision —
which regex matched, which check fired, the per-field diff table — used by
the Streamlit inspector below.

## Running the pipeline

```bash
./.venv/bin/python3 pipeline/run.py sdoc-hackathon-bundle submission.json
```

Reads only the participant bundle (no ground truth involved) and writes a
`submission.json` shaped like `sdoc-hackathon-bundle/sample_submission.json`.

## Scoring

Ground truth lives in `sdoc-hackathon-docker/data_v2/ground_truth.json`
(organizer-only — never shipped to participants). Score any submission
against it, no Docker required:

```bash
cd sdoc-hackathon-docker/server
python3 score_cli.py /path/to/submission.json
```

Final score = 30% Stage-1 macro-F1 + 20% Stage-3 defect-F1 + 50% end-to-end
(defects caught all the way through, exact field match). `NEEDS_REVIEW`
handling is scored separately as an escalation-precision/recall
"reliability" axis.

If you have Docker installed, `sdoc-hackathon-docker/docker-compose.yml`
also stands up an HTTP server (`docker compose up --build`, serves on
`localhost:8080`) exposing `POST /submit` for remote scoring — useful for
simulating the real participant flow, where ground truth never leaves the
server. See `sdoc-hackathon-docker/README.md`.

## Streamlit app

```bash
./.venv/bin/streamlit run app.py
```

Two views:

- **📊 Dashboard** — final score and per-stage metrics as KPI cards (scored
  live against `ground_truth.json`), category/status distributions, which
  of the 7 fields mismatches most often, `NEEDS_REVIEW` reason breakdown, a
  Stage-1 confusion-matrix heatmap, and a filterable table of every email
  (by category/status, or "only mismatches vs ground truth").
- **🔍 Email Inspector** — pick any email and watch the pipeline reason
  through it: which classification rule fired and on what text, then for
  `BL_COMPARISON` emails a step-by-step trace of the reliability checks,
  a side-by-side SI-vs-BL field diff table (raw + normalized, mismatches
  highlighted), the raw extracted `(label, value)` pairs per attachment,
  and a ground-truth comparison badge.

Both the inbox path and ground-truth path are editable in the sidebar;
ground-truth scoring can be toggled off to browse label-free.

## Setup

Everything runs inside a project-local virtualenv so nothing pollutes the
system Python:

```bash
python3 -m venv .venv
./.venv/bin/pip install pdfplumber python-docx openpyxl streamlit plotly pandas
```

(`score_cli.py` and the Docker server only need the standard library /
`fastapi`+`uvicorn` respectively — see `sdoc-hackathon-docker/server/requirements.txt`.)
