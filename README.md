# Shipping Document Verification (SDOC)

A pipeline that reads a shipping-logistics inbox, classifies each email, and
for comparison requests, cross-checks the **Shipping Instruction (SI)**
against the **draft Bill of Lading (BL)** attachments to catch discrepancies
before documents go out. Built for the SDOC hackathon dataset in
`sdoc-hackathon-bundle/` (participant data) and `sdoc-hackathon-docker/`
(organizer data + scoring server).

Current score against the full 520-email set with the default **rule-based**
classifier: **1.0000** (perfect on every axis — see [Scoring](#scoring)). An
optional **Gemini-backed** classifier is also available (see
[Classifier backends](#classifier-backends)) — it trades a little of that
synthetic-benchmark score for a classifier that should generalize better to
phrasing this dataset's generator never produced.

## Contents

```
pipeline/
  classify.py            Stage 1 dispatcher: rule-based scorer, or Gemini (see below)
  gemini_classify.py      Gemini client, prompt/schema, disk cache, retry/backoff
  extract.py              Stage 2: per-format attachment -> (label, value) pairs
  fields.py / normalize.py   label-synonym vocabulary + cross-format value normalization
  compare.py              Stages 2b/3: reliability gate + SI-vs-BL field diff
  review_store.py         JSON-backed human-review overrides
  run.py                  CLI: builds a submission.json from a bundle
  tools/smoke_test_gemini.py   cheap ~15-call sanity check before a full Gemini run
app.py                    Streamlit dashboard + inspector + human review queue
tests/test_app.py          headless Streamlit checks (streamlit.testing.v1.AppTest)
sdoc-hackathon-bundle/     participant data: inbox/, attachments/, loader.py (no labels)
sdoc-hackathon-docker/     organizer data: ground_truth.json, scoring.py, score_cli.py, Docker server
.venv/                     project-local virtualenv
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

- **`classify.py`** — Stage 1, and a thin dispatcher (see
  [Classifier backends](#classifier-backends)). Its default path is a
  weighted keyword-signal scorer: not a literal template matcher, but a set
  of *concept* phrases per category (what a request like this actually
  says), so it has a chance of transferring to phrasing it's never seen —
  with a `confidence` (`high`/`low`, from the margin between the top two
  categories' scores) exposed alongside the winning label, so a genuinely
  ambiguous email can be routed to a human instead of forcing a guess.
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
which signal/regex matched, which check fired, the per-field diff table —
used by the Streamlit inspector below.

## Classifier backends

`pipeline/classify.py`'s `classify_trace()` is a dispatcher controlled by
the `SDOC_CLASSIFIER` env var:

- **`rules`** (default) — the keyword-signal scorer described above. No API
  key, no network calls, no cost. Also always runs regardless of backend,
  since its output feeds into the Gemini prompt as a hint and is the
  fallback on any Gemini failure.
- **`gemini`** — routes through `pipeline/gemini_classify.py`, which calls
  the Gemini API with the email content *and* the rule scorer's guess/scores
  embedded in the prompt as a labeled "heuristic hint" (explicitly flagged
  as possibly wrong, not ground truth). Returns a 0–100 score per category,
  a category pick, and a one-sentence `reasoning` string.

**On any Gemini failure** (missing key, network error, rate limit, malformed
response after retries) — it falls back to the rule-based result, marked
`decided_by: "rule_fallback"` with the error attached in `llm_error`, rather
than crashing or guessing blind. This has been verified against a *real*
failure, not just a simulated one — see the quota note below.

**Caching**: every Gemini result is cached to
`pipeline/.cache/gemini_classify/<hash>.json` (gitignored), one file per
email, keyed by the email's content plus the prompt/model version — so
re-running the pipeline never re-calls the API for an email it's already
classified, and editing the prompt automatically invalidates just the
affected entries. Confidence is recomputed on every read (not cached), so
tuning the confidence margin doesn't require busting the cache.

**Concurrency**: `gemini_classify.warm_cache(emails)` pre-fills the cache
for a batch using a small thread pool (`SDOC_GEMINI_MAX_WORKERS`, default 6)
before the normal one-email-at-a-time loop runs — both `app.py` and
`pipeline/run.py` call this automatically when the backend is `gemini` and
a key is present.

**A real lesson learned**: Google's free tier for the Flash-tier model is a
**daily** quota (as of testing, 20 requests/day for this project/model, via
`generativelanguage.googleapis.com/generate_content_free_tier_requests`),
not a per-minute rate limit — a single full 520-email run will exhaust it
almost immediately. Enable billing on the Google AI Studio project backing
your key for a realistic quota before running the full inbox through
Gemini; small smoke tests are fine on the free tier.

### Configuration (env vars)

Set these directly in your shell, in a project-root `.env` file (gitignored,
auto-loaded — see `.env.example`), or in `.streamlit/secrets.toml` for the
Streamlit app specifically (see [Streamlit app](#streamlit-app)).

| Var | Default | Meaning |
|---|---|---|
| `SDOC_CLASSIFIER` | `rules` | `rules` or `gemini` |
| `GEMINI_API_KEY` | — | required to actually call the API |
| `GEMINI_MODEL` | `gemini-flash-latest` | Google's rolling fast/cheap alias; set a dated model to pin one |
| `SDOC_GEMINI_MAX_WORKERS` | `6` | thread pool size for `warm_cache` |
| `SDOC_GEMINI_MAX_RETRIES` | `4` | retries on transient errors (429/5xx) before falling back |
| `SDOC_GEMINI_CONFIDENT_MARGIN` | `20` | Gemini's 0–100 score scale needs its own high/low threshold (the rule scorer uses a much smaller one) |
| `SDOC_GEMINI_CACHE_DIR` | `pipeline/.cache/gemini_classify/` | override for tests/CI |

### Sanity-check before a full run

```bash
export GEMINI_API_KEY=...
./.venv/bin/python3 pipeline/tools/smoke_test_gemini.py
```

~15 calls (3 per category, against ground truth), prints
`email_id | gold | predicted | confidence | decided_by | llm_error` per row
— cheap feedback before spending a full 520-email budget.

## Running the pipeline

```bash
./.venv/bin/python3 pipeline/run.py sdoc-hackathon-bundle submission.json
```

Reads only the participant bundle (no ground truth involved) and writes a
`submission.json` shaped like `sdoc-hackathon-bundle/sample_submission.json`.
Set `SDOC_CLASSIFIER=gemini` to use the Gemini backend for this run.

## Scoring

Ground truth lives in `sdoc-hackathon-docker/data_v2/ground_truth.json` —
**organizer-only, gitignored, and not tracked in this repo** (it's the
answer key; see that folder's own README for why it must never be handed to
participants). You'll need your own copy locally (e.g. from the original
dataset zip) for the commands below to work. Score any submission against
it, no Docker required:

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

Three pages, selectable in the sidebar (which also holds the inbox/ground-truth
paths, a **classifier backend** toggle, and an "apply human corrections to
report" switch):

- **📊 Dashboard** — final score and per-stage metrics as KPI cards (scored
  live against `ground_truth.json`), a **classification confidence** section
  (low/high-confidence counts, average margin, a per-category confidence
  breakdown and margin-distribution histogram — useful for tuning
  `SDOC_GEMINI_CONFIDENT_MARGIN`), category/status distributions, which of
  the 7 fields mismatches most often, `NEEDS_REVIEW` reason breakdown, a
  Stage-1 confusion-matrix heatmap, and a filterable table of every email.
- **🔍 Email Inspector** — pick any email and watch the pipeline reason
  through it: which classifier decided (rule signal fired, or Gemini's
  per-category scores + `reasoning`, with a visible warning if it had to
  fall back), then for `BL_COMPARISON` emails a step-by-step trace of the
  reliability checks, a side-by-side SI-vs-BL field diff table, the raw
  extracted `(label, value)` pairs per attachment, and a ground-truth
  comparison badge.
- **🧑‍⚖️ Review Queue** — see [Human review loop](#human-review-loop).

## Human review loop

Anything the pipeline escalated (`NEEDS_REVIEW`), choked on
(`PROCESSING_ERROR` — an unexpected exception is caught and surfaced
visibly rather than crashing the run), or wasn't confident about even
though it resolved (`confidence: "low"`) shows up in the **Review Queue**
page with its full evidence (extracted fields, trace steps, Gemini's
reasoning if applicable). Three actions, each persisted to
`review_overrides.json` (gitignored) via `pipeline/review_store.py`:

- **Confirm** — accept the pipeline's own result as-is.
- **Correct** — pick the right category/status/fields yourself.
- **Retry** — re-run extraction/classification live (useful after a
  transient failure); accept the new result if it looks right.

Whatever's decided there overrides the pipeline's raw output in **the
report** — the Dashboard's score and tables reflect human-reviewed results
by default (toggleable in the sidebar), with an "undo" per reviewed item.

## Testing

```bash
./.venv/bin/pip install -r requirements-dev.txt   # adds pytest
./.venv/bin/pytest tests/test_app.py -v
```

Headless Streamlit checks via `streamlit.testing.v1.AppTest`: the app loads
with no exceptions and defaults to the `rules` backend, and — without
needing any API key — switching to `gemini` with no key set degrades
visibly (a sidebar warning, a fallback banner in the Inspector) rather than
hanging or crashing. A third test (needs a real `GEMINI_API_KEY`) is
skipped automatically when the key isn't set.

## Setup (new users start here)

Requires Python 3.9+. Everything runs inside a project-local virtualenv so
nothing pollutes your system Python.

1. **Clone/open the project, then create the virtualenv:**

   ```bash
   cd shipping-document-verification
   python3 -m venv .venv
   ```

2. **Install dependencies** from `requirements.txt`:

   ```bash
   ./.venv/bin/pip install -r requirements.txt
   ```

3. **Verify it works** by running the pipeline over the sample inbox:

   ```bash
   ./.venv/bin/python3 pipeline/run.py sdoc-hackathon-bundle submission.json
   ```

   You should see `wrote 520 predictions to submission.json`.

4. **(Optional) Score it** against the organizer's ground truth — no Docker
   needed, just the standard library:

   ```bash
   cd sdoc-hackathon-docker/server
   python3 score_cli.py ../../submission.json
   cd ../..
   ```

5. **(Optional) Set up the Gemini backend** — copy `.env.example` to `.env`
   and fill in `GEMINI_API_KEY` (see [Classifier backends](#classifier-backends)
   for the full config; `.env` is gitignored and auto-loaded by both the CLI
   and the app). For the Streamlit app specifically, `.streamlit/secrets.toml`
   works the same way and is what you'd use once deployed to Streamlit
   Community Cloud (see `.streamlit/secrets.toml.example`).

6. **Launch the Streamlit app:**

   ```bash
   ./.venv/bin/streamlit run app.py
   ```

   Opens at `http://localhost:8501` — see [Streamlit app](#streamlit-app)
   above for what's on each page.

Everything after step 2 assumes the venv's Python (`./.venv/bin/python3` /
`./.venv/bin/streamlit`), not whatever `python3` resolves to on your `PATH`.

`score_cli.py` and the Docker server (step 4's alternative, see
[Scoring](#scoring)) only need the standard library / `fastapi`+`uvicorn`
respectively — their own `requirements.txt` lives in
`sdoc-hackathon-docker/server/`.
