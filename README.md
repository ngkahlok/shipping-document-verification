# Team LegacyOS - Shipping Document Verification

An automated pipeline that reads a shipping-logistics inbox, classifies each email, and cross-checks Shipping Instructions (SI) against draft Bills of Lading (BL) to catch discrepancies before documents go out.

Classification uses a hybrid rules + Gemini approach: a weighted keyword-signal scorer runs first as a fast, always-available baseline, then Gemini (when configured) makes the final call, using the rule output as a hint it can override. Each email is sorted into one of five categories, and comparison requests are checked across seven key fields (shipper, consignee, ports, container count, gross weight), flagging matches, mismatches, or cases needing human review.

Attachments in four formats (.txt, .pdf, .docx, .xlsx) are parsed and normalized so differently-labeled fields compare correctly. A Streamlit dashboard provides live scoring, a per-email inspector showing the pipeline's reasoning, and a human review queue for low-confidence or ambiguous cases.

Scores against the full 520-email set (see [Scoring](#scoring)):

| | Final score | Stage-1 macro-F1 | End-to-end |
|---|---|---|---|
| Rules + Gemini (combined) | **0.9935** | 0.978 | 1.000 |

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

3. **Set up the Gemini backend** — copy `.env.example` to `.env`
   and fill in `GEMINI_API_KEY` (see
   [How classification works](#how-classification-works-rules--gemini-combined)
   for the full config; `.env` is gitignored and auto-loaded by both the CLI
   and the app). For the Streamlit app specifically, `.streamlit/secrets.toml`
   works the same way and is what you'd use once deployed to Streamlit
   Community Cloud (see `.streamlit/secrets.toml.example`).

4. **Activate venv**
  
   ```bash
   source .venv/bin/activate
   ```

5. **Launch the Streamlit app:**

   ```bash
   streamlit run app.py
   ```

   Opens at `http://localhost:8501` — see [Streamlit app](#streamlit-app)
   above for what's on each page.

## Streamlit app

Three pages, selectable in the sidebar (which also holds the inbox/ground-truth
paths, a read-only Gemini status line, and an "apply human corrections to
report" switch):

- **📊 Dashboard** — final score and per-stage metrics as KPI cards (scored
  live against `ground_truth.json`), a **classification confidence** section
  (low/high-confidence counts, average margin, a per-category confidence
  breakdown and margin-distribution histogram — useful for tuning
  `SDOC_GEMINI_CONFIDENT_MARGIN`), category/status distributions, which of
  the 7 fields mismatches most often, `NEEDS_REVIEW` reason breakdown, a
  Stage-1 confusion-matrix heatmap, and a filterable table of every email.
  ![Dashboard](Dashboard.png)
- **🔍 Email Inspector** — pick any email and watch the pipeline reason
  through it: which classifier decided (rule signal fired, or Gemini's
  per-category scores + `reasoning`, with a visible warning if it had to
  fall back), then for `BL_COMPARISON` emails a step-by-step trace of the
  reliability checks, a side-by-side SI-vs-BL field diff table, the raw
  extracted `(label, value)` pairs per attachment, and a ground-truth
  comparison badge.
  ![Email Inspector](Email_inspection.png)
- **🧑‍⚖️ Review Queue** — see [Human review loop](#human-review-loop).
![Review Queue](Review_queue.png)

## Technical architecture

In plain terms, the system is a straight line an email travels down, with
one branch point:

```
Inbox email
   │
   ▼
Stage 1 — Classify (rules, or Gemini if unsure)
   │
   ├── Not a comparison request → done, category recorded
   │
   └── BL_COMPARISON → Stage 2 — Extract fields from SI + BL attachments
                            │
                            ▼
                        Stage 2b — Reliability check
                        (can we trust what we extracted?)
                            │
                            ▼
                        Stage 3 — Compare SI vs BL field-by-field
                            │
                            ▼
                        OK / MISMATCH / NEEDS_REVIEW
```

- **Classification** decides what kind of email it is.
- **Extraction** pulls structured data out of whatever file format the
  attachment happens to be (text, PDF, Word, Excel).
- **Comparison** checks whether the two documents agree.
- **The Streamlit app** (`app.py`) sits on top of all of it as a
  dashboard, so a person can see the results, drill into any single
  email, and correct the pipeline when it gets something wrong.

Everything in between is glue: a shared vocabulary (`fields.py`,
`normalize.py`) so "Port of Loading" and "Load Port" are recognized as the
same thing, a cache so repeated runs don't re-pay for Gemini calls, and a
review store so human corrections stick.

## Implementation details

A few of the more interesting decisions behind the code:

- **Rules first, AI second.** Every email is scored by a cheap, local
  keyword-matching classifier first. Gemini is only called when that
  local classifier isn't confident — which keeps the system fast and
  cheap for the easy cases, and accurate for the ambiguous ones.
- **PDFs are read as raw character streams, not laid-out text.** Normal
  PDF text extraction groups words by their position on the page, but in
  these documents some labels are long enough to visually run into the
  answer next to them. Reading the underlying character order (instead of
  page position) avoids that garbling.
- **Everything is normalized before comparing.** Values are cleaned up
  (extra address lines, port codes, weight units, formatting) so that two
  fields which *mean* the same thing still *look* the same when compared,
  regardless of which document or file format they came from.
- **A reliability gate runs before any comparison.** The pipeline checks
  for missing attachments, unreadable files, wrong document types, and
  blank values first — so a `MISMATCH` verdict always means "the data
  genuinely disagrees," not "we failed to read the data."
- **Every decision is traceable.** Each stage can explain itself (which
  rule fired, which fields didn't match, why something needs review) so
  the Streamlit inspector can show a human exactly why the pipeline
  concluded what it did, instead of just handing over a verdict.

## Challenges faced

- **The same field, said differently across documents.** The Shipping
  Instruction and Bill of Lading don't use the same labels for the same
  data (e.g. "Port of Loading" vs "Load Port"), so a simple text match
  wasn't enough — the pipeline needed a synonym vocabulary and value
  normalization to compare like with like.
- **PDF attachments were the hardest format.** Their fixed-position
  layout meant long labels could visually overlap the answer next to
  them, which standard text-extraction libraries misread as jumbled text.
  This required extracting at the character-stream level instead of
  relying on layout-based extraction.
- **Telling "no mismatch" apart from "couldn't read the document."** Early
  on, a missing attachment or an extraction failure could look
  indistinguishable from a genuine field mismatch. The reliability gate
  (Stage 2b) was added specifically to separate "the documents disagree"
  from "we don't have enough information to know."
- **Balancing cost/speed against accuracy.** Calling an AI model for
  every single email would be slow and expensive; relying only on simple
  rules would miss ambiguous phrasing. The two-tier
  rules-first-then-Gemini approach, with a confidence score and a local
  cache, was the answer.
- **Keeping humans in the loop without slowing things down.** Not every
  decision should be fully automatic. The review queue and override
  system let a person correct the pipeline's mistakes without needing to
  re-run anything, while the pipeline keeps working normally for
  everything else.

## Future roadmap

- **Expand beyond 7 fields** — support additional SI/BL fields as new
  document types or customer requirements come in.
- **Support more attachment formats** — e.g. scanned/image-based PDFs via
  OCR, which the current extractor doesn't handle.
- **Active learning from human corrections** — feed confirmed/corrected
  reviews back into the classifier so accuracy improves automatically
  over time, instead of only fixing the one email in front of you.
- **Batch and API access** — expose the pipeline as an API endpoint or
  batch job runner for integration into a real inbox/ops workflow, rather
  than a manual Streamlit session.
- **Multi-model support** — allow swapping Gemini for other LLM providers
  as a configuration choice, rather than a hardcoded dependency.
- **Alerting** — notify a human directly (email/Slack) when something
  lands in the review queue, instead of requiring someone to check the
  dashboard.
