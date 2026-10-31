# MedRAGEval

Builds RAG evaluation datasets from medical PDF documents.

## Setup

Requires Python 3.13. Dependency versions are pinned exactly, because parser
output (tables, OCR text) changes between versions.

With [uv](https://docs.astral.sh/uv/) (recommended, uses `uv.lock`):

```bash
uv sync
```

With pip:

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

Then verify the parser environment:

```bash
uv run python -m medrageval.parse.convert --check
```

The pipeline stops with an error, rather than producing degraded output, if the
PDF layout model or RapidOCR is missing. No system packages are needed: RapidOCR's
models ship inside the pip package, and Tesseract is not used.

## Configuration

All settings that affect the output live in [configs/default.yaml](configs/default.yaml).

## Usage

Parse a PDF into documents, sections and elements, and print a report:

```bash
uv run medrageval parse data/raw/source_document.pdf --out data/interim/ta1063
```

Conversion (a few minutes) is cached on the PDF hash, settings and parser versions;
the other steps rerun in about a second. `--force` re-converts. Outputs in the
directory: `elements.jsonl` (the element store), `sections.json`, `documents.json`,
`parse_report.json`, and `pages_clean.md` for reading the cleaned text.

Tests: `uv run pytest` (unit, fast) and `uv run pytest -m regression` (end to end
on the PDFs in `data/raw`, checking facts verified by hand).

The individual steps can also be run on their own (each reads the previous step's output):

```bash
D=data/interim/ta1063
# 1. PDF -> per-page layout boxes (+ pages.meta.json: hash, settings, versions, bookmarks)
uv run python -m medrageval.parse.convert data/raw/source_document.pdf $D/pages.jsonl
# 2. drop running headers/footers, normalise redactions and superscripts
uv run python -m medrageval.parse.clean $D/pages.jsonl $D/pages_clean.jsonl --md $D/pages_clean.md
# 3. split the bundle into its documents
uv run python -m medrageval.parse.segment $D/pages_clean.jsonl $D/pages.meta.json $D/documents.json
# 4. section tree per document + element list (paragraphs, tables, figures with captions)
uv run python -m medrageval.parse.structure $D/pages_clean.jsonl $D/documents.json $D
```

Step 4 also repairs the tables, which the converter writes one row per printed
line: a two-line column header becomes one, a row label wrapped onto its own
line or across its value rows is joined (and spans those rows as a rowspan),
a note split across cells becomes one cell, and repeated header rows and
empty columns are dropped. See `src/medrageval/parse/tables.py`.

### Generate the evaluation dataset

Question generation picks the evidence programmatically and asks an LLM, through
[OpenRouter](https://openrouter.ai), to write each item. Put a key in `.env`:

```bash
cp .env.example .env            # then set OPENROUTER_API_KEY=...
uv run medrageval generate data/interim/ta1063 --dry-run     # roles, noise and candidate counts; no LLM calls
uv run medrageval generate data/interim/ta1063 --per-type 2  # smoke run: 2 items per type
uv run medrageval generate data/interim/ta1063               # full run: 150 items into data/datasets/ta1063/
```

Outputs: `dataset.jsonl` (one item per line), `rejects.jsonl` (every rejected
candidate with the reason) and `generate_report.json` (counts per type and
reason, document roles, LLM usage). Responses are cached in `data/cache/llm/`
by model, temperature and prompt, so rerunning the same configuration makes no
calls and writes an identical dataset; `--no-cache` disables that. The model,
the per-type targets and every selection threshold live under `generate:` in
[configs/default.yaml](configs/default.yaml).

Seven item types, each with its own evidence selector and validity rules:

| Type | Evidence the system picks | Checks before the LLM judge |
|---|---|---|
| Factual | one complete paragraph | the quote is in the element and contains the answer keys |
| Numerical | a paragraph with a number and unit, or a captioned table | the answer's number is in the quote and among the evidence's numbers |
| Comparative | a table with two arm columns (first), or labelled rows of one column (the two rows with the closest differing values, within a merged group), or a paragraph with two values of one unit | both values quoted, direction consistent with the values, answer names the winner |
| Multi-hop | two parts of one document linked by "see Table N", or a sentence that describes a trial paired with a passage that gives its value | quotes from both, bridge not named or identified by its place in a list |
| Cross-document | a reviewing passage (EAG, clarification) paired with the company passage it refers to, shares numbers with, or best matches lexically | quotes from two documents, not copies of each other |
| Attribution | a stance sentence ("The EAG considers..."), an expert's form answer, or a company response; claims another party disputes come first, claims two parties share are skipped | answer is the speaker, speaker not named in the question, claim not restated unattributed elsewhere (BM25) |
| Unanswerable | a value shown as `[REDACTED]` that no other passage prints, or a plausible question the bundle does not cover | the quote carries the marker; or BM25 hits plus an LLM check find no answer, not even partly |

The generation prompt tells the model that the answerer will search the whole
bundle, so a question has to say what sets its fact apart (population, arm,
data cut, whose figure), and asks for the question in its own words with short
answer keys; the model may also decline evidence it cannot write a sound item
on, and the reason is kept in `rejects.jsonl`. Every selector skips OCR'd text
(its glyph errors would end up in questions and answer keys) and figures that
OCR turned into tables, and uses the summary for patients and background
sections only when nothing else is left. A table that has a twin anywhere in
the bundle (same rows and columns, as in per-arm AIC tables or a table the
clarification response reprints with new values) is used only when its
caption or its document tells it apart, and the question must then contain
those words ("fulvestrant arm", "company submission"); a table reprinted
unchanged in another document is recorded on the item as `also_in`.

After these checks an LLM judge reviews the comparative, multi-hop and
cross-document items (the types where a check needs reading, such as a table
that answers a multi-hop question alone); the other types are settled by the
programmatic checks. Rejected candidates are replaced from the selector's
queue, up to `attempts_per_item` per accepted item.

Document roles (company submission, EAG report, clarification, expert
statements...) are detected from headings and front matter with the patterns
under `generate.roles`; `--dry-run` prints them so a new bundle can be checked.

Run the tests with `uv run pytest`.
