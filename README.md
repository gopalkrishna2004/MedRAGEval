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

### Build the fact sheet

Questions are written from facts, not from raw passages, following RAGEval
(schema → config → questions → references → keypoints). RAGEval generates
its documents from a config of key facts, so the config is true by
construction. Here the documents are real, so the config is **extracted from
them and verified**:

1. **Schema** (once per document family). [configs/schema/nice_sta.yaml](configs/schema/nice_sta.yaml)
   lists the facts that matter in a NICE single technology appraisal: trial
   design, endpoint results, safety, indirect comparisons, model inputs,
   cost-effectiveness results, the company's and EAG's positions on key
   issues, and stakeholders' views. For each group it gives the document roles
   it is read from, section patterns and keywords, and a weight that sets its
   share of the dataset. Slots are disease-agnostic, and qualifiers (trial,
   endpoint, population, data cut, scenario, owner) tell two facts of a slot
   apart. For a new document family, `medrageval schema draft` asks an LLM for
   a first version to review.
2. **Extract** reads each group's passages in batches and records the facts
   they state: slot, qualifiers, value, unit id and a verbatim quote. A fact is
   kept only when its quote is a verbatim span of the unit, every number in its
   value is in the quote, and a text value shares words with its quote.
   Duplicates within a document are merged. `also_in` records the units of other
   documents that state the same fact.

```bash
cp .env.example .env            # then set OPENROUTER_API_KEY=...
uv run medrageval extract data/interim/ta1063 --dry-run    # batches and characters per group; no LLM calls
uv run medrageval extract data/interim/ta1063              # factsheet.jsonl, factsheet_rejects.jsonl, factsheet_report.json
uv run medrageval schema draft data/interim/ta1063 data/interim/document2 --out configs/schema/draft.yaml   # new family only
```

The fact sheet knows every fact of a slot. When two facts of a slot differ
(the DCO1 and DCO2 medians, the company's and the EAG's ICER), it works out
which terms a question must name to mean only one of them. A term contained in
a sibling's term does not count: TA1063 prints two tables captioned
"Scenario 1", and naming "Scenario 1" names both. A fact nothing tells apart
is never used.

### Generate the evaluation dataset

```bash
uv run medrageval generate data/interim/ta1063 --dry-run     # facts per group and candidates per type; no LLM calls
uv run medrageval generate data/interim/ta1063 --per-type 2  # smoke run: 2 items per type
uv run medrageval generate data/interim/ta1063               # full run: 150 items into data/datasets/ta1063/
```

Outputs:
- `dataset.jsonl`: one item per line, with the facts it was written from, its group, its evidence and `also_in`.
- `rejects.jsonl`: every rejected candidate, with the reason.
- `generate_report.json`: counts per type, group and reason, fact-sheet stats and LLM usage.

Responses are cached in `data/cache/llm/` by model, temperature and prompt, so
rerunning the same configuration makes no calls and writes an identical
dataset; `--no-cache` disables that.

Eight item types, each built from facts. Selectors draw across schema groups
in proportion to their weights, so the topic mix follows what matters rather
than page count.

| Type | Built from | Checks before the LLM judge |
|---|---|---|
| Factual | one text fact | key terms from the fact; the question names what sets the fact apart from its siblings |
| Numerical | one value of a number fact | the key is the fact's number with its unit (`78.1%` is accepted where the table prints `121 (78.1)`) |
| Comparative | two arms of one fact, or two facts of a slot in one document that differ in one qualifier | both values from the facts, a difference of at least 10% (no 3.25 vs 3.26), direction consistent, answer names the winner |
| Multi-hop | a schema link: fact A identifies what fact B is about, by its value ("the primary endpoint of CAPItello-291" → its median) or by describing it ("the comparator the company ruled out because of TA239" → its cost); B's passage must hold a rival the description rules out | the bridged thing is not named or spelled out; quotes from both; after the judge, neither excerpt alone answers it by its own words (an answer the model reaches from what it knows of the drug does not count) |
| Cross-document | two documents of different roles giving different values for one subject (the company's ICER against the EAG's) | quotes from both documents, not copies; neither excerpt alone answers it |
| Attribution | one party's side of a disputed point, or a stakeholder's view in its own document | the answer is the party, or the organisation by name; not named in the question; the claim not restated elsewhere |
| Summary | 3–6 facts of one group in one document (the EAG's key issues, a respondent's views) | one keypoint per fact, each stating its fact; scored on keypoints |
| Unanswerable | a redacted fact with no disclosed sibling, or a near miss on a fact (a data cut, population or scenario the bundle does not report) | the quote holds the marker; or BM25 hits plus an LLM check find no answer |

Every question is checked for layout talk: table or figure numbers, column
headers such as "(N=155)", rows and columns are rejected. Keys must be
specific: no bare small numbers, no "weak" or "plausible". An LLM judge then
reviews every answerable item, including whether a clinician, health
economist or committee member would plausibly ask it. A rejected item gets one
rewrite: the generator sees its item and why each check failed, and the
rewrite is checked again in full.

Document roles (company submission, EAG report, clarification, expert
statements...) are detected from headings and front matter with the patterns
under `generate.roles`.

Run the tests with `uv run pytest` (unit, synthetic) and `uv run pytest -m regression`
(the real PDFs, including hand-verified facts when a fact sheet exists).
