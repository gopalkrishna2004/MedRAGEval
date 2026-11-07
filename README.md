# MedRAGEval

Builds RAG evaluation datasets from medical PDF documents (`medrageval`), and a
RAG application to evaluate with them (`medrag`).

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

Questions are written from facts, not from raw passages (schema → config →
questions → references → keypoints). The documents are real, so the config of
key facts is **extracted from them and verified**:

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

TA1063's parse output and fact sheet are committed in `data/interim/ta1063/`, so
`generate` runs from a fresh clone without paying for `extract` again. The dataset
cites them: an item's `facts` are ids in `factsheet.jsonl`, and its evidence
`element_id`s are in `elements.jsonl`.

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
uv run medrageval generate data/interim/ta1063               # full run: 146 items targeted, 142 for TA1063, into data/datasets/ta1063/
```

Outputs:
- `dataset.jsonl`: one item per line, with the facts it was written from, its group, its evidence and `also_in`. An unanswerable item also carries `source_item_id`, `withheld` (the answer, keys and evidence taken away) and `distractors` (the passages it is read with), plus the screening verdict on every candidate passage.
- Every item also carries `question_messy`, the **second question setting**: the same question as a real user would type it. The generator rewrites the clean question in two to four user habits picked by the seed (informal, shorthand, lay terms, keyword query, statement, non-native grammar, rambling, vague, misspelt names, unfinished fragments), told to be rougher than feels natural, with a severity (light, medium, heavy) that also sets how much surface noise code then adds: typos, text-speak spellings, missing spaces, lower case, dropped punctuation. Numbers are never altered. Nothing checks the rewrite: the setting measures what sloppy wording costs, and a rewrite that drifts is part of that cost. The field holds `text` (what is asked), `rewrite` (before the surface noise), `severity`, `persona`, `habits` and `surface`; an unanswerable item copies its source item's twin. `--no-messy` or `messy.enabled: false` skips the pass; `generate_report.json` has the mix under `messy`. `uv run medrageval messy data/datasets/ta1063` runs the pass again on an existing dataset, replacing the twins and nothing else (one generator call per answerable item), for a changed `messy` config without regenerating the items.
- `rejects.jsonl`: every rejected candidate, with the reason.
- `generate_report.json`: counts per type, group and reason, fact-sheet stats and LLM usage.

Responses are cached in `data/cache/llm/` by model, temperature and prompt, so
rerunning the same configuration makes no calls and writes an identical
dataset; `--no-cache` disables that.

Seven item types, each built from facts. Selectors draw across schema groups
in proportion to their weights, so the topic mix follows what matters rather
than page count. The weights hold over the whole dataset: each type continues
the group rotation from what earlier types accepted, so a group one type
cannot supply (multi-hop links mostly reach cost-effectiveness results) is
made up by the others. One bridge fact carries at most
`multihop.max_per_bridge` items, so a bundle with few describable bridges
gives fewer multi-hop items rather than the same hop asked five ways. The
report's `group_mix` lists answerable items per group against the weights'
targets.

| Type | Built from | Checks before the LLM judge |
|---|---|---|
| Factual | one text fact that says more than its own qualifiers (a scenario whose definition is its name is skipped) | key terms from the fact, not already in the question in any word form; the question names what sets the fact apart from its siblings |
| Numerical | one value of a number fact, not a count below 5 | the key is the fact's number with its unit (`78.1%` is accepted where the table prints `121 (78.1)`) |
| Comparative | two arms of one fact, or two facts of a slot in one document that differ in one qualifier | both values from the facts, a difference of at least 10% (no 3.25 vs 3.26), direction consistent, answer names the winner |
| Multi-hop | a schema link: fact A identifies what fact B is about, by its value ("the primary endpoint of CAPItello-291" → its median) or by describing it ("the comparator the company ruled out because of TA239" → its cost); B's passage must hold a rival the description rules out | the bridged thing is not named or spelled out; quotes from both; after the judge, neither excerpt alone answers it by its own words (an answer the model reaches from what it knows of the drug does not count) |
| Attribution | one party's side of a disputed point, or a stakeholder's view in its own document | the answer is the party, or the organisation by name; not named in the question; the claim not restated elsewhere |
| Summary | 3–6 facts of one group in one section of one document, the section heading being the theme (the company's scenario analyses, the EAG's comparators critique) | one keypoint per fact, each stating its fact; scored on keypoints |
| Unanswerable | an accepted item of another type, asked again with its evidence withheld; the passages that stand in for it are the units BM25 ranks closest to the evidence and to the question, minus the evidence and its copies | the judge screens every candidate passage against the withheld answer in one call and only those that do not answer are kept; a passage holding every answer key is dropped regardless; at least `min_distractors` must remain |

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

## Part 2: the RAG application, and evaluating it with the dataset

The system under test is a separate application, the `medrag` package
([src/medrag/](src/medrag/), config [configs/rag.yaml](configs/rag.yaml)). It
shares nothing with the benchmark but the PDF: it parses the file itself, chunks
its own way and keeps its chunks in its own database. The benchmark scores it
from the outside, by page numbers and verbatim quotes.

### The application

- **Parsing** ([convert.py](src/medrag/convert.py)): [Docling](https://github.com/docling-project/docling)
  with its table-structure model and RapidOCR for the scanned pages (the patient
  and expert statements). The DoclingDocument is cached as JSON under
  `data/rag/cache/docling/`, keyed on the PDF hash. TA1063's 578 pages convert in
  about eight minutes on a laptop.
- **Document boundaries** ([pdf.py](src/medrag/pdf.py)): a committee-papers bundle
  is many documents in one file. The PDF's top-level bookmarks name them and give
  their first pages, so every chunk says which document it comes from without any
  content analysis ("Capivasertib for breast cancer - EAG Report ..."). No role
  classification: the printed title only.
- **Chunking** ([elements.py](src/medrag/elements.py), [chunk.py](src/medrag/chunk.py)):
  structure-aware. Docling's reading-order items become elements with a heading
  path; page furniture is dropped. Prose under one heading is packed to 400 tokens
  (never past 512, counted with the embedding model's tokenizer), a long paragraph
  is cut at sentence ends, and the last sentence of a chunk opens the next one.
  A table is one chunk, Markdown with its caption (taken from the "Table N" line
  printed above or below it) and the footnotes under it; a table that continues on
  the next page with the same header row is merged; past 1,500 tokens the rows are
  split with the caption and header repeated. Every chunk starts with
  "document title | heading path | pages", which is what the model cites.
- **Store** ([store.py](src/medrag/store.py)): Postgres with
  [pgvector](https://github.com/pgvector/pgvector). One `collections` row per
  (PDF, chunking settings, embedding model), so a chunk-size sweep ingests side by
  side; chunks with metadata in jsonb and a `vector(1024)` column under an HNSW
  cosine index. An in-memory store (`store.backend: memory`) serves tests and
  machines without Postgres.
- **Retrieval** ([lexical.py](src/medrag/lexical.py), [retrieve.py](src/medrag/retrieve.py),
  [rerank.py](src/medrag/rerank.py)): BM25 for exact names, trial ids and figures
  (in process: Postgres's `ts_rank` has no inverse document frequency, so a common
  word would outrank a rare trial id); dense vectors from `BAAI/bge-m3` for
  paraphrase, 30 candidates each, fused by reciprocal rank; then the
  `BAAI/bge-reranker-v2-m3` cross-encoder reorders the fused top 30 and the top 6
  are shown. Modes `bm25 | dense | hybrid`, reranking on or off. Four settings,
  off in `rag.yaml` so that file stays the baseline, answer the failure modes
  the first full run showed (`retrieval` section of the config): `rrf_weights`
  (dense counts for less in the fusion: every chunk is about the same drug, so
  dense similarity separates sections poorly), `document_hints` + `hint_boost`
  (a question naming a party, "the EAG report", raises that document's chunks),
  `neighbours` (each shown chunk brings its same-section neighbours, for
  overview questions whose evidence spans adjacent chunks) and `rewrite` (the
  answering model restates a question that describes a thing instead of naming
  it, using the first pass's passages; the restated queries' rankings are fused
  in). `medrageval eval --set retrieval.<key>=<value>` turns any of them on for a run.
- **Answering** ([generate.py](src/medrag/generate.py), [llm.py](src/medrag/llm.py)):
  the top 6 chunks, up to 12,000 chars, as numbered passages to the model on
  [vLLM](https://docs.vllm.ai)'s OpenAI-compatible server, temperature 0. One
  prompt for every question (a deployed RAG does not know the item type): answer
  from the passages, cite them, never guess a [REDACTED] value, and say "Not found
  in the documents." otherwise. Replies are cached by (model, temperature, messages).

```bash
brew install postgresql@17 pgvector && brew services start postgresql@17   # or any Postgres 16+ with pgvector
createdb medrag                                  # store.dsn in configs/rag.yaml (default port 5433 here)
uv run medrag db init                            # tables and indexes
uv run medrag ingest                             # Docling (cached) -> chunks -> bge-m3 embeddings -> Postgres; ~10 min
uv run medrag search "EAG base-case ICER versus everolimus" --mode hybrid
vllm serve Qwen/Qwen2.5-3B-Instruct --port 8003                                # on a GPU machine; VLLM_BASE_URL in .env points at it
uv run medrag ask "What was the company's base-case ICER for capivasertib plus fulvestrant versus everolimus plus exemestane?"
```

The embedding and reranking models run locally through sentence-transformers
(Apple MPS or CUDA when available). `embedding.provider: openai` points the
embedder at any `/v1/embeddings` server instead.

### The demo

A browser view of the application and of an evaluation run:

```bash
uv run medrag serve                              # http://127.0.0.1:8000; needs Postgres, and the reader for new questions
uv run medrag serve --run data/runs/<run>        # show another run (default: demo.run in configs/rag.yaml)
```

**Ask** answers a live question and shows the passages it read, with each
retriever's rank, beside the PDF page each one came from. A retrieval control
switches between hybrid with reranking, hybrid, dense and BM25. **Evaluation**
reads the run's `eval_report.json` and `responses.jsonl`: overall metrics, score
by question type, outcomes, and every question with its answer, reference, judge
claims and evidence pages. It makes no model calls.

If the reader is down, the header says so and sources still appear. A new question
fails within `demo.llm_timeout` seconds, and one asked before answers from the
cache, so run the `demo.examples` once before presenting.

### Evaluating it

```bash
uv run medrageval eval data/datasets/ta1063 --limit 10 --out data/runs/smoke              # a few items; judge on OpenRouter
uv run medrageval eval data/datasets/ta1063 --limit 10 --no-judge --mode bm25 --out data/runs/smoke-bm25   # keys only, no OpenRouter calls
uv run medrageval eval data/datasets/ta1063 --types unanswerable --out data/runs/unans    # one type only (the rest are cached anyway)
uv run medrageval ablate data/datasets/ta1063 --limit 10 --chunk-tokens 256,400,512 --out-root data/runs/ablation-smoke   # every mode x chunk size, summary table
uv run medrageval eval data/datasets/ta1063                                               # full run, into data/runs/ta1063
uv run medrageval eval data/datasets/ta1063 --question messy                              # the user-typed twins, into data/runs/ta1063-messy
```

Without `--out`, a run writes to `data/runs/<dataset>` and `ablate` to
`data/runs/ablation`, replacing what is there, so give a partial run its own directory.

`eval` builds the application from `configs/rag.yaml` (`--rag-config`) and asks
it each question through its Python interface. Options: `--mode`, `--top-k`,
`--no-rerank`, `--set retrieval.candidates=100` (any value of the application's
config, repeatable: a retrieval setting is then an arm of the ablation), `--model`
and `--base-url` (the answering server), `--judge-model` (an OpenRouter id; the
judge never runs on the answering server), `--no-judge`, `--limit N`. Answers
and verdicts are cached, so a rerun makes no calls and a judge change makes no
answering calls. `ablate` runs `bm25`, `dense`, `hybrid`
and `hybrid+rerank` (and, with `--chunk-tokens`, ingests one collection per
size) into `data/runs/ablation/<run>/` with a summary table in `ablation.json`.
Whether hybrid beats BM25 is a result of that table, not a precondition of
anything.

**Two question settings.** `--question clean` (default) asks each item's
question as generated; `--question messy` asks its `question_messy` twin, the
same question as a user would type it. Only what the application is sent
changes (`question_asked` on each row): the stand-in screening, the judge and
the keys work from the clean question, which is the intent the twin was written
from, so the two runs are paired item for item and their difference is what
sloppy wording costs. The default run directory gets a `-messy` suffix, the
report records `question_setting`, and an item without a twin is asked its
clean question and listed under `messy_missing`.

**Gold evidence is matched by page and text**, since the application's chunks
share no ids with the benchmark: a passage holds an evidence entry when its page
span overlaps the evidence pages and the normalised quote (letters and digits
only; a redaction placeholder of any length matches `[REDACTED]`) is inside the
passage text, whole or interrupted once by a running header the PDF parse left
at a page break. A passage that merely overlaps the pages is
counted as `page_only` and never credited; a quote found across two passages
adjacent in the document is credited and counted as `quote_split`. Both counts
are in the report under `evidence_matching`.

**Unanswerable items are read over stand-in passages**, not a search, and the
stand-ins come from the application's own chunks. The chunks holding the
withheld quotes are found (`gold_chunks` in `responses.jsonl`), the
application's retrieval is asked for the `standin_search_k` chunks nearest to
each, the gold chunks are removed, and the judge screens the rest against the
question and the withheld answer in one call (the generator's screening prompt;
the verdicts are stored under `screening`). The first `top_k` it finds do not
answer are the passages shown. Screening the exact text the reader sees means no
string rule has to decide what "holds the answer"; an earlier version matched
the stored distractors to chunks and dropped any holding an answer key, which
emptied the context of attribution items whose key was a party named everywhere
and let answers through in chunks wider than their unit. An item with no passage
left is run but not scored (`excluded`, listed in the report), since an empty
context makes a decline a free point; `--no-judge` excludes them all, as there
is nothing to screen with. `context` in `responses.jsonl` says which path an
item took.

Outputs in the run directory:
- `responses.jsonl`: per item, the reply, the passages shown (with BM25, dense, fused and rerank ranks), the candidate pool, and the scores below.
- `eval_report.json`: the application's config and collection, per-type and overall metrics with 95% bootstrap intervals, outcomes, agreement and evidence-matching counts.

| Metric | What it measures |
|---|---|
| `hit`, `all_evidence`, `evidence_recall`, `MRR` | Retrieval: was the item's gold evidence among the passages shown? Multi-hop and summary items need `all_evidence`. A copy of the fact in another document is not credited, so these are conservative. |
| `gold_prec` (`gold_prec@k`) | The share of the passages shown that hold gold evidence: how much of what reaches the model is irrelevant. Its ceiling is the evidence count over `top_k`, since an on-topic passage without the evidence counts as irrelevant. |
| `candidate_recall` (`cand@n`) | Evidence recall over the fused candidate pool (`candidates` chunks) before the reranker and the top-k cut. Low `candidate_recall` means the retrievers never found it; high `candidate_recall` with low `evidence_recall` means the ordering lost it. Not computed for fixed-context items. |
| `keys` | Deterministic: every answer key in the reply (numbers compared as numbers: "£52,681" = "52681", "0.50" = "0.5"), the party for attribution, a decline for unanswerable items. |
| `score` | The judge's verdict (correct 1, partial 0.5, incorrect or declined 0). An unanswerable item scores 1 only when the reply declines (an abstain pattern, or the judge's "abstained" verdict) and the judge finds no answer stated: a judge "correct" on a reply that states a supported answer does not count. Keys when run with `--no-judge`. |
| `grounded` | The judge: every claim of the reply is supported by the passages shown. |
| `faithfulness` | The judge lists the reply's factual claims, each supported or not by the passages shown; this is the supported share (0 to 1). None when the judge lists no claims (a bare decline); a decline that goes on to describe the passages is scored on what it says. |
| keypoints | Summaries: share of keypoints covered and contradicted (completeness and hallucination). |
| outcome | `correct`; `retrieval_miss` (wrong, evidence not shown); `reader_error` (wrong, evidence shown); `false_abstention` (declined, evidence shown); `correct_abstention` / `hallucinated_answer` for unanswerable items. |
| judge vs keys | Agreement on items both can decide, with the disagreeing ids: a check on the scorer, and a pointer to keys or items worth a look. |

## Tests

`uv run pytest` runs the fast tests on synthetic documents: the parser, fact sheet
and generator; the application's chunker, caption and continuation handling, BM25,
retrieval and fusion over the in-memory store, and the Postgres store (skipped
without a database; it uses its own schema); evidence matching, stand-in screening
and the evaluator end to end with a fake reader and judge.

`uv run pytest -m regression` runs end to end on the real PDFs in `data/raw`
(minutes on the first run), checking facts verified by hand, including the fact
sheet when one exists.
