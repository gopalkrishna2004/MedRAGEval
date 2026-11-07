# MedRAGEval

> Build evaluation datasets from your medical documents, and find out whether your RAG actually works on them.

Give it a PDF and it writes questions with gold answers and evidence, then runs them through a RAG and scores the replies.

---

## Pipeline

```
your PDF
  ├── medrageval            — builds the dataset and scores the RAG
  │     ├── parse           — documents, sections, tables, OCR
  │     ├── extract         — facts, each verified against a verbatim quote
  │     ├── generate        — questions written from facts, then checked and judged
  │     └── eval / ablate   — runs the dataset through the RAG and scores it
  └── medrag                — the RAG under test
        ├── ingest          — Docling → chunks → embeddings → Postgres + pgvector
        ├── retrieve        — BM25 + dense → fusion → reranker
        └── answer          — a small LLM on an OpenAI-compatible server
```

---

## Question types

| Type | Tests |
|:-----|:------|
| `factual` | A stated fact |
| `numerical` | A value with its unit |
| `comparative` | Two arms or facts compared |
| `multihop` | One fact needed to identify another |
| `attribution` | Which party said something |
| `summary` | An overview, scored on keypoints |
| `unanswerable` | An answerable question with its evidence withheld |

Every question also has a messy version, worded the way a real user might type it. For example, from the TA1063 dataset:

| Version | Question |
|:--------|:---------|
| Clean | In the overall CAPItello-291 trial population, how many patients were randomised to placebo plus fulvestrant? |
| Messy | in oevrall capitello-291 trial how many pts got assigned to llacebo + fulv |

---

## Setup

**1. Install dependencies** (Python 3.13)

```bash
uv sync
```

Or with pip, then drop `uv run` from the commands below:

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

**2. Configure keys**

```bash
cp .env.example .env
```

```
OPENROUTER_API_KEY=      # required for extract, generate and the eval judge
VLLM_BASE_URL=           # if the reader is not at http://localhost:8003/v1
MEDRAG_DSN=              # if Postgres is not at postgresql://localhost:5433/medrag
```

**3. Start Postgres** (only for the RAG)

```bash
brew install postgresql@17 pgvector
brew services start postgresql@17
"$(brew --prefix postgresql@17)/bin/createdb" medrag
```

Homebrew uses port 5432, so set `MEDRAG_DSN=postgresql://localhost:5432/medrag`.

**4. Start a reader** (only for the RAG, on a GPU machine)

```bash
pip install vllm
vllm serve <model> --port 8003
```

---

## Run

**Build a dataset**

```bash
uv run medrageval parse path/to/document.pdf --out data/interim/my-doc
uv run medrageval extract data/interim/my-doc
uv run medrageval generate data/interim/my-doc    # writes data/datasets/my-doc
```

`extract` and `generate` make paid API calls. Add `--dry-run` to either one to see the plan first.

**Build the RAG**

```bash
uv run medrag ingest path/to/document.pdf

# Optional: a quick check before running eval
uv run medrag ask "your question" --pdf path/to/document.pdf
```

**Evaluate it**

Pass the same PDF with `--pdf`. Without it, a command uses the `pdf` set in [configs/rag.yaml](configs/rag.yaml), which is TA1063.

```bash
# Full run, writes data/runs/my-doc
uv run medrageval eval data/datasets/my-doc --pdf path/to/document.pdf

# The same questions, worded the way users type them
uv run medrageval eval data/datasets/my-doc --question messy --pdf path/to/document.pdf

# Compare retrieval modes
uv run medrageval ablate data/datasets/my-doc --out-root data/runs/my-doc-ablation --pdf path/to/document.pdf

# Browse a run in the browser, at http://127.0.0.1:8000
uv run medrag serve --run data/runs/my-doc --pdf path/to/document.pdf
```

---

## Evaluation metrics

| Metric | What it measures |
|:-------|:-----------------|
| Score | Answer quality from 0 to 1: the judge marks each answer correct (1), partial (0.5), or incorrect or declined (0). An unanswerable question scores 1 only if the RAG declines. Reported with a 95% bootstrap CI. |
| Hit rate | Share of answerable questions with at least one gold evidence quote in the passages shown to the reader |
| Evidence recall | Share of a question's gold evidence quotes found in those passages |
| MRR | How high the first passage holding gold evidence ranks |
| Key recall | Share of a question's answer keys found in the reply, checked by code rather than the judge |
| Faithfulness | Share of the reply's factual claims that the passages support |
| Keypoint completeness | Share of a summary question's keypoints the reply covers |
| Outcome category | Why an answer scored as it did: `correct`, `retrieval_miss`, `reader_error`, `false_abstention`, `correct_abstention` or `hallucinated_answer` |

Each run writes these to `eval_report.json`, overall and per question type.

---

## Go deeper

**[docs/WRITEUP.md](docs/WRITEUP.md)** explains how the dataset is built and the RAG is tested, with metrics and results across the tested settings.

---

## Stack

`pymupdf` · `rapidocr` · `docling` · `sentence-transformers` · `postgres` · `pgvector` · `vllm` · `openrouter`
