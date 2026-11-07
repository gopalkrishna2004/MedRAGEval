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

## 
## Scores: NICE TA1063

The repo includes a full run on [TA1063](https://www.nice.org.uk/guidance/ta1063/evidence/draft-guidance-consultation-committee-papers-pdf-15313389517), a 578-page bundle of 13 documents. It produced [142 questions](data/datasets/ta1063/dataset.jsonl), tested on a RAG using Qwen2.5-3B. The [runs](data/runs/) are committed.

| Run | Score (95% CI) | Hit rate |
|:----|:--------------:|:--------:|
| Hybrid retrieval + reranker | 0.542 (0.468–0.616) | 0.603 |
| Same, messy questions | 0.412 (0.345–0.482) | 0.569 |
| Boosted retrieval | 0.592 (0.525–0.669) | 0.707 |

To try the evaluation on this dataset, use the default `configs/rag.yaml` and pick a new `--out`, so the committed run isn't replaced:

```bash
uv run medrag ingest
uv run medrageval eval data/datasets/ta1063 --out data/runs/my-rerun
```


---

## Go Deeper

**[docs/WRITEUP.md](docs/WRITEUP.md)** covers the methodology, design decisions, dataset format, metrics and analysis of the results.

---

## Stack

`pymupdf` · `rapidocr` · `docling` · `sentence-transformers` · `postgres` · `pgvector` · `vllm` · `openrouter`
