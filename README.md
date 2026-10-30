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

Parse a PDF (each step reads the previous step's output):

```bash
D=data/interim/ta1063
# 1. PDF -> per-page layout boxes (+ pages.meta.json: hash, settings, versions, bookmarks)
uv run python -m medrageval.parse.convert data/raw/source_document.pdf $D/pages.jsonl
# 2. drop running headers/footers, normalise redactions and superscripts
uv run python -m medrageval.parse.clean $D/pages.jsonl $D/pages_clean.jsonl --md $D/pages_clean.md
# 3. split the bundle into its documents
uv run python -m medrageval.parse.segment $D/pages_clean.jsonl $D/pages.meta.json $D/documents.json
```

Run the tests with `uv run pytest`.
