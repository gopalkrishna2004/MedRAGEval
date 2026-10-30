"""Convert a PDF into per-page layout boxes with pymupdf4llm.

Each page becomes a list of boxes typed by the layout model (text, table,
caption, section-header, page-footer, ...) with their Markdown/HTML text and
page coordinates. Facts that need the PDF itself are attached here so later
steps work on JSON alone: whether a box's text came from OCR, and black
redaction rectangles with any placeholder text drawn under them.

pymupdf4llm degrades silently when optional pieces are missing: without
pymupdf.layout it uses a legacy parser with no OCR, without RapidOCR it
switches to Tesseract, and with no OCR engine at all it skips OCR. Any of
these gives a worse dataset with no error, so check_environment() turns
each of them into a hard failure.
"""

import argparse
import hashlib
import json
import re
from importlib.metadata import version
from pathlib import Path

import pymupdf
import pymupdf4llm
from pymupdf4llm.ocr.detect_rapidocr import detect_rapidocr_backend

from medrageval.config import load_config

SUPPORTED_OCR_ENGINES = ("rapidocr",)
PARSER_PACKAGES = ("pymupdf", "pymupdf4llm", "pymupdf-layout", "rapidocr", "onnxruntime")

# Redactions are drawn as near-black filled rectangles. Redactors usually leave
# placeholder text underneath ("xx", "****", "X,XXX"), which is what we record.
_NEAR_BLACK = 0.15
_PLACEHOLDER = re.compile(r"[xX*]+(?:[.,][xX*]+)*")
_TAG = re.compile(r"<[^>]+>")


class ParserEnvironmentError(RuntimeError):
    pass


def check_environment(settings: dict) -> None:
    try:
        import pymupdf.layout  # noqa: F401
    except ImportError as e:
        raise ParserEnvironmentError(
            "pymupdf.layout is not importable, so pymupdf4llm would use its legacy "
            "parser without OCR. Run `uv sync`."
        ) from e

    ocr = settings["ocr"]
    if not ocr["enabled"]:
        return
    if ocr["engine"] not in SUPPORTED_OCR_ENGINES:
        raise ValueError(f"Unsupported OCR engine {ocr['engine']!r}; expected one of {SUPPORTED_OCR_ENGINES}")
    backend = detect_rapidocr_backend()
    if backend != "rapidocr":
        raise ParserEnvironmentError(
            f"RapidOCR is required but the detected backend is {backend!r}. Run `uv sync`."
        )


def parser_versions() -> dict[str, str]:
    """Versions that determine parser output; stored alongside parsed data for provenance."""
    return {pkg: version(pkg) for pkg in PARSER_PACKAGES}


def _black_rects(page: pymupdf.Page) -> list[pymupdf.Rect]:
    rects, seen = [], set()
    page_area = page.rect.get_area()
    for d in page.get_drawings():
        fill, r = d.get("fill"), d["rect"]
        if not fill or max(fill) > _NEAR_BLACK or (d.get("fill_opacity") or 1) < 0.9:
            continue
        if r.width < 8 or r.height < 5 or r.get_area() > 0.5 * page_area:
            continue
        key = tuple(round(v) for v in r)
        if key not in seen:
            seen.add(key)
            rects.append(r)
    return rects


def _visible_chars(text: str) -> int:
    return sum(c.isalnum() for c in _TAG.sub("", text))


def _page_record(page: pymupdf.Page, chunk: dict) -> dict:
    text = chunk["text"]
    boxes = []
    for b in chunk["page_boxes"]:
        rect = pymupdf.Rect(b["bbox"])
        start, end = b["pos"]
        box_text = text[start:end]
        native = page.get_text(clip=rect)
        if b["class"] in ("page-header", "page-footer") and not box_text.strip():
            # pymupdf4llm blanks these; keep the text so cleaning can read page labels.
            box_text = native
        boxes.append({
            "class": b["class"],
            "bbox": [round(v, 1) for v in rect],
            "text": box_text,
            # OCR only covers images, so a box whose text is missing from the
            # PDF's own text layer was produced by OCR.
            "source": "ocr" if _visible_chars(native) < 0.5 * _visible_chars(box_text) else "native",
            "redaction_boxes": 0,
            "redacted_tokens": [],
        })

    words = page.get_text("words")
    for r in _black_rects(page):
        overlaps = [(pymupdf.Rect(b["bbox"]) & r).get_area() for b in boxes]
        if not overlaps or max(overlaps) < 0.5 * r.get_area():
            continue
        box = boxes[overlaps.index(max(overlaps))]
        box["redaction_boxes"] += 1
        for w in words:
            wr = pymupdf.Rect(w[:4])
            if (wr & r).get_area() >= 0.5 * wr.get_area() and _PLACEHOLDER.fullmatch(w[4]):
                box["redacted_tokens"].append(w[4])

    return {
        "page": page.number + 1,
        "width": round(page.rect.width, 1),
        "height": round(page.rect.height, 1),
        "boxes": boxes,
    }


def convert_pdf(pdf_path: Path, settings: dict) -> list[dict]:
    """Return one record per page: {"page", "width", "height", "boxes": [...]}."""
    check_environment(settings)
    ocr = settings["ocr"]
    ocr_function = None
    if ocr["enabled"]:
        from pymupdf4llm.ocr import rapidocr_api

        ocr_function = rapidocr_api.exec_ocr

    # pymupdf4llm opens its own copy of the file and writes OCR text into it,
    # so the document opened below still holds only the native text layer.
    chunks = pymupdf4llm.to_markdown(
        str(pdf_path),
        page_chunks=True,
        table_output=settings["table_output"],
        use_ocr=ocr["enabled"],
        ocr_function=ocr_function,
        ocr_dpi=ocr["dpi"],
        # These only catch some running headers/footers; clean.py removes the rest.
        header=False,
        footer=False,
    )
    with pymupdf.open(pdf_path) as doc:
        return [_page_record(doc[c["metadata"]["page_number"] - 1], c) for c in chunks]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert a PDF to per-page layout boxes (JSONL).")
    ap.add_argument("pdf", type=Path, nargs="?")
    ap.add_argument("out", type=Path, nargs="?")
    ap.add_argument("--config", type=Path)
    ap.add_argument("--check", action="store_true", help="only verify the parser environment")
    args = ap.parse_args()

    settings = load_config(args.config)["parse"]
    if args.check:
        check_environment(settings)
        print("Parser environment OK:", json.dumps(parser_versions()))
        return
    if not (args.pdf and args.out):
        ap.error("pdf and out are required unless --check is given")

    records = convert_pdf(args.pdf, settings)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {"pdf": str(args.pdf), "sha256": _sha256(args.pdf), "settings": settings, "versions": parser_versions()}
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {len(records)} pages to {args.out}")


if __name__ == "__main__":
    main()
