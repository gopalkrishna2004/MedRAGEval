"""PDF -> DoclingDocument, cached as JSON.

Conversion of a 600-page bundle takes the best part of an hour on a laptop
CPU, so the result is cached under `cache_dir`, keyed on the PDF's hash, the
Docling version and the conversion settings. Anything downstream (chunking,
embedding) reads the JSON and never touches the PDF again.
"""

import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path

from medrag.config import load_config
from medrag.pdf import pdf_sha256


def docling_version() -> str:
    return version("docling")


def cache_paths(cfg: dict, pdf: str | Path) -> tuple[Path, Path]:
    d = cfg["docling"]
    tag = f"{pdf_sha256(pdf)[:16]}-docling{docling_version()}-{'ocr' if d.get('ocr', True) else 'noocr'}"
    base = Path(cfg["cache_dir"]) / "docling"
    return base / f"{tag}.json", base / f"{tag}.meta.json"


def _converter(d: dict):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    opts = PdfPipelineOptions(do_ocr=bool(d.get("ocr", True)), do_table_structure=bool(d.get("table_structure", True)))
    if opts.do_ocr:
        opts.ocr_options = RapidOcrOptions()
    opts.table_structure_options.do_cell_matching = bool(d.get("cell_matching", True))
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


def convert(cfg: dict, pdf: str | Path | None = None, force: bool = False, page_range: tuple[int, int] | None = None,
            log=print) -> Path:
    """Path of the DoclingDocument JSON for `pdf`, converting unless cached. A page range is never cached."""
    pdf = Path(pdf or cfg["pdf"])
    json_path, meta_path = cache_paths(cfg, pdf)
    if page_range:
        json_path = json_path.with_name(json_path.stem + f"-p{page_range[0]}-{page_range[1]}.json")
        meta_path = json_path.with_suffix(".meta.json")
    if json_path.exists() and not force and not page_range:
        return json_path
    log(f"converting {pdf} with Docling {docling_version()} (ocr={cfg['docling'].get('ocr', True)})...")
    t0 = time.time()
    conv = _converter(cfg["docling"])
    kwargs = {"page_range": page_range} if page_range else {}
    res = conv.convert(str(pdf), **kwargs)
    status = str(res.status)
    if "SUCCESS" not in status and "PARTIAL" not in status:
        raise RuntimeError(f"Docling conversion failed: {status}")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    res.document.save_as_json(json_path)
    meta = {"pdf": str(pdf), "sha256": pdf_sha256(pdf), "docling": docling_version(), "settings": cfg["docling"],
            "status": status, "pages": len(res.document.pages), "seconds": round(time.time() - t0, 1),
            "page_range": list(page_range) if page_range else None}
    meta_path.write_text(json.dumps(meta, indent=2))
    log(f"  {meta['pages']} pages in {meta['seconds']}s ({status}) -> {json_path}")
    return json_path


def load_document(json_path: str | Path):
    from docling_core.types.doc import DoclingDocument
    return DoclingDocument.load_from_json(json_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="convert a PDF with Docling into the cache")
    ap.add_argument("pdf", nargs="?")
    ap.add_argument("--config")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--pages", help="a-b: convert only these pages (not cached as the full document)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    pr = tuple(int(x) for x in a.pages.split("-")) if a.pages else None
    convert(cfg, a.pdf, a.force, pr)


if __name__ == "__main__":
    main()
