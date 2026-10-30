"""Run the whole parse stage on one PDF and write a report.

convert -> clean -> segment -> structure, each writing its output into one
directory. Conversion is the only slow step (minutes), so it is skipped when
pages.meta.json shows the same PDF, parse settings and parser versions; the
other steps take seconds and always rerun.
"""

import json
from collections import Counter
from pathlib import Path

from medrageval.config import load_config
from medrageval.parse.clean import clean_pages, to_markdown
from medrageval.parse.convert import _sha256, convert_pdf, parser_versions, write_pages
from medrageval.parse.segment import score_against_outline, segment
from medrageval.parse.structure import build


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _conversion_is_current(pdf: Path, out_dir: Path, settings: dict) -> bool:
    meta_path = out_dir / "pages.meta.json"
    if not (meta_path.exists() and (out_dir / "pages.jsonl").exists()):
        return False
    meta = json.loads(meta_path.read_text())
    return (meta["sha256"], meta["settings"], meta["versions"]) == (_sha256(pdf), settings, parser_versions())


def parse_pdf(pdf: Path, out_dir: Path, config: dict | None = None, force: bool = False) -> dict:
    """Parse one PDF into out_dir and return the parse report."""
    config = config or load_config()
    out_dir.mkdir(parents=True, exist_ok=True)

    if force or not _conversion_is_current(pdf, out_dir, config["parse"]):
        write_pages(pdf, convert_pdf(pdf, config["parse"]), out_dir / "pages.jsonl", config["parse"])
    pages = [json.loads(line) for line in (out_dir / "pages.jsonl").open()]
    outline = json.loads((out_dir / "pages.meta.json").read_text()).get("outline", [])

    cleaned = clean_pages(pages, config["clean"])
    _write_jsonl(out_dir / "pages_clean.jsonl", cleaned)
    (out_dir / "pages_clean.md").write_text(to_markdown(cleaned))

    docs = segment(cleaned, outline, config["segment"])
    (out_dir / "documents.json").write_text(json.dumps(docs, indent=2, ensure_ascii=False))

    sections, elements = build(cleaned, docs)
    keep = ("section_id", "doc_id", "parent_id", "depth", "title", "page")
    (out_dir / "sections.json").write_text(
        json.dumps([{k: s[k] for k in keep} for s in sections], indent=2, ensure_ascii=False))
    _write_jsonl(out_dir / "elements.jsonl", elements)

    report = build_report(pdf, cleaned, docs, sections, elements, outline)
    (out_dir / "parse_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def build_report(pdf, pages, docs, sections, elements, outline) -> dict:
    per_doc = []
    for d in docs:
        els = [e for e in elements if e["doc_id"] == d["doc_id"]]
        types = Counter(e["type"] for e in els)
        per_doc.append({
            "doc_id": d["doc_id"],
            "pages": f"{d['start_page']}-{d['end_page']}",
            "label": (d["headings"] or [d["running_text"] or ""])[0][:80],
            "sections": sum(s["doc_id"] == d["doc_id"] for s in sections),
            "elements": len(els),
            "tables": types["table"],
            "figures": types["figure"],
            "ocr_elements": sum(e["source"] == "ocr" for e in els),
            "redactions": sum(e["redactions"] for e in els),
        })

    warnings = []
    pages_with_elements = {p for e in elements for p in range(e["page_start"], e["page_end"] + 1)}
    if empty := [p["page"] for p in pages if p["page"] not in pages_with_elements]:
        warnings.append(f"{len(empty)} pages have no content after cleaning: {empty[:20]}")
    for d in per_doc:
        if d["sections"] == 0 and d["elements"] > 20:
            warnings.append(f"{d['doc_id']} has {d['elements']} elements but no headings were found")
        if d["elements"] and d["ocr_elements"] / d["elements"] > 0.3:
            warnings.append(f"{d['doc_id']} is {d['ocr_elements']}/{d['elements']} OCR text; expect more noise")
    unlabelled = sum(p["page_label"] is None for p in pages)
    if unlabelled > 0.5 * len(pages):
        warnings.append(f"{unlabelled}/{len(pages)} pages have no printed page number; document splitting is weaker")

    report = {
        "pdf": str(pdf),
        "pages": len(pages),
        "documents": len(docs),
        "sections": len(sections),
        "elements": len(elements),
        "element_types": dict(Counter(e["type"] for e in elements)),
        "header_footer_boxes_removed": sum(len(p["removed"]) for p in pages),
        "pages_without_page_number": unlabelled,
        "redactions": sum(e["redactions"] for e in elements),
        "per_document": per_doc,
        "warnings": warnings,
    }
    if outline:
        report["segmentation_vs_bookmarks"] = score_against_outline(docs, outline)
    return report


def print_report(report: dict) -> None:
    print(f"{report['pdf']}: {report['pages']} pages -> {report['documents']} documents, "
          f"{report['sections']} sections, {report['elements']} elements, {report['redactions']} redactions")
    for d in report["per_document"]:
        print(f"  {d['doc_id']} p{d['pages']:>9} {d['sections']:4d} sec {d['elements']:5d} el "
              f"{d['tables']:4d} tab {d['ocr_elements']:3d} ocr {d['redactions']:5d} red  {d['label'][:50]}")
    if "segmentation_vs_bookmarks" in report:
        s = report["segmentation_vs_bookmarks"]
        print(f"  document split vs. bookmarks: precision {s['precision']:.2f}, recall {s['recall']:.2f}")
    for w in report["warnings"]:
        print(f"  WARNING: {w}")
