"""Split a bundle of cleaned pages into its separate documents.

Committee papers, evidence packs and similar bundles concatenate several
documents into one PDF. Each page votes on whether it starts a new document:

- page_reset:    the printed page number restarts or its total changes
                 ("11 of 11" -> "1 of 12"), or the previous page was the last
                 one ("178 of 178") even if this page has no number. Missing
                 labels are inferred from neighbours, so an unnumbered cover
                 before "2" counts as 1.
- title_page:    the layout model found a title box on the page.
- footer_change: the running header/footer text differs from the previous page.
- outline:       a top-level PDF bookmark points here (when the PDF has any).

A page starts a document when its weighted votes reach the threshold. Only a
page reset is strong enough alone; the weaker signals must agree, so none of
them can split a document by itself. When most pages carry no page number,
page resets cannot fire, so a title page becomes decisive instead.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from medrageval.config import load_config
from medrageval.parse.clean import _norm

_LABEL = re.compile(r"(\d+)(?: of (\d+))?")


def _parse_label(label: str | None) -> tuple[int, int | None] | None:
    if label and (m := _LABEL.fullmatch(label)):
        return int(m[1]), int(m[2]) if m[2] else None
    return None


def infer_page_numbers(pages: list[dict]) -> list[tuple[int, int | None] | None]:
    nums = [_parse_label(p["page_label"]) for p in pages]
    # Backward first: an unnumbered page before "2" is the cover of a new
    # document, which a forward fill would wrongly number as previous + 1.
    for i in range(len(nums) - 2, -1, -1):
        if nums[i] is None and nums[i + 1] and nums[i + 1][0] > 1:
            nums[i] = (nums[i + 1][0] - 1, nums[i + 1][1])
    for i in range(1, len(nums)):
        prev = nums[i - 1]
        if nums[i] is None and prev and (prev[1] is None or prev[0] < prev[1]):
            nums[i] = (prev[0] + 1, prev[1])
    return nums


def _page_reset(prev, cur) -> bool:
    if prev is not None and prev[1] is not None and prev[0] == prev[1]:
        return True  # previous page was "N of N": that document is complete
    if cur is None:
        return False
    if prev is None:
        return cur[0] == 1
    return cur[0] <= prev[0] or (cur[1] is not None and prev[1] is not None and cur[1] != prev[1])


def _footer_signature(page: dict) -> set[str]:
    # Letters only: page numbers inside a footer must not make every page differ.
    sig = {re.sub(r"[^a-z]", "", _norm(r["text"])) for r in page["removed"]}
    return {s for s in sig if len(s) >= 5}


def boundary_votes(pages: list[dict], outline: list[dict], settings: dict) -> list[dict]:
    """Return, per page, the signals that fired."""
    nums = infer_page_numbers(pages)
    outline_starts = {o["page"] for o in outline if o["level"] == 1} if settings["use_outline"] else set()
    votes = []
    for i, page in enumerate(pages):
        signals = []
        if i and _page_reset(nums[i - 1], nums[i]):
            signals.append("page_reset")
        if any(b["class"] == "title" for b in page["boxes"]):
            signals.append("title_page")
        if i and (prev := _footer_signature(pages[i - 1])) and (cur := _footer_signature(page)) and not prev & cur:
            signals.append("footer_change")
        if page["page"] in outline_starts:
            signals.append("outline")
        votes.append({"page": page["page"], "signals": signals, "page_number": nums[i]})
    return votes


def _headings(pages: list[dict], limit: int = 4) -> list[str]:
    seen = []
    for page in pages[:2]:
        for b in page["boxes"]:
            if b["class"] in ("title", "section-header"):
                text = re.sub(r"[#*_]+", "", b["text"]).strip()
                if text and text not in seen:
                    seen.append(text)
    return seen[:limit]


def segment(pages: list[dict], outline: list[dict], settings: dict) -> list[dict]:
    weights, threshold = settings["weights"], settings["threshold"]
    votes = boundary_votes(pages, outline, settings)
    numbered = sum(v["page_number"] is not None for v in votes) / len(votes)
    if numbered < settings["min_numbered_pages"]:
        weights = {**weights, "title_page": threshold}
    starts = [0] + [
        i for i, v in enumerate(votes) if i and sum(weights[s] for s in v["signals"]) >= threshold
    ]

    docs = []
    for n, (a, b) in enumerate(zip(starts, starts[1:] + [len(pages)]), start=1):
        doc_pages = pages[a:b]
        running = Counter(r["text"] for p in doc_pages for r in p["removed"] if re.search(r"[A-Za-z]{5}", r["text"]))
        docs.append({
            "doc_id": f"d{n:02d}",
            "start_page": doc_pages[0]["page"],
            "end_page": doc_pages[-1]["page"],
            "n_pages": len(doc_pages),
            "signals": votes[a]["signals"],
            "headings": _headings(doc_pages),
            "running_text": running.most_common(1)[0][0] if running else None,
        })

    # Headings shared by most documents (e.g. "Single Technology Appraisal") say
    # nothing about which document this is.
    common = Counter(h for d in docs for h in set(d["headings"]))
    for d in docs:
        d["headings"] = [h for h in d["headings"] if common[h] <= len(docs) / 2]
    return docs


def score_against_outline(docs: list[dict], outline: list[dict]) -> dict:
    found = {d["start_page"] for d in docs}
    truth = {o["page"] for o in outline if o["level"] == 1}
    return {
        "precision": len(found & truth) / len(found),
        "recall": len(found & truth) / len(truth),
        "missed": sorted(truth - found),
        "extra": sorted(found - truth),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Split cleaned pages into documents.")
    ap.add_argument("pages", type=Path, help="output of medrageval.parse.clean")
    ap.add_argument("meta", type=Path, help="pages.meta.json written by medrageval.parse.convert")
    ap.add_argument("out", type=Path)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--no-outline", action="store_true", help="ignore PDF bookmarks")
    args = ap.parse_args()

    settings = load_config(args.config)["segment"]
    if args.no_outline:
        settings["use_outline"] = False
    pages = [json.loads(line) for line in args.pages.open()]
    outline = json.loads(args.meta.read_text()).get("outline", [])

    docs = segment(pages, outline, settings)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(docs, indent=2, ensure_ascii=False))

    for d in docs:
        label = d["headings"][0] if d["headings"] else d["running_text"] or ""
        print(f"{d['doc_id']} p{d['start_page']}-{d['end_page']} ({d['n_pages']}p) {d['signals']} {label[:60]!r}")
    if outline:
        print("vs. top-level bookmarks:", score_against_outline(docs, outline))


if __name__ == "__main__":
    main()
