"""Clean converted pages before structure detection.

- Running headers/footers: a box is dropped when the same text repeats on
  nearby pages and sits in the top/bottom margin on (almost) every repeat.
  Notes repeated in the page body survive, and so do headings of a form
  template that recur once per document. Dropped boxes are kept under "removed"
  because they help split a bundle into its documents, and the printed page
  label ("41 of 146") is read from them.
- Redactions: placeholder text ("****", "X,XXX", or anything under a black
  box) becomes one [REDACTED] marker. Footnote asterisks attached to a word
  or bracket ("trials***", "(Scenario 1)***") are left alone.
- Superscripts: reference numbers become "[6,57]" so they never merge into a
  preceding number ("2023<sup>6</sup>" would otherwise read "20236").
"""

import argparse
import difflib
import json
import re
from collections import defaultdict
from pathlib import Path

from medrageval.config import load_config

REDACTED = "[REDACTED]"
ALWAYS_HEADER_FOOTER = {"page-header", "page-footer"}
NEVER_HEADER_FOOTER = {"table", "picture"}

_TAG = re.compile(r"<[^>]+>")
_STYLE_TAG = re.compile(r"</?(?:mark|u)>")
_SUP = re.compile(r"<sup>(.*?)</sup>", re.S)
_EMPHASIS = re.compile(r"(\*\*|\*|_)(.+)\1", re.S)
_ORDINAL = re.compile(r"st|nd|rd|th")
_CITATION = re.compile(r"[\d\s,–-]*\d[\d\s,–-]*")

_BREAK = r"(?:\s|<br\s*/?>)+"
_STARS = r"\*+(?:[.,]\*+)*"
_XS = r"[xX]+(?:[.,][xX]+)*"
# [^\W_] is a letter or digit: "_" is Markdown italics, not part of a word.
# Asterisks attached to a word or closing bracket are footnote markers.
_STAR_SPAN = re.compile(rf"(?<![^\W_])(?<![*)\]]){_STARS}(?:{_BREAK}{_STARS})*(?![^\W_]|\*)")
_X_SPAN = re.compile(rf"(?<![^\W_]){_XS}(?:{_BREAK}{_XS})*(?![^\W_])")
_REDACTED_RUN = re.compile(rf"{re.escape(REDACTED)}(?:{_BREAK}{re.escape(REDACTED)})+")

_PAGE_OF = re.compile(r"(?:page\s+)?(\d+)\s+of\s+(\d+)", re.I)
_PAGE_NUM = re.compile(r"(?:page\s+)?(\d{1,4})", re.I)


# --- inline text --------------------------------------------------------------

def _unwrap_emphasis(s: str) -> str:
    while m := _EMPHASIS.fullmatch(s):
        s = m.group(2)
    return s


def _superscript(m: re.Match) -> str:
    content = _unwrap_emphasis(m.group(1).strip())
    if not content:
        return ""
    if _ORDINAL.fullmatch(content):
        return content
    if _CITATION.fullmatch(content):
        return "[" + re.sub(r"\s+", "", content) + "]"
    if not any(c.isalnum() for c in content):  # footnote symbols: * † ‡ §
        return content
    return " " + content


def _redact_spans(text: str, pattern: re.Pattern, char: str) -> str:
    # A span may include short runs ("** *") from a word-by-word redacted
    # passage, but must contain at least one run of 3+ placeholder characters.
    def repl(m: re.Match) -> str:
        runs = re.split(_BREAK, m.group(0))
        return REDACTED if max(r.lower().count(char) for r in runs) >= 3 else m.group(0)

    return pattern.sub(repl, text)


def _tidy(text: str) -> str:
    lines = [line.rstrip() for line in text.strip().splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines))


def clean_text(text: str, redacted_tokens: list[str] = ()) -> str:
    # Placeholders known to sit under a black box go first, while still intact.
    for token in redacted_tokens:
        text = re.sub(rf"(?<![^\W_])(?<!\*){re.escape(token)}(?![^\W_]|\*)", REDACTED, text, count=1)
    text = _SUP.sub(_superscript, text)
    text = _STYLE_TAG.sub("", text)
    text = _redact_spans(text, _STAR_SPAN, "*")
    text = _redact_spans(text, _X_SPAN, "x")
    text = _REDACTED_RUN.sub(REDACTED, text)
    return _tidy(text)


# --- running headers/footers ----------------------------------------------------

def _norm(text: str) -> str:
    t = re.sub(r"(?m)^[#>\s]+", "", _TAG.sub("", text))  # heading/quote marks vary per page
    t = re.sub(r"\d+", "#", t.lower())
    return "".join(c for c in t if c.isalnum() or c == "#")


def _key(text: str) -> str:
    # Order-insensitive, so reflowed or OCR'd copies of one footer
    # ("NICE National Institute for..." vs "National Institute for NICE...") match.
    return "".join(sorted(_norm(text)))


def _in_margin(box: dict, page: dict, margin: float) -> bool:
    _, y0, _, y1 = box["bbox"]
    return y1 <= margin * page["height"] or y0 >= (1 - margin) * page["height"]


def _runs(occurrences: list[tuple], max_gap: int):
    """Split page-ordered occurrences wherever consecutive ones are more than max_gap pages apart."""
    run = [occurrences[0]]
    for occ in occurrences[1:]:
        if occ[0] - run[-1][0] > max_gap:
            yield run
            run = []
        run.append(occ)
    yield run


def find_running_boxes(
    pages: list[dict], margin: float, min_pages: int, max_gap: int, min_in_margin: float, fuzzy_ratio: float
) -> set[tuple[int, int]]:
    """Return (page index, box index) of every running header/footer box."""
    occurrences = defaultdict(list)
    for pi, page in enumerate(pages):
        for bi, box in enumerate(page["boxes"]):
            if box["class"] not in NEVER_HEADER_FOOTER and (key := _key(box["text"])):
                occurrences[key].append((pi, bi, _in_margin(box, page, margin)))

    running, known = set(), []
    for occ in occurrences.values():
        for run in _runs(occ, max_gap):
            in_margin = [(pi, bi) for pi, bi, m in run if m]
            if len({pi for pi, _, _ in run}) >= min_pages and len(in_margin) >= min_in_margin * len(run):
                running.update(in_margin)
                pi, bi = in_margin[0]
                known.append((_norm(pages[pi]["boxes"][bi]["text"]), run[0][0] - max_gap, run[-1][0] + max_gap))

    for pi, page in enumerate(pages):
        for bi, box in enumerate(page["boxes"]):
            if box["class"] in ALWAYS_HEADER_FOOTER:
                running.add((pi, bi))
            elif (pi, bi) in running or box["class"] in NEVER_HEADER_FOOTER or not _in_margin(box, page, margin):
                continue
            # OCR misreads a header slightly differently on each page. Only match
            # near the header's own run: a document title elsewhere can read
            # almost the same as a footer that quotes it.
            elif len(norm := _norm(box["text"])) >= 10 and any(
                first <= pi <= last and difflib.SequenceMatcher(None, norm, k).ratio() >= fuzzy_ratio
                for k, first, last in known
            ):
                running.add((pi, bi))
    return running


def _page_label(removed_texts: list[str]) -> str | None:
    for t in removed_texts:
        if m := _PAGE_OF.search(t):
            return f"{m[1]} of {m[2]}"
    for t in removed_texts:
        if m := _PAGE_NUM.fullmatch(t.strip()):
            return m[1]
    return None


# --- pages ------------------------------------------------------------------------

def clean_pages(pages: list[dict], settings: dict) -> list[dict]:
    running = find_running_boxes(pages, **settings["header_footer"])
    cleaned = []
    for pi, page in enumerate(pages):
        boxes, removed = [], []
        for bi, box in enumerate(page["boxes"]):
            if (pi, bi) in running:
                removed.append({"class": box["class"], "bbox": box["bbox"], "text": _tidy(_TAG.sub("", box["text"]))})
                continue
            text = clean_text(box["text"], box["redacted_tokens"])
            if not text and box["redaction_boxes"]:
                text = REDACTED  # e.g. a figure blacked out entirely
            if not text and box["class"] != "picture":
                continue
            boxes.append({
                "class": box["class"],
                "bbox": box["bbox"],
                "text": text,
                "source": box["source"],
                "redactions": text.count(REDACTED),
            })
        cleaned.append({
            "page": page["page"],
            "page_label": _page_label([r["text"] for r in removed]),
            "width": page["width"],
            "height": page["height"],
            "boxes": boxes,
            "removed": removed,
        })
    return cleaned


def to_markdown(pages: list[dict]) -> str:
    """Readable dump for eyeballing the cleaned output."""
    parts = []
    for p in pages:
        label = f" (printed {p['page_label']})" if p["page_label"] else ""
        body = "\n\n".join(b["text"] for b in p["boxes"] if b["text"])
        parts.append(f"<!-- page {p['page']}{label} -->\n\n{body}")
    return "\n\n".join(parts) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean converted pages (JSONL in, JSONL out).")
    ap.add_argument("pages", type=Path, help="output of medrageval.parse.convert")
    ap.add_argument("out", type=Path)
    ap.add_argument("--md", type=Path, help="also write a readable Markdown dump")
    ap.add_argument("--config", type=Path)
    args = ap.parse_args()

    settings = load_config(args.config)["clean"]
    pages = [json.loads(line) for line in args.pages.open()]
    cleaned = clean_pages(pages, settings)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for p in cleaned:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    if args.md:
        args.md.write_text(to_markdown(cleaned))

    removed = sum(len(p["removed"]) for p in cleaned)
    redactions = sum(b["redactions"] for p in cleaned for b in p["boxes"])
    ocr_boxes = sum(b["source"] == "ocr" for p in cleaned for b in p["boxes"])
    labelled = sum(p["page_label"] is not None for p in cleaned)
    print(
        f"{len(cleaned)} pages: removed {removed} header/footer boxes, {redactions} redactions, "
        f"{ocr_boxes} OCR boxes, page labels on {labelled} pages -> {args.out}"
    )


if __name__ == "__main__":
    main()
