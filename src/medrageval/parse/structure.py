"""Build each document's section tree and place every content box in it.

The converter marks every heading the same ("######"), so nesting is rebuilt
with a stack: a new heading closes every open section that cannot contain it.

- Numbered headings nest by prefix: "B.2" contains "B.2.6.1" but "B.3" does
  not. This needs no knowledge of the numbering scheme's absolute depth.
- Named headings ("Section A", "Part 2", "Appendix K", "Issue 3") contain
  numbered ones; named headings with the same keyword are siblings.
- Everything else nests by typography: a heading contains another only if its
  font is larger, or the same size but bold where the other is not.

Headings the layout model got wrong are fixed first: table/figure titles
become captions, and a heading split over two boxes is joined. Captions are
then attached to their table or figure, and a table continuing at the top of
the next page is merged into one element.
"""

import argparse
import json
import re
from pathlib import Path

from medrageval.parse.tables import repair_tables
from medrageval.text import to_plain

HEADING_CLASSES = {"title", "section-header"}
ELEMENT_TYPES = {"text": "paragraph", "list-item": "list_item", "footnote": "footnote",
                 "table": "table", "picture": "figure", "caption": "caption"}

_MARKUP = re.compile(r"[#*_]+")
_CAPTION = re.compile(r"(?i)^(table|figure|fig\.?|box|chart)\s*[A-Z]?\d")
_NAMED = re.compile(r"(?i)^(section|part|appendix|annex|chapter|issue|question)\s+([A-Z]|\d+)\b")
_NUMBER = re.compile(r"^((?:[A-Z]\.)?\d+(?:\.\d+)*[a-z]?|[A-Z]\d+(?:\.\d+)*)[.):]?(?:\s|$)")
_ROW = re.compile(r"<tr>(.*?)</tr>", re.S)
_CELL = re.compile(r"<t[hd](?: colspan=\"(\d+)\")?[^>]*>")


def heading_text(text: str) -> str:
    return re.sub(r"\s+", " ", _MARKUP.sub("", text)).strip()


def _number(title: str) -> tuple[str, ...] | None:
    if m := _NUMBER.match(title):
        parts = re.split(r"\.", m.group(1))
        if len(parts) == 1 and parts[0].isdigit() and int(parts[0]) > 99:
            return None  # a year ("2021 SLR update"), not a section number
        # "2a" -> ("2", "a") so that "2a)" sits under "2"
        return tuple(p for part in parts for p in re.fullmatch(r"(\d+)([a-z]?)", part).groups() if p) \
            if all(re.fullmatch(r"\d+[a-z]?", p) for p in parts) else tuple(parts)
    return None


def _style(h: dict) -> tuple[float, bool]:
    return (h["font_size"] or 0.0, h["bold"])


def _contains(top: dict, h: dict) -> bool:
    """Can the open section `top` contain the new heading `h`?"""
    if top["number"] and h["number"]:
        return len(top["number"]) < len(h["number"]) and h["number"][: len(top["number"])] == top["number"]
    if top["named"] and (h["number"] or (h["named"] and h["named"] != top["named"] and _style(top) > _style(h))):
        return True
    if top["named"] and h["named"]:
        return False
    ts, hs = _style(top), _style(h)
    return ts[0] > hs[0] or (ts[0] == hs[0] and ts[1] and not hs[1])


# --- heading fixes ----------------------------------------------------------------

def _is_heading(box: dict) -> bool:
    return box["class"] in HEADING_CLASSES and not _CAPTION.match(heading_text(box["text"]))


def _join_split_headings(boxes: list[dict]) -> list[dict]:
    if not boxes:
        return boxes
    column = max(b["bbox"][2] for b in boxes) - min(b["bbox"][0] for b in boxes)
    out = []
    for box in boxes:
        prev = out[-1] if out else None
        if (
            prev and _is_heading(prev) and _is_heading(box)
            and (prev["font_size"], prev["bold"]) == (box["font_size"], box["bold"])
            and box["bbox"][1] - prev["bbox"][3] < (box["font_size"] or 10)
            and not _number(heading_text(box["text"])) and not _NAMED.match(heading_text(box["text"]))
            and not heading_text(prev["text"]).endswith((".", ":", "?"))
            # A wrapped heading fills its line or continues mid-sentence; otherwise
            # these are two headings ("5.5.2 Results" then "Results of ...").
            and (prev["bbox"][2] - prev["bbox"][0] >= 0.75 * column or heading_text(box["text"])[:1].islower())
        ):
            out[-1] = {**prev, "text": f"{heading_text(prev['text'])} {heading_text(box['text'])}",
                       "bbox": [prev["bbox"][0], prev["bbox"][1], max(prev["bbox"][2], box["bbox"][2]), box["bbox"][3]]}
        else:
            out.append(box)
    return out


# --- tables -------------------------------------------------------------------------

def _columns(table_html: str) -> int:
    row = _ROW.search(table_html)
    return sum(int(span or 1) for span in _CELL.findall(row.group(1))) if row else 0


def _merge_tables(first: str, second: str) -> str:
    rows_a, rows_b = _ROW.findall(first), _ROW.findall(second)
    if rows_a and rows_b and rows_a[0] == rows_b[0]:  # header row repeated on the new page
        second = second.replace(f"<tr>{rows_b[0]}</tr>", "", 1)
    return first.removesuffix("</table>") + second.removeprefix("<table>")


# --- document -----------------------------------------------------------------------

def structure_document(doc: dict, pages: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return (sections, elements) for one document."""
    doc_id = doc["doc_id"]
    sections, elements, stack = [], [], []

    for page in pages:
        for box in _join_split_headings(page["boxes"]):
            if _is_heading(box):
                title = heading_text(box["text"])
                named = _NAMED.match(title)
                h = {
                    "section_id": f"{doc_id}-s{len(sections) + 1:03d}",
                    "doc_id": doc_id,
                    "title": title,
                    "number": _number(title),
                    "named": named.group(1).lower() if named else None,
                    "font_size": box["font_size"],
                    "bold": box["bold"],
                    "page": page["page"],
                }
                while stack and not _contains(stack[-1], h):
                    stack.pop()
                h["parent_id"] = stack[-1]["section_id"] if stack else None
                h["depth"] = len(stack) + 1
                stack.append(h)
                sections.append(h)
                continue

            etype = "caption" if box["class"] in HEADING_CLASSES else ELEMENT_TYPES.get(box["class"], "paragraph")
            elements.append({
                "type": etype,
                "text": heading_text(box["text"]) if box["class"] in HEADING_CLASSES else box["text"],
                "page_start": page["page"],
                "page_end": page["page"],
                "bbox": box["bbox"],
                "section_id": stack[-1]["section_id"] if stack else None,
                "section_path": [s["title"] for s in stack],
                "source": box["source"],
                "redactions": box["redactions"],
            })

    elements = _join_continued_tables(elements)
    for el in elements:
        if el["type"] == "table":
            el["text"] = repair_tables(el["text"])
    return sections, _attach_captions(elements)


def _join_continued_tables(elements: list[dict]) -> list[dict]:
    out = []
    for el in elements:
        # Last element on the previous page is a table with the same columns,
        # and this table opens the page with no caption: one table split by a page break.
        prev = out[-1] if out else None
        if (
            prev and el["type"] == "table" and prev["type"] == "table"
            and el["page_start"] == prev["page_end"] + 1
            and _columns(el["text"]) == _columns(prev["text"]) > 0
        ):
            prev["text"] = _merge_tables(prev["text"], el["text"])
            prev["page_end"] = el["page_end"]
            prev["redactions"] += el["redactions"]
        else:
            out.append(el)
    return out


def _attach_captions(elements: list[dict]) -> list[dict]:
    out, pending = [], None
    for el in elements:
        if el["type"] == "caption":
            if pending:
                out.append(pending)
            pending = el
            continue
        if el["type"] in ("table", "figure") and pending and pending["page_end"] == el["page_start"] and "caption" not in el:
            el["caption"] = pending["text"]
            pending = None
        elif pending:
            out.append(pending)
            pending = None
        out.append(el)
    if pending:
        out.append(pending)
    # Caption printed below its figure: no caption came first, so take the next one.
    for i, el in enumerate(out[:-1]):
        nxt = out[i + 1]
        if el["type"] in ("table", "figure") and "caption" not in el and nxt["type"] == "caption" \
                and nxt["page_start"] == el["page_end"] and _CAPTION.match(nxt["text"]):
            el["caption"] = nxt["text"]
            nxt["type"] = "attached_caption"
    return [el for el in out if el["type"] != "attached_caption"]


def build(pages: list[dict], docs: list[dict]) -> tuple[list[dict], list[dict]]:
    by_page = {p["page"]: p for p in pages}
    all_sections, all_elements = [], []
    for doc in docs:
        doc_pages = [by_page[n] for n in range(doc["start_page"], doc["end_page"] + 1)]
        sections, elements = structure_document(doc, doc_pages)
        for i, el in enumerate(elements, start=1):
            el["element_id"] = f"{doc['doc_id']}-e{i:04d}"
            el["doc_id"] = doc["doc_id"]
            # Flattened text for keyword search and quote matching; the caption is
            # part of what a reader sees as the table, so it is included.
            el["plain"] = to_plain(f"{el['caption']}\n{el['text']}" if el.get("caption") else el["text"])
        all_sections += sections
        all_elements += elements
    return all_sections, all_elements


def outline_text(sections: list[dict], doc_id: str, max_depth: int = 3) -> str:
    return "\n".join(
        f"{'  ' * (s['depth'] - 1)}{s['title'][:90]}  (p{s['page']})"
        for s in sections if s["doc_id"] == doc_id and s["depth"] <= max_depth
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build section trees and the element list.")
    ap.add_argument("pages", type=Path, help="output of medrageval.parse.clean")
    ap.add_argument("documents", type=Path, help="output of medrageval.parse.segment")
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args()

    pages = [json.loads(line) for line in args.pages.open()]
    docs = json.loads(args.documents.read_text())
    sections, elements = build(pages, docs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    keep = ("section_id", "doc_id", "parent_id", "depth", "title", "page")
    (args.out_dir / "sections.json").write_text(
        json.dumps([{k: s[k] for k in keep} for s in sections], indent=2, ensure_ascii=False))
    with open(args.out_dir / "elements.jsonl", "w") as f:
        for el in elements:
            f.write(json.dumps(el, ensure_ascii=False) + "\n")

    tables = [e for e in elements if e["type"] == "table"]
    print(f"{len(docs)} documents, {len(sections)} sections, {len(elements)} elements, "
          f"{len(tables)} tables ({sum(t['page_end'] > t['page_start'] for t in tables)} span pages, "
          f"{sum('caption' in t for t in tables)} captioned)")


if __name__ == "__main__":
    main()
