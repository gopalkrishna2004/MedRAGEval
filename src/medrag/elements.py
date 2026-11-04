"""DoclingDocument -> a flat list of elements the chunker understands.

Docling gives reading-order items with labels (section_header, text,
list_item, footnote, caption, table, picture, page_header...) and page
provenance. This adapter keeps the heading path as it walks, drops page
furniture, renders tables as Markdown, and attaches to each table the loose
caption next to it and the footnotes right under it, so a table chunk carries
its own title and notes.

The chunker works on these elements only, so tests build them directly.
"""

import re
from dataclasses import dataclass, field

from medrag.pdf import title_for_page

SKIP_LABELS = {"page_header", "page_footer", "picture", "document_index", "form", "key_value_region", "code"}
_CAPTION_START = re.compile(r"^\s*\**\s*(table|figure|fig\.)\s*\d", re.I)


@dataclass
class Element:
    kind: str                            # heading | text | table
    text: str                            # heading text, paragraph text, or the table's Markdown (without caption)
    page_start: int
    page_end: int
    heading_path: list[str] = field(default_factory=list)
    document_title: str | None = None
    level: int | None = None             # headings only
    caption: str | None = None           # tables only
    notes: list[str] = field(default_factory=list)   # tables only: footnotes printed under the table
    label: str = ""                      # Docling label, for diagnostics


def _pages(item) -> tuple[int, int] | None:
    pages = [p.page_no for p in (getattr(item, "prov", None) or [])]
    return (min(pages), max(pages)) if pages else None


def compress_markdown_table(md: str) -> str:
    """Docling pads cells to align columns; the padding is tokens with no meaning. One space per gap, "---" separators."""
    out = []
    for line in md.split("\n"):
        line = re.sub(r" {2,}", " ", line.strip())
        if line and set(line.replace("|", "").strip()) <= set("-: "):
            line = "|" + "---|" * max(1, line.count("|") - 1)
        out.append(line)
    return "\n".join(out)


def table_markdown(item, doc) -> tuple[str, str]:
    """(caption, Markdown of the table alone). Docling puts the caption in front of the Markdown when it has one."""
    caption = " ".join((item.caption_text(doc) or "").split())
    md = item.export_to_markdown(doc).strip()
    if caption and md.startswith(caption):
        md = md[len(caption):].strip()
    return caption, compress_markdown_table(md)


def elements_from_docling(doc, documents: list[dict] | None = None) -> list[Element]:
    """Elements in reading order. `documents` is the PDF outline (medrag.pdf.outline_documents) for document titles."""
    documents = documents or []
    out: list[Element] = []
    stack: list[tuple[int, str]] = []    # (level, heading text)
    last_page = 1
    for item, _depth in doc.iterate_items():
        label = str(getattr(item, "label", "") or "")
        label = label.split(".")[-1].lower()
        pages = _pages(item)
        if pages:
            last_page = pages[0]
        p0, p1 = pages or (last_page, last_page)
        title = title_for_page(documents, p0)
        tname = type(item).__name__
        if label in SKIP_LABELS:
            continue
        if tname == "TableItem":
            caption, md = table_markdown(item, doc)
            if not md:
                continue
            out.append(Element("table", md, p0, p1, [h for _, h in stack], title, caption=caption or None, label=label))
            continue
        text = " ".join(str(getattr(item, "text", "") or "").split())
        if not text:
            continue
        if tname == "TitleItem" or label == "title":
            stack = [(0, text)]
            out.append(Element("heading", text, p0, p1, [text], title, level=0, label=label))
            continue
        if tname == "SectionHeaderItem" or label == "section_header":
            level = int(getattr(item, "level", 1) or 1)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
            out.append(Element("heading", text, p0, p1, [h for _, h in stack], title, level=level, label=label))
            continue
        if label == "checkbox_selected":
            text = "[x] " + text.lstrip("☒☑✓ ")
        elif label == "checkbox_unselected":
            text = "[ ] " + text.lstrip("□☐ ")
        out.append(Element("text", text, p0, p1, [h for _, h in stack], title, label=label))
    return attach_table_context(out)


def _caption_like(el: Element) -> bool:
    return el.kind == "text" and len(el.text) < 300 and (el.label == "caption" or bool(_CAPTION_START.match(el.text)))


def _norm(text: str) -> str:
    return " ".join(text.replace("*", "").split()).lower()


def attach_table_context(elements: list[Element], window: int = 5) -> list[Element]:
    """Tables get their context. A caption-less table takes the nearest unclaimed "Table N ..." line before it
    (captions are printed above tables), else the nearest after it, within `window` elements on its pages and
    with no other table in between. Footnotes right under a table become its notes, a loose duplicate of its
    caption is dropped, and a table continuing on the next page with the same header row is merged."""
    n = len(elements)
    used: set[int] = set()

    def same_page(j: int, el: Element) -> bool:
        return elements[j].page_start <= el.page_end and elements[j].page_end >= el.page_start

    def candidates(i: int, el: Element, step: int) -> list[int]:
        out = []
        j = i + step
        while 0 <= j < n and abs(j - i) <= window and same_page(j, el):
            if elements[j].kind == "table":
                break
            if j not in used and _caption_like(elements[j]):
                out.append(j)
            j += step
        return out

    tables = [i for i, el in enumerate(elements) if el.kind == "table"]
    for i in tables:                      # pass 1: the caption above
        el = elements[i]
        if not el.caption:
            before = candidates(i, el, -1)
            if before:
                el.caption = elements[before[0]].text.strip("* ")
                used.add(before[0])
    for i in tables:                      # pass 2: the caption below, when nothing above claimed one
        el = elements[i]
        if not el.caption:
            after = candidates(i, el, +1)
            if after:
                el.caption = elements[after[0]].text.strip("* ")
                used.add(after[0])
    for i in tables:                      # duplicates of the caption, and footnotes under the table
        el = elements[i]
        if el.caption:
            for j in candidates(i, el, -1) + candidates(i, el, +1):
                if _norm(elements[j].text) == _norm(el.caption):
                    used.add(j)
        k = i + 1
        while k < n and k not in used and elements[k].kind == "text" and elements[k].label == "footnote" and same_page(k, el):
            el.notes.append(elements[k].text)
            used.add(k)
            k += 1
    kept = [el for i, el in enumerate(elements) if i not in used]
    return merge_continuations(kept)


def merge_continuations(elements: list[Element]) -> list[Element]:
    out: list[Element] = []
    for el in elements:
        prev = out[-1] if out else None
        if (el.kind == "table" and prev is not None and prev.kind == "table" and not el.caption
                and el.page_start in (prev.page_end, prev.page_end + 1) and el.heading_path == prev.heading_path):
            head_prev, head_el = prev.text.split("\n", 1)[0], el.text.split("\n", 1)[0]
            if head_prev == head_el and head_prev.count("|") > 2:
                rows = el.text.split("\n")[2:]
                if rows:
                    prev.text = prev.text + "\n" + "\n".join(rows)
                prev.page_end = max(prev.page_end, el.page_end)
                prev.notes.extend(el.notes)
                continue
        out.append(el)
    return out
