"""What the application reads from the PDF file itself: its hash and its bookmark outline.

A committee-papers bundle is many documents in one file. The PDF's top-level
bookmarks name them and give their first pages, so every chunk can say which
document it comes from without any content analysis. Without an outline the
caller falls back to top-level headings.
"""

import hashlib
import re
from pathlib import Path

_TAGS = re.compile(r"\[[^\]]*\]|\([^)]*\)")            # [CIC_REDACTED], (1)
_ENUM = re.compile(r"^\s*\d+[a-z]?\s*[.)-]\s*")         # "1a. ", "3) "
_ID = re.compile(r"\bID\s?\d{3,}\b", re.I)
_REDACTED = re.compile(r"[_\s-]*redacted\b", re.I)
_DATE_VERSION = re.compile(r"\b(v\d+(\.\d+)*|\d{1,2}[A-Za-z]{3}\d{2,4}|\d{1,2}\.\d{1,2}\.\d{2,4})\b")


def pdf_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clean_title(raw: str) -> str:
    """A bookmark title fit for a chunk header: byte-order marks (a journal's bookmarks carry one), enumerators, ids, tags,
    version and date stamps and underscores removed."""
    t = _TAGS.sub(" ", raw.replace("\ufeff", ""))
    t = _ENUM.sub("", t)
    t = _ID.sub(" ", t)
    t = _REDACTED.sub(" ", t)
    t = _DATE_VERSION.sub(" ", t)
    t = t.replace("_", " ")
    t = re.sub(r"\s+", " ", t).strip(" -.,")
    return t or raw.strip()


def outline_documents(path: str | Path) -> list[dict]:
    """Top-level bookmarks as documents: [{title, raw_title, page_start, page_end}], in page order. [] without an outline."""
    import pymupdf
    with pymupdf.open(str(path)) as doc:
        toc = doc.get_toc()
        n_pages = doc.page_count
    tops = [(page, title) for level, title, page in toc if level == 1 and page > 0]
    tops.sort()
    out = []
    for i, (page, title) in enumerate(tops):
        end = tops[i + 1][0] - 1 if i + 1 < len(tops) else n_pages
        if end < page:
            continue
        out.append({"title": clean_title(title), "raw_title": title, "page_start": page, "page_end": end})
    return out


def title_for_page(documents: list[dict], page: int) -> str | None:
    for d in documents:
        if d["page_start"] <= page <= d["page_end"]:
            return d["title"]
    return None
