"""Structure-aware chunking.

Prose under one heading is packed to `target_tokens`, never past `max_tokens`;
a paragraph longer than that is cut at sentence ends, then at spaces. The last
`overlap_sentences` of a chunk open the next chunk of the same section, so a
fact straddling a boundary is whole in one of them. A table is one chunk with
its caption and notes; past `max_table_tokens` its rows are split, each piece
repeating the caption and header row so values keep their column names.

Every chunk starts with a header line "document title | heading path | pages":
the same number printed by the company and by the assessment group is told
apart by it, and it is what the model cites.
"""

import re
from dataclasses import asdict, dataclass, field

from medrag.elements import Element

_SENT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9(\["\'“])')


@dataclass
class Chunk:
    chunk_id: str
    document_title: str | None
    heading_path: list[str]
    page_start: int
    page_end: int
    kind: str                     # text | table
    body: str
    text: str                     # header + body: what is embedded, indexed and shown to the model
    tokens: int
    part: int | None = None       # piece number when one element was split
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "Chunk":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__ if k in d})


class TokenCounter:
    """Token counts with the embedding model's tokenizer, or about four characters per token without one."""

    def __init__(self, model: str | None = None):
        self.tok = None
        if model:
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(model)

    def count(self, text: str) -> int:
        if self.tok is None:
            return max(1, (len(text) + 3) // 4)
        return len(self.tok.encode(text, add_special_tokens=False))


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENT.split(text.strip()) if s]


def split_long(text: str, limit: int, count) -> list[str]:
    """Pieces of at most `limit` tokens: by sentence, then a sentence longer than the limit by words."""
    pieces: list[str] = []
    cur: list[str] = []
    cur_n = 0
    for s in split_sentences(text):
        n = count(s)
        if n > limit:
            if cur:
                pieces.append(" ".join(cur))
                cur, cur_n = [], 0
            words = s.split()
            lo = 0
            while lo < len(words):
                hi = len(words)
                while hi > lo + 1 and count(" ".join(words[lo:hi])) > limit:
                    hi = lo + max(1, (hi - lo) * 3 // 4)
                pieces.append(" ".join(words[lo:hi]))
                lo = hi
            continue
        if cur and cur_n + n > limit:
            pieces.append(" ".join(cur))
            cur, cur_n = [], 0
        cur.append(s)
        cur_n += n
    if cur:
        pieces.append(" ".join(cur))
    return pieces or [text]


def header(title: str | None, heading_path: list[str], p0: int, p1: int, max_levels: int = 3) -> str:
    path = " > ".join(" ".join(h.split())[:80] for h in heading_path[-max_levels:]) or "-"
    pages = f"p. {p0}" if p0 == p1 else f"pp. {p0}-{p1}"
    return f"{title or 'Document'} | {path} | {pages}"


def table_lines(md: str) -> tuple[list[str], list[str]]:
    """(header lines, row lines) of a Markdown table: the first row and its separator, then the body rows."""
    lines = [ln for ln in md.split("\n") if ln.strip()]
    if len(lines) >= 2 and set(lines[1].replace("|", "").strip()) <= set("-: "):
        return lines[:2], lines[2:]
    return lines[:1], lines[1:]


def build_chunks(elements: list[Element], cfg: dict, counter: TokenCounter) -> list[Chunk]:
    """Chunks in document order. `cfg` is the `chunking` config section."""
    target, limit = cfg["target_tokens"], cfg["max_tokens"]
    max_table, overlap_n, min_body = cfg["max_table_tokens"], cfg.get("overlap_sentences", 1), cfg.get("min_body_chars", 15)
    chunks: list[Chunk] = []

    def emit(kind: str, title, path, p0, p1, body: str, part=None, meta=None) -> None:
        body = body.strip()
        if len(body) < min_body:
            return
        text = f"{header(title, path, p0, p1)}\n{body}"
        chunks.append(Chunk(f"c{len(chunks) + 1:05d}", title, list(path), p0, p1, kind, body, text, counter.count(text), part, meta or {}))

    # ----- prose: buffered per section -----
    buf: list[Element] = []

    def flush() -> None:
        if not buf:
            return
        title, path = buf[0].document_title, buf[0].heading_path
        paras: list[tuple[str, int, int]] = []
        for el in buf:
            for piece in split_long(el.text, limit, counter.count) if counter.count(el.text) > limit else [el.text]:
                paras.append((piece, el.page_start, el.page_end))
        cur: list[tuple[str, int, int]] = []
        cur_n = 0
        carry: str | None = None
        for text, p0, p1 in paras:
            n = counter.count(text)
            if cur and cur_n + n > target:
                emit("text", title, path, min(p for _, p, _ in cur), max(p for _, _, p in cur), "\n\n".join(t for t, _, _ in cur))
                tail = split_sentences(cur[-1][0])[-overlap_n:] if overlap_n else []
                carry = " ".join(tail) if tail and counter.count(" ".join(tail)) <= target // 4 else None
                cur, cur_n = [], 0
                if carry:
                    cur.append((carry, p0, p1))
                    cur_n = counter.count(carry)
            cur.append((text, p0, p1))
            cur_n += n
        if cur:
            emit("text", title, path, min(p for _, p, _ in cur), max(p for _, _, p in cur), "\n\n".join(t for t, _, _ in cur))
        buf.clear()

    def emit_table(el: Element) -> None:
        caption = f"{el.caption}\n" if el.caption else ""
        notes = ("\n" + "\n".join(el.notes)) if el.notes else ""
        body = f"{caption}{el.text}{notes}"
        meta = {"caption": el.caption} if el.caption else {}
        if counter.count(header(el.document_title, el.heading_path, el.page_start, el.page_end) + "\n" + body) <= max_table:
            emit("table", el.document_title, el.heading_path, el.page_start, el.page_end, body, meta=meta)
            return
        head, rows = table_lines(el.text)
        prefix = caption + "\n".join(head)
        base = counter.count(prefix) + 8
        groups: list[list[str]] = []
        cur, cur_n = [], base
        for r in rows:
            n = counter.count(r)
            if cur and cur_n + n > max_table:
                groups.append(cur)
                cur, cur_n = [], base
            cur.append(r)
            cur_n += n
        if cur:
            groups.append(cur)
        for i, g in enumerate(groups):
            tail = notes if i == len(groups) - 1 else ""
            emit("table", el.document_title, el.heading_path, el.page_start, el.page_end,
                 f"{prefix}\n" + "\n".join(g) + tail, part=i, meta={**meta, "parts": len(groups)})

    for el in elements:
        if el.kind == "heading":
            flush()
            continue
        if el.kind == "table":
            flush()
            emit_table(el)
            continue
        if len(el.text) < min_body:
            continue   # a stray page number, an empty cell
        if buf and (el.heading_path != buf[-1].heading_path or el.document_title != buf[-1].document_title):
            flush()
        buf.append(el)
    flush()
    return chunks
