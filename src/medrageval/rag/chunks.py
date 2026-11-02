"""Chunks: the passages the RAG retrieves and the small model reads.

Built from the corpus that question generation used (`load_corpus`), so the
noise filter (reference lists, contents pages, form instructions, OCR'd
figures), the document roles and the row groups of long tables are shared
with it. Consecutive units of one section are packed up to `max_chars`; a
table or row group is a chunk of its own, because packed with prose or cut
mid-table its values lose their column headers. Every chunk starts with a
header naming its document, section and pages: "the company's ICER" and "the
EAG's ICER" are told apart by the document, which a bare table row never says.

Each chunk records the units and elements it holds, so retrieval can be
scored against an item's gold evidence.
"""

import re
from dataclasses import dataclass, field

from medrageval.generate.corpus import Corpus, Unit
from medrageval.text import to_plain

_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+(?=[A-Z(\[\"'0-9•-])")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    role: str
    section_path: list[str]
    page_start: int
    page_end: int
    unit_ids: list[str]
    element_ids: list[str]
    text: str                       # header + body: what is indexed and what the model reads
    part: int | None = None         # piece number when one unit was split over several chunks
    tokens: list[str] = field(default_factory=list, repr=False)

    def to_json(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "tokens"}


def split_text(text: str, limit: int) -> list[str]:
    """Pieces of at most `limit` chars, cut at line ends, then sentence ends, then spaces."""
    if len(text) <= limit:
        return [text]
    pieces: list[str] = []
    for line in text.split("\n"):
        parts = [line] if len(line) <= limit else _SENTENCE_END.split(line)
        for part in parts:
            while len(part) > limit:   # one sentence longer than the limit: cut at the last space before it
                cut = part.rfind(" ", 0, limit)
                cut = cut if cut > limit // 2 else limit
                pieces.append(part[:cut].strip())
                part = part[cut:].strip()
            pieces.append(part)
    out, cur = [], ""
    for p in pieces:
        if not p:
            continue
        if cur and len(cur) + 1 + len(p) > limit:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur}\n{p}" if cur else p
    return out + [cur] if cur else out


def header(corpus: Corpus, unit: Unit, page_start: int, page_end: int) -> str:
    d = corpus.docs[unit.doc_id]
    section = " > ".join(" ".join(t.split())[:80] for t in unit.section_path[-3:]) or "-"
    pages = f"p. {page_start}" if page_start == page_end else f"pp. {page_start}-{page_end}"
    return f"{d.label} | {section} | {pages}"


def _table_body(unit: Unit) -> str:
    """A whole table's plain text already starts with its caption; a row group's does not."""
    caption = to_plain(unit.caption or "").strip()
    if caption and unit.row_index is not None:
        return f"{caption}\n{unit.plain}"
    return unit.plain


def build_chunks(corpus: Corpus, cfg: dict) -> list[Chunk]:
    """Chunks in document order. `cfg` is the `rag` config section."""
    ch = cfg["chunking"]
    max_chars, max_table = ch["max_chars"], ch["max_table_chars"]
    chunks: list[Chunk] = []

    def emit(units: list[Unit], body: str, part: int | None = None) -> None:
        u0 = units[0]
        p0, p1 = min(u.page_start for u in units), max(u.page_end for u in units)
        chunks.append(Chunk(chunk_id=f"c{len(chunks) + 1:05d}", doc_id=u0.doc_id, role=corpus.docs[u0.doc_id].role,
                            section_path=u0.section_path, page_start=p0, page_end=p1,
                            unit_ids=[u.unit_id for u in units], element_ids=list(dict.fromkeys(u.element_id for u in units)),
                            text=f"{header(corpus, u0, p0, p1)}\n{body}", part=part))

    def emit_split(unit: Unit, body: str, limit: int) -> None:
        pieces = split_text(body, limit)
        if len(pieces) == 1:
            emit([unit], body)
            return
        # A table's caption and header row go on every piece, so its values keep their column names. Not a
        # long one: a form table's first "row" can be a 6,000-char answer (TA1063's expert statements).
        lines = body.split("\n")[:2 if unit.caption else 1] if unit.type == "table" else []
        head = "\n".join(x for x in lines if len(x) <= limit // 4)
        for i, piece in enumerate(pieces):
            emit([unit], piece if i == 0 or not head else f"{head}\n{piece}", part=i)

    pending: list[Unit] = []

    def flush() -> None:
        if pending:
            emit(list(pending), "\n\n".join(u.plain for u in pending))
            pending.clear()

    for uid in corpus.order:
        u = corpus.units[uid]
        if pending and (u.section_id != pending[-1].section_id or u.doc_id != pending[-1].doc_id):
            flush()
        if u.type == "table":
            flush()
            emit_split(u, _table_body(u), max_table)
            continue
        if len(u.plain) > max_chars:
            flush()
            emit_split(u, u.plain, max_chars)
            continue
        if pending and sum(len(p.plain) + 2 for p in pending) + len(u.plain) > max_chars:
            flush()
        pending.append(u)
    flush()
    return chunks
