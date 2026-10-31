"""Plain-text rendering of element text and the quote matching built on it.

Element text keeps HTML tables and Markdown because LLMs read structure well.
`to_plain` flattens it for keyword search and display; `normalize` goes
further, to tokens, so a quote copied by an LLM still matches when spacing,
punctuation, dash/quote styles or table cell separators differ. Decimal
points stay inside numbers, so "7.3" never matches "73".
"""

import html
import re
import unicodedata

_PICTURE = re.compile(r"<!--\s*(?:Start|End) of picture text\s*-->")
_ROW = re.compile(r"<tr>(.*?)</tr>", re.S)
_CELL = re.compile(r"<t[hd]([^>]*)>(.*?)</t[hd]>", re.S)
_SPAN = re.compile(r"\b(rowspan|colspan)=\"(\d+)\"")
_TABLE = re.compile(r"<table>.*?</table>", re.S)
_BR = re.compile(r"<br\s*/?>")
_KNOWN_TAG = re.compile(r"</?(?:table|tr|td|th|br|sup|sub|mark|u)\b[^>]*>")  # leaves "<insert deadline>"
_BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)
_ITALIC = re.compile(r"(?<![^\W_])_(.+?)_(?![^\W_])", re.S)
_LINE_MARKER = re.compile(r"(?m)^[ \t]*(?:#{1,6}|>|[-•])[ \t]+")

_CITATION = re.compile(r"\[\d[\d,\s–-]*\]")
_DASHES = str.maketrans({c: "-" for c in "‐‑‒–—―−"})
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})
_TOKEN = re.compile(r"\d+(?:\.\d+)?|[^\W\d_]+|[%<>≤≥=±+]")


def _grid(table_html: str) -> list[list[str]]:
    """Raw cell contents of an HTML table as a rectangular grid.

    A cell with rowspan is repeated in the rows it spans, so a merged group
    label ("CAPItello-291" over two arm rows) stays on every row it covers
    and a row quoted on its own still says which group it belongs to. A cell
    with colspan keeps its text in its first column and leaves the others
    empty, so later columns line up with the header.
    """
    rows: list[list[str]] = []
    pending: dict[int, tuple[str, int]] = {}   # column -> (text, rows still to fill)
    for row in _ROW.findall(table_html):
        cells: list[str] = []

        def take_pending():
            while len(cells) in pending:
                text, left = pending.pop(len(cells))
                if left > 1:
                    pending[len(cells)] = (text, left - 1)
                cells.append(text)

        for attrs, content in _CELL.findall(row):
            take_pending()
            spans = {k: int(v) for k, v in _SPAN.findall(attrs)}
            for i in range(max(spans.get("colspan", 1), 1)):
                text = content if i == 0 else ""
                if spans.get("rowspan", 1) > 1:
                    pending[len(cells)] = (text, spans["rowspan"] - 1)
                cells.append(text)
        take_pending()
        rows.append(cells)
    return rows


def _table_to_plain(m: re.Match) -> str:
    return "\n".join(" | ".join(_BR.sub(" ", c).strip() for c in row) for row in _grid(m.group(0)))


def table_rows(text: str) -> list[list[str]]:
    """Cells of the first HTML table in text, one list per row, cleaned like `to_plain` does."""
    m = _TABLE.search(text)
    if not m:
        return []
    return [[" ".join(to_plain(_BR.sub(" ", c)).split()) for c in row] for row in _grid(m.group(0))]


def to_plain(text: str) -> str:
    text = _PICTURE.sub("", text)
    text = _TABLE.sub(_table_to_plain, text)
    text = _BR.sub("\n", text)
    text = _KNOWN_TAG.sub("", text)
    text = _BOLD.sub(r"\1", text)
    text = _ITALIC.sub(r"\1", text)
    text = _LINE_MARKER.sub("", text)
    text = html.unescape(text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def normalize(text: str) -> str:
    """Token string used for matching: case, punctuation and spacing removed."""
    text = unicodedata.normalize("NFKC", to_plain(text)).translate(_DASHES).translate(_QUOTES).casefold()
    text = _CITATION.sub(" ", text)
    return " ".join(_TOKEN.findall(text))


def contains_quote(quote: str, text: str) -> bool:
    q = normalize(quote)
    return bool(q) and f" {q} " in f" {normalize(text)} "
