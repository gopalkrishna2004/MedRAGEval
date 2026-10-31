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
_CELL = re.compile(r"<t[hd][^>]*>(.*?)</t[hd]>", re.S)
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


def _table_to_plain(m: re.Match) -> str:
    rows = []
    for row in _ROW.findall(m.group(0)):
        cells = [_BR.sub(" ", c).strip() for c in _CELL.findall(row)]
        rows.append(" | ".join(cells))
    return "\n".join(rows)


def table_rows(text: str) -> list[list[str]]:
    """Cells of the first HTML table in text, one list per row, cleaned like `to_plain` does."""
    m = _TABLE.search(text)
    if not m:
        return []
    return [[" ".join(to_plain(_BR.sub(" ", c)).split()) for c in _CELL.findall(row)] for row in _ROW.findall(m.group(0))]


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
