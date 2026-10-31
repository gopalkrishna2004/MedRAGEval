"""Repairs for HTML tables as the layout model emits them.

The converter writes a table row per printed line, so what a reader sees as
one cell often arrives as two rows, and a two-line column header arrives as
two header rows. Each repair below is a rule over the cell grid:

- a header split over rows ("PI3K/AKT pathway-altered" over "AIC | BIC")
  becomes one header row, each column headed by its group and its own name;
- a row label wrapped onto a line of its own ("Patients starting" /
  "subsequent medication | | |") is joined to its row;
- a label wrapped across rows that each carry values ("Capivasertib plus
  fulvestrant (PI3K/AKT" / "pop from CAPItello-291) | PFS | ...") is joined
  and spans both rows, and a blank label under a label with values ("| PFS
  | 9.5 | 6.3") is covered by the label above, as a rowspan;
- a note or legend split across cells ("PFS = progression-free survival; PH
  = | proportional hazards") becomes one cell across the row;
- a header row repeated in the body (a table continued on the next page)
  and a column that is empty throughout are dropped.

Spans are kept as rowspan/colspan attributes, so `text.to_plain` and
`text.table_rows` see a merged label on every row it covers.
"""

import re
from dataclasses import dataclass

_ROW = re.compile(r"<tr>(.*?)</tr>", re.S)
_CELL = re.compile(r"<(t[hd])([^>]*)>(.*?)</t[hd]>", re.S)
_SPAN = re.compile(r"\b(rowspan|colspan)=\"(\d+)\"")
_BR = re.compile(r"<br\s*/?>")
_TABLE = re.compile(r"<table>.*?</table>", re.S)
# a cell that holds a value rather than a label: a number (with unit, CI, footnote mark), a redaction, or a "not reported"
VALUE = re.compile(r"^(?:[£$€]?\s?[<>≤≥]?[-−]?\d[\d,]*(?:\.\d+)?\s?%?(?:\s?\([^()]*\))?[*†‡a-c]*|\[REDACTED\]|NR|NE|N/?A|n/?a|[-–—])$")
MAX_HEADER_ROWS = 3


@dataclass
class Cell:
    tag: str
    text: str
    rowspan: int = 1
    colspan: int = 1

    @property
    def flat(self) -> str:
        return " ".join(_BR.sub(" ", self.text).split())

    @property
    def is_value(self) -> bool:
        return bool(VALUE.match(self.flat))

    @property
    def empty(self) -> bool:
        return not self.flat


def _parse(table_html: str) -> list[list[Cell]]:
    """The table as a rectangular grid; a spanning cell is the same object at every position it covers."""
    grid: list[list[Cell]] = []
    pending: dict[int, tuple[Cell, int]] = {}
    for row in _ROW.findall(table_html):
        cells: list[Cell] = []

        def take_pending():
            while len(cells) in pending:
                cell, left = pending.pop(len(cells))
                if left > 1:
                    pending[len(cells)] = (cell, left - 1)
                cells.append(cell)

        for tag, attrs, content in _CELL.findall(row):
            take_pending()
            spans = {k: int(v) for k, v in _SPAN.findall(attrs)}
            cell = Cell(tag, content, max(spans.get("rowspan", 1), 1), max(spans.get("colspan", 1), 1))
            for _ in range(cell.colspan):
                if cell.rowspan > 1:
                    pending[len(cells)] = (cell, cell.rowspan - 1)
                cells.append(cell)
        take_pending()
        grid.append(cells)
    width = max((len(r) for r in grid), default=0)
    for r in grid:
        r += [Cell("td", "") for _ in range(width - len(r))]
    return grid


def _serialize(grid: list[list[Cell]]) -> str:
    """Rows of cells back to HTML; spans are recomputed from where each cell object sits."""
    out = []
    seen: set[int] = set()
    for i, row in enumerate(grid):
        parts = []
        for j, cell in enumerate(row):
            if id(cell) in seen:
                continue
            seen.add(id(cell))
            rows = 1
            while i + rows < len(grid) and j < len(grid[i + rows]) and grid[i + rows][j] is cell:
                rows += 1
            cols = 1
            while j + cols < len(row) and row[j + cols] is cell:
                cols += 1
            attrs = (f' rowspan="{rows}"' if rows > 1 else "") + (f' colspan="{cols}"' if cols > 1 else "")
            parts.append(f"<{cell.tag}{attrs}>{cell.text}</{cell.tag}>")
        out.append("<tr>" + "".join(parts) + "</tr>")
    return "<table>" + "".join(out) + "</table>"


def _header_rows(grid: list[list[Cell]]) -> int:
    """How many leading rows make up the header.

    The first row always; then rows without a value that are header-marked,
    that name the columns under a spanning group header, or that carry on
    wrapped header text ("ICER* (capivasertib plus" / "fulvestrant vs.
    alpelisib" / "plus fulvestrant)").
    """
    if not grid:
        return 0
    n = 1
    width = len(grid[0])
    named = [not grid[0][j].empty and grid[0][j].colspan == 1 for j in range(width)]
    grouped = any(c.colspan > 1 for c in grid[0])
    while n < min(len(grid), MAX_HEADER_ROWS):
        row, above = grid[n], grid[n - 1]
        if any(c.is_value for c in row if not c.empty):
            break
        wrapped = any(not c.empty and _continues(above[j].flat, c.flat) for j, c in enumerate(row))
        if not (all(c.tag == "th" for c in row) or (grouped and not all(named)) or wrapped):
            break
        n += 1
        named = [named[j] or not row[j].empty for j in range(width)]
        grouped = any(c.colspan > 1 for c in row)
    return n


def _merge_header(grid: list[list[Cell]], n: int) -> tuple[list[Cell], list[str]]:
    """One header row from n: each column is its group text(s) then its own name. Returns the row and the leaf texts."""
    width = len(grid[0])
    header, leaves = [], []
    for j in range(width):
        texts = []
        for i in range(n):
            t = grid[i][j].flat
            if t and t not in texts:
                texts.append(t)
        header.append(Cell("th", " ".join(texts)))
        leaves.append(grid[n - 1][j].flat)
    return header, leaves


def _closes_bracket(prev: str, label: str) -> bool:
    return prev.count("(") > prev.count(")") and ")" in label


def _continues(prev: str, label: str) -> bool:
    """A label fragment that belongs to the label above: starts lowercase, or closes a bracket the label above opened."""
    return bool(label) and bool(prev) and (label[0].islower() or _closes_bracket(prev, label))


def _fragment(above: str, below: str) -> bool:
    """`below` carries on a wrapped prose cell: the cell above runs to three or more words with no end punctuation."""
    return bool(below) and len(above.split()) >= 3 and not re.search(r"[.;:!?]$", above)


def _repair_body(grid: list[list[Cell]], header_rows: list[list[str]]) -> list[list[Cell]]:
    body: list[list[Cell]] = []
    for row in grid:
        texts = [c.flat for c in row]
        if texts in header_rows:
            continue   # the header printed again where the table continued on the next page
        label, rest = row[0], row[1:]
        prev = body[-1] if body else None
        origin = label.colspan == 1 and (prev is None or prev[0] is not label)
        if prev is not None and origin:
            above = prev[0]
            if above.colspan > 1:
                pass   # a note across the row does not take the row below
            elif all(c.empty for c in rest) and _continues(above.flat, label.flat) and any(not c.empty for c in prev[1:]):
                above.text += " " + label.text          # wrapped label on a line of its own
                continue
            elif any(not c.empty for c in rest) and not above.empty and (label.empty or _closes_bracket(above.flat, label.flat) or (
                    label.flat[0].islower() and all(_fragment(prev[j].flat, c.flat) for j, c in enumerate(row) if j and not c.empty))):
                # a label wrapped across rows that carry values: "Capivasertib plus fulvestrant (PI3K/AKT" / "pop from CAPItello-291) | PFS | ..."
                # A lowercase label alone is not enough ("trt + PFSFLAG (model 3)" is a label): the cells must be wrapped prose too.
                if not label.empty:
                    above.text += " " + label.text
                if all(_fragment(prev[j].flat, c.flat) for j, c in enumerate(row) if j and not c.empty):
                    for j, c in enumerate(row):          # every cell is the tail of the cell above: one wrapped row
                        if j and not c.empty:
                            prev[j].text += " " + c.text
                    continue
                body.append([above] + rest)              # separate rows under one label: the label spans them
                continue
        if (len(row) > 1 and not label.empty and label.colspan == 1 and not any(c.is_value for c in row if not c.empty)
                and any(c.flat[0].islower() for c in rest if not c.empty)):
            note = Cell(label.tag, " ".join(c.text for c in row if not c.empty), colspan=len(row))   # a note split across cells
            body.append([note] * len(row))
            continue
        body.append(row)
    return body


def _drop_empty_columns(header: list[Cell], leaves: list[str], body: list[list[Cell]]) -> tuple[list[Cell], list[list[Cell]]]:
    keep = [j for j in range(len(header)) if leaves[j] or any(j < len(r) and not r[j].empty for r in body)]
    if len(keep) == len(header) or not keep:
        return header, body
    return [header[j] for j in keep], [[r[j] for j in keep] for r in body]


def repair_table(table_html: str) -> str:
    grid = _parse(table_html)
    if len(grid) < 2 or len(grid[0]) < 2:
        return table_html
    n = _header_rows(grid)
    header, leaves = _merge_header(grid, n)
    header_texts = [[c.flat for c in grid[i]] for i in range(n)] + [[c.flat for c in header]]
    body = _repair_body(grid[n:], header_texts)
    header, body = _drop_empty_columns(header, leaves, body)
    return _serialize([header] + body)


def repair_tables(text: str) -> str:
    """Repair every HTML table in an element's text."""
    return _TABLE.sub(lambda m: repair_table(m.group(0)), text)
