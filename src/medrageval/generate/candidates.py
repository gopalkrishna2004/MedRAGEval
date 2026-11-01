"""Evidence selection: one seeded selector per question type, no LLM.

The system, not the model, decides what each item is about. Every selector
yields Candidate objects lazily, in an order that is deterministic for the
seed and spread across documents and sections, so the orchestrator can stop
as soon as a type has enough accepted items. The selectors only use the
corpus indexes; the LLM later writes a question about the chosen evidence.
"""

import re
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from random import Random

from medrageval.generate.corpus import _RANGE_DASHES, REDACTED, Corpus, Unit, entity_key, mentions, significant_numbers, tokenize
from medrageval.text import normalize, table_rows

DESCRIPTIVE = re.compile(r"\b(randomi[sz]ed|phase|double[- ]blind|placebo[- ]controlled|enrolled|recruited|multicentre|"
                         r"multicenter|open[- ]label|primary endpoint|primary outcome|was an?\b|is an?\b|were randomi)", re.I)
TICKBOX = re.compile(r"[☐☒☑✓✔]|\[ ?x? ?\]|\byes/no\b", re.I)
REDACTED_LABEL = [
    re.compile(r"([A-Za-z][^|\n:]{3,80}?)\s*[:|]\s*\[REDACTED\]"),
    re.compile(r"(£|\$|€)\s?\[REDACTED\]"),
    re.compile(r"\[REDACTED\]\s*(%|patients|participants|months|years|weeks|QALYs?|per cent|percent)\b", re.I),
]
# A word mixing ASCII letters with two or more non-ASCII ones ("yéárs") is an OCR glyph error. One accent ("naïve") is a word.
GARBLED = re.compile(r"\b(?=[^\W\d_]*[A-Za-z])(?=(?:[^\W\d_]*[^\x00-\x7F\W\d_]){2})[^\W\d_]+\b")
SENTENCE = re.compile(r"(?<=[.;!?])\s+(?=[A-Z(\[\"'])")


def garbled(text: str) -> bool:
    return bool(GARBLED.search(text))


@dataclass
class Candidate:
    type: str
    subtype: str | None
    units: list[Unit]
    hint: dict = field(default_factory=dict)

    @property
    def candidate_id(self) -> str:
        return f"{self.type}:{'+'.join(u.unit_id for u in self.units)}"


def spread(items: list, key: Callable, rng: Random, cap: int | None = None) -> Iterator:
    """Seeded shuffle, then round-robin over key buckets (at most cap per bucket)."""
    items = list(items)
    rng.shuffle(items)
    buckets: dict = defaultdict(list)
    for it in items:
        if cap is None or len(buckets[key(it)]) < cap:
            buckets[key(it)].append(it)
    queues = list(buckets.values())
    while queues:
        nxt = []
        for q in queues:
            yield q.pop(0)
            if q:
                nxt.append(q)
        queues = nxt


def interleave(queues: list[list]) -> Iterator:
    """Round-robin over several lists until all are empty."""
    queues = [list(q) for q in queues if q]
    while queues:
        nxt = []
        for q in queues:
            yield q.pop(0)
            if q:
                nxt.append(q)
        queues = nxt


def section_key(u: Unit) -> tuple:
    return (u.doc_id, u.section_id)


def demoted(corpus: Corpus, u: Unit, cfg: dict) -> bool:
    """Evidence to use only when nothing better is left: lay summaries and background sections (`demote` in the config)."""
    dm = cfg.get("demote", {})
    if corpus.docs[u.doc_id].role in dm.get("roles", []):
        return True
    title = " | ".join(u.section_path)
    return any(re.search(pat, title, re.I) for pat in dm.get("section_patterns", []))


def ordered(corpus: Corpus, items: list, key: Callable, rng: Random, cap: int | None, cfg: dict, unit: Callable = lambda x: x) -> Iterator:
    """`spread` over the items that are not demoted, then over the demoted ones."""
    main = [it for it in items if not demoted(corpus, unit(it), cfg)]
    rest = [it for it in items if demoted(corpus, unit(it), cfg)]
    yield from spread(main, key, rng, cap)
    yield from spread(rest, key, rng, cap)


def native(u: Unit, cfg: dict) -> bool:
    return cfg["limits"]["allow_ocr"] or u.source != "ocr"


def clean_prose(u: Unit, cfg: dict, allow_redacted: bool = False) -> bool:
    """A complete paragraph of evidence: right length, starts and ends like a sentence, no redactions, not OCR."""
    lim = cfg["limits"]
    p = u.plain
    return (u.evidence and u.type in ("paragraph", "list_item", "footnote") and native(u, cfg)
            and lim["min_chars"] <= len(p) <= lim["max_chars"]
            and bool(re.match(r"[A-Z0-9\"'(\[]", p)) and bool(re.search(r"[.!?)\"'%\]]$", p))
            and (allow_redacted or u.redactions == 0))


def whole_table(u: Unit, cfg: dict, allow_redacted: bool = False) -> bool:
    return (u.evidence and u.type == "table" and u.row_index is None and native(u, cfg)
            and len(u.plain) <= cfg["limits"]["max_table_chars"] and (allow_redacted or u.redactions == 0))


def twin_hint(corpus: Corpus, u: Unit) -> dict | None:
    """Hint fields for a table: {"must_mention": [...]} when something tells it from its twins, None when nothing does.

    `also_in` lists copies of the table in other documents: the same cells, so
    a retriever that finds the copy has found the evidence. Only the first
    copy is used (None for the others): two items on one table's data test
    the same thing twice.
    """
    if any(corpus.position[c] < corpus.position[u.unit_id] for c in corpus.copies.get(u.unit_id, [])):
        return None
    phrases = corpus.discriminator(u)
    if phrases is not None and any(not p for p in phrases):
        return None
    hint = {"must_mention": phrases} if phrases else {}
    if u.unit_id in corpus.copies:
        hint["also_in"] = corpus.copies[u.unit_id]
    return hint


def bridge_sentence(plain: str, key: str, corpus: Corpus) -> str | None:
    """The sentence that characterises an entity: names it, says what it is or how it was run, and lists at most one other entity.

    A sentence that merely lists trials ("data from SOLAR-1 or BOLERO-2")
    leaves the question nothing to describe the bridge by except its place
    in the list.
    """
    for sent in SENTENCE.split(plain):
        folded = entity_key(sent)
        if key in folded and DESCRIPTIVE.search(sent):
            others = sum(1 for k in corpus.entities if k != key and k in folded)
            if others <= 1:
                return sent.strip()
    return None


def shown_with(u: Unit, key: str) -> list[str]:
    """What unit B itself says about the entity: its row's other labelled cells, its column header, or its sentence.

    A question that describes the bridge by any of these is answered by B
    alone: Table 15's FAKTION row says "UK" and "ET, no CDK", so "the UK trial
    without prior CDK4/6 inhibitors" finds the row without the other excerpt.
    """
    if u.type != "table":
        return [s.strip()[:300] for s in SENTENCE.split(u.plain) if key in entity_key(s)]
    rows = table_rows(u.text)
    header = rows[0] if rows else []
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            if key not in entity_key(cell):
                continue
            if i == 0:     # the entity heads a column
                return [cell]
            if j == 0:     # the entity labels a row
                return [(f"{h}: {c}" if h else c)[:150] for h, c in zip(header[1:], row[1:]) if re.search(r"[A-Za-z]", c) and REDACTED not in c]
            return [x for x in (header[j] if j < len(header) else "", row[0]) if x]   # named in the body ("Source" row)
    return []


def alternatives(corpus: Corpus, u: Unit, key: str) -> set[str]:
    """Other trials unit B gives values for in the bridge's place: what a reader holding only B would have to choose between.

    With none, the hop is no hop: TA1063's Table 4.6 cites CAPItello-291 on
    every row, so "the trial whose prior CDK4/6i use was 72.9%" points at
    the only row there is. A name inside the bridge's own cell or brackets
    ("CAPItello-291 (NCT04305496)") is the same trial, not an alternative.
    """
    if u.type != "table":
        return {k for k in corpus.entities if k != key and k in entity_key(re.sub(r"\([^)]*\)", "", u.plain))}
    rows = table_rows(u.text)
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            if key in entity_key(cell):
                line = row if i == 0 else [r[j] for r in rows if j < len(r)]   # the other column heads, or the other cells of its column
                return {k for c in line if c != cell for k in corpus.entities if k != key and k in entity_key(c)}
    return set()


def new_words(bridge: str, name: str, shown: list[str]) -> set[str]:
    """Content words of the bridge sentence that neither the entity's name nor what B shows contains."""
    return set(tokenize(bridge)) - set(tokenize(name)) - set(tokenize(" ".join(shown)))


def numeric_hits(plain: str, cfg: dict) -> list[tuple[str, str]]:
    """(number, unit) pairs: a number, or a range ("40-50"), followed by a unit pattern or preceded by a currency sign."""
    out = []
    for m in re.finditer(r"\d[\d,]*(?:\.\d+)?(?:\s?[-–‒—]\s?\d[\d,]*(?:\.\d+)?)?", plain):
        number = re.sub(r"[\s,]", "", m.group(0)).translate(_RANGE_DASHES)
        before = plain[max(0, m.start() - 2):m.start()].strip()
        tail = plain[m.end():m.end() + 14]
        if before in ("£", "$", "€"):
            out.append((number, before))
            continue
        for pat in cfg["numeric_patterns"]:
            if re.match(r"\s?" + pat, tail, re.I):
                out.append((number, re.match(r"\s?" + pat, tail, re.I).group(0).strip()))
                break
    return out


def speaker_for(alias_text: str, cfg: dict) -> str | None:
    text = alias_text.lower().strip()
    for label, aliases in cfg["attribution"]["speakers"].items():
        if text == label.lower() or text in (a.lower() for a in aliases) or text.rstrip("s") in (a.lower().rstrip("s") for a in aliases):
            return label
    return None


def agreement_patterns(cfg: dict) -> list[re.Pattern]:
    """A second party that shares the claim: "agrees with the company that", "which the EAG accepts"."""
    aliases = sorted({a for al in cfg["attribution"]["speakers"].values() for a in al} | set(cfg["attribution"]["speakers"]), key=len, reverse=True)
    names = "|".join(re.escape(a) for a in aliases)
    return [re.compile(rf"\b(?:agree[sd]?|concur(?:s|red)?|in line)\s+with\s+(?:the\s+)?({names})\b", re.I),
            re.compile(rf"\b({names})\s+(?:also\s+)?(?:accept(?:s|ed)?|agree[sd]?|concur(?:s|red)?)\b", re.I)]


def other_parties(text: str, speaker: str, stance: list[re.Pattern], agreement: list[re.Pattern], cfg: dict) -> set[str]:
    """Parties other than the speaker that hold the claim too: the subject of a stance verb, or of "agrees with".

    "The company's base-case ICER" names the company as the owner of a
    figure, not as a holder of the claim, so a bare mention does not count.
    """
    found = {speaker_for(m.group(1), cfg) for pat in stance + agreement for m in pat.finditer(text)}
    return found - {speaker, None}


def contested_by(corpus: Corpus, u: Unit, claim: str, speaker: str, stance: list[re.Pattern], cfg: dict) -> str | None:
    """A unit in which another party takes a position on the claim's subject: its rare words, under another speaker's stance verb."""
    ac = cfg["attribution"]
    limit = ac["rare_df_max"] * len(corpus.units)
    rare = {t for t in tokenize(claim) if len(t) >= 3 and corpus.df.get(t, 0) <= limit}
    if len(rare) < ac["min_rare_tokens"]:
        return None
    own = set(corpus.by_element.get(u.element_id, ())) | {u.unit_id}
    for hit, _ in corpus.bm25.search(sorted(rare), k=ac["contested_top_k"], exclude=own):
        h = corpus.units[hit]
        if len(rare & set(h.tokens)) / len(rare) < ac["contested_containment"]:
            continue
        for pat in stance:
            if any(speaker_for(m.group(1), cfg) not in (None, speaker) for m in pat.finditer(h.plain)):
                return hit
    return None


def speaker_cue(plain: str, cfg: dict) -> bool:
    aliases = [a for al in cfg["attribution"]["speakers"].values() for a in al]
    return any(re.search(rf"\b{re.escape(a)}\b", plain, re.I) for a in aliases)


# ----- selectors -----

def select_factual(corpus: Corpus, cfg: dict, rng: Random) -> Iterator[Candidate]:
    units = [u for u in corpus.units.values() if clean_prose(u, cfg)]
    for u in ordered(corpus, units, section_key, rng, cfg["limits"]["max_per_section"], cfg):
        yield Candidate("factual", None, [u])


def select_numerical(corpus: Corpus, cfg: dict, rng: Random) -> Iterator[Candidate]:
    items = []
    for u in corpus.units.values():
        if clean_prose(u, cfg):
            hits = numeric_hits(u.plain, cfg)
            if hits:
                items.append((u, "prose", sorted({n for n, _ in hits} | significant_numbers(u.plain)), {}))
        elif whole_table(u, cfg) and u.caption:
            cells = [c for r in table_rows(u.text) for c in r]
            numbers = sorted({c for c in cells if re.fullmatch(r"[£$€]?\s?-?\d[\d,]*(?:\.\d+)?\s?%?", c.strip())})
            twin = twin_hint(corpus, u)
            if len(numbers) >= 4 and twin is not None:
                items.append((u, "table", [re.sub(r"[£$€%,\s]", "", n) for n in numbers], twin))
    for u, sub, numbers, twin in ordered(corpus, items, lambda x: section_key(x[0]), rng, cfg["limits"]["max_per_section"], cfg, lambda x: x[0]):
        yield Candidate("numerical", sub, [u], {"numbers": numbers[:40], **twin})


NUMCELL = re.compile(r"^[£$€]?\s?[<>≤≥]?-?\d[\d,]*(?:\.\d+)?\s?%?(?:\s?\([^()]*\))?[*†‡a-c]?$")   # a value, maybe with (95% CI) and a footnote mark


def arm_columns(rows: list[list[str]], cfg: dict, extra: list[str] = ()) -> tuple[int, list[int], list[str]] | None:
    """(header row index, two column indexes, labels) when a header row names two arms, groups or scenarios."""
    pats = list(cfg["comparative"]["arm_header_patterns"]) + list(extra)
    for h in range(min(3, len(rows))):
        cols = [i for i, c in enumerate(rows[h]) if i > 0 and c.strip() and any(re.search(p, c, re.I) for p in pats)][:2]
        if len(cols) == 2:
            labels = []
            for i in cols:
                label = rows[h][i]
                if h > 0 and i < len(rows[h - 1]) and rows[h - 1][i].strip() and rows[h - 1][i] != label:
                    label = f"{rows[h - 1][i]} {label}"
                labels.append(label)
            return h, cols, labels
    return None


def table_comparison(rows: list[list[str]], cfg: dict, rng: Random, extra: list[str] = ()) -> dict | None:
    """How a table supports a comparison: two arm columns ("columns"), or several labelled rows with a value in one column ("rows")."""
    cc = cfg["comparative"]
    arms = arm_columns(rows, cfg, extra)
    if arms:
        h, cols, labels = arms
        body = [r for r in rows[h + 1:] if max(cols) < len(r)
                and all(re.search(r"\d", r[i]) and REDACTED not in r[i] for i in cols)]
        if len(body) >= cc["min_numeric_rows"]:
            if len(body) > cc["max_rows"]:
                start = rng.randrange(0, len(body) - cc["max_rows"] + 1)
                body = body[start:start + cc["max_rows"]]
            return {"mode": "columns", "labels": labels, "header": " | ".join(rows[h]), "rows": [" | ".join(r) for r in body]}
    if not rows:
        return None
    header = rows[0]
    width = label_width(rows)
    for j in range(width, max(len(r) for r in rows)):
        good = [r for r in rows[1:] if j < len(r) and all(c.strip() for c in r[:width]) and NUMCELL.match(r[j].strip())
                and REDACTED not in r[j] and not garbled(" ".join(r[:width]))]
        if len(good) < cc["min_numeric_rows"] or not (j < len(header) and header[j].strip()):
            continue
        if width > 1:
            # a group column (the study over its arms): compare rows of one group only, under group-qualified labels
            groups = defaultdict(list)
            for r in good:
                groups[r[0]].append(r)
            pick = [g for g in groups.values() if len({" ".join(r[:width]) for r in g}) >= 2]
            chosen = pick[rng.randrange(len(pick))] if pick else good   # no repeating group: both columns just name the row
        else:
            chosen = good
        pair = closest_pair([(" ".join(r[:width]), r[j]) for r in chosen])
        if not pair:
            continue
        return {"mode": "rows", "labels": list(pair), "measure": header[j],
                "header": " | ".join(header), "rows": [" | ".join(r) for r in good][:cc["max_rows"]]}
    return None


def closest_pair(labelled: list[tuple[str, str]]) -> tuple[str, str] | None:
    """The two labels whose values differ by the least, both non-zero when any such pair exists.

    Left to itself the model compared diarrhoea (7.7%) with maculo-papular
    rash (0.0%): true, and nothing to work out. Close values make the
    answerer read both rows; a zero is kept only when every pair has one.
    """
    values = []
    for label, cell in labelled:
        n = _number(cell)
        if n is not None and label.strip():
            values.append((label, n))
    pairs = []
    for i in range(len(values)):
        for k in range(i + 1, len(values)):
            (la, a), (lb, b) = values[i], values[k]
            if la != lb and a != b:
                pairs.append((a == 0 or b == 0, abs(a - b) / max(abs(a), abs(b)), la, lb))
    if not pairs:
        return None
    zero, _, la, lb = min(pairs)
    return la, lb


def _number(cell: str) -> float | None:
    m = re.match(r"[£$€]?\s?[<>≤≥]?(-?\d[\d,]*(?:\.\d+)?)", cell.strip())
    try:
        return float(m.group(1).replace(",", "")) if m else None
    except ValueError:
        return None


def label_width(rows: list[list[str]]) -> int:
    """How many leading columns name a row: the usual count of cells before the first numeric one (2 when a group column spans arms)."""
    counts = []
    for r in rows[1:]:
        n = next((i for i, c in enumerate(r) if NUMCELL.match(c.strip())), None)
        if n:
            counts.append(n)
    return max(set(counts), key=counts.count) if counts else 1


def select_comparative(corpus: Corpus, cfg: dict, rng: Random) -> Iterator[Candidate]:
    drug = technology(corpus)
    extra = [rf"\b{re.escape(drug)}\b"] if drug else []
    items = []
    for u in corpus.units.values():
        if whole_table(u, cfg, allow_redacted=True):
            hint = table_comparison(table_rows(u.text), cfg, rng, extra)
            twin = twin_hint(corpus, u)
            if hint and twin is not None:
                items.append((u, "table", {**hint, **twin}))
        elif clean_prose(u, cfg):
            hits = numeric_hits(u.plain, cfg)
            by_unit = defaultdict(list)
            for n, unit in hits:
                by_unit[unit.lower()].append(n)
            pairs = {k: v for k, v in by_unit.items() if len(set(v)) >= 2}
            if pairs:
                items.append((u, "prose", {"mode": "prose", "numbers": sorted({n for v in pairs.values() for n in v})}))
    # the same measure across arms or scenarios (columns) matches the spec best; rows of one column next; prose last
    for sub, mode in (("table", "columns"), ("table", "rows"), ("prose", "prose")):
        for u, _, hint in ordered(corpus, [i for i in items if i[1] == sub and i[2]["mode"] == mode], lambda x: section_key(x[0]), rng,
                                  cfg["limits"]["max_per_section"], cfg, lambda x: x[0]):
            yield Candidate("comparative", sub, [u], hint)


def printed_together(corpus: Corpus, a: str, b: str, exclude: set[str]) -> str | None:
    """A unit with one line that prints both values: it answers a comparison of the two on its own."""
    pa, pb = (re.compile(rf"(?<![\d.]){re.escape(x)}(?!\d|\.\d)") for x in (a, b))   # "7.3" is not in "17.3" or "7.31"
    for uid, u in corpus.units.items():
        if uid not in exclude and any(pa.search(line) and pb.search(line) for line in u.plain.splitlines()):
            return uid
    return None


def twin_comparisons(corpus: Corpus, cfg: dict, rng: Random) -> list[Candidate]:
    """Two twin tables and one cell they print differently: the same measure under another scenario, arm, price or document.

    Each table holds one of the two values, so neither answers the comparison
    alone, and a cell is passed over when some other unit prints both values
    on one line. Trials make poor subjects here: in TA1063 every value
    sentence that names a single trial names CAPItello-291, and the other
    trials' figures sit in tables that list them all, so one unit answers.
    """
    mc = cfg["multihop"]
    drug = technology(corpus)
    out, seen, used = [], set(), set()
    for a, twins in sorted(corpus.twins.items()):
        for b in twins:
            pair = tuple(sorted((a, b)))
            if pair in seen or used & set(pair):   # each table in one comparison: six transition matrices would make fifteen
                continue
            seen.add(pair)
            ua, ub = corpus.units[pair[0]], corpus.units[pair[1]]
            ha = twin_hint(corpus, ua) if whole_table(ua, cfg, allow_redacted=True) else None
            hb = twin_hint(corpus, ub) if whole_table(ub, cfg, allow_redacted=True) else None
            va, vb = corpus.distinguish(ua, ub), corpus.distinguish(ub, ua)
            if ha is None or hb is None or not va or not vb:
                continue
            ra, rb = table_rows(ua.text), table_rows(ub.text)
            header, by_key = ra[0], keyed_rows(rb)
            cells = []
            for key, row in keyed_rows(ra).items():
                other = by_key.get(key)
                # the same row in both: its text cells agree too (document2's two UP/C tables list their sub-rows in another order)
                if not other or not all(normalize(x) == normalize(y) for x, y in zip(row, other) if not value_cell(x) and not value_cell(y)):
                    continue
                if any(re.search(p, f"{row[0]} {ua.caption}", re.I) for p in mc["incomparable_measures"]):
                    continue
                for j in range(1, min(len(row), len(other))):
                    # a column without a name is no measure (sub-columns of document2's three-level transition matrices)
                    if j < len(header) and re.search(r"[A-Za-z]", header[j]) and value_cell(row[j]) and value_cell(other[j]) and not same_value(row[j], other[j]):
                        cells.append((row, other, j))
            rng.shuffle(cells)
            cells.sort(key=lambda c: not (drug and re.search(rf"\b{re.escape(drug)}\b", c[0][0], re.I)))   # the appraised drug's row first
            own = {ua.unit_id, ub.unit_id} | set(corpus.copies.get(ua.unit_id, [])) | set(corpus.copies.get(ub.unit_id, []))
            for row, other, j in cells:
                if printed_together(corpus, row[j], other[j], own):
                    continue
                # the row's own words ("Comparison", "Ratio of GMRs (95% CI)") and the column's name the measure
                label = " / ".join(dict.fromkeys(c.strip() for c in row if c.strip() and not value_cell(c) and REDACTED not in c))
                # what tells each table from all its twins, not only from this one: "Base-case" alone fits the list and PAS tables
                phrases = list(dict.fromkeys(ha.get("must_mention", []) + hb.get("must_mention", []) + [va, vb]))
                out.append(Candidate("multihop", "comparison", [ua, ub], {
                    "link": f"{va} / {vb}", "link_kind": "comparison", "versions": [va, vb], "values": [row[j], other[j]],
                    "measure": f"{label} ({header[j]})", "must_mention": phrases}))
                used.update(pair)
                break
    return list(spread(out, lambda c: (c.units[0].doc_id, c.units[1].doc_id), rng))


def keyed_rows(rows: list[list[str]]) -> dict[tuple[str, int], list[str]]:
    """Body rows by (label, occurrence): a label can repeat ("Comparison" under month 9 and again under 2 years)."""
    seen, out = defaultdict(int), {}
    for r in rows[1:]:
        if r and re.search(r"\w", r[0]):
            label = r[0].strip()
            out[(label, seen[label])] = r
            seen[label] += 1
    return out


def value_cell(cell: str) -> bool:
    """A number with its unit or interval ("£24,718", "-18% (-32%, -2%); p=0.03"), not a sentence that happens to contain one."""
    return bool(re.search(r"\d", cell)) and len(cell) <= 40 and len(re.findall(r"[A-Za-z]", cell)) <= 4 and REDACTED not in cell


def same_value(a: str, b: str) -> bool:
    """The same figure printed to different precision: "-18%" and "-18.1%" are one result, not two to compare."""
    fa, fb = (re.search(r"-?\d[\d,]*(?:\.\d+)?", x) for x in (a, b))
    if not fa or not fb:
        return False
    sa, sb = fa.group(0).replace(",", ""), fb.group(0).replace(",", "")
    places = min(len(s.split(".")[1]) if "." in s else 0 for s in (sa, sb))
    return round(float(sa), places) == round(float(sb), places)


def select_multihop(corpus: Corpus, cfg: dict, rng: Random) -> Iterator[Candidate]:
    """Comparisons across twin tables and entity bridges, alternating; then "see Table N" pairs.

    An entity bridge is a unit that characterises a trial and a unit that gives its value.
    """
    comparisons = twin_comparisons(corpus, cfg, rng)
    seen = set()
    pairs = []
    for key, uids in corpus.entities.items():
        by_doc = defaultdict(list)
        for uid in uids:
            by_doc[corpus.units[uid].doc_id].append(corpus.units[uid])
        for doc_id, us in by_doc.items():
            desc = [(u, b) for u in us if clean_prose(u, cfg) and (b := bridge_sentence(u.plain, key, corpus))]
            num = [u for u in us if (clean_prose(u, cfg) and numeric_hits(u.plain, cfg)) or (whole_table(u, cfg, allow_redacted=True) and u.caption and mentions(u.plain, key))]
            rng.shuffle(desc); rng.shuffle(num)
            for d, bridge in desc[:6]:
                for n in num:
                    twin = twin_hint(corpus, n) if n.type == "table" else {}
                    if n.unit_id == d.unit_id or n.section_id == d.section_id or (d.unit_id, n.unit_id) in seen or twin is None:
                        continue
                    shown = shown_with(n, key)
                    # B must hold other trials to choose between, and the bridge sentence must say something about
                    # this one that B does not: otherwise B answers alone
                    if alternatives(corpus, n, key) and len(new_words(bridge, corpus.entity_names[key], shown)) >= cfg["multihop"]["min_new_tokens"]:
                        seen.add((d.unit_id, n.unit_id))
                        pairs.append((d, n, corpus.entity_names[key], twin, bridge, shown))
                        break
    bridges = [Candidate("multihop", "entity", [d, n], {"link": name, "link_kind": "entity", "bridge": bridge, "shown": shown, **twin})
               for d, n, name, twin, bridge, shown in spread(pairs, lambda x: (x[0].doc_id, x[2]), rng, 4)]
    # a comparison needs both tables by construction, so it leads; bridges keep the type from being all comparisons
    yield from interleave([comparisons, bridges])

    xrefs = []
    for u in corpus.units.values():
        if not clean_prose(u, cfg):
            continue
        for kind, number, element_id in corpus.resolve_references(u):
            t = corpus.unit_for_element(element_id)
            if not t or t.element_id == u.element_id or not whole_table(t, cfg, allow_redacted=True):
                continue
            twin = twin_hint(corpus, t)
            if (u.unit_id, t.unit_id) in seen or twin is None:
                continue
            seen.add((u.unit_id, t.unit_id))
            xrefs.append((u, t, f"{kind.capitalize()} {number}", twin))
    for u, t, link, twin in spread(xrefs, lambda x: (x[0].doc_id, x[2]), rng, 2):
        yield Candidate("multihop", "xref", [u, t], {"link": link, "link_kind": "xref", **twin})


def _role_sets(corpus: Corpus, cfg: dict) -> tuple[set[str], set[str]]:
    """Unit ids of reviewing documents (A) and reviewed documents (B); any two documents when roles are unknown."""
    rc = cfg["roles"]
    a = {uid for uid, u in corpus.units.items() if corpus.docs[u.doc_id].role in rc["secondary"]}
    b = {uid for uid, u in corpus.units.items() if corpus.docs[u.doc_id].role in rc["primary"]}
    if not a or not b:
        a = b = {uid for uid, u in corpus.units.items() if corpus.docs[u.doc_id].role != "cover"}
    return a, b


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(len(sa | sb), 1)


def select_crossdoc(corpus: Corpus, cfg: dict, rng: Random) -> Iterator[Candidate]:
    cd = cfg["cross_doc"]
    a_ids, b_ids = _role_sets(corpus, cfg)
    seen = set()

    def ok_pair(a: Unit, b: Unit) -> bool:
        if a.doc_id == b.doc_id or frozenset((a.unit_id, b.unit_id)) in seen:
            return False
        if jaccard(a.tokens, b.tokens) >= cd["max_overlap"]:
            return False
        seen.add(frozenset((a.unit_id, b.unit_id)))
        return True

    anchors = [corpus.units[i] for i in sorted(a_ids) if clean_prose(corpus.units[i], cfg)]
    rng.shuffle(anchors)

    # 1. explicit pointers: "CS Table 12", "Document B Figure 3"
    refs = []
    primary_docs = {corpus.units[i].doc_id for i in b_ids}
    for a in anchors:
        for pat in cd["reference_patterns"]:
            for m in re.finditer(pat, a.plain, re.I):
                if m.lastindex and m.lastindex >= 2:
                    for doc_id in primary_docs:
                        t = corpus.unit_for_element(corpus.captions.get((doc_id, m.group(1).lower(), m.group(2)), ""))
                        twin = twin_hint(corpus, t) if t else None
                        if t and t.unit_id in b_ids and whole_table(t, cfg, allow_redacted=True) and twin is not None and ok_pair(a, t):
                            refs.append((a, t, f"{m.group(1).capitalize()} {m.group(2)}", twin))
    for a, t, link, twin in spread(refs, lambda x: section_key(x[0]), rng, cfg["limits"]["max_per_section"]):
        yield Candidate("crossdoc", "reference", [a, t], {"link": link, **twin})

    # 2. shared significant numbers between a stance-bearing passage and a reviewed passage
    numbers_index = defaultdict(set)
    for i in b_ids:
        u = corpus.units[i]
        if u.evidence and (clean_prose(u, cfg) or whole_table(u, cfg)):
            for n in significant_numbers(u.plain):
                numbers_index[n].add(i)
    cued = [a for a in anchors if speaker_cue(a.plain, cfg)]
    fingerprints = []
    for a in cued:
        sig = significant_numbers(a.plain)
        if not sig:
            continue
        shared = defaultdict(set)
        for n in sig:
            for i in numbers_index.get(n, ()):
                shared[i].add(n)
        rare_a = corpus.rare_tokens(a)
        best = None
        for i, nums in shared.items():
            b = corpus.units[i]
            rare_shared = len(rare_a & set(b.tokens))
            if len(nums) >= cd["min_shared_numbers"] or (nums and rare_shared >= cd["min_shared_rare"]):
                score = (len(nums), rare_shared)
                if best is None or score > best[0]:
                    best = (score, b, sorted(nums))
        if best and ok_pair(a, best[1]):
            fingerprints.append((a, best[1], best[2]))
    for a, b, nums in spread(fingerprints, lambda x: section_key(x[0]), rng, cfg["limits"]["max_per_section"]):
        yield Candidate("crossdoc", "numbers", [a, b], {"link": ", ".join(nums[:5])})

    # 3. lexical search for the reviewed passage
    lexical = []
    for a in cued:
        rare_a = corpus.rare_tokens(a)
        terms = sorted(set(a.tokens), key=lambda t: -corpus.bm25.idf.get(t, 0))[:cd["query_terms"]]
        hits = corpus.bm25.search(terms, k=cd["top_k"], only=b_ids, exclude={a.unit_id})
        if not hits:
            continue
        (top, s1), s2 = hits[0], hits[1][1] if len(hits) > 1 else 0.0
        b = corpus.units[top]
        if not (clean_prose(b, cfg) or whole_table(b, cfg)):
            continue
        shared = sorted(rare_a & set(b.tokens))
        if len(shared) >= cd["min_shared_rare"] and s1 >= s2 * (1 + cd["min_margin"]) and ok_pair(a, b):
            lexical.append((a, b, shared))
    for a, b, shared in spread(lexical, lambda x: section_key(x[0]), rng, cfg["limits"]["max_per_section"]):
        yield Candidate("crossdoc", "bm25", [a, b], {"link": ", ".join(shared[:6])})


def select_attribution(corpus: Corpus, cfg: dict, rng: Random) -> Iterator[Candidate]:
    ac = cfg["attribution"]
    stance = [re.compile(p, re.I) for p in ac["stance_patterns"]]
    agreement = agreement_patterns(cfg)
    has_guidance = any(d.role == "guidance" for d in corpus.docs.values())
    items = []
    for u in corpus.units.values():
        role = corpus.docs[u.doc_id].role
        if clean_prose(u, cfg):
            for pat in stance:
                m = pat.search(u.plain)
                if m:
                    speaker = speaker_for(m.group(1), cfg)
                    if speaker == "the committee" and not has_guidance:
                        speaker = None   # "the committee agreed" here is about an earlier appraisal, not this bundle
                    if speaker in ac["own_voice_only"] and ac["role_speakers"].get(role) != speaker:
                        speaker = None   # "UK clinical experts expect ..." in the company's documents is the company's position too
                    if speaker:
                        prev = u.plain.rfind(". ", 0, m.start())
                        start = prev + 2 if prev >= 0 else 0
                        end = u.plain.find(". ", m.end())
                        claim = u.plain[start: end + 1 if end > 0 else None].strip()
                        # "The EAG agrees with the company that ...": both parties hold the claim, so "who" has two answers
                        if not other_parties(claim, speaker, stance, agreement, cfg):
                            items.append((u, "stance", speaker, claim))
                    break
            else:
                if (role == "clarification" and any(re.search(p, t, re.I) for p in ac["response_section_patterns"] for t in u.section_path[-1:])
                        and not re.match(r"^[A-C]\s?\d{1,2}\b", u.plain) and not u.plain.rstrip().endswith("?")):
                    items.append((u, "response", ac["role_speakers"].get(role, "the company"), u.plain[:400]))
        elif u.row_index is not None and role in ac["role_speakers"] and role not in cfg["roles"]["primary"] + cfg["roles"]["secondary"]:
            cells = u.plain.split("\n")[-1].split(" | ")
            answer = cells[-1].strip()
            about_respondent = any(re.search(p, cells[0], re.I) for p in ac["form_skip_questions"])
            if len(answer) >= ac["min_answer_chars"] and not TICKBOX.search(answer) and REDACTED not in answer and not about_respondent:
                items.append((u, "form", ac["role_speakers"][role], answer[:400]))
    by_speaker = defaultdict(lambda: [[], [], [], []])
    for u, sub, speaker, claim in items:
        other = contested_by(corpus, u, claim, speaker, stance, cfg)
        by_speaker[speaker][2 * demoted(corpus, u, cfg) + (0 if other else 1)].append((u, sub, speaker, claim, other))
    # a claim another party disputes comes first: the answerer has to tell the positions apart, not guess from the subject;
    # lay summaries and background sections come last
    queues = [[x for tier in tiers for x in spread(tier, lambda x: section_key(x[0]), rng, 3)] for _, tiers in sorted(by_speaker.items())]
    for u, sub, speaker, claim, other in interleave(queues):
        hint = {"speaker": speaker, "claim": claim}
        if other:
            hint["contested_by"] = other
        yield Candidate("attribution", sub, [u], hint)


def stated_elsewhere(corpus: Corpus, u: Unit, cfg: dict) -> str | None:
    """A unit that gives a value for something this unit redacts: a line with the redacted line's words and a number.

    TA1063's Table 14 redacts body surface area by sex, but Table 42 (model
    inputs) prints both values. A question built on the redaction would mark
    a correct answer wrong.
    """
    uc = cfg["unanswerable"]
    own = set(corpus.by_element.get(u.element_id, ())) | {u.unit_id}
    limit = uc["stated_rare_df_max"] * len(corpus.units)
    for line in u.plain.splitlines():
        if REDACTED not in line:
            continue
        # the label's distinctive words ("surface", "male"), not the drug and trial names on every line
        rare = {t for t in tokenize(line.replace(REDACTED, " ")) if len(t) >= 3 and not t.isdigit() and corpus.df.get(t, 0) <= limit}
        if len(rare) < 2:
            continue
        for hit, _ in corpus.bm25.search(sorted(rare), k=uc["stated_top_k"], exclude=own):
            for hl in corpus.units[hit].plain.splitlines():
                toks = tokenize(hl)
                if REDACTED not in hl and len(toks) <= uc["stated_max_tokens"] and re.search(r"\d", hl) and rare <= set(toks):
                    return hit
    return None


def select_unanswerable_redacted(corpus: Corpus, cfg: dict, rng: Random) -> Iterator[Candidate]:
    uc = cfg["unanswerable"]
    items = []
    for u in corpus.units.values():
        if not (1 <= u.redactions <= uc["max_redactions"] and u.evidence and native(u, cfg)):
            continue
        if not (clean_prose(u, cfg, allow_redacted=True) or whole_table(u, cfg, allow_redacted=True) or (u.row_index is not None and len(u.plain) <= cfg["limits"]["max_table_chars"])):
            continue
        if len(u.plain.replace(REDACTED, "").strip()) < uc["min_clear_chars"]:
            continue
        labels = []
        for pat in REDACTED_LABEL:
            labels += [m.group(1).strip() for m in pat.finditer(u.plain) if m.group(1).strip() and "REDACTED" not in m.group(1)]
        if labels and not stated_elsewhere(corpus, u, cfg):
            items.append((u, sorted(set(labels))[:6]))
    for u, labels in spread(items, lambda x: section_key(x[0]), rng, cfg["limits"]["max_per_section"]):
        yield Candidate("unanswerable", "redacted", [u], {"labels": labels})


def technology(corpus: Corpus) -> str | None:
    """The appraised drug: the topic's first word ("Capivasertib with fulvestrant for ...")."""
    first = corpus.topic.split()[0] if corpus.topic else ""
    return first if len(first) > 3 else None


def about_the_appraisal(corpus: Corpus, u: Unit, cfg: dict) -> bool:
    """Names the drug or a trial, or states a number with a unit.

    A missing fact can be invented for any text, form instructions included:
    7 of TA1063's first 18 "not covered" anchors were deadlines, form help and
    acknowledgements ("The deadline for your response is 5pm on ..."), which
    this drops along with half of all paragraphs (887 to 435).
    """
    drug = technology(corpus)
    return bool((drug and re.search(rf"\b{re.escape(drug)}\b", u.plain, re.I)) or corpus.unit_entities.get(u.unit_id) or numeric_hits(u.plain, cfg))


def select_unanswerable_not_covered(corpus: Corpus, cfg: dict, rng: Random) -> Iterator[Candidate]:
    units = [u for u in corpus.units.values() if clean_prose(u, cfg) and about_the_appraisal(corpus, u, cfg)]
    for u in ordered(corpus, units, section_key, rng, cfg["limits"]["max_per_section"], cfg):
        yield Candidate("unanswerable", "not_covered", [u], {"topic": corpus.topic})


SELECTORS: dict[str, Callable[[Corpus, dict, Random], Iterator[Candidate]]] = {
    "factual": select_factual,
    "numerical": select_numerical,
    "comparative": select_comparative,
    "multihop": select_multihop,
    "crossdoc": select_crossdoc,
    "attribution": select_attribution,
    "unanswerable_redacted": select_unanswerable_redacted,
    "unanswerable_not_covered": select_unanswerable_not_covered,
}


def candidate_counts(corpus: Corpus, cfg: dict, seed: int, examples: int = 2) -> dict:
    """Exhaust every selector (no LLM): counts by subtype and a few example candidates, for --dry-run."""
    out = {}
    for name, sel in SELECTORS.items():
        counts, samples = defaultdict(int), defaultdict(list)
        for c in sel(corpus, cfg, Random(seed)):
            counts[c.subtype or "-"] += 1
            if len(samples[c.subtype or "-"]) < examples:
                samples[c.subtype or "-"].append({"units": [u.unit_id for u in c.units], "hint": {k: v for k, v in c.hint.items() if k != "rows"}})
        out[name] = {"total": sum(counts.values()), "by_subtype": dict(counts), "examples": dict(samples)}
    return out
