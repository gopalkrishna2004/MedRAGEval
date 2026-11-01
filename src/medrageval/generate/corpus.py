"""The corpus the selectors draw evidence from.

Loads one parse output directory (elements.jsonl, sections.json,
documents.json) and builds what question generation needs:

- evidence units: elements that passed the noise filter, with long tables
  split into row groups so a form's "question | answer" row can be cited on
  its own. A row unit keeps its parent's element_id, and its lines are lines
  of the parent's plain text, so a quote from it is also a quote from the
  element.
- a role for every document (company submission, EAG report, ...), scored
  from headings, running text and the first elements against config patterns.
- a BM25 index over the units, used for cross-document pairing, attribution
  exclusivity and the "is it really unanswerable" check.
- indexes of captions ("Table 7" -> element) and trial names ("CAPItello-291"
  -> units), the links behind multi-hop questions.
"""

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from medrageval.text import normalize, table_rows

STOPWORDS = set("""
a an and are as at be been being by for from has have had he her his in into is it its of on or that the their them
there these they this those to was were which while with within will would can could should may might not no nor
than then so such but if also any all each both into onto over under between about after before during per vs versus
""".split())

CAPTION_RE = re.compile(r"^[\W_]*(Table|Figure)\s+([A-Z]?\d+(?:\.\d+)?)\b", re.I)   # "_Table 14: ..._": an italic caption
REFERENCE_RE = re.compile(
    r"\b(?:see|in|shown in|presented in|reported in|summari[sz]ed in|described in|outlined in|detailed in|listed in)"
    r"\s+(Table|Figure)\s+([A-Z]?\d+(?:\.\d+)?)\b", re.I)
YEAR_RE = re.compile(r"^(?:19|20)\d\d$")
NUMBER_RE = re.compile(r"[£$€]\s?\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?\s?%|\d+\.\d+|\d[\d,]{2,}")
RANGE_RE = re.compile(r"[£$€]?\s?(\d[\d,]*(?:\.\d+)?\s?[-–‒—]\s?\d[\d,]*(?:\.\d+)?)(?=\s?%|\s?(?:months?|years?|weeks?|days?|mg|patients|participants)\b)|[£$€]\s?(\d[\d,]*(?:\.\d+)?\s?[-–‒—]\s?\d[\d,]*(?:\.\d+)?)")
REDACTED = "[REDACTED]"
_NUMBERING = re.compile(r"^(?:appendix\s+)?[A-Z]?\d*(?:\.\d+)*\.?\s+(?=\S)", re.I)


@dataclass
class Unit:
    unit_id: str
    element_id: str
    doc_id: str
    type: str
    text: str
    plain: str
    caption: str | None
    section_id: str | None
    section_path: list[str]
    page_start: int
    page_end: int
    redactions: int
    row_index: int | None = None
    evidence: bool = True                     # False: searchable, but too long to cite
    source: str = "native"                    # "ocr" text carries glyph errors, so selectors avoid it
    tokens: list[str] = field(default_factory=list)

    @property
    def pages(self) -> list[int]:
        return [self.page_start, self.page_end]


@dataclass
class Document:
    doc_id: str
    role: str
    label: str
    title: str
    start_page: int
    end_page: int
    n_units: int = 0
    role_scores: dict = field(default_factory=dict)


def tokenize(text: str) -> list[str]:
    return [t for t in normalize(text).split() if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


class BM25:
    def __init__(self, docs: dict[str, list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids = list(docs)
        self.tf = {i: Counter(t) for i, t in docs.items()}
        self.len = {i: len(t) for i, t in docs.items()}
        self.avg = sum(self.len.values()) / max(len(docs), 1)
        self.df = Counter()
        for tf in self.tf.values():
            self.df.update(tf.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in self.df.items()}
        self.postings = defaultdict(list)
        for i, tf in self.tf.items():
            for t in tf:
                self.postings[t].append(i)

    def search(self, query: list[str], k: int = 10, only: set[str] | None = None,
               exclude: set[str] = frozenset()) -> list[tuple[str, float]]:
        scores = defaultdict(float)
        for t in set(query):
            if t not in self.idf:
                continue
            for i in self.postings[t]:
                if i in exclude or (only is not None and i not in only):
                    continue
                f = self.tf[i][t]
                scores[i] += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]


def is_noise(el: dict, cfg: dict) -> str | None:
    """Reason an element must not be evidence or a search hit, or None."""
    plain = el["plain"]
    if el["type"] in ("figure", "caption"):
        return el["type"]
    if CAPTION_RE.match(el.get("caption") or "") and CAPTION_RE.match(el["caption"]).group(1).lower() == "figure":
        return "figure"   # an OCR'd chart stored as a table (a forest plot's value column) is still a figure
    for title in el["section_path"]:
        core = _NUMBERING.sub("", title)   # "B.4 References" -> "References"
        for pat in cfg["noise"]["section_title_patterns"]:
            if re.search(pat, title, re.I) or re.search(pat, core, re.I):
                return f"section:{pat}"
    for pat in cfg["noise"]["text_patterns"]:
        if re.search(pat, plain, re.I | re.M):
            return f"text:{pat}"
    if len(plain) < cfg["noise"]["min_chars"]:
        return "short"
    if len(plain.replace(REDACTED, "").strip()) < 10:
        return "redacted_only"
    return None


def template_elements(elements: list[dict], roles: dict[str, str], cfg: dict) -> set[str]:
    """Element ids of form text: a paragraph that `template_min_docs` or more documents of one role repeat word for word.

    Submission forms print the same instructions in every copy ("You do not
    have to answer every question ..."); the answers differ.
    """
    n = cfg["noise"].get("template_min_docs")
    if not n:
        return set()
    docs = defaultdict(set)
    keyed = {}
    for e in elements:
        if e["type"] in ("paragraph", "list_item"):
            keyed[e["element_id"]] = (roles[e["doc_id"]], normalize(e["plain"]))
            docs[keyed[e["element_id"]]].add(e["doc_id"])
    return {eid for eid, k in keyed.items() if len(docs[k]) >= n}


def make_units(el: dict, cfg: dict) -> list[Unit]:
    """One unit per element, or row groups for a long table."""
    lim = cfg["limits"]
    common = dict(element_id=el["element_id"], doc_id=el["doc_id"], type=el["type"], caption=el.get("caption"),
                  section_id=el["section_id"], section_path=el["section_path"],
                  page_start=el["page_start"], page_end=el["page_end"], source=el.get("source", "native"))
    whole = Unit(unit_id=el["element_id"], text=el["text"], plain=el["plain"], redactions=el["redactions"], **common)
    if el["type"] == "table" and len(el["plain"]) > lim["max_table_chars"]:
        rows = table_rows(el["text"])
        if len(rows) < 3:
            whole.evidence = False
            return [whole]
        header = " | ".join(rows[0])
        units, group, size = [], [], 0
        def flush():
            if group:
                lines = [header] + [" | ".join(r) for r in group]
                plain = "\n".join(lines)
                units.append(Unit(unit_id=f"{el['element_id']}#r{len(units):02d}", text=plain, plain=plain,
                                  redactions=plain.count(REDACTED), row_index=len(units), **common))
        for row in rows[1:]:
            line = " | ".join(row)
            if size and size + len(line) > lim["row_group_chars"]:
                flush(); group, size = [], 0
            group.append(row); size += len(line)
        flush()
        return units
    if el["type"] != "table" and len(el["plain"]) > lim["max_paragraph_chars"]:
        whole.evidence = False
    return [whole]


def detect_roles(docs: list[dict], elements: list[dict], cfg: dict) -> dict[str, tuple[str, dict]]:
    """doc_id -> (role, scores). Heading hits count double; the best role wins, ties by pattern order."""
    rc = cfg["roles"]
    by_doc = defaultdict(list)
    for e in elements:
        by_doc[e["doc_id"]].append(e)
    out = {}
    for d in docs:
        headings = "\n".join(d["headings"])
        body = "\n".join([d.get("running_text") or ""] + [e["plain"] for e in by_doc[d["doc_id"]][:30]])
        scores = {}
        for role, pats in rc["patterns"].items():
            s = 0
            for p in pats:
                if re.search(p, headings, re.I | re.M):
                    s += 2
                elif re.search(p, body, re.I | re.M):
                    s += 1
            scores[role] = s
        role = max(scores, key=lambda r: scores[r]) if any(scores.values()) else "other"
        if scores.get(role, 0) < 2 and len(by_doc[d["doc_id"]]) <= rc["cover_max_units"]:
            role = "cover"  # no heading hit and few elements: a cover or contents page
        role = rc.get("override", {}).get(d["doc_id"], role)
        out[d["doc_id"]] = (role, {r: s for r, s in scores.items() if s})
    return out


_GENERIC_TITLE = re.compile(r"^(national institute|single technology|committee papers|document [a-z]\b|produced by)", re.I)


def document_title(d: dict) -> str:
    for h in d["headings"]:
        if len(h) >= 20 and not _GENERIC_TITLE.match(h):
            return h[:160]
    return (d["headings"] or [d.get("running_text") or d["doc_id"]])[0][:160]


def corpus_topic(titles: list[str]) -> str:
    """The appraisal title: the longest wording shared by the most documents.

    Statements and submissions repeat the appraisal title with a suffix
    ("... [ID6308] Patient organisation submission"), so titles are grouped by
    their first words and the group's common prefix is the topic.
    """
    groups: dict[str, list[list[str]]] = defaultdict(list)
    for t in titles:
        if _GENERIC_TITLE.match(t):
            continue
        words = t.split()
        if len(words) >= 3:
            groups[" ".join(words[:3]).lower()].append(words)
    best, best_key = "", (0, 0)
    for members in groups.values():
        prefix = members[0]
        for w in members[1:]:
            n = 0
            while n < min(len(prefix), len(w)) and prefix[n] == w[n]:
                n += 1
            prefix = prefix[:n]
        text = " ".join(prefix).rstrip(" :-[(")
        if (len(members), len(text)) > best_key:
            best, best_key = text, (len(members), len(text))
    return best


def split_ranges(plain: str) -> tuple[set[str], str]:
    """Ranges with a unit ("40-50%", "£1,000-2,000", "3-6 months") as single values, and the text with them blanked out."""
    out = set()

    def take(m: re.Match) -> str:
        out.add(re.sub(r"[\s,]", "", m.group(1) or m.group(2)).translate(_RANGE_DASHES))
        return " "
    return out, RANGE_RE.sub(take, plain)


_RANGE_DASHES = str.maketrans({c: "-" for c in "–‒—"})


def significant_numbers(plain: str) -> set[str]:
    """Numbers that identify a figure: money, percentages, decimals, integers of 3+ digits (not years), ranges as one value."""
    out, plain = split_ranges(plain)
    for m in NUMBER_RE.findall(plain):
        n = re.sub(r"[£$€%,\s]", "", m)
        if n and not YEAR_RE.match(n):
            out.add(n)
    return out


def entity_key(name: str) -> str:
    return re.sub(r"[- ]", "", name).upper()


def extract_entities(plain: str, cfg: dict) -> dict[str, str]:
    """canonical key -> surface form for study names: a name with 2+ capitals or a digit, right before "trial"/"study"."""
    ec = cfg["entities"]
    found = {}
    for pat in ec["patterns"]:
        for m in re.finditer(pat, plain):
            name = m.group(1)
            if any(re.search(s, name, re.I) for s in ec["stoplist"]):
                continue
            if sum(c.isupper() for c in name) < 2 and not any(c.isdigit() for c in name):
                continue
            found.setdefault(entity_key(name), name)
    return found


def mentions(plain: str, key: str) -> bool:
    """Does the text mention the entity, ignoring hyphen/space variants (CAPItello-291, CAPItello291)?"""
    return key in entity_key(plain)


def _caption_words(caption: str | None) -> list[str]:
    text = CAPTION_RE.sub("", caption or "", count=1).strip(" :.-*_")
    return re.sub(r"\*\*|(?<!\w)_|_(?!\w)", "", text).split()


def _distinct_phrase(mine: list[str], theirs: list[str]) -> str:
    """The shortest run of `mine` around its first difference from `theirs` that is not a run of `theirs`."""
    fold = lambda ws: [re.sub(r"[^\w]", "", w.lower()) for w in ws]   # noqa: E731  ("targeted-release" is "targetedrelease")
    a, b = fold(mine), fold(theirs)
    ops = [op for op in SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if op[0] in ("replace", "delete")]
    if not ops:
        return ""
    lo, hi = ops[0][1], ops[0][1] + 1   # the first differing word, widened until the run is not in the twin's caption

    def in_theirs(lo: int, hi: int) -> bool:
        run = a[lo:hi]
        return any(b[i:i + len(run)] == run for i in range(len(b) - len(run) + 1))

    while in_theirs(lo, hi) and (hi < len(a) or lo > 0):
        if hi < len(a):
            hi += 1
        if in_theirs(lo, hi) and lo > 0:
            lo -= 1
    while lo > 0 and not re.search(r"[^\W\d_]", " ".join(a[lo:hi])):
        lo -= 1   # "1" or "0-12" alone is no phrase to put in a question: "Scenario 1", "Weeks 0-12"
    return " ".join(w.strip("(),;:.") for w in mine[lo:hi]).strip()


class Corpus:
    def __init__(self, interim_dir: Path, docs: list[dict], sections: list[dict], elements: list[dict], cfg: dict):
        self.interim_dir = interim_dir
        self.cfg = cfg
        self.sections = {s["section_id"]: s for s in sections}
        self.elements = {e["element_id"]: e for e in elements}
        self.element_order = [e["element_id"] for e in elements]

        roles = detect_roles(docs, elements, cfg)
        self.docs: dict[str, Document] = {}
        for d in docs:
            role, scores = roles[d["doc_id"]]
            self.docs[d["doc_id"]] = Document(d["doc_id"], role, cfg["roles"]["labels"].get(role, role),
                                              document_title(d), d["start_page"], d["end_page"], role_scores=scores)
        self.topic = corpus_topic([d.title for d in self.docs.values()])

        self.noise = Counter()
        self.units: dict[str, Unit] = {}
        templates = template_elements(elements, {doc_id: role for doc_id, (role, _) in roles.items()}, cfg)
        for e in elements:
            reason = is_noise(e, cfg) or ("template" if e["element_id"] in templates else None)
            if reason:
                self.noise[reason.split(":")[0]] += 1
                continue
            for u in make_units(e, cfg):
                u.tokens = tokenize(u.plain)
                self.units[u.unit_id] = u
        self.order = list(self.units)
        self.position = {uid: i for i, uid in enumerate(self.order)}
        self.by_doc: dict[str, list[str]] = defaultdict(list)
        self.by_section: dict[str, list[str]] = defaultdict(list)
        self.by_element: dict[str, list[str]] = defaultdict(list)
        for uid, u in self.units.items():
            self.by_doc[u.doc_id].append(uid)
            self.by_section[u.section_id].append(uid)
            self.by_element[u.element_id].append(uid)
        for d in self.docs.values():
            d.n_units = len(self.by_doc[d.doc_id])

        self.bm25 = BM25({uid: u.tokens for uid, u in self.units.items()})
        self.df = self.bm25.df
        self.captions = self._caption_index(elements)
        self.entities, self.unit_entities, self.entity_names = self._entity_index()
        self.twins, self.copies = self._twin_index()

    # ----- indexes -----
    def _caption_index(self, elements: list[dict]) -> dict[tuple[str, str, str], str]:
        idx = {}
        for i, e in enumerate(elements):
            m = CAPTION_RE.match(e["caption"] or "") if e.get("caption") else None
            if m and e["type"] in ("table", "figure"):
                idx.setdefault((e["doc_id"], m.group(1).lower(), m.group(2)), e["element_id"])
                continue
            # an unattached caption: a short "Table N ..." element right before a table/figure
            m = CAPTION_RE.match(e["plain"])
            if m and len(e["plain"]) < 300 and i + 1 < len(elements):
                nxt = elements[i + 1]
                if nxt["doc_id"] == e["doc_id"] and nxt["type"] in ("table", "figure"):
                    idx.setdefault((e["doc_id"], m.group(1).lower(), m.group(2)), nxt["element_id"])
        return idx

    def _entity_index(self):
        ec = self.cfg["entities"]
        names: dict[str, Counter] = defaultdict(Counter)
        for u in self.units.values():
            for key, surface in extract_entities(u.plain, self.cfg).items():
                names[key][surface] += 1
        hits: dict[str, set[str]] = defaultdict(set)
        for uid, u in self.units.items():
            folded = entity_key(u.plain)
            for key in names:
                if key in folded:
                    hits[key].add(uid)
        keep, per_unit, surface = {}, defaultdict(set), {}
        for key, uids in hits.items():
            if len(uids) < ec["min_units"]:
                continue
            per_doc = Counter(self.units[x].doc_id for x in uids)
            ok = {x for x in uids if per_doc[self.units[x].doc_id] <= ec["max_doc_share"] * max(self.docs[self.units[x].doc_id].n_units, 1)}
            if len(ok) < ec["min_units"]:
                continue
            keep[key] = ok
            surface[key] = names[key].most_common(1)[0][0]
        for key in list(keep):
            longer = [k for k in keep if k != key and k.startswith(key) and len(keep[key]) <= 1.1 * len(keep[k])]
            if longer:
                del keep[key], surface[key]
        for key, uids in keep.items():
            for x in uids:
                per_unit[x].add(key)
        return keep, per_unit, surface

    def _twin_index(self) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """unit_id -> other tables with the same row labels under the same columns, and unit_id -> copies of it.

        Appraisals repeat a table per arm, population or data cut (TA1063's
        clarification response has AIC tables for the capivasertib arm and
        the fulvestrant arm with identical rows), and across documents: the
        clarification response reprints the submission's goodness-of-fit
        table with new values, the EAG report reprints tables unchanged. A
        question about a twin has to say which, so selectors ask
        `discriminator` for the words that tell it apart. A twin in another
        document with the same cells is a copy, not a twin: either is valid
        evidence for the same question, so `copies` records it instead.
        """
        groups: dict[tuple, list[str]] = defaultdict(list)
        by_values: dict[tuple, list[str]] = defaultdict(list)
        body: dict[str, list[list[str]]] = {}
        for uid, u in self.units.items():
            if u.type != "table" or u.row_index is not None or not u.caption:
                continue
            rows = table_rows(u.text)
            # rows with a label and values, leaving out notes and sub-headings (a label with nothing after it):
            # the EAG's reprint of FAKTION's SAE table adds a "Based on Jones 2020" note row
            body[uid] = [[normalize(c) for c in r] for r in rows[1:] if r and r[0].strip() and any(c.strip() for c in r[1:])]
            labels = frozenset(r[0] for r in body[uid])
            # same rows under the same columns; a different header is told apart by naming the measure. The
            # label column's own title is left out: the same table heads it "Serious adverse event" and "SAE"
            columns = " | ".join(rows[0][1:])
            if len(labels) >= 3:
                groups[(columns, labels)].append(uid)
            # a reprint may also shorten its labels (EAG Table 3.8 has "HR" for Table 7's "Hazard ratio"): the same
            # numbers under the same columns are the same table. Rows of redactions alone match anything, so need a digit.
            values = tuple(tuple(r[1:]) for r in body[uid])
            if sum(any(re.search(r"\d", c) for c in r) for r in values) >= 3:
                by_values[(columns, values)].append(uid)
        twins: dict[str, list[str]] = {}
        copies: dict[str, list[str]] = {}
        for members in groups.values():
            if len(members) < 2:
                continue
            cells = {m: body[m] for m in members}
            for uid in members:
                same = [o for o in members if o != uid and self.units[o].doc_id != self.units[uid].doc_id and cells[o] == cells[uid]]
                other = [o for o in members if o != uid and o not in same]
                if same:
                    copies[uid] = same
                if other:
                    twins[uid] = other
        for members in by_values.values():
            for uid in members:
                same = [o for o in members if self.units[o].doc_id != self.units[uid].doc_id]
                if same:
                    copies[uid] = sorted(set(copies.get(uid, [])) | set(same))
                    twins[uid] = [t for t in twins.get(uid, []) if t not in same]
                    if not twins[uid]:
                        del twins[uid]
        return twins, copies

    def discriminator(self, unit: Unit) -> list[str] | None:
        """Phrases that set this table apart from each twin; None without twins; "" for a twin it cannot be told from.

        Each phrase is the shortest run of caption words around a difference
        that does not occur in the twin's caption ("fulvestrant arm" against
        "capivasertib arm"). The table number is left out: questions must not
        rely on it. When the captions say nothing and the twin is in another
        document, the document does ("company submission" against
        "clarification", from roles.mention).
        """
        twins = self.twins.get(unit.unit_id)
        if not twins:
            return None
        phrases = {self.distinguish(unit, self.units[t]) for t in twins}
        # "Scenario" against one twin and "Scenario 1" against another: the longer phrase covers both
        return sorted(p for p in phrases if not any(o != p and p.lower() in o.lower() for o in phrases))

    def distinguish(self, unit: Unit, other: Unit) -> str:
        """The phrase that sets this table apart from one twin: caption words, or failing those its document; "" when nothing does."""
        phrase = _distinct_phrase(_caption_words(unit.caption), _caption_words(other.caption))
        if not phrase and other.doc_id != unit.doc_id:
            mention = self.cfg["roles"].get("mention", {})
            own, theirs = mention.get(self.docs[unit.doc_id].role, ""), mention.get(self.docs[other.doc_id].role, "")
            phrase = own if own and own != theirs else ""
        return phrase

    # ----- queries -----
    def unit_for_element(self, element_id: str) -> Unit | None:
        """The whole-element unit, or None when the element is noise or was split into rows."""
        u = self.units.get(element_id)
        return u if u and u.evidence else None

    def resolve_references(self, unit: Unit) -> list[tuple[str, str, str]]:
        """(kind, number, element_id) for every "see Table N" in the unit that resolves in its document."""
        out, seen = [], set()
        for m in REFERENCE_RE.finditer(unit.plain):
            key = (unit.doc_id, m.group(1).lower(), m.group(2))
            if key in seen:
                continue
            seen.add(key)
            target = self.captions.get(key)
            if target and target != unit.element_id:
                out.append((key[1], key[2], target))
        return out

    def rare_tokens(self, unit: Unit) -> set[str]:
        limit = self.cfg["attribution"]["rare_df_max"] * len(self.units)
        return {t for t in set(unit.tokens) if len(t) >= 3 and self.df[t] <= limit}

    def neighbours(self, unit: Unit, n: int) -> list[Unit]:
        sibs = self.by_section.get(unit.section_id, [])
        if unit.unit_id not in sibs:
            return []
        i = sibs.index(unit.unit_id)
        picked = [uid for uid in sibs[max(0, i - n // 2): i + n // 2 + 1] if uid != unit.unit_id][:n]
        return [self.units[u] for u in picked]

    def preceding(self, unit: Unit, n: int) -> list[Unit]:
        """Up to n units right before this one in its section, from other elements (not earlier rows of its own table)."""
        sibs = self.by_section.get(unit.section_id, [])
        if unit.unit_id not in sibs:
            return []
        before = [self.units[u] for u in sibs[: sibs.index(unit.unit_id)] if self.units[u].element_id != unit.element_id]
        return before[-n:] if n else []

    def doc(self, unit: Unit) -> Document:
        return self.docs[unit.doc_id]

    def render(self, unit: Unit, max_chars: int = 3000) -> str:
        """Context header plus the unit's plain text, cut at a line boundary."""
        d = self.doc(unit)
        section = " > ".join(" ".join(t.split())[:80] for t in unit.section_path[-3:]) or "-"
        head = f"[{unit.unit_id}] Document: {d.label} ({unit.doc_id}) | Section: {section} | Pages: {unit.page_start}-{unit.page_end}"
        body = unit.plain
        if unit.caption and not body.startswith(normalize(unit.caption)[:10]) and unit.row_index is not None:
            body = f"{unit.caption}\n{body}"
        if len(body) > max_chars:
            cut = body.rfind("\n", 0, max_chars)
            body = body[: cut if cut > max_chars // 2 else max_chars] + "\n[...]"
        return f"{head}\n{body}"

    def stats(self) -> dict:
        return {
            "units": len(self.units),
            "evidence_units": sum(u.evidence for u in self.units.values()),
            "row_units": sum(u.row_index is not None for u in self.units.values()),
            "noise": dict(self.noise),
            "roles": {d.doc_id: d.role for d in self.docs.values()},
            "captions": len(self.captions),
            "entities": {self.entity_names[k]: len(v) for k, v in sorted(self.entities.items(), key=lambda kv: -len(kv[1]))[:15]},
            "topic": self.topic,
        }


def load_corpus(interim_dir: Path, cfg: dict) -> Corpus:
    interim_dir = Path(interim_dir)
    for name in ("elements.jsonl", "sections.json", "documents.json"):
        if not (interim_dir / name).exists():
            raise FileNotFoundError(f"{interim_dir / name} is missing; run `medrageval parse` first")
    elements = [json.loads(line) for line in (interim_dir / "elements.jsonl").open()]
    sections = json.loads((interim_dir / "sections.json").read_text())
    docs = json.loads((interim_dir / "documents.json").read_text())
    return Corpus(interim_dir, docs, sections, elements, cfg)
