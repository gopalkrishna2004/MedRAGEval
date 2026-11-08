"""The corpus that fact extraction reads and question generation quotes.

Loads one parse output directory (elements.jsonl, sections.json,
documents.json) and builds what the later stages need:

- evidence units: elements that passed the noise filter, with long tables
  split into row groups so a form's "question | answer" row can be cited on
  its own. A row unit keeps its parent's element_id, and its lines are lines
  of the parent's plain text, so a quote from it is also a quote from the
  element.
- a role for every document (company submission, EAG report, ...), scored
  from headings, running text and the first elements against config patterns.
- a BM25 index over the units, used to pick passages for extraction, to find
  copies of a fact in other documents, for attribution exclusivity and for the
  "is it really unanswerable" check.
"""

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from medrageval.text import normalize, table_rows

STOPWORDS = set("""
a an and are as at be been being by for from has have had he her his in into is it its of on or that the their them
there these they this those to was were which while with within will would can could should may might not no nor
than then so such but if also any all each both into onto over under between about after before during per vs versus
""".split())

CAPTION_RE = re.compile(r"^[\W_]*(Table|Figure)\s+([A-Z]?\d+(?:\.\d+)?)\b", re.I)   # "_Table 14: ..._": an italic caption
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
        if len(docs) > 1 and scores.get(role, 0) < 2 and len(by_doc[d["doc_id"]]) <= rc["cover_max_units"]:
            role = "cover"  # no heading hit and few elements in a bundle: its cover or contents page (a PDF of one short document is that document)
        role = rc.get("override", {}).get(d["doc_id"], role)
        out[d["doc_id"]] = (role, {r: s for r, s in scores.items() if s})
    return out


_GENERIC_TITLE = re.compile(r"^(national institute|single technology|committee papers|document [a-z]\b|produced by)", re.I)


def document_title(d: dict, section_titles: list[str] = ()) -> str:
    """The document's first telling heading. A document whose first pages gave no heading (a journal article set in
    columns) takes its first telling section title before its running text, which is a citation line
    ("Pape et al. BMC Psychology (2024) 12:626") rather than the title."""
    for h in d["headings"] or section_titles:
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


class Corpus:
    def __init__(self, interim_dir: Path, docs: list[dict], sections: list[dict], elements: list[dict], cfg: dict):
        self.interim_dir = interim_dir
        self.cfg = cfg
        self.sections = {s["section_id"]: s for s in sections}
        self.elements = {e["element_id"]: e for e in elements}

        roles = detect_roles(docs, elements, cfg)
        self.docs: dict[str, Document] = {}
        for d in docs:
            role, _ = roles[d["doc_id"]]
            titles = [" ".join(s["title"].split()) for s in sections if s["doc_id"] == d["doc_id"] and s.get("depth", 1) <= 2]
            self.docs[d["doc_id"]] = Document(d["doc_id"], role, cfg["roles"]["labels"].get(role, role),
                                              document_title(d, titles), d["start_page"], d["end_page"])
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
