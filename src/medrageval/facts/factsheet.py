"""Facts extracted from one bundle, and what tells each one apart from its siblings.

A fact is one slot of the schema filled from one passage: its qualifiers
(trial, population, data cut, scenario, whose analysis), its value or values,
and the verbatim quote it rests on. The fact sheet is what questions are
written from. The question writer sees a clean fact instead of a table's
layout, and the sheet knows every other fact of the same slot, so it can say
which qualifiers a question must name to mean exactly one of them.

TA1063 prints two tables captioned "Scenario 1" with different life years.
The fact sheet gives them qualifiers "Scenario 1" and "Scenario 1 – generalised
gamma for OS". Naming "Scenario 1" names both, so the first fact cannot be
told apart and is not used; the second can, by its longer name.
"""

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from medrageval.facts.schema import Schema
from medrageval.generate.corpus import REDACTED, Corpus, Unit, tokenize
from medrageval.text import contains_quote, normalize

NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def numbers_in(text: str | None) -> set[str]:
    """Numbers as written, without thousands separators: "£24,567" -> "24567"."""
    return {n.replace(",", "") for n in NUMBER.findall(text or "")}


def content_tokens(text: str | None) -> set[str]:
    return set(tokenize(text or ""))


def jaccard_tokens(a: str, b: str) -> float:
    sa, sb = content_tokens(a), content_tokens(b)
    return len(sa & sb) / max(len(sa | sb), 1)


def _stem(t: str) -> str:
    return t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t


def spells_acronym(acronym: str, text: str) -> bool:
    """"OS" is named by "overall survival", "PFS" by "progression-free survival": consecutive words' initials."""
    a = acronym.lower()
    words = [w for w in normalize(text).split() if w.isalpha()]
    return any("".join(w[0] for w in words[i:i + len(a)]) == a for i in range(len(words) - len(a) + 1))


def term_mentioned(term: str, text: str) -> bool:
    """Does the text name the term? Every token with a digit ("1" in "Scenario 1", "DCO1"), and most of its words.

    Questions paraphrase: "PI3K/AKT-altered" for "PI3K/AKT pathway-altered
    population" still names the population, "overall survival" names "OS",
    "event" names "events". Identifiers may not be paraphrased: "Scenario 2"
    does not name "Scenario 1".
    """
    if contains_quote(term, text):
        return True
    if re.fullmatch(r"[A-Z]{2,6}", term.strip()) and spells_acronym(term.strip(), text):
        return True
    toks = tokenize(term)
    if not toks:
        return False
    have = {_stem(t) for t in content_tokens(text)}
    codes = [t for t in toks if any(c.isdigit() for c in t)]
    words = [_stem(t) for t in toks if t not in codes]
    if any(c not in have for c in codes):
        return False
    return not words or sum(w in have for w in words) / len(words) >= 0.6


@dataclass
class Fact:
    fact_id: str
    slot: str                      # "group.name"
    kind: str
    doc_id: str
    role: str
    evidence: list[dict]           # [{"unit_id": str, "quote": str}]
    qualifiers: dict[str, str] = field(default_factory=dict)
    value: str | None = None
    values: dict[str, str] | None = None
    positions: dict[str, str] | None = None
    redacted: bool = False
    also_in: list[str] = field(default_factory=list)
    speaker: str | None = None     # the respondent named in a stakeholder document, set when the sheet is loaded

    @property
    def group(self) -> str:
        return self.slot.split(".", 1)[0]

    @property
    def name(self) -> str:
        return self.slot.split(".", 1)[1]

    @property
    def unit_ids(self) -> list[str]:
        return [e["unit_id"] for e in self.evidence]

    @property
    def quote(self) -> str:
        return "\n".join(e["quote"] for e in self.evidence)

    def value_text(self) -> str:
        if self.values:
            return "; ".join(f"{k}: {v}" for k, v in self.values.items())
        if self.positions:
            return "; ".join(f"{k}: {v}" for k, v in self.positions.items())
        return self.value or ""

    def numbers(self) -> set[str]:
        return numbers_in(self.value_text().replace(REDACTED, " "))

    def numeric_values(self) -> list[tuple[str, str]]:
        """(label, value) pairs with a number, not redacted; a single value is labelled by the fact's own name."""
        pairs = list(self.values.items()) if self.values else ([(self.name, self.value)] if self.value else [])
        return [(k, v) for k, v in pairs if v and REDACTED not in v and NUMBER.search(v)]

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k != "speaker"}

    @classmethod
    def from_dict(cls, d: dict) -> "Fact":
        return cls(**{k: d.get(k) for k in ("fact_id", "slot", "kind", "doc_id", "role", "evidence")},
                   qualifiers=d.get("qualifiers") or {}, value=d.get("value"), values=d.get("values"),
                   positions=d.get("positions"), redacted=bool(d.get("redacted")), also_in=d.get("also_in") or [])


class FactSheet:
    """The facts of one bundle, indexed by slot, group and document, with sibling and discriminator queries."""

    def __init__(self, facts: list[Fact], schema: Schema, corpus: Corpus, cfg: dict, fcfg: dict):
        self.schema, self.corpus, self.cfg, self.fcfg = schema, corpus, cfg, fcfg
        missing = [f.fact_id for f in facts for uid in f.unit_ids if uid not in corpus.units]
        if missing:
            raise ValueError(f"{len(missing)} facts cite units the corpus does not have (e.g. {missing[:3]}); "
                             "the parse output changed since `medrageval extract` ran: run it again")
        unknown = sorted({f.slot for f in facts} - set(schema.slots))
        if unknown:
            raise ValueError(f"facts use slots the schema does not have: {unknown}; extract with the same schema")
        self.facts = {f.fact_id: f for f in facts}
        self.by_slot: dict[str, list[Fact]] = defaultdict(list)
        self.by_group: dict[str, list[Fact]] = defaultdict(list)
        self.by_doc: dict[str, list[Fact]] = defaultdict(list)
        for f in facts:
            self.by_slot[f.slot].append(f)
            self.by_group[f.group].append(f)
            self.by_doc[f.doc_id].append(f)
        self.respondents: dict[str, str] = {}
        rs = fcfg.get("respondent_slot")
        for f in self.by_slot.get(rs, []) if rs else []:
            if f.value and not f.redacted:
                self.respondents.setdefault(f.doc_id, f.value)
        for f in facts:
            f.speaker = self.respondents.get(f.doc_id)
        self._mentions: dict[str, list[list[str]] | None] = {}

    @classmethod
    def load(cls, path: Path, schema: Schema, corpus: Corpus, cfg: dict, fcfg: dict) -> "FactSheet":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; run `medrageval extract {path.parent}` first")
        facts = [Fact.from_dict(json.loads(line)) for line in path.open() if line.strip()]
        return cls(facts, schema, corpus, cfg, fcfg)

    def __len__(self) -> int:
        return len(self.facts)

    # ----- per fact -----
    def units(self, f: Fact) -> list[Unit]:
        return [self.corpus.units[u] for u in dict.fromkeys(f.unit_ids)]

    def describe(self, f: Fact) -> str:
        return self.schema.slot(f.slot).describe

    def doc_label(self, f: Fact) -> str:
        return self.corpus.docs[f.doc_id].label

    def askable(self, f: Fact) -> bool:
        return self.schema.slot(f.slot).ask

    # ----- siblings -----
    def _copies(self, f: Fact, g: Fact) -> bool:
        return bool(set(g.unit_ids) & set(f.also_in) or set(f.unit_ids) & set(g.also_in))

    def conflicts(self, f: Fact) -> list[Fact]:
        """Other facts of the same slot with a different value: a question about f must not fit them.

        Two text or stance facts of one slot in one document are not rivals:
        they are complementary statements about one subject (the trial's
        primary endpoint phrased twice, the company's and the EAG's sides of
        one issue). Numbers are rivals whenever they differ.
        """
        out = []
        for g in self.by_slot.get(f.slot, []):
            if g is f or g.redacted or f.redacted or self._copies(f, g):
                continue
            if f.kind == "number":
                differ = f.numbers() != g.numbers()
            else:
                differ = g.doc_id != f.doc_id and jaccard_tokens(f.value_text(), g.value_text()) < self.fcfg["text_conflict_jaccard"]
            if differ:
                out.append(g)
        return out

    def must_mention(self, f: Fact) -> list[list[str]] | None:
        """For each conflicting sibling, the terms of f a question could name to exclude it; None when one cannot be excluded.

        A qualifier excludes a sibling when its value differs and is not
        contained in the sibling's value ("Scenario 1" does not exclude
        "Scenario 1 – generalised gamma for OS"). A sibling in a document of
        another role is also excluded by naming f's document ("EAG",
        "company submission"). Sets that contain a smaller set are dropped:
        naming the smaller one satisfies both.
        """
        if f.fact_id in self._mentions:
            return self._mentions[f.fact_id]
        mention = self.cfg["roles"].get("mention", {})
        sets: list[frozenset[str]] = []
        result: list[list[str]] | None = None
        for g in self.conflicts(f):
            terms = set()
            for k, v in f.qualifiers.items():
                gv = g.qualifiers.get(k)
                if not v:
                    continue
                if gv and content_tokens(v) <= content_tokens(gv):
                    continue
                if gv is None or normalize(v) != normalize(gv):
                    terms.add(v)
            if f.role != g.role and mention.get(f.role) and mention.get(f.role) != mention.get(g.role):
                terms.add(mention[f.role])
            if f.speaker and g.speaker != f.speaker:   # two patient organisations: name the one
                terms.add(f.speaker)
            # rows of one table read as separate facts: alpelisib's 1.78 QALYs against capivasertib's 2.40 differ in their arm
            other_keys = {normalize(k) for k in (g.values or {})}
            terms |= {k for k in (f.values or {}) if other_keys and normalize(k) not in other_keys}
            if not terms:
                self._mentions[f.fact_id] = None
                return None
            sets.append(frozenset(terms))
        minimal = {s for s in sets if not any(o < s for o in sets)}
        result = sorted(sorted(s) for s in minimal)
        self._mentions[f.fact_id] = result
        return result

    def ambiguous(self, f: Fact) -> bool:
        return self.must_mention(f) is None

    def reported(self, f: Fact) -> dict[str, list[str]]:
        """Qualifier values the bundle reports for f's slot: what a near-miss question must avoid."""
        out: dict[str, set[str]] = defaultdict(set)
        for g in self.by_slot.get(f.slot, []):
            for k, v in g.qualifiers.items():
                if v:
                    out[k].add(v)
        return {k: sorted(v) for k, v in out.items()}

    def stats(self) -> dict:
        return {"facts": len(self.facts),
                "by_group": {g: len(fs) for g, fs in sorted(self.by_group.items())},
                "redacted": sum(f.redacted for f in self.facts.values()),
                "ambiguous": sum(self.ambiguous(f) for f in self.facts.values()),
                "respondents": dict(self.respondents)}


def write_facts(path: Path, facts: list[Fact]) -> None:
    with open(path, "w") as f:
        for fact in facts:
            f.write(json.dumps(fact.to_dict(), ensure_ascii=False) + "\n")
