"""Candidate selection from the fact sheet: one seeded selector per question type, no LLM.

The system decides what each item is about, and the LLM writes it. Every
selector yields Candidates lazily, in an order that is deterministic for the
seed. The order is weighted across schema groups, so the schema's weights set
the topic mix rather than page count, and spread across documents and
sections.

A fact that cannot be told apart from a sibling (`FactSheet.must_mention` is
None) is never used: a question about it would have two right answers, as
TA1063's two "Scenario 1" tables gave the v14 dataset.
"""

import re
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from itertools import combinations
from random import Random

from medrageval.facts.factsheet import NUMBER, Fact, FactSheet, content_tokens, jaccard_tokens
from medrageval.generate.corpus import REDACTED, Corpus, Unit, tokenize
from medrageval.text import normalize

# A word mixing ASCII letters with two or more non-ASCII ones ("yéárs") is an OCR glyph error. One accent ("naïve") is a word.
GARBLED = re.compile(r"\b(?=[^\W\d_]*[A-Za-z])(?=(?:[^\W\d_]*[^\x00-\x7F\W\d_]){2})[^\W\d_]+\b")


def garbled(text: str) -> bool:
    return bool(GARBLED.search(text))


@dataclass
class Candidate:
    type: str
    subtype: str | None
    units: list[Unit]
    hint: dict = field(default_factory=dict)
    facts: list[Fact] = field(default_factory=list)

    @property
    def candidate_id(self) -> str:
        ids = [f.fact_id for f in self.facts] or [u.unit_id for u in self.units]
        return f"{self.type}:{'+'.join(ids)}"


# ----- ordering -----
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


def weighted(items: list, group_of: Callable, weights: dict[str, float], rng: Random, key: Callable, cap: int | None) -> Iterator:
    """Interleave schema groups in proportion to their weights (smooth weighted round robin); within a group, `spread` by key.

    A group that runs out leaves the rotation and the others share its turns,
    so a bundle without, say, an indirect comparison still fills its target.
    Groups of weight 0 are left out.
    """
    by_group: dict[str, list] = defaultdict(list)
    for it in items:
        by_group[group_of(it)].append(it)
    iters = {g: spread(by_group[g], key, rng, cap) for g in sorted(by_group) if weights.get(g, 0) > 0}
    current = {g: 0.0 for g in iters}
    while current:
        total = sum(weights[g] for g in current)
        for g in current:
            current[g] += weights[g]
        g = max(sorted(current), key=lambda k: current[k])
        current[g] -= total
        try:
            yield next(iters[g])
        except StopIteration:
            del current[g]


def section_key(u: Unit) -> tuple:
    return (u.doc_id, u.section_id)


# ----- shared tests -----
def native(u: Unit, cfg: dict) -> bool:
    return cfg["limits"]["allow_ocr"] or u.source != "ocr"


def usable(f: Fact, sheet: FactSheet, cfg: dict) -> bool:
    """Askable, distinguishable from its siblings, from native text, readable."""
    return (sheet.askable(f) and not sheet.ambiguous(f) and all(native(u, cfg) for u in sheet.units(f))
            and not garbled(f.value_text()))


def first_number(value: str) -> float | None:
    m = NUMBER.search((value or "").replace(REDACTED, " "))
    try:
        return float(m.group(0).replace(",", "")) if m else None
    except ValueError:
        return None


def meaningful(a: float | None, b: float | None, threshold: float) -> bool:
    """Two values a reader would care to compare: both numbers, differing by at least `threshold` of the larger."""
    if a is None or b is None or a == b:
        return False
    return abs(a - b) / max(abs(a), abs(b)) >= threshold


_CODE = re.compile(r"\b[A-Za-z]+-?\d+[A-Za-z]?\b")


def _codes(text: str) -> set[str]:
    """Identifiers written as letters and digits ("PFS2", "DCO1", "SOLAR-1"); the tokenizer splits them, so read the raw text."""
    return {m.replace("-", "").lower() for m in _CODE.findall(text or "")}


def small_count(value: str, limit: float) -> bool:
    """A count below `limit`: an integer not followed by "%" ("1 (1%)", "0"), not a rate, a median or a cost."""
    m = re.search(r"\d[\d,]*(?:\.\d+)?", value or "")
    if not m or "." in m.group(0) or value[m.end():].lstrip().startswith("%"):
        return False
    return float(m.group(0).replace(",", "")) < limit


def too_small(a: str, b: str, cfg: dict) -> bool:
    """Both sides are small counts: 1 pleural-effusion SAE against 0 (a fresh smoke-run comparison) is no finding."""
    return small_count(a, cfg["comparative"]["min_count"]) and small_count(b, cfg["comparative"]["min_count"])


def labels_match(a: str, b: str) -> bool:
    """One label's words contain the other's ("PFS" in "investigator-assessed progression-free survival (PFS)").
    Identifiers must agree: "PFS" does not match "PFS2" (a second smoke-run cross-document item set the two side by side)."""
    if _codes(a) != _codes(b) and (_codes(a) or _codes(b)):
        return False
    ta, tb = content_tokens(a), content_tokens(b)
    if not ta or not tb or not (ta <= tb or tb <= ta):
        return False
    return any(not t.replace(".", "").isdigit() for t in ta & tb)


_CONNECTORS = {"+", "plus", "with", "and"}


def same_arm(a: str, b: str) -> bool:
    """Two arm or technology labels for the same regimen: "Everolimus + exemestane", "everolimus with exemestane"."""
    ta, tb = content_tokens(a) - _CONNECTORS, content_tokens(b) - _CONNECTORS
    return bool(ta and tb) and (ta <= tb or tb <= ta) and _codes(a) == _codes(b)


def same_topic(a: Fact, b: Fact) -> bool:
    """Two stance facts about one issue: their topics share most words (each document phrases the issue its own way)."""
    ta, tb = a.qualifiers.get("topic", ""), b.qualifiers.get("topic", "")
    return bool(ta and tb) and jaccard_tokens(ta, tb) >= 0.4


def link_match(value: str, label: str) -> bool:
    """A bridge value names a qualifier: one's words contain the other's ("PFS" in "progression-free survival (PFS)"),
    or they share two words ("generalised gamma distribution" in "Scenario 1 – generalised gamma for OS").
    Identifiers must agree: "PFS2 (second progression-free survival)" shares two words with "progression-free
    survival (PFS)" and is another endpoint; the first smoke run bridged to it."""
    ca, cb = _codes(value), _codes(label)
    if cb - ca or (ca - cb and cb):
        return False
    ta, tb = content_tokens(value), content_tokens(label)
    shared = {t for t in ta & tb if not t.replace(".", "").isdigit()}
    return bool(shared) and (ta <= tb or tb <= ta or len(shared) >= 2)


def compatible(a: Fact, b: Fact, skip: set[str] = frozenset()) -> bool:
    """Qualifiers both facts give agree (one contains the other's words): the two are about the same trial, population, owner."""
    for k in (set(a.qualifiers) & set(b.qualifiers)) - set(skip):
        va, vb = a.qualifiers[k], b.qualifiers[k]
        if normalize(va) != normalize(vb) and not labels_match(va, vb):
            return False
    return True


def shares_subject(a: Fact, b: Fact) -> bool:
    """At least one qualifier both give, and it agrees; or neither has any."""
    common = set(a.qualifiers) & set(b.qualifiers)
    if not (a.qualifiers or b.qualifiers):
        return True
    return any(normalize(a.qualifiers[k]) == normalize(b.qualifiers[k]) or labels_match(a.qualifiers[k], b.qualifiers[k]) for k in common)


def without(sets: list[list[str]] | None, *terms: str) -> list[list[str]]:
    """Discriminator sets that naming any of `terms` does not already satisfy."""
    terms = [t for t in terms if t]
    out = []
    for s in sets or []:
        if not any(normalize(t) == normalize(x) or labels_match(t, x) for t in terms for x in s) and s not in out:
            out.append(s)
    return out


def fill(template: str, quals: dict[str, str]) -> str | None:
    """A link's description with the source fact's qualifiers; None when one it needs is missing."""
    names = re.findall(r"\{(\w+)\}", template)
    if any(not quals.get(n) for n in names):
        return None
    return template.format(**{n: quals[n] for n in names})


def _hint(sheet: FactSheet, f: Fact, **extra) -> dict:
    return {"group": f.group, "slot": f.slot, "must_mention": sheet.must_mention(f) or [], **extra}


def _units(sheet: FactSheet, *facts: Fact) -> list[Unit]:
    out: dict[str, Unit] = {}
    for f in facts:
        for u in sheet.units(f):
            out.setdefault(u.unit_id, u)
    return list(out.values())


def _order(sheet: FactSheet, items: list, cfg: dict, rng: Random, fact_of: Callable = lambda x: x) -> Iterator:
    return weighted(items, lambda it: fact_of(it).group, sheet.schema.weights(), rng,
                    lambda it: section_key(sheet.units(fact_of(it))[0]), cfg["limits"]["max_per_section"])


# ----- selectors -----
def select_factual(corpus: Corpus, sheet: FactSheet, cfg: dict, rng: Random) -> Iterator[Candidate]:
    facts = [f for f in sheet.facts.values() if f.kind == "text" and not f.redacted and usable(f, sheet, cfg)]
    for f in _order(sheet, facts, cfg, rng):
        yield Candidate("factual", f.group, sheet.units(f), _hint(sheet, f), [f])


def select_numerical(corpus: Corpus, sheet: FactSheet, cfg: dict, rng: Random) -> Iterator[Candidate]:
    items = []
    for f in sorted(sheet.facts.values(), key=lambda f: f.fact_id):
        if f.kind == "number" and not f.redacted and usable(f, sheet, cfg) and f.numeric_values():
            label, value = rng.choice(f.numeric_values())
            items.append((f, label, value))
    for f, label, value in _order(sheet, items, cfg, rng, lambda x: x[0]):
        numbers = sorted({n.replace(",", "") for n in NUMBER.findall(value)})
        yield Candidate("numerical", f.group, sheet.units(f),
                        _hint(sheet, f, label=label if f.values else None, value=value, numbers=numbers), [f])


def select_comparative(corpus: Corpus, sheet: FactSheet, cfg: dict, rng: Random) -> Iterator[Candidate]:
    """Two values of one measure: two arms of one fact, or two facts of a slot in one document that differ in one qualifier."""
    threshold = cfg["comparative"]["min_relative_difference"]
    items = []
    for f in sorted(sheet.facts.values(), key=lambda f: f.fact_id):
        if f.kind != "number" or f.redacted or not f.values or not usable(f, sheet, cfg):
            continue
        pairs = [(a, b) for a, b in combinations(f.numeric_values(), 2)
                 if meaningful(first_number(a[1]), first_number(b[1]), threshold) and not too_small(a[1], b[1], cfg)]
        if pairs:
            (la, va), (lb, vb) = rng.choice(pairs)
            items.append(([f], {"labels": [la, lb], "values": [va, vb], "measure": sheet.describe(f), "mode": "arms",
                                "must_mention": sheet.must_mention(f) or []}))
    for slot in sorted(sheet.by_slot):
        singles = [f for f in sheet.by_slot[slot] if f.kind == "number" and not f.redacted and not f.values and f.value and usable(f, sheet, cfg)]
        for a, b in combinations(singles, 2):
            if a.doc_id != b.doc_id or set(a.qualifiers) != set(b.qualifiers):
                continue
            diff = [k for k in a.qualifiers if normalize(a.qualifiers[k]) != normalize(b.qualifiers[k])]
            # one measure in two arms, data cuts, populations or scenarios; not two different things (alpelisib's
            # acquisition cost against fulvestrant's, a first smoke-run item)
            if len(diff) != 1 or diff[0] not in cfg["comparative"]["comparable_qualifiers"]:
                continue
            if not meaningful(first_number(a.value), first_number(b.value), threshold) or too_small(a.value, b.value, cfg):
                continue
            k = diff[0]
            la, lb = a.qualifiers[k], b.qualifiers[k]
            items.append(([a, b], {"labels": [la, lb], "values": [a.value, b.value], "measure": sheet.describe(a), "mode": "facts",
                                   "differ_by": k, "must_mention": without((sheet.must_mention(a) or []) + (sheet.must_mention(b) or []), la, lb)}))
    for facts, hint in _order(sheet, items, cfg, rng, lambda x: x[0][0]):
        yield Candidate("comparative", hint["mode"], _units(sheet, *facts), {"group": facts[0].group, "slot": facts[0].slot, **hint}, facts)


def select_multihop(corpus: Corpus, sheet: FactSheet, cfg: dict, rng: Random) -> Iterator[Candidate]:
    """A schema link: fact A's value names what fact B is qualified by; the question reaches B through A's description."""
    items = []
    for link in sheet.schema.links:
        for a in sheet.by_slot.get(link.source, []):
            if a.redacted or not a.value or not usable(a, sheet, cfg):
                continue
            bridge = fill(link.describe, a.qualifiers)
            if not bridge:
                continue
            bridge_words = {t for t in content_tokens(re.sub(r"\{\w+\}", " ", link.describe)) if len(t) > 3}
            for b in sheet.by_slot.get(link.target, []):
                via = b.qualifiers.get(link.via)
                if (not via or b.redacted or not usable(b, sheet, cfg) or not link_match(a.value, via)
                        or not compatible(a, b, {link.via}) or set(a.unit_ids) & set(b.unit_ids)):
                    continue
                # B's passage already says what the bridge describes ("the primary endpoint ... PFS ... 0.50"): one excerpt
                # answers. In the first smoke run the judge failed or the writer skipped 13 of 17 such pairs.
                if bridge_words and all(bridge_words <= set(u.tokens) for u in sheet.units(b)):
                    continue
                # without a sibling of another `via` value, the slot alone gives the answer and A is no hop
                if not any(g is not b and g.qualifiers.get(link.via) and not link_match(a.value, g.qualifiers[link.via])
                           for g in sheet.by_slot[link.target]):
                    continue
                items.append((a, b, link, bridge, via))
    by_link: dict[str, list] = defaultdict(list)
    for it in items:
        by_link[f"{it[2].source}->{it[2].target}"].append(it)
    # round robin over links: the second smoke run took all four items through one link
    orders = [list(_order(sheet, by_link[k], cfg, rng, lambda x: x[1])) for k in sorted(by_link)]
    for a, b, link, bridge, via in interleave(orders):
        hint = {"group": b.group, "slot": b.slot, "link": f"{link.source} -> {link.target}", "bridge": bridge, "bridge_value": a.value,
                "via": link.via, "via_value": via, "must_mention": without(sheet.must_mention(b), via, a.value)}
        yield Candidate("multihop", "bridge", _units(sheet, a, b), hint, [a, b])


def select_crossdoc(corpus: Corpus, sheet: FactSheet, cfg: dict, rng: Random) -> Iterator[Candidate]:
    """Two documents of different roles giving different values for the same subject: the company's figure against the EAG's."""
    mention = cfg["roles"].get("mention", {})
    items, seen = [], set()
    for slot in sorted(sheet.by_slot):
        # figures and positions, which can disagree; two documents' free-text descriptions of a topic only differ in
        # wording, and the first smoke run turned them into "how do X and Y characterise ..." items
        ok = [f for f in sheet.by_slot[slot] if f.kind in ("number", "stance") and not f.redacted and usable(f, sheet, cfg)]
        for f in ok:
            conflicts = {g.fact_id for g in sheet.conflicts(f)}
            for g in ok:
                pair = tuple(sorted((f.fact_id, g.fact_id)))
                # whose analysis it is differs by design: that is the comparison
                if g.doc_id == f.doc_id or g.role == f.role or pair in seen or g.fact_id not in conflicts:
                    continue
                if f.kind == "stance":
                    if not same_topic(f, g):
                        continue
                elif not compatible(f, g, {"owner"}) or not shares_subject(f, g):
                    continue
                seen.add(pair)
                if f.kind == "number" and not (f.numbers() - g.numbers() and g.numbers() - f.numbers()):
                    continue   # one document only adds figures to the other's: they agree on what both report
                if f.values and g.values and not any(same_arm(a, b) for a in f.values for b in g.values):
                    continue   # different arms: alpelisib's diarrhoea count against capivasertib's is no disagreement
                docs = [mention.get(f.role, sheet.doc_label(f)), mention.get(g.role, sheet.doc_label(g))]
                sets = without((sheet.must_mention(f) or []) + (sheet.must_mention(g) or []), *docs)
                items.append((f, g, {"docs": docs, "must_mention": sets}))
    for f, g, hint in _order(sheet, items, cfg, rng, lambda x: x[0]):
        yield Candidate("crossdoc", f.group, _units(sheet, f, g), {"group": f.group, "slot": f.slot, **hint}, [f, g])


def speaker_for(party: str, cfg: dict) -> str | None:
    """The canonical speaker label ("the EAG") for a party name or alias."""
    p = party.strip().lower()
    for label, aliases in cfg["attribution"]["speakers"].items():
        if p in {label.lower(), label.lower().removeprefix("the ")} | {a.lower() for a in aliases}:
            return label
    return None


def select_attribution(corpus: Corpus, sheet: FactSheet, cfg: dict, rng: Random) -> Iterator[Candidate]:
    """Who holds a position: a party's side of a disputed point, or a stakeholder's view in its own document.

    Disputed positions come first: telling two parties apart is the test,
    guessing the speaker from the topic is not.
    """
    ac = cfg["attribution"]
    contested, rest = [], []
    for f in sorted(sheet.facts.values(), key=lambda f: f.fact_id):
        if f.redacted or not sheet.askable(f) or not all(native(u, cfg) for u in sheet.units(f)):
            continue
        if f.kind == "stance" and f.positions:
            for party, pos in f.positions.items():
                speaker = speaker_for(party, cfg)
                others = [p for k, p in f.positions.items() if k != party]
                if speaker and not any(jaccard_tokens(pos, o) >= 0.5 for o in others):   # parties that agree are no attribution item
                    (contested if others else rest).append((f, "stance", {"speaker": speaker, "claim": pos, "contested": bool(others)}))
        elif f.kind == "text" and f.value:
            speaker = ac["role_speakers"].get(f.role)
            if speaker and speaker in ac["own_voice_only"]:
                rest.append((f, "form", {"speaker": speaker, "claim": f.value, "speaker_name": f.speaker}))
    for items in (contested, rest):
        for f, sub, hint in _order(sheet, items, cfg, rng, lambda x: x[0]):
            yield Candidate("attribution", sub, sheet.units(f), {"group": f.group, "slot": f.slot, **hint}, [f])


def _by_group_doc(sheet: FactSheet) -> dict[tuple[str, str], list[Fact]]:
    out: dict[tuple[str, str], list[Fact]] = defaultdict(list)
    for f in sheet.facts.values():
        out[(f.group, f.doc_id)].append(f)
    return out


def select_summary(corpus: Corpus, sheet: FactSheet, cfg: dict, rng: Random) -> Iterator[Candidate]:
    """Several facts of one group in one document: a question whose full answer covers them all."""
    sc = cfg["summary"]
    items = []
    for (group, doc_id), facts in sorted(_by_group_doc(sheet).items()):
        ok = [f for f in facts if not f.redacted and usable(f, sheet, cfg)]
        if len(ok) < sc["min_facts"]:
            continue
        by_slot: dict[str, list[Fact]] = defaultdict(list)
        for f in ok:
            by_slot[f.slot].append(f)
        picked: list[Fact] = []
        queues = [list(v) for _, v in sorted(by_slot.items())]
        while queues and len(picked) < sc["max_facts"]:   # spread the picks over slots
            for q in list(queues):
                if len(picked) < sc["max_facts"]:
                    picked.append(q.pop(rng.randrange(len(q))))
                if not q:
                    queues.remove(q)
        picked.sort(key=lambda f: corpus.position[f.unit_ids[0]])
        items.append(picked)
    for facts in _order(sheet, items, cfg, rng, lambda x: x[0]):
        f = facts[0]
        hint = {"group": f.group, "group_describe": sheet.schema.groups[f.group].describe, "doc": sheet.doc_label(f), "speaker_name": f.speaker}
        yield Candidate("summary", f.group, _units(sheet, *facts), hint, facts)


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


def select_unanswerable_redacted(corpus: Corpus, sheet: FactSheet, cfg: dict, rng: Random) -> Iterator[Candidate]:
    """A value the bundle redacts and prints nowhere else: the right response is that it is not disclosed."""
    items = []
    for f in sorted(sheet.facts.values(), key=lambda f: f.fact_id):
        if not f.redacted or not sheet.askable(f) or not all(native(u, cfg) for u in sheet.units(f)):
            continue
        quals = {k: normalize(v) for k, v in f.qualifiers.items()}
        if any(not g.redacted and {k: normalize(v) for k, v in g.qualifiers.items()} == quals for g in sheet.by_slot[f.slot]):
            continue   # the same fact, disclosed in another table
        if any(stated_elsewhere(corpus, u, cfg) for u in sheet.units(f)):
            continue
        label = sheet.describe(f) + (" (" + "; ".join(f"{k}: {v}" for k, v in f.qualifiers.items()) + ")" if f.qualifiers else "")
        items.append((f, label))
    for f, label in _order(sheet, items, cfg, rng, lambda x: x[0]):
        yield Candidate("unanswerable", "redacted", sheet.units(f), {"group": f.group, "slot": f.slot, "labels": [label]}, [f])


def select_unanswerable_not_covered(corpus: Corpus, sheet: FactSheet, cfg: dict, rng: Random) -> Iterator[Candidate]:
    """A fact as anchor: near misses ask for the same measure for a version of it the bundle does not report."""
    facts = [f for f in sheet.facts.values() if f.kind == "number" and not f.redacted and f.qualifiers and usable(f, sheet, cfg)]
    for f in _order(sheet, sorted(facts, key=lambda f: f.fact_id), cfg, rng):
        yield Candidate("unanswerable", "not_covered", sheet.units(f)[:1],
                        {"group": f.group, "slot": f.slot, "topic": corpus.topic, "reported": sheet.reported(f)}, [f])


SELECTORS: dict[str, Callable[[Corpus, FactSheet, dict, Random], Iterator[Candidate]]] = {
    "factual": select_factual,
    "numerical": select_numerical,
    "comparative": select_comparative,
    "multihop": select_multihop,
    "crossdoc": select_crossdoc,
    "attribution": select_attribution,
    "summary": select_summary,
    "unanswerable_redacted": select_unanswerable_redacted,
    "unanswerable_not_covered": select_unanswerable_not_covered,
}


def candidate_counts(corpus: Corpus, sheet: FactSheet, cfg: dict, seed: int, examples: int = 2) -> dict:
    """Exhaust every selector (no LLM): counts by group and a few examples, for --dry-run."""
    out = {}
    for name, sel in SELECTORS.items():
        counts, samples = defaultdict(int), []
        for c in sel(corpus, sheet, cfg, Random(seed)):
            counts[c.hint.get("group") or "-"] += 1
            if len(samples) < examples:
                samples.append({"facts": [f.fact_id for f in c.facts], "subtype": c.subtype})
        out[name] = {"total": sum(counts.values()), "by_group": dict(counts), "examples": samples}
    return out
