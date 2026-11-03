"""Candidate selection from the fact sheet: one seeded selector per question type, no LLM.

The system decides what each item is about, and the LLM writes it. Every
selector yields Candidates lazily, in an order that is deterministic for the
seed. The order is weighted across schema groups, so the schema's weights set
the topic mix rather than page count, and spread across documents and
sections. The rotation continues from the items earlier types accepted
(`taken`), so the weights hold over the whole dataset rather than within each
type.

A fact that cannot be told apart from a sibling (`FactSheet.must_mention` is
None) is never used: a question about it would have two right answers, as
TA1063's two "Scenario 1" tables gave the v14 dataset.
"""

import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from itertools import combinations
from random import Random

from medrageval.facts.factsheet import NUMBER, Fact, FactSheet, content_tokens, jaccard_tokens, roots
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
def spread(items: list, key: Callable, rng: Random, cap: int | None = None, prefer: Callable | None = None) -> Iterator:
    """Seeded shuffle, then round-robin over key buckets (at most cap per bucket). `prefer` ranks the items to put first (stable)."""
    items = list(items)
    rng.shuffle(items)
    if prefer is not None:
        items.sort(key=prefer)
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


def weighted(items: list, group_of: Callable, weights: dict[str, float], rng: Random, key: Callable, cap: int | None,
             taken: Counter | None = None, prefer: Callable | None = None) -> Iterator:
    """Interleave schema groups in proportion to their weights (smooth weighted round robin); within a group, `spread` by key.

    `taken` counts the items earlier types accepted per group. The rotation
    starts as if it had made those picks, so the group furthest below its
    share comes first and the weights hold across the dataset, not within
    each type. Started from zero, every type opened with its heaviest group,
    and a smoke run of two items per type was cost-effectiveness almost
    throughout. Only groups present here count, so a group that this type
    cannot supply is neither owed nor charged. Ties fall to a seeded order,
    not to the alphabet.
    A group that runs out leaves the rotation and the others share its turns,
    so a bundle without, say, an indirect comparison still fills its target.
    Groups of weight 0 are left out.
    """
    by_group: dict[str, list] = defaultdict(list)
    for it in items:
        by_group[group_of(it)].append(it)
    iters = {g: spread(by_group[g], key, rng, cap, prefer) for g in sorted(by_group) if weights.get(g, 0) > 0}
    order = list(iters)
    rng.shuffle(order)
    rank = {g: i for i, g in enumerate(order)}
    taken = taken or Counter()
    total = sum(weights[g] for g in iters)
    before = sum(taken.get(g, 0) for g in iters)
    current = {g: before * weights[g] - total * taken.get(g, 0) for g in iters}
    while current:
        total = sum(weights[g] for g in current)
        for g in current:
            current[g] += weights[g]
        g = max(current, key=lambda k: (current[k], -rank[k]))
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
    Identifiers must agree: "PFS" does not match "PFS2" (a second smoke run set the two side by side)."""
    if _codes(a) != _codes(b) and (_codes(a) or _codes(b)):
        return False
    ta, tb = content_tokens(a), content_tokens(b)
    if not ta or not tb or not (ta <= tb or tb <= ta):
        return False
    return any(not t.replace(".", "").isdigit() for t in ta & tb)


CONNECTORS = {"+", "plus", "with", "and"}


def same_arm(a: str, b: str) -> bool:
    """Two arm or technology labels for the same regimen: "Everolimus + exemestane", "everolimus with exemestane"."""
    ta, tb = content_tokens(a) - CONNECTORS, content_tokens(b) - CONNECTORS
    return bool(ta and tb) and (ta <= tb or tb <= ta) and _codes(a) == _codes(b)


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


def _order(sheet: FactSheet, items: list, cfg: dict, rng: Random, fact_of: Callable = lambda x: x,
           taken: Counter | None = None, prefer: Callable | None = None) -> Iterator:
    return weighted(items, lambda it: fact_of(it).group, sheet.schema.weights(), rng,
                    lambda it: section_key(sheet.units(fact_of(it))[0]), cfg["limits"]["max_per_section"], taken, prefer)


# ----- selectors -----
def restates_qualifiers(f: Fact) -> bool:
    """The value only repeats the fact's qualifiers: the "Exclusion of subsequent treatment costs" scenario "excludes subsequent
    treatment costs". A question has to name the qualifiers, so it would carry its own answer (v17's factual-001)."""
    value = roots(f.value)
    return bool(value) and len(value & roots(" ".join(f.qualifiers.values()))) / len(value) >= 0.8


def select_factual(corpus: Corpus, sheet: FactSheet, cfg: dict, rng: Random, taken: Counter | None = None) -> Iterator[Candidate]:
    facts = [f for f in sheet.facts.values() if f.kind == "text" and not f.redacted and usable(f, sheet, cfg) and not restates_qualifiers(f)]
    for f in _order(sheet, facts, cfg, rng, taken=taken):
        yield Candidate("factual", f.group, sheet.units(f), _hint(sheet, f), [f])


def select_numerical(corpus: Corpus, sheet: FactSheet, cfg: dict, rng: Random, taken: Counter | None = None) -> Iterator[Candidate]:
    items = []
    for f in sorted(sheet.facts.values(), key=lambda f: f.fact_id):
        if f.kind == "number" and not f.redacted and usable(f, sheet, cfg) and f.numeric_values():
            # not a count of one or two events (v17 asked for the rate of non-cardiac chest pain SAEs: 1 (1%))
            values = [(l, v) for l, v in f.numeric_values() if not small_count(v, cfg["comparative"]["min_count"])]
            if not values:
                continue
            label, value = rng.choice(values)
            items.append((f, label, value))
    for f, label, value in _order(sheet, items, cfg, rng, lambda x: x[0], taken):
        numbers = sorted({n.replace(",", "") for n in NUMBER.findall(value)})
        yield Candidate("numerical", f.group, sheet.units(f),
                        _hint(sheet, f, label=label if f.values else None, value=value, numbers=numbers), [f])


def select_comparative(corpus: Corpus, sheet: FactSheet, cfg: dict, rng: Random, taken: Counter | None = None) -> Iterator[Candidate]:
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
    for facts, hint in _order(sheet, items, cfg, rng, lambda x: x[0][0], taken):
        yield Candidate("comparative", hint["mode"], _units(sheet, *facts), {"group": facts[0].group, "slot": facts[0].slot, **hint}, facts)


def _via_tokens(label: str) -> set[str]:
    return content_tokens(label) - CONNECTORS


def label_names(bridged: str, label: str, mode: str) -> bool:
    """Does a bridge name this label? By the words the source's value shares with it (`value` links) or as one regimen (`qualifier` links)."""
    return link_match(bridged, label) if mode == "value" else (same_arm(bridged, label) or labels_match(bridged, label))


def agree_but_via(b: Fact, g: Fact, via: str, lb: str, lg: str) -> bool:
    """b and g agree on every qualifier both give once each side's own bridged label is taken out of the words:
    "median PFS" and "median PFS2" are one measure, "OS rate" is another."""
    for k in (set(b.qualifiers) & set(g.qualifiers)) - {via}:
        vb, vg = b.qualifiers[k], g.qualifiers[k]
        if normalize(vb) == normalize(vg) or labels_match(vb, vg):
            continue
        tb, tg = content_tokens(vb) - _via_tokens(lb), content_tokens(vg) - _via_tokens(lg)
        if not (tb and tg and (tb <= tg or tg <= tb)):
            return False
    return True


def _overlap(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(len(a | b), 1)


def _labels(f: Fact, via: str, per_keyed: bool) -> list[str]:
    return [l for l, v in (f.values or {}).items() if v] if per_keyed else [f.qualifiers[via]] if f.qualifiers.get(via) else []


def rival_of(sheet: FactSheet, b: Fact, via: str, per_keyed: bool, bridged: str, mode: str, label: str) -> dict | None:
    """What a reader of b's passage could take for b while the bridge is unresolved: another label of b's own values, a fact
    of the slot in the same table or paragraph ("unit"), or one in the same section ("section"). None when the passage
    answers alone, as the PFS tables of TA1063 do for "the primary endpoint": nothing else in them has a median."""
    for other in _labels(b, via, per_keyed):
        if other != label and not label_names(bridged, other, mode):
            return {"tier": "unit", "id": b.fact_id, "label": other}
    elements = {u.element_id for u in sheet.units(b)}
    sections = {(u.doc_id, u.section_id) for u in sheet.units(b)}
    best = None
    for g in sheet.by_slot[b.slot]:
        if g is b or sheet.copies(b, g):
            continue
        for other in _labels(g, via, per_keyed):
            if label_names(bridged, other, mode) or not agree_but_via(b, g, via, label, other):
                continue
            units = sheet.units(g)
            if any(u.element_id in elements for u in units):
                return {"tier": "unit", "id": g.fact_id, "label": other}
            if best is None and any((u.doc_id, u.section_id) in sections for u in units):
                best = {"tier": "section", "id": g.fact_id, "label": other}
    return best


def visible_tokens(sheet: FactSheet, f: Fact) -> set[str]:
    """Words a reader of f's passage sees: the units, their captions and section titles, and the context shown before them."""
    out: set[str] = set()
    for u in sheet.units(f):
        heads = f"{u.caption or ''} {' '.join(u.section_path)}"
        out |= set(u.tokens) | content_tokens(heads) | idents(u.plain) | idents(heads)
        for p in sheet.corpus.preceding(u, sheet.cfg["context"]["units"]):
            out |= set(p.tokens) | idents(p.plain)
    return out


_IDENT = re.compile(r"\b[A-Za-z]+\d+[A-Za-z]*\b")   # "PIK3CA", "TA239", "CDK4": the tokenizer splits them


def idents(text: str) -> set[str]:
    return {m.lower() for m in _IDENT.findall(text or "")}


def description_tokens(sheet: FactSheet, a: Fact, link) -> set[str]:
    """The words of a `qualifier` link's description that carry it: identifiers and words rare in the bundle, without the
    bridged name, the topic's names or connectors. "is a relevant comparator" describes nothing; "only recommended in the
    PIK3CA mutated population" does."""
    corpus = sheet.corpus
    limit = sheet.cfg["multihop"]["rare_df_max"] * len(corpus.units)
    label = a.qualifiers.get(link.source_via, "")
    own = content_tokens(label) | CONNECTORS | content_tokens(corpus.topic)
    words = {t for t in content_tokens(a.value) - own if len(t) > 3 and corpus.df.get(t, 0) <= limit}
    return words | (idents(a.value) - idents(f"{label} {corpus.topic}"))


def select_multihop(corpus: Corpus, sheet: FactSheet, cfg: dict, rng: Random, taken: Counter | None = None) -> Iterator[Candidate]:
    """A schema link: fact A identifies what fact B is about; the question reaches B through A's description.

    A pair is kept only when B's passage holds a rival the description rules
    out, so that passage alone cannot answer: the third smoke run put all 12
    candidates through "the primary endpoint" onto PFS tables that report
    nothing but PFS, and the judge failed every one. Pairs whose rival is in
    B's own table or paragraph come before those whose rival is elsewhere in
    the section (the judge sees the section's context; a retriever may not).
    """
    tiers: dict[str, dict[str, list]] = {"unit": defaultdict(list), "section": defaultdict(list)}
    for link in sheet.schema.links:
        per_keyed = sheet.schema.slot(link.target).per == link.via
        targets = [b for b in sheet.by_slot.get(link.target, []) if not b.redacted and usable(b, sheet, cfg)]
        sources = [a for a in sheet.by_slot.get(link.source, []) if not a.redacted and a.value and usable(a, sheet, cfg)]
        bridge_words = {t for t in content_tokens(re.sub(r"\{\w+\}", " ", link.describe)) if len(t) > 3}
        for a in sources:
            if link.match == "qualifier":
                bridged = a.qualifiers.get(link.source_via, "")
                desc = description_tokens(sheet, a, link)
                # a description two comparators share ("was considered in the model") picks neither
                if not bridged or len(desc) < cfg["multihop"]["min_description_tokens"] or any(
                        o is not a and not label_names(bridged, o.qualifiers.get(link.source_via, ""), "qualifier")
                        and _overlap(desc, description_tokens(sheet, o, link)) >= 0.5 for o in sources):
                    continue
                bridge = fill(link.describe, {**a.qualifiers, "value": a.value})
            else:
                bridged, desc = a.value, bridge_words
                bridge = fill(link.describe, a.qualifiers)
            if not bridge:
                continue
            for b in targets:
                if set(a.unit_ids) & set(b.unit_ids) or not compatible(a, b, {link.via, link.source_via or ""}):
                    continue
                matched = [l for l in _labels(b, link.via, per_keyed) if label_names(bridged, l, link.match)]
                if len(matched) != 1 or (per_keyed and REDACTED in (b.values or {}).get(matched[0], "")):
                    continue
                label = matched[0]
                # B's passage already says what the description says ("Primary endpoint: PFS" above the table): one excerpt answers
                seen = visible_tokens(sheet, b)
                if desc and len(desc & seen) / len(desc) >= (1.0 if link.match == "value" else 0.5):
                    continue
                rival = rival_of(sheet, b, link.via, per_keyed, bridged, link.match, label)
                if rival is None:
                    continue
                value = b.values[label] if per_keyed else (b.value or b.value_text())
                tiers[rival["tier"]][f"{link.source}->{link.target}"].append((a, b, link, bridge, label, value, rival, desc))
    asked: set[tuple[str, str]] = set()
    for tier in ("unit", "section"):
        by_link = tiers[tier]
        # round robin over links: the second smoke run took all four items through one link
        orders = [list(_order(sheet, by_link[k], cfg, rng, lambda x: x[1], taken)) for k in sorted(by_link)]
        for a, b, link, bridge, label, value, rival, desc in interleave(orders):
            if (b.fact_id, label) in asked:   # one question per value: two descriptions of alpelisib asked the same cost twice
                continue
            asked.add((b.fact_id, label))
            hint = {"group": b.group, "slot": b.slot, "link": f"{link.source} -> {link.target}", "mode": link.match, "bridge": bridge,
                    "bridge_value": a.value, "via": link.via, "via_value": label, "value": value, "rival": rival,
                    "description": sorted(desc), "must_mention": without(sheet.must_mention(b), label, a.value)}
            yield Candidate("multihop", "bridge" if link.match == "value" else "described", _units(sheet, a, b), hint, [a, b])


def speaker_for(party: str, cfg: dict) -> str | None:
    """The canonical speaker label ("the EAG") for a party name or alias."""
    p = party.strip().lower()
    for label, aliases in cfg["attribution"]["speakers"].items():
        if p in {label.lower(), label.lower().removeprefix("the ")} | {a.lower() for a in aliases}:
            return label
    return None


def select_attribution(corpus: Corpus, sheet: FactSheet, cfg: dict, rng: Random, taken: Counter | None = None) -> Iterator[Candidate]:
    """Who holds a position: a party's side of a disputed point, or a stakeholder's view in its own document.

    Within a group, disputed positions come first: telling two parties apart
    is the test, guessing the speaker from the topic is not. Across groups
    the weighted rotation decides, as for every type: taken first outright,
    disputed positions filled the whole type from the key issues, which in an
    appraisal are mostly points of economic modelling.
    """
    ac = cfg["attribution"]
    items = []
    for f in sorted(sheet.facts.values(), key=lambda f: f.fact_id):
        if f.redacted or not sheet.askable(f) or not all(native(u, cfg) for u in sheet.units(f)):
            continue
        if f.kind == "stance" and f.positions:
            for party, pos in f.positions.items():
                speaker = speaker_for(party, cfg)
                others = [p for k, p in f.positions.items() if k != party]
                if speaker and not any(jaccard_tokens(pos, o) >= 0.5 for o in others):   # parties that agree are no attribution item
                    items.append((f, "stance", {"speaker": speaker, "claim": pos, "contested": bool(others)}))
        elif f.kind == "text" and f.value:
            speaker = ac["role_speakers"].get(f.role)
            if speaker and speaker in ac["own_voice_only"]:
                items.append((f, "form", {"speaker": speaker, "claim": f.value, "speaker_name": f.speaker, "contested": False}))
    for f, sub, hint in _order(sheet, items, cfg, rng, lambda x: x[0], taken, prefer=lambda x: not x[2]["contested"]):
        yield Candidate("attribution", sub, sheet.units(f), {"group": f.group, "slot": f.slot, **hint}, [f])


# "B.3.11.3. ", "3.2.5.4 ", "A 13. Priority question: ", "b) ", "iv. ": each alternative consumes something, or the first would match nothing and win
_NUMBERING = re.compile(r"^(?:A\s?\d+\.\s*(?:Priority question:\s*)?|[a-z]\)\s*|[ivx]+\.\s*|(?:[A-Z]\.)?\d+(?:\.\d+)*\.?\s*)", re.I)


def heading(u: Unit) -> str:
    """A section's title without its numbering: "B.3.11.3. Scenario analysis" is "Scenario analysis"."""
    title = u.section_path[-1] if u.section_path else ""
    return " ".join(_NUMBERING.sub("", title, count=1).split())[:120]


def _by_theme(sheet: FactSheet) -> dict[tuple[str, str, str], list[Fact]]:
    """Facts of one group, one document and one section: the section is the document's own statement of their theme."""
    out: dict[tuple[str, str, str], list[Fact]] = defaultdict(list)
    for f in sheet.facts.values():
        out[(f.group, f.doc_id, sheet.units(f)[0].section_id or "")].append(f)
    return out


def select_summary(corpus: Corpus, sheet: FactSheet, cfg: dict, rng: Random, taken: Counter | None = None) -> Iterator[Candidate]:
    """Several facts of one group in one section of one document: a question whose full answer covers them all.

    The section is the theme. Drawn from a whole document, the third smoke
    run's summaries put an RDI scenario, testing costs, the severity modifier
    and progressed-disease QALYs under one question, and exemestane's £4.18
    beside the model's time horizon.
    """
    sc = cfg["summary"]
    items = []
    for (group, doc_id, section), facts in sorted(_by_theme(sheet).items()):
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
    for facts in _order(sheet, items, cfg, rng, lambda x: x[0], taken):
        f = facts[0]
        hint = {"group": f.group, "group_describe": sheet.schema.groups[f.group].describe, "doc": sheet.doc_label(f), "speaker_name": f.speaker,
                "theme": heading(sheet.units(f)[0]), "section_path": sheet.units(f)[0].section_path}
        yield Candidate("summary", f.group, _units(sheet, *facts), hint, facts)


SELECTORS: dict[str, Callable[..., Iterator[Candidate]]] = {
    "factual": select_factual,
    "numerical": select_numerical,
    "comparative": select_comparative,
    "multihop": select_multihop,
    "attribution": select_attribution,
    "summary": select_summary,
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
