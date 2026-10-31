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

from medrageval.generate.corpus import REDACTED, Corpus, Unit, mentions, significant_numbers, tokenize
from medrageval.text import normalize, table_rows

DESCRIPTIVE = re.compile(r"\b(randomi[sz]ed|phase|double[- ]blind|placebo[- ]controlled|enrolled|recruited|multicentre|"
                         r"multicenter|open[- ]label|primary endpoint|primary outcome|was a|is a|were randomi)", re.I)
TICKBOX = re.compile(r"[☐☒☑✓✔]|\[ ?x? ?\]|\byes/no\b", re.I)
REDACTED_LABEL = [
    re.compile(r"([A-Za-z][^|\n:]{3,80}?)\s*[:|]\s*\[REDACTED\]"),
    re.compile(r"(£|\$|€)\s?\[REDACTED\]"),
    re.compile(r"\[REDACTED\]\s*(%|patients|participants|months|years|weeks|QALYs?|per cent|percent)\b", re.I),
]
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
    """Hint fields for a table: {} without twins, {"must_mention": [...]} when its caption tells it apart, None when nothing does."""
    phrases = corpus.discriminator(u)
    if phrases is None:
        return {}
    if any(not p for p in phrases):
        return None
    return {"must_mention": phrases}


def numeric_hits(plain: str, cfg: dict) -> list[tuple[str, str]]:
    """(number, unit) pairs: a number followed by a unit pattern, or preceded by a currency sign."""
    out = []
    for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", plain):
        before = plain[max(0, m.start() - 2):m.start()].strip()
        tail = plain[m.end():m.end() + 14]
        if before in ("£", "$", "€"):
            out.append((m.group(0).replace(",", ""), before))
            continue
        for pat in cfg["numeric_patterns"]:
            if re.match(r"\s?" + pat, tail, re.I):
                out.append((m.group(0).replace(",", ""), re.match(r"\s?" + pat, tail, re.I).group(0).strip()))
                break
    return out


def speaker_for(alias_text: str, cfg: dict) -> str | None:
    text = alias_text.lower().strip()
    for label, aliases in cfg["attribution"]["speakers"].items():
        if text == label.lower() or text in (a.lower() for a in aliases) or text.rstrip("s") in (a.lower().rstrip("s") for a in aliases):
            return label
    return None


def speaker_cue(plain: str, cfg: dict) -> bool:
    aliases = [a for al in cfg["attribution"]["speakers"].values() for a in al]
    return any(re.search(rf"\b{re.escape(a)}\b", plain, re.I) for a in aliases)


# ----- selectors -----

def select_factual(corpus: Corpus, cfg: dict, rng: Random) -> Iterator[Candidate]:
    units = [u for u in corpus.units.values() if clean_prose(u, cfg)]
    for u in spread(units, section_key, rng, cfg["limits"]["max_per_section"]):
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
    for u, sub, numbers, twin in spread(items, lambda x: section_key(x[0]), rng, cfg["limits"]["max_per_section"]):
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
            labels = [" ".join(r[:width]) for r in (pick[rng.randrange(len(pick))] if pick else good)]   # no repeating group: both columns just name the row
            if len(set(labels)) < 2:
                continue
        else:
            labels = [r[0] for r in good]
            if len(set(labels)) < 2:
                continue
        return {"mode": "rows", "labels": labels[:8], "measure": header[j],
                "header": " | ".join(header), "rows": [" | ".join(r) for r in good][:cc["max_rows"]]}
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
    drug = corpus.topic.split()[0] if corpus.topic and len(corpus.topic.split()[0]) > 3 else None
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
    for sub in ("table", "prose"):  # tables match the spec best, so they come first
        for u, _, hint in spread([i for i in items if i[1] == sub], lambda x: section_key(x[0]), rng, cfg["limits"]["max_per_section"]):
            yield Candidate("comparative", sub, [u], hint)


def select_multihop(corpus: Corpus, cfg: dict, rng: Random) -> Iterator[Candidate]:
    """Entity bridges first (a unit that characterises a trial, a unit that gives its value), then "see Table N" pairs."""
    seen = set()
    pairs = []
    for key, uids in corpus.entities.items():
        by_doc = defaultdict(list)
        for uid in uids:
            by_doc[corpus.units[uid].doc_id].append(corpus.units[uid])
        for doc_id, us in by_doc.items():
            desc = [u for u in us if clean_prose(u, cfg) and DESCRIPTIVE.search(u.plain)]
            num = [u for u in us if (clean_prose(u, cfg) and numeric_hits(u.plain, cfg)) or (whole_table(u, cfg, allow_redacted=True) and u.caption and mentions(u.plain, key))]
            rng.shuffle(desc); rng.shuffle(num)
            for d in desc[:6]:
                for n in num:
                    twin = twin_hint(corpus, n) if n.type == "table" else {}
                    if n.unit_id != d.unit_id and n.section_id != d.section_id and (d.unit_id, n.unit_id) not in seen and twin is not None:
                        seen.add((d.unit_id, n.unit_id))
                        pairs.append((d, n, corpus.entity_names[key], twin))
                        break
    for d, n, name, twin in spread(pairs, lambda x: (x[0].doc_id, x[2]), rng, 4):
        yield Candidate("multihop", "entity", [d, n], {"link": name, "link_kind": "entity", **twin})

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
                    if speaker:
                        start = max(u.plain.rfind(". ", 0, m.start()) + 2, 0)
                        end = u.plain.find(". ", m.end())
                        claim = u.plain[start: end + 1 if end > 0 else None].strip()
                        items.append((u, "stance", speaker, claim))
                    break
            else:
                if (role == "clarification" and any(re.search(p, t, re.I) for p in ac["response_section_patterns"] for t in u.section_path[-1:])
                        and not re.match(r"^[A-C]\s?\d{1,2}\b", u.plain) and not u.plain.rstrip().endswith("?")):
                    items.append((u, "response", ac["role_speakers"].get(role, "the company"), u.plain[:400]))
        elif u.row_index is not None and role in ac["role_speakers"] and role not in cfg["roles"]["primary"] + cfg["roles"]["secondary"]:
            answer = u.plain.split("\n")[-1].split(" | ")[-1].strip()
            if len(answer) >= ac["min_answer_chars"] and not TICKBOX.search(answer) and REDACTED not in answer:
                items.append((u, "form", ac["role_speakers"][role], answer[:400]))
    by_speaker = defaultdict(list)
    for it in items:
        by_speaker[it[2]].append(it)
    queues = [list(spread(v, lambda x: section_key(x[0]), rng, 3)) for _, v in sorted(by_speaker.items())]
    for u, sub, speaker, claim in interleave(queues):
        yield Candidate("attribution", sub, [u], {"speaker": speaker, "claim": claim})


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


def select_unanswerable_not_covered(corpus: Corpus, cfg: dict, rng: Random) -> Iterator[Candidate]:
    units = [u for u in corpus.units.values() if clean_prose(u, cfg)]
    for u in spread(units, lambda u: (u.doc_id, u.section_path[:1][0] if u.section_path else None), rng, 2):
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
