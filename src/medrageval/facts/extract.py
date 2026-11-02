"""Extract the fact sheet from one parse output directory.

For each schema group and each document of a role the group reads, the
passages to read are the document's units whose section path matches the
group's patterns, plus the units that best match its keywords (BM25 within
the document). They are packed into batches, and the LLM records the facts
each batch states.

Every fact is checked against the batch before it is kept:
- the quote is a verbatim span of the unit it cites;
- every number in its value is in the quote;
- a text value shares words with the quote;
- a "[REDACTED]" value has the marker in its quote.

Facts that fail go to factsheet_rejects.jsonl with the reason. Duplicates
within a document are merged. Each fact records the units in other documents
that state it too (`also_in`), so a retriever that finds the copy is not
marked wrong. Outputs: factsheet.jsonl, factsheet_rejects.jsonl and
factsheet_report.json in the interim directory.
"""

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from medrageval.config import load_config
from medrageval.facts.factsheet import Fact, content_tokens, numbers_in, write_facts
from medrageval.facts.prompts import EXTRACT_PROMPT_VERSION, build_extraction
from medrageval.facts.schema import Group, Schema, Slot, known_roles, load_schema
from medrageval.generate.candidates import garbled
from medrageval.generate.corpus import REDACTED, Corpus, Unit, load_corpus, tokenize
from medrageval.generate.llm import LLMError, LLMJsonError, make_llm
from medrageval.generate.validate import Verdict, quote_in
from medrageval.text import normalize


@dataclass
class Batch:
    group: str
    doc_id: str
    index: int
    units: list[Unit]

    @property
    def key(self) -> str:
        return f"{self.group}:{self.doc_id}:{self.index}"


def _native(u: Unit, cfg: dict) -> bool:
    return cfg["limits"]["allow_ocr"] or u.source != "ocr"


def in_sections(u: Unit, patterns: tuple[str, ...]) -> bool:
    """The unit's own section or its parent matches a pattern.

    Not the whole path: "B.2 Clinical effectiveness" heads the trial methods as
    well as the results, so a pattern for results matched against the whole
    path filled the efficacy batches with eligibility criteria.
    """
    titles = " | ".join(u.section_path[-2:])
    return any(re.search(p, titles, re.I) for p in patterns)


def select_units(corpus: Corpus, group: Group, doc_id: str, cfg: dict, fcfg: dict) -> list[Unit]:
    """Units of one document to read for one group, in document order: section matches first, then keyword matches."""
    bc = fcfg["batch"]
    units = [corpus.units[uid] for uid in corpus.by_doc.get(doc_id, []) if corpus.units[uid].evidence and _native(corpus.units[uid], cfg)]
    if not units:
        return []
    ids = {u.unit_id for u in units}
    scores = dict(corpus.bm25.search(tokenize(" ".join(group.keywords)), k=bc["bm25_top_k"], only=ids)) if group.keywords else {}
    in_section = {u.unit_id for u in units if in_sections(u, group.sections)}
    # in a matching section and on topic first, then the rest of the matching sections, then keyword hits elsewhere
    ranked = sorted(ids & (in_section | set(scores)),
                    key=lambda uid: (uid not in in_section, uid not in scores, -scores.get(uid, 0.0), corpus.position[uid]))
    picked = ranked[: bc["max_units"]]
    return sorted((corpus.units[uid] for uid in picked), key=lambda u: corpus.position[u.unit_id])


def plan_batches(corpus: Corpus, schema: Schema, cfg: dict, fcfg: dict) -> list[Batch]:
    bc = fcfg["batch"]
    batches = []
    for group in schema.groups.values():
        if group.weight <= 0:
            continue
        for doc in corpus.docs.values():
            if doc.role not in group.roles:
                continue
            chunk, size = [], 0
            n = 0
            for u in select_units(corpus, group, doc.doc_id, cfg, fcfg):
                length = min(len(u.plain), bc["max_unit_chars"]) + 120
                if chunk and size + length > bc["max_batch_chars"]:
                    batches.append(Batch(group.id, doc.doc_id, n, chunk))
                    chunk, size, n = [], 0, n + 1
                chunk.append(u)
                size += length
            if chunk:
                batches.append(Batch(group.id, doc.doc_id, n, chunk))
    return batches


def _clean_qualifiers(raw, slot: Slot) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    out = {}
    for k, v in raw.items():
        k = str(k).strip().lower().replace(" ", "_")
        if (k in slot.qualifiers or k == slot.per) and isinstance(v, (str, int, float)) and str(v).strip() and REDACTED not in str(v) and len(str(v)) <= 160:
            out[k] = " ".join(str(v).split())
    return out


def _grounded(text: str, quote: str, n: int) -> bool:
    return len(content_tokens(text) & content_tokens(quote)) >= min(n, max(len(content_tokens(text)), 1))


def verify(raw: dict, batch: Batch, group: Group, corpus: Corpus, fcfg: dict) -> tuple[dict | None, Verdict]:
    """Check one extracted fact against its batch. Returns the cleaned fact (without an id) and the verdict."""
    v = Verdict()
    name = str(raw.get("slot") or "").split(".")[-1].strip()
    slot = group.slots.get(name)
    if not v.check("slot_known", slot is not None, f"unknown_slot:{name[:40]}"):
        return None, v
    units = {u.unit_id: u for u in batch.units}
    uid = str(raw.get("unit_id") or "").strip()
    if uid not in units:
        prefix = [k for k in units if k.startswith(uid)] if uid else []
        uid = prefix[0] if len(prefix) == 1 else uid
    u = units.get(uid)
    if not v.check("unit_known", u is not None, f"unit_unknown:{uid[:40]}"):
        return None, v
    quote = " ".join(str(raw.get("quote") or "").split())
    n = len(normalize(quote).split())
    lo, hi = fcfg["quote_tokens"]
    v.check("quote_length", lo <= n <= hi, f"quote_length:{n}")
    if not v.check("quote_found", bool(quote) and quote_in(quote, u.plain), f"quote_not_found:{uid}"):
        return None, v
    fact = {"slot": slot.id, "kind": slot.kind, "doc_id": batch.doc_id, "role": corpus.docs[batch.doc_id].role,
            "evidence": [{"unit_id": uid, "quote": quote}], "qualifiers": _clean_qualifiers(raw.get("qualifiers"), slot),
            "value": None, "values": None, "positions": None, "redacted": False}
    max_chars, ground = fcfg["max_value_chars"], fcfg["min_grounding_tokens"]
    if slot.kind == "number":
        values = raw.get("values") if isinstance(raw.get("values"), dict) else None
        values = {" ".join(str(k).split()): " ".join(str(x).split()) for k, x in (values or {}).items() if str(k).strip() and str(x).strip()}
        value = " ".join(str(raw.get("value") or "").split()) if not values else None
        texts = list(values.values()) if values else ([value] if value else [])
        if not v.check("value_present", bool(texts), "no_value"):
            return None, v
        v.check("value_length", all(len(t) <= max_chars for t in texts), "value_too_long")
        redacted = [t for t in texts if REDACTED in t]
        if redacted:
            v.check("redaction_in_quote", REDACTED in quote, "redacted_value_not_in_quote")
        for t in texts:
            if REDACTED in t:
                continue
            nums = numbers_in(t)
            if v.check("value_has_number", bool(nums), f"value_without_number:{t[:40]}"):
                v.check("value_in_quote", nums <= numbers_in(quote), f"value_not_in_quote:{t[:40]}")
        fact["values"], fact["value"] = (values or None), value
        fact["redacted"] = len(redacted) == len(texts)
    elif slot.kind == "text":
        value = " ".join(str(raw.get("value") or "").split())
        if not v.check("value_present", bool(value), "no_value"):
            return None, v
        v.check("value_length", len(value) <= max_chars, "value_too_long")
        v.check("value_grounded", _grounded(value, quote, ground), "value_not_grounded")
        fact["value"] = value
        fact["redacted"] = value == REDACTED
    else:
        parties = {p.lower(): p for p in slot.parties}
        pos = raw.get("positions") if isinstance(raw.get("positions"), dict) else {}
        positions = {parties[str(k).strip().lower()]: " ".join(str(x).split()) for k, x in pos.items()
                     if str(k).strip().lower() in parties and str(x).strip()}
        if not v.check("positions_present", bool(positions), "no_positions"):
            return None, v
        v.check("value_length", all(len(t) <= max_chars for t in positions.values()), "value_too_long")
        v.check("positions_grounded", any(_grounded(t, quote, ground) for t in positions.values()), "positions_not_grounded")
        fact["positions"] = positions
    text = " ".join(filter(None, [fact["value"], *(fact["values"] or {}).values(), *(fact["positions"] or {}).values()]))
    v.check("value_clean", not garbled(text), "value_garbled")
    return (fact if v.ok else None), v


def _key(f: dict) -> tuple:
    vals = f["values"] or f["positions"] or {"": f["value"] or ""}
    sig = tuple(sorted((normalize(k), " ".join(sorted(numbers_in(x))) if f["kind"] == "number" else normalize(x)) for k, x in vals.items()))
    return f["doc_id"], f["slot"], tuple(sorted((k, normalize(x)) for k, x in f["qualifiers"].items())), sig


def find_also_in(fact: Fact, corpus: Corpus, fcfg: dict) -> list[str]:
    """Units in other documents that state the same fact: every value number, and enough of the quote's rare words."""
    ac = fcfg["also_in"]
    quote = fact.quote
    limit = ac["rare_df_max"] * len(corpus.units)
    rare = {t for t in tokenize(quote) if len(t) >= 3 and not any(c.isdigit() for c in t) and corpus.df.get(t, 0) <= limit}
    nums = fact.numbers()
    same_doc = set(corpus.by_doc.get(fact.doc_id, ()))
    out = []
    for hit, _ in corpus.bm25.search(sorted(set(tokenize(quote))), k=ac["top_k"], exclude=same_doc):
        u = corpus.units[hit]
        shared = len(rare & set(u.tokens))
        if nums:
            ok = nums <= numbers_in(u.plain) and shared >= min(ac["min_shared_rare"], len(rare))
        else:
            ok = bool(rare) and shared / len(rare) >= ac["text_containment"]
        if ok:
            out.append(hit)
    return out


def assemble(accepted: list[dict], corpus: Corpus, fcfg: dict) -> list[Fact]:
    """Merge duplicates within a document, order by position, assign ids, find copies in other documents."""
    merged: dict[tuple, dict] = {}
    for f in accepted:
        k = _key(f)
        if k in merged:
            ev = merged[k]["evidence"]
            if all(e["unit_id"] != f["evidence"][0]["unit_id"] for e in ev):
                ev.append(f["evidence"][0])
        else:
            merged[k] = {**f, "evidence": list(f["evidence"])}
    ordered = sorted(merged.values(), key=lambda f: (corpus.position[f["evidence"][0]["unit_id"]], f["slot"]))
    counters: Counter = Counter()
    facts = []
    for f in ordered:
        counters[(f["doc_id"], f["slot"])] += 1
        fact = Fact.from_dict({**f, "fact_id": f"{f['doc_id']}-{f['slot']}-{counters[(f['doc_id'], f['slot'])]:03d}"})
        if not fact.redacted:
            fact.also_in = find_also_in(fact, corpus, fcfg)
        facts.append(fact)
    return facts


def extract_factsheet(interim_dir: Path, cfg: dict, fcfg: dict, llm, schema: Schema, out_dir: Path | None = None) -> dict:
    """Run extraction into out_dir (default: the interim directory) and return the report."""
    t0 = time.time()
    interim_dir = Path(interim_dir)
    out_dir = Path(out_dir or interim_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus(interim_dir, cfg)
    batches = plan_batches(corpus, schema, cfg, fcfg)
    print(f"{interim_dir}: {len(batches)} batches over {len(schema.groups)} groups; topic: {corpus.topic[:70]}")

    def call(batch: Batch):
        msgs = build_extraction(schema.groups[batch.group], batch.units, corpus, batch.doc_id, fcfg["max_facts_per_batch"], fcfg["batch"]["max_unit_chars"])
        try:
            return batch, llm.complete_json(msgs, f"extract:{batch.key}"), None
        except (LLMJsonError, LLMError) as e:
            return batch, None, e

    accepted, rejects = [], []
    per_group = defaultdict(lambda: {"batches": 0, "facts_raw": 0, "accepted": 0, "rejected": Counter(), "failed_batches": 0})
    failures = 0
    with ThreadPoolExecutor(max_workers=max(1, fcfg["max_workers"])) as pool:
        for batch, raw, err in pool.map(call, batches):
            st = per_group[batch.group]
            st["batches"] += 1
            if err is not None:
                st["failed_batches"] += 1
                rejects.append({"batch": batch.key, "stage": "llm", "reasons": [str(err)[:200]]})
                failures = failures + 1 if isinstance(err, LLMError) else failures
                if failures >= fcfg["abort_after_consecutive_failures"]:
                    raise RuntimeError(f"{failures} extraction calls failed in a row; check the model and key: {err}")
                continue
            failures = 0
            items = raw.get("facts") if isinstance(raw.get("facts"), list) else []
            for item in items[: fcfg["max_facts_per_batch"]]:
                if not isinstance(item, dict):
                    continue
                st["facts_raw"] += 1
                fact, verdict = verify(item, batch, schema.groups[batch.group], corpus, fcfg)
                if fact:
                    accepted.append(fact)
                    st["accepted"] += 1
                else:
                    st["rejected"][verdict.reasons[0].split(":")[0] if verdict.reasons else "invalid"] += 1
                    rejects.append({"batch": batch.key, "stage": "verify", "reasons": verdict.reasons, "fact": item})
    facts = assemble(accepted, corpus, fcfg)
    write_facts(out_dir / "factsheet.jsonl", facts)
    with open(out_dir / "factsheet_rejects.jsonl", "w") as f:
        for r in rejects:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_group_doc = defaultdict(Counter)
    for fact in facts:
        by_group_doc[fact.group][fact.doc_id] += 1
    warnings = [f"{g}: no facts" for g, grp in schema.groups.items() if grp.weight > 0 and not by_group_doc.get(g)]
    report = {
        "interim_dir": str(interim_dir), "topic": corpus.topic,
        "schema": {"name": schema.name, "version": schema.version, "path": str(schema.path)},
        "prompt_version": EXTRACT_PROMPT_VERSION, "model": llm.model,
        "batches": len(batches), "facts": len(facts), "redacted": sum(f.redacted for f in facts),
        "with_copies": sum(bool(f.also_in) for f in facts),
        "per_group": {g: {"batches": s["batches"], "failed_batches": s["failed_batches"], "facts_raw": s["facts_raw"], "accepted": s["accepted"],
                          "facts": sum(by_group_doc[g].values()), "by_doc": dict(by_group_doc[g]), "rejected": dict(s["rejected"])}
                      for g, s in per_group.items()},
        "per_slot": dict(Counter(f.slot for f in facts)),
        "llm": dict(llm.stats), "elapsed_seconds": round(time.time() - t0, 1),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "warnings": warnings,
    }
    (out_dir / "factsheet_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def print_report(report: dict) -> None:
    print(f"{report['facts']} facts ({report['redacted']} redacted, {report['with_copies']} also stated elsewhere) "
          f"from {report['batches']} batches in {report['elapsed_seconds']}s with {report['model']} (extract prompt v{report['prompt_version']})")
    for g, s in report["per_group"].items():
        rej = ", ".join(f"{k} {v}" for k, v in sorted(s["rejected"].items(), key=lambda kv: -kv[1])[:4])
        print(f"  {g:22s} {s['facts']:4d} facts from {s['facts_raw']:4d} proposed in {s['batches']:3d} batches  {dict(s['by_doc'])}  rejected: {rej or '-'}")
    llm = report["llm"]
    print(f"  LLM calls: {llm.get('calls', 0)} ({llm.get('cache_hits', 0)} cached), tokens in/out: {llm.get('prompt_tokens', 0)}/{llm.get('completion_tokens', 0)}")
    for w in report["warnings"]:
        print(f"  WARNING: {w}")


def dry_run(interim_dir: Path, cfg: dict, fcfg: dict, schema: Schema) -> dict:
    corpus = load_corpus(interim_dir, cfg)
    batches = plan_batches(corpus, schema, cfg, fcfg)
    out = defaultdict(lambda: {"batches": 0, "units": 0, "chars": 0, "docs": set()})
    for b in batches:
        s = out[b.group]
        s["batches"] += 1
        s["units"] += len(b.units)
        s["chars"] += sum(min(len(u.plain), fcfg["batch"]["max_unit_chars"]) for u in b.units)
        s["docs"].add(b.doc_id)
    print(f"{interim_dir}: {len(batches)} batches; roles {dict(Counter(d.role for d in corpus.docs.values()))}")
    for g, s in out.items():
        print(f"  {g:22s} {s['batches']:3d} batches  {s['units']:4d} units  {s['chars']:7d} chars  docs {sorted(s['docs'])}")
    missing = [g for g in schema.groups if g not in out and schema.groups[g].weight > 0]
    total = sum(s["chars"] for s in out.values())
    print(f"  total {total} chars (~{total // 4} prompt tokens plus instructions)")
    for g in missing:
        print(f"  WARNING: {g}: no document of roles {list(schema.groups[g].roles)}")
    return {"batches": len(batches), "per_group": {g: {**s, "docs": sorted(s["docs"])} for g, s in out.items()}, "chars": total, "missing": missing}


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("interim_dir", type=Path, help="output directory of `medrageval parse`")
    ap.add_argument("--config", type=Path)
    ap.add_argument("--schema", type=Path, help="schema YAML (default: facts.schema in the config)")
    ap.add_argument("--out", type=Path, help="where to write factsheet.jsonl (default: the interim directory)")
    ap.add_argument("--no-cache", action="store_true", help="ignore and do not write the LLM response cache")
    ap.add_argument("--dry-run", action="store_true", help="only plan the batches and count characters; no LLM calls")


def run_cli(args: argparse.Namespace) -> None:
    full = load_config(args.config)
    cfg, fcfg = full["generate"], full["facts"]
    schema = load_schema(args.schema or fcfg["schema"], known_roles(cfg))
    if args.dry_run:
        dry_run(args.interim_dir, cfg, fcfg, schema)
        return
    llm = make_llm(fcfg, cfg, use_cache=not args.no_cache)
    report = extract_factsheet(args.interim_dir, cfg, fcfg, llm, schema, args.out)
    print_report(report)
    print(f"outputs in {args.out or args.interim_dir}/")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Extract the fact sheet from parse output.")
    add_arguments(ap)
    run_cli(ap.parse_args(argv))


if __name__ == "__main__":
    main()
