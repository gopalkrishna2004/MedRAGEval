"""Run question generation on one parse output directory.

For each question type the selector proposes evidence, the generator LLM
writes an item about it, the programmatic checks and then the judge LLM
decide whether to keep it, until the type's target is met or the attempt
budget is spent. Accepted items are appended to dataset.jsonl as they come
(a crash loses nothing), rejected ones to rejects.jsonl with their reasons,
and generate_report.json summarises the run. If a type ends far short of its
target the run fails after writing, because a dataset with a hole in it
should not pass silently.
"""

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from random import Random

from medrageval.config import load_config
from medrageval.generate.candidates import SELECTORS, Candidate, candidate_counts
from medrageval.generate.corpus import Corpus, Unit, load_corpus
from medrageval.generate.llm import LLMError, LLMJsonError, make_llms
from medrageval.generate.prompts import PROMPT_VERSION, build_generation, build_judge, build_unanswerable_check
from medrageval.generate.validate import Verdict, is_duplicate, token_set, validate

TYPES = ["factual", "numerical", "comparative", "multihop", "crossdoc", "attribution", "unanswerable"]
NEEDS_SINGLE_EXCERPT_CHECK = {"multihop"}   # cross-document items only need quotes from two documents


def attempts_per_item(cfg: dict, type_: str) -> int:
    a = cfg["attempts_per_item"]
    return a if isinstance(a, int) else a.get(type_, a["default"])


class Run:
    """Shared state of one generation run."""

    def __init__(self, corpus: Corpus, cfg: dict, gen, judge, out_dir: Path, seed: int):
        self.corpus, self.cfg, self.gen, self.judge, self.out_dir, self.seed = corpus, cfg, gen, judge, out_dir, seed
        self.accepted: list[dict] = []
        self.questions: list[str] = []
        self.stats: dict[str, dict] = defaultdict(lambda: {"attempted": 0, "accepted": 0, "rejected": Counter(), "subtypes": Counter()})
        self.speakers = Counter()
        self.failures = 0
        self.dataset = (out_dir / "dataset.jsonl").open("w")
        self.rejects = (out_dir / "rejects.jsonl").open("w")

    def close(self) -> None:
        self.dataset.close()
        self.rejects.close()

    # ----- per candidate work, runs in a worker thread -----
    def process(self, cand: Candidate) -> list[dict]:
        tag = cand.candidate_id
        try:
            raw = self.gen.complete_json(build_generation(cand, self.corpus, self.cfg), tag)
        except LLMJsonError as e:
            return [{"cand": cand, "item": {}, "stage": "generation", "reasons": [f"bad_json:{e}"[:200]]}]
        if raw.get("skip"):   # the model may decline evidence it cannot write a sound item on; the reason goes to rejects.jsonl
            return [{"cand": cand, "item": raw, "stage": "skipped", "reasons": [f"skip:{str(raw['skip'])[:120]}"]}]
        if cand.subtype == "not_covered":
            return self._check_not_covered(cand, [q for q in (raw.get("questions") or [])[:self.cfg["unanswerable"]["questions_per_anchor"]] if isinstance(q, dict)])
        verdict = validate(raw, cand, self.corpus, self.cfg, [])
        if not verdict.ok:
            return [{"cand": cand, "item": raw, "stage": "programmatic", "reasons": verdict.reasons, "verdict": verdict}]
        if cand.type not in self.cfg["judge_types"]:
            return [{"cand": cand, "item": raw, "stage": "accepted", "reasons": [], "verdict": verdict, "judge": None}]
        judge = self.judge.complete_json(build_judge(raw, cand, self.corpus, self.cfg), f"judge:{tag}")
        reasons = []
        if not judge.get("pass"):
            reasons.append("judge_failed")
        for k in ("answer_supported", "self_contained", "type_fits"):
            if judge.get(k) is False:
                reasons.append(f"judge:{k}")
        if cand.type in NEEDS_SINGLE_EXCERPT_CHECK:
            single = judge.get("answerable_from_single_excerpt") or {}
            if any(bool(v) for v in single.values()):
                reasons.append("judge:answerable_from_single_excerpt")
        return [{"cand": cand, "item": raw, "stage": "judge" if reasons else "accepted", "reasons": reasons, "verdict": verdict, "judge": judge}]

    def _check_not_covered(self, cand: Candidate, qds: list[dict]) -> list[dict]:
        """Programmatic checks on each proposed question, then one coverage call for the anchor's surviving questions."""
        uc = self.cfg["unanswerable"]
        lim = self.cfg["limits"]["question_words"]
        out, pending = [], []
        context = set(cand.units[0].tokens).union(*(set(n.tokens) for n in self.corpus.neighbours(cand.units[0], uc["neighbours"])))
        for qd in qds:
            q = str(qd.get("question") or "").strip()
            item = {"question": q, "answer": None, "answer_keys": [], "quotes": [], "missing_fact": qd.get("missing_fact")}
            v = Verdict()
            v.check("question_present", bool(q), "no_question")
            v.check("question_length", lim[0] <= len(q.split()) <= lim[1], "question_length")
            v.check("no_deictic", not any(re.search(p, q, re.I) for p in self.cfg["validate"]["deictic_patterns"]), "deictic_question")
            v.check("in_domain", len(token_set(q) & context) >= 2, "question_off_topic")
            if v.ok:
                pending.append((item, v))
            else:
                out.append({"cand": cand, "item": item, "stage": "programmatic", "reasons": v.reasons, "verdict": v})
        if not pending:
            return out
        passages, seen = [], set()
        for item, _ in pending:
            for h, _ in self.corpus.bm25.search(sorted(token_set(item["question"])), k=uc["top_k"]):
                if h not in seen:
                    seen.add(h); passages.append(self.corpus.units[h])
        for n in self.corpus.neighbours(cand.units[0], uc["neighbours"]) + [cand.units[0]]:
            if n.unit_id not in seen:
                seen.add(n.unit_id); passages.append(n)
        check = self.judge.complete_json(build_unanswerable_check([i["question"] for i, _ in pending], passages, self.corpus), f"check:{cand.candidate_id}")
        results = {int(r.get("n", 0)): r for r in (check.get("results") or []) if isinstance(r, dict)}
        ok_verdicts = {"no", "partly"} if uc["accept_partly"] else {"no"}
        for n, (item, v) in enumerate(pending, start=1):
            r = results.get(n, {})
            item["search"] = {"hits": [p.unit_id for p in passages], "verdict": r.get("answered"), "answer_if_any": r.get("answer_if_any"),
                              "passage_ids": r.get("passage_ids")}
            reasons = [] if r.get("answered") in ok_verdicts else [f"answered:{r.get('answered')}"]
            out.append({"cand": cand, "item": item, "stage": "unanswerable_check" if reasons else "accepted", "reasons": reasons, "verdict": v, "judge": r})
        return out

    # ----- main thread -----
    def run_type(self, selector: str, type_: str, target: int) -> None:
        cfg = self.cfg
        st = self.stats[type_]
        rng = Random(f"{self.seed}:{selector}")
        it = SELECTORS[selector](self.corpus, cfg, rng)
        budget = target * attempts_per_item(cfg, type_)
        accepted_here, attempts = 0, 0
        quota = math.ceil(cfg["attribution"]["max_share_per_speaker"] * target) if type_ == "attribution" else None
        window = max(1, cfg["max_workers"]) * 2
        with ThreadPoolExecutor(max_workers=max(1, cfg["max_workers"])) as pool:
            while accepted_here < target and attempts < budget:
                batch = []
                for cand in it:
                    if quota and self.speakers[cand.hint.get("speaker")] >= quota:
                        continue
                    batch.append(cand)
                    if len(batch) >= min(window, target - accepted_here, budget - attempts):   # no over-fetch past the target
                        break
                if not batch:
                    break
                attempts += len(batch)
                st["attempted"] += len(batch)
                try:
                    results = list(pool.map(self.process, batch))
                except LLMError as e:
                    self.failures += 1
                    if self.failures >= cfg["abort_after_consecutive_failures"]:
                        raise RuntimeError(f"{self.failures} LLM calls failed in a row; check the model and key: {e}") from e
                    print(f"  LLM error: {e}")
                    continue
                self.failures = 0
                for res in (r for group in results for r in group):
                    if accepted_here >= target:
                        break
                    if res["stage"] == "accepted" and is_duplicate(res["item"].get("question", ""), self.questions, cfg["dedupe_jaccard"]):
                        res["stage"], res["reasons"] = "duplicate", ["duplicate_question"]
                    if res["stage"] == "accepted" and quota and self.speakers[res["cand"].hint.get("speaker")] >= quota:
                        res["stage"], res["reasons"] = "speaker_quota", ["speaker_quota"]
                    rec = self.record(res)
                    if res["stage"] == "accepted":
                        accepted_here += 1
                        st["accepted"] += 1
                        st["subtypes"][res["cand"].subtype or "-"] += 1
                        self.accepted.append(rec)
                        self.questions.append(rec["question"])
                        if type_ == "attribution":
                            self.speakers[res["cand"].hint.get("speaker")] += 1
                        self.dataset.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    else:
                        st["rejected"][res["reasons"][0].split(":")[0] if res["reasons"] else res["stage"]] += 1
                        self.rejects.write(json.dumps(rec, ensure_ascii=False) + "\n")
                self.dataset.flush()
                self.rejects.flush()
        print(f"  {type_:12s} {selector:26s} accepted {accepted_here:3d}/{target}  attempts {attempts}")

    def record(self, res: dict) -> dict:
        cand, item = res["cand"], res["item"]
        units = {u.unit_id: u for u in cand.units}
        evidence = []
        for qt in item.get("quotes") or []:
            u = units.get(qt.get("unit_id")) if isinstance(qt, dict) else None
            if u:
                evidence.append(self._evidence(u, qt.get("quote")))
        if not evidence and cand.subtype == "not_covered":
            evidence = []
        rec = {
            "item_id": f"{cand.type}-{self.stats[cand.type]['accepted'] + 1:03d}" if res["stage"] == "accepted" else None,
            "type": cand.type,
            "subtype": cand.subtype,
            "question": item.get("question"),
            "answer": item.get("answer") if cand.type != "unanswerable" else None,
            "answer_keys": item.get("answer_keys") if cand.type != "unanswerable" else [],
            "answerable": cand.type != "unanswerable",
            "evidence": evidence,
            "anchor": [self._evidence(u, None) for u in cand.units] if cand.type == "unanswerable" and cand.subtype == "not_covered" else None,
            "hint": {k: v for k, v in cand.hint.items() if k != "rows"},
            "provenance": {"candidate_id": cand.candidate_id, "model": self.gen.model, "judge_model": self.judge.model,
                           "prompt_version": PROMPT_VERSION, "seed": self.seed, "interim_dir": str(self.corpus.interim_dir),
                           "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
            "validation": {"programmatic": res.get("verdict").checks if res.get("verdict") else {}, "judge": res.get("judge"),
                           "metrics": res.get("verdict").metrics if res.get("verdict") else {}},
        }
        if cand.type == "attribution":
            sp = cand.hint.get("speaker", "")
            rec["answer_aliases"] = list(dict.fromkeys([sp.removeprefix("the ")] + list(self.cfg["attribution"]["speakers"].get(sp, []))))
        if cand.type == "comparative":
            rec["comparison"] = {k: item.get(k) for k in ("measure", "label_a", "value_a", "label_b", "value_b", "higher")}
        if cand.type == "multihop":
            rec["hops"] = {"a": item.get("hop_a"), "b": item.get("hop_b")}
        if "search" in item:
            rec["search"] = item["search"]
        if res["stage"] != "accepted":
            rec["stage"] = res["stage"]
            rec["reasons"] = res["reasons"]
        return rec

    def _evidence(self, u: Unit, quote: str | None) -> dict:
        d = self.corpus.docs[u.doc_id]
        return {"unit_id": u.unit_id, "element_id": u.element_id, "row_index": u.row_index, "doc_id": u.doc_id,
                "doc_role": d.role, "doc_label": d.label, "section_path": u.section_path, "pages": u.pages,
                "caption": u.caption, "quote": quote}


def generate_dataset(interim_dir: Path, out_dir: Path, cfg: dict, gen, judge, targets: dict[str, int] | None = None,
                     types: list[str] | None = None, seed: int | None = None) -> dict:
    """Generate the dataset into out_dir and return the report. Raises after writing if a type fell short."""
    t0 = time.time()
    seed = cfg["seed"] if seed is None else seed
    targets = {**cfg["targets"], **(targets or {})}
    types = [t for t in TYPES if t in (types or TYPES) and targets.get(t, 0) > 0]
    corpus = load_corpus(interim_dir, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{interim_dir}: {len(corpus.units)} evidence units in {len(corpus.docs)} documents; topic: {corpus.topic[:70]}")
    run = Run(corpus, cfg, gen, judge, out_dir, seed)
    error = None
    try:
        for t in types:
            if t == "unanswerable":
                half = math.ceil(targets[t] / 2)
                run.run_type("unanswerable_redacted", t, half)
                run.run_type("unanswerable_not_covered", t, targets[t] - half)
            else:
                run.run_type(t, t, targets[t])
    except RuntimeError as e:
        error = e
    finally:
        run.close()
    report = build_report(run, targets, types, time.time() - t0, error)
    (out_dir / "generate_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if error:
        raise error
    if short := {t: (run.stats[t]["accepted"], targets[t]) for t in types
                 if run.stats[t]["accepted"] < cfg["min_share_of_target"] * targets[t]}:
        raise RuntimeError(f"short of target after writing {out_dir}: {short}")
    return report


def build_report(run: Run, targets: dict, types: list[str], elapsed: float, error: Exception | None) -> dict:
    per_type = {t: {"target": targets[t], "attempted": run.stats[t]["attempted"], "accepted": run.stats[t]["accepted"],
                    "subtypes": dict(run.stats[t]["subtypes"]), "rejected_by_reason": dict(run.stats[t]["rejected"])} for t in types}
    warnings = [f"{t}: {s['accepted']}/{s['target']} accepted" for t, s in per_type.items() if s["accepted"] < s["target"]]
    if error:
        warnings.append(f"aborted: {error}")
    return {
        "interim_dir": str(run.corpus.interim_dir),
        "topic": run.corpus.topic,
        "model": run.gen.model, "judge_model": run.judge.model, "prompt_version": PROMPT_VERSION, "seed": run.seed,
        "items": len(run.accepted),
        "per_type": per_type,
        "items_per_document": dict(Counter(e["doc_id"] for r in run.accepted for e in (r["evidence"] or r.get("anchor") or []))),
        "speakers": dict(run.speakers),
        "corpus": run.corpus.stats(),
        "llm": {"generator": dict(run.gen.stats), "judge": dict(run.judge.stats)},
        "elapsed_seconds": round(elapsed, 1),
        "warnings": warnings,
    }


def print_report(report: dict) -> None:
    print(f"{report['items']} items in {report['elapsed_seconds']}s with {report['model']} (prompt v{report['prompt_version']}, seed {report['seed']})")
    for t, s in report["per_type"].items():
        rej = ", ".join(f"{k} {v}" for k, v in sorted(s["rejected_by_reason"].items(), key=lambda kv: -kv[1])[:4])
        print(f"  {t:12s} {s['accepted']:3d}/{s['target']:<3d} of {s['attempted']:3d} tried  {dict(s['subtypes'])}  rejected: {rej}")
    g, j = report["llm"]["generator"], report["llm"]["judge"]
    print(f"  LLM calls: {g.get('calls', 0) + j.get('calls', 0)} ({g.get('cache_hits', 0) + j.get('cache_hits', 0)} cached), "
          f"tokens in/out: {g.get('prompt_tokens', 0) + j.get('prompt_tokens', 0)}/{g.get('completion_tokens', 0) + j.get('completion_tokens', 0)}")
    for w in report["warnings"]:
        print(f"  WARNING: {w}")


def dry_run(interim_dir: Path, cfg: dict, seed: int | None = None) -> dict:
    corpus = load_corpus(interim_dir, cfg)
    s = corpus.stats()
    print(f"{interim_dir}: {s['units']} units ({s['row_units']} table rows), noise {s['noise']}")
    print(f"  topic: {s['topic']}")
    for d in corpus.docs.values():
        print(f"  {d.doc_id} {d.role:20s} {d.n_units:4d} units  {d.title[:60]}")
    print(f"  entities: {s['entities']}")
    counts = candidate_counts(corpus, cfg, cfg["seed"] if seed is None else seed, examples=2)
    for sel, info in counts.items():
        print(f"  {sel:26s} {info['total']:5d} candidates {info['by_subtype']}")
        for sub, exs in info["examples"].items():
            for e in exs[:1]:
                desc = " || ".join(corpus.units[u].plain[:60].replace("\n", " / ") for u in e["units"])
                print(f"      e.g. [{sub}] {e['units']}: {desc}")
    return {"corpus": s, "candidates": counts}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Generate an evaluation dataset from parse output.")
    add_arguments(ap)
    run_cli(ap.parse_args(argv))


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("interim_dir", type=Path, help="output directory of `medrageval parse`")
    ap.add_argument("--out", type=Path, help="dataset directory (default: data/datasets/<interim dir name>)")
    ap.add_argument("--config", type=Path)
    ap.add_argument("--per-type", type=int, help="override every type's target (smoke runs)")
    ap.add_argument("--types", help="comma-separated subset of types to run")
    ap.add_argument("--no-cache", action="store_true", help="ignore and do not write the LLM response cache")
    ap.add_argument("--dry-run", action="store_true", help="only load the corpus and count candidates; no LLM calls")
    ap.add_argument("--seed", type=int)


def run_cli(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)["generate"]
    if args.dry_run:
        dry_run(args.interim_dir, cfg, args.seed)
        return
    gen, judge = make_llms(cfg, use_cache=not args.no_cache)
    out = args.out or Path("data/datasets") / args.interim_dir.name
    targets = {t: args.per_type for t in TYPES} if args.per_type else None
    types = args.types.split(",") if args.types else None
    try:
        report = generate_dataset(args.interim_dir, out, cfg, gen, judge, targets, types, args.seed)
    except RuntimeError as e:
        print_report(json.loads((out / "generate_report.json").read_text()))
        raise SystemExit(f"error: {e}")
    print_report(report)
    print(f"outputs in {out}/")


if __name__ == "__main__":
    main()
