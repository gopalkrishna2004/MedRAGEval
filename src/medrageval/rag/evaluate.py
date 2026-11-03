"""Run the RAG over evaluation datasets and score every answer.

Two passes. First the small model on vLLM answers every question (the first
one alone, so a server that is down or a model that is not served stops the
run before the rest are sent). An unanswerable item that carries distractors
is read with those passages instead of a search; an unanswerable item without
them (an older dataset) goes through retrieval like the rest. Then the judge on OpenRouter grades each
answer against the item and the passages the model was shown. Both are
cached, so rescoring after a judge-prompt change makes no RAG calls, and a
rerun of the same config makes none at all.

Writes responses.jsonl (one row per item: the reply, the passages, retrieval,
keys, judge, outcome) and eval_report.json (per type and overall, with 95%
bootstrap intervals), and prints the summary.
"""

import argparse
import copy
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from medrageval.config import load_config
from medrageval.generate.generate import TYPES
from medrageval.generate.llm import LLMError, LLMJsonError, make_llm, make_local_llm
from medrageval.rag.index import MODES, load_index
from medrageval.rag.pipeline import RAG, distractor_chunks
from medrageval.rag.prompts import PROMPT_VERSION, build_judge
from medrageval.rag.score import agreement, score_item, summarize


def load_items(dataset_dirs: list[Path], limit: int | None = None) -> list[dict]:
    """Items of one or more datasets. With several, ids are prefixed by the dataset directory, since each run numbers from 001."""
    items = []
    for d in dataset_dirs:
        path = d / "dataset.jsonl" if d.is_dir() else d
        name = (path.parent if path.name == "dataset.jsonl" else path).name
        for line in path.open():
            if line.strip():
                it = json.loads(line)
                it["dataset"] = name
                it["uid"] = f"{name}/{it['item_id']}" if len(dataset_dirs) > 1 else it["item_id"]
                items.append(it)
    return items[:limit] if limit else items


def resolve_interim(items: list[dict], override: Path | None) -> Path:
    if override:
        return override
    dirs = {it.get("provenance", {}).get("interim_dir") for it in items} - {None}
    if len(dirs) != 1:
        raise SystemExit(f"error: the datasets were generated from {sorted(dirs) or 'unknown'} parse outputs; pass --interim")
    return Path(dirs.pop())


def apply_overrides(full: dict, args: argparse.Namespace) -> dict:
    full = copy.deepcopy(full)
    r = full["rag"]
    if getattr(args, "mode", None):
        r["retrieval"]["mode"] = args.mode
    if getattr(args, "top_k", None):
        r["retrieval"]["top_k"] = args.top_k
    if getattr(args, "model", None):
        r["llm"]["model"] = args.model
    if getattr(args, "judge_model", None):
        r["judge"]["model"] = args.judge_model
    return full


def _map(fn, xs: list, workers: int) -> list:
    """fn over xs: the first alone, so a dead server or a bad key fails once instead of once per item."""
    if not xs:
        return []
    first = fn(xs[0])
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return [first] + list(pool.map(fn, xs[1:]))


def evaluate(items: list[dict], rag: RAG, judge, full: dict, out_dir: Path, index_stats: dict | None = None) -> dict:
    rcfg, ecfg = full["rag"], full["rag"]["eval"]
    patterns, speakers = ecfg["abstain_patterns"], full["generate"]["attribution"]["speakers"]
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    def answer(it: dict) -> dict:
        fixed = distractor_chunks(it, rag.index) if it.get("distractors") and not it.get("answerable", True) else None
        try:
            return rag.answer(it["question"], tag=f"rag:{it['uid']}", passages=fixed)
        except LLMError as e:
            if it is items[0]:
                raise
            return {"error": str(e)[:300]}

    print(f"answering {len(items)} questions with {rag.llm.model} ({rcfg['retrieval']['mode']}, top {rcfg['retrieval']['top_k']})")
    answers = _map(answer, items, rcfg["llm"]["max_workers"])

    def grade(pair: tuple[dict, dict]) -> dict | None:
        it, a = pair
        if judge is None or a.get("error"):
            return None
        try:
            return judge.complete_json(build_judge(it, a["answer"], a["passages"], rag.index.topic, rcfg["judge"]["max_chars_per_passage"]),
                                       f"judge:{it['uid']}")
        except LLMJsonError as e:
            return {"error": str(e)[:300]}
        except LLMError as e:
            if it is items[0]:
                raise
            return {"error": str(e)[:300]}

    if judge is not None:
        print(f"grading with {judge.model}")
    verdicts = _map(grade, list(zip(items, answers)), rcfg["judge"]["max_workers"]) if judge is not None else [None] * len(items)

    rows = []
    with (out_dir / "responses.jsonl").open("w") as f:
        for it, a, j in zip(items, answers, verdicts):
            row = make_row(it, a, j if j and not j.get("error") else None, patterns, speakers)
            if j and j.get("error"):
                row["judge_error"] = j["error"]
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = build_report(rows, items, rag, judge, full, index_stats, time.time() - t0)
    (out_dir / "eval_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def make_row(it: dict, a: dict, judge: dict | None, patterns: list[str], speakers: dict) -> dict:
    ref = {k: it[k] for k in ("answer", "answer_keys", "answer_aliases", "keypoints", "comparison") if it.get(k)}
    row = {"uid": it["uid"], "dataset": it["dataset"], "item_id": it["item_id"], "type": it["type"], "subtype": it.get("subtype"),
           "answerable": it.get("answerable", True), "question": it["question"], "reference": ref,
           "evidence": [{"unit_id": e["unit_id"], "doc_id": e["doc_id"], "pages": e["pages"]} for e in it.get("evidence") or []]}
    if a.get("error"):
        return {**row, "error": a["error"], "score": 0.0, "category": "error"}
    scored = score_item(it, a["answer"], a["passages"], judge, patterns, speakers)
    return {**row, "response": a["answer"], "context": a.get("context", "retrieved"), **scored, "judge": judge,
            "retrieved": [{k: h[k] for k in ("chunk_id", "rank", "score", "ranks", "in_context", "doc_id", "pages")} for h in a["retrieved"]],
            "context_chars": a["context_chars"], "usage": a["usage"], "latency_s": a["latency_s"]}


def build_report(rows: list[dict], items: list[dict], rag: RAG, judge, full: dict, index_stats: dict | None, elapsed: float) -> dict:
    rcfg, ecfg = full["rag"], full["rag"]["eval"]
    n_boot, seed = ecfg["bootstrap"], ecfg["seed"]
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    prompt_tokens = [r["usage"].get("prompt_tokens", 0) for r in rows if r.get("usage")]
    latencies = [r["latency_s"] for r in rows if r.get("latency_s") is not None and not r["usage"].get("cached")]
    warnings = []
    if errors := sum(bool(r.get("error")) for r in rows):
        warnings.append(f"{errors} items failed to answer (see `error` in responses.jsonl)")
    if judge_errors := sum(bool(r.get("judge_error")) for r in rows):
        warnings.append(f"{judge_errors} items have no judge verdict (see `judge_error`); they are scored on keys")
    window = rcfg["llm"]["context_window"] - rcfg["llm"]["max_tokens"]
    if prompt_tokens and max(prompt_tokens) > window:
        warnings.append(f"a prompt of {max(prompt_tokens)} tokens leaves less than max_tokens in a {rcfg['llm']['context_window']}-token window")
    fixed = [r for r in rows if r.get("context") == "fixed"]
    if empty := sum(1 for r in fixed if not r.get("retrieved")):
        warnings.append(f"{empty} unanswerable items had no distractor chunk left after dropping those holding the evidence")
    return {
        "datasets": sorted({it["dataset"] for it in items}),
        "items": len(rows),
        "rag": {"model": rag.llm.model, "base_url": rcfg["llm"]["base_url"], "temperature": rcfg["llm"]["temperature"],
                "embedding_model": rcfg["embedding"]["model"] if rag.index.embedder else None,
                "retrieval": rcfg["retrieval"], "chunking": rcfg["chunking"], "prompt_version": PROMPT_VERSION},
        "judge_model": judge.model if judge is not None else None,
        "index": index_stats,
        "fixed_context_items": len(fixed),   # unanswerable items read with their stored distractors, not a search
        "overall": summarize(rows, n_boot, seed),
        "answerable": summarize([r for r in rows if r["answerable"]], n_boot, seed),
        "unanswerable": summarize([r for r in rows if not r["answerable"]], n_boot, seed),
        "per_type": {t: summarize(by_type[t], n_boot, seed) for t in TYPES if by_type.get(t)},
        "agreement": agreement(rows),
        "usage": {"prompt_tokens_max": max(prompt_tokens, default=0),
                  "prompt_tokens_mean": round(sum(prompt_tokens) / len(prompt_tokens)) if prompt_tokens else 0,
                  "latency_s_mean": round(sum(latencies) / len(latencies), 2) if latencies else None},
        "llm": {"rag": dict(rag.llm.stats), "judge": dict(judge.stats) if judge is not None else None},
        "elapsed_seconds": round(elapsed, 1),
        "warnings": warnings,
    }


def _f(x: float | None) -> str:
    return "  -  " if x is None else f"{x:.2f}"


def print_report(report: dict) -> None:
    r = report["rag"]
    print(f"{report['items']} items from {', '.join(report['datasets'])}")
    print(f"  RAG: {r['model']}, {r['retrieval']['mode']} top {r['retrieval']['top_k']} (prompt v{r['prompt_version']}); "
          f"judge: {report['judge_model'] or 'none (keys only)'}")
    print(f"  {'':14s} {'n':>3s}  {'score':>5s}  {'95% CI':13s}  {'hit':>4s}  {'all-ev':>6s}  {'MRR':>4s}  {'keys':>4s}  {'grounded':>8s}")

    def line(name: str, s: dict) -> None:
        ci = f"[{s['ci95'][0]:.2f}, {s['ci95'][1]:.2f}]" if s["ci95"] else "-"
        print(f"  {name:14s} {s['items']:3d}  {_f(s['score']):>5s}  {ci:13s}  {_f(s['hit']):>4s}  {_f(s['all_evidence']):>6s}  "
              f"{_f(s['mrr']):>4s}  {_f(s['key_recall']):>4s}  {_f(s['grounded']):>8s}")

    for t, s in report["per_type"].items():
        line(t, s)
    for name in ("answerable", "unanswerable", "overall"):
        if report[name]["items"]:
            line(name, report[name])
    o = report["overall"]
    if o["keypoint_completeness"] is not None:
        print(f"  summaries: keypoints covered {o['keypoint_completeness']:.2f}, contradicted {o['keypoint_hallucination']:.2f}")
    print("  outcomes: " + ", ".join(f"{k} {v}" for k, v in sorted(o["categories"].items(), key=lambda kv: -kv[1])))
    ag = report["agreement"]
    if ag["items"]:
        print(f"  judge vs keys: agree on {ag['agree']:.2f} of {ag['items']} items; keys only: {ag['keys_only'] or '-'}; judge only: {ag['judge_only'] or '-'}")
    u, llm = report["usage"], report["llm"]
    print(f"  RAG calls {llm['rag'].get('calls', 0)} ({llm['rag'].get('cache_hits', 0)} cached), prompt tokens max {u['prompt_tokens_max']} "
          f"mean {u['prompt_tokens_mean']}, latency mean {u['latency_s_mean']}s"
          + (f"; judge calls {llm['judge'].get('calls', 0)} ({llm['judge'].get('cache_hits', 0)} cached)" if llm["judge"] else ""))
    for w in report["warnings"]:
        print(f"  WARNING: {w}")


# ----- CLI -----

def _common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--config", type=Path)
    ap.add_argument("--mode", choices=MODES, help="retrieval mode (default: rag.retrieval.mode)")
    ap.add_argument("--top-k", type=int, help="passages shown to the model (default: rag.retrieval.top_k)")


def add_index_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("interim_dir", type=Path, help="output directory of `medrageval parse`")
    _common(ap)


def run_index_cli(args: argparse.Namespace) -> None:
    full = apply_overrides(load_config(args.config), args)
    t0 = time.time()
    try:
        _, corpus, stats = load_index(args.interim_dir, full, args.mode)
    except LLMError as e:
        raise SystemExit(f"error: {e}\n(is the embedding server at {full['rag']['embedding']['base_url']} running?)")
    emb = stats.get("embedding") or {}
    print(f"{args.interim_dir}: {stats['chunks']} chunks from {len(corpus.units)} units in {len(corpus.docs)} documents "
          f"({stats['chars']:,} chars, longest {stats['max_chars']}, {stats['split_parts']} pieces of split units)")
    if emb:
        how = "loaded from cache" if emb.get("cached") else f"embedded in {emb.get('embed_seconds')}s"
        print(f"  embeddings: {emb['embedding_model']}, {how}")
    print(f"  index in {Path(full['rag']['index_dir']) / args.interim_dir.name}/ ({time.time() - t0:.1f}s)")


def add_ask_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("interim_dir", type=Path, help="output directory of `medrageval parse`")
    ap.add_argument("question")
    ap.add_argument("--model", help="model id on the vLLM server (default: rag.llm.model)")
    ap.add_argument("--no-cache", action="store_true")
    _common(ap)


def run_ask_cli(args: argparse.Namespace) -> None:
    full = apply_overrides(load_config(args.config), args)
    rcfg = full["rag"]
    try:
        index, _, _ = load_index(args.interim_dir, full, args.mode)
        res = RAG(index, make_local_llm(rcfg["llm"], rcfg["cache_dir"], not args.no_cache), rcfg).answer(args.question)
    except LLMError as e:
        raise SystemExit(f"error: {e}\n(are the servers at {rcfg['llm']['base_url']} and {rcfg['embedding']['base_url']} running?)")
    print(res["answer"])
    print(f"\n{len(res['passages'])} passages ({res['context_chars']} chars, {res['latency_s']}s):")
    for i, c in enumerate(res["passages"], start=1):
        head, _, body = c.text.partition("\n")
        print(f"  [{i}] {c.chunk_id} {head}\n      {' '.join(body.split())[:160]}")


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("datasets", type=Path, nargs="+", help="dataset directories (or dataset.jsonl files) from `medrageval generate`")
    ap.add_argument("--out", type=Path, help="run directory (default: data/runs/<first dataset>)")
    ap.add_argument("--interim", type=Path, help="parse output to index (default: the datasets' provenance.interim_dir)")
    ap.add_argument("--model", help="model id on the vLLM server (default: rag.llm.model)")
    ap.add_argument("--judge-model", help="OpenRouter model id for the judge (default: rag.judge.model, else generate.model)")
    ap.add_argument("--no-judge", action="store_true", help="score on answer keys only; no OpenRouter calls")
    ap.add_argument("--limit", type=int, help="only the first N items (smoke runs)")
    ap.add_argument("--no-cache", action="store_true", help="ignore and do not write the response cache")
    _common(ap)


def run_cli(args: argparse.Namespace) -> None:
    full = apply_overrides(load_config(args.config), args)
    rcfg = full["rag"]
    items = load_items(args.datasets, args.limit)
    if not items:
        raise SystemExit("error: no items in the given datasets")
    interim = resolve_interim(items, args.interim)
    judge = None
    if not args.no_judge:   # before any answering: a missing key fails here, not after the RAG pass
        try:
            judge = make_llm(rcfg["judge"], full["generate"], use_cache=not args.no_cache)
        except RuntimeError as e:
            raise SystemExit(f"error: {e} (the judge runs on OpenRouter; --no-judge scores on keys only)")
    out = args.out or Path("data/runs") / args.datasets[0].name
    try:
        index, corpus, stats = load_index(interim, full)
        print(f"{interim}: {stats['chunks']} chunks over {len(corpus.docs)} documents")
        rag = RAG(index, make_local_llm(rcfg["llm"], rcfg["cache_dir"], not args.no_cache), rcfg)
        report = evaluate(items, rag, judge, full, out, stats)
    except LLMError as e:
        where = f"the vLLM servers at {rcfg['llm']['base_url']} and {rcfg['embedding']['base_url']}"
        raise SystemExit(f"error: {e}\n(check {where}{', and the judge model on OpenRouter' if judge is not None else ''})")
    print_report(report)
    print(f"outputs in {out}/")
