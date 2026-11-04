"""Run the RAG application over evaluation datasets and score every answer.

The application (`medrag`) is built from its own config and asked each
question through its Python interface, exactly as a user would. Two passes:
first the answering model on every question (the first one alone, so a server
that is down stops the run before the rest are sent); then the judge on
OpenRouter grades each answer against the item and the passages shown. Both
are cached, so rescoring after a judge change makes no answering calls.

Unanswerable items are the exception to the application's independence: they
are read over the benchmark's stored distractors, mapped to the application's
chunks by page and text overlap (`distractor_passages`), with any chunk that
holds the withheld evidence dropped, so the evidence is certainly absent.

Writes responses.jsonl (one row per item: the reply, the passages, retrieval,
keys, judge, outcome) and eval_report.json (per type and overall, with 95%
bootstrap intervals), and prints the summary.
"""

import argparse
import copy
import json
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from medrag.app import App, collection_name
from medrag.chunk import Chunk
from medrag.generate import PROMPT_VERSION, RAG
from medrag.llm import LLMError as RagLLMError
from medrag.retrieve import MODES

from medrageval.config import load_config
from medrageval.generate.generate import TYPES
from medrageval.generate.llm import LLMError, LLMJsonError, make_llm
from medrageval.rag.judge import JUDGE_VERSION, build_judge
from medrageval.rag.score import agreement, key_found, norm_alnum, pages_overlap, score_item, summarize

ABLATION_MODES = ("bm25", "dense", "hybrid", "hybrid+rerank")


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


def interim_dir(items: list[dict], override: Path | None) -> Path | None:
    """The benchmark's parse output the datasets came from (for the unit texts of distractors), if it is on disk."""
    if override:
        return override
    dirs = {it.get("provenance", {}).get("interim_dir") for it in items} - {None}
    if len(dirs) == 1 and Path(next(iter(dirs))).exists():
        return Path(dirs.pop())
    return None


def load_unit_texts(interim: Path | None) -> dict[str, str]:
    """element id -> plain text, from the benchmark's element store. A row group "e0034#r02" maps to its element's text."""
    if not interim or not (interim / "elements.jsonl").exists():
        return {}
    out = {}
    for line in (interim / "elements.jsonl").open():
        if line.strip():
            e = json.loads(line)
            out[e["element_id"]] = e.get("plain") or e.get("text") or ""
    return out


def _shingles(text: str, n: int = 12, step: int = 4) -> set[str]:
    t = norm_alnum(text)
    return {t[i:i + n] for i in range(0, max(1, len(t) - n + 1), step)} if len(t) >= n else ({t} if t else set())


def unit_chunks(d: dict, chunks: list[Chunk], unit_texts: dict[str, str], min_overlap: float, per_unit: int) -> list[Chunk]:
    """The application's chunks standing for one benchmark unit: on its pages, ranked by shared text when the unit's
    text is known, else holding its caption, else the first chunks of the page."""
    on_pages = [c for c in chunks if pages_overlap(d.get("pages"), c)]
    if not on_pages:
        return []
    text = unit_texts.get(str(d.get("unit_id", "")).split("#")[0]) or unit_texts.get(str(d.get("element_id", "")))
    if text:
        sh = _shingles(text)
        if sh:
            scored = sorted(((len(sh & _shingles(c.text, step=1)) / len(sh), c) for c in on_pages), key=lambda x: -x[0])
            return [c for s, c in scored[:per_unit] if s >= min_overlap]
    cap = norm_alnum(d.get("caption") or "")
    if cap:
        with_cap = [c for c in on_pages if cap in norm_alnum(c.text)]
        if with_cap:
            return with_cap[:per_unit]
    return on_pages[:per_unit]


def distractor_passages(item: dict, chunks: list[Chunk], unit_texts: dict[str, str], ecfg: dict, top_k: int) -> tuple[list[Chunk], int]:
    """(passages, unmapped): the stored distractors as application chunks, minus any holding the withheld evidence."""
    withheld = item.get("withheld") or {}
    quotes = [norm_alnum(e.get("quote") or "") for e in withheld.get("evidence") or [] if e.get("quote")]
    keys = [str(k) for k in withheld.get("answer_keys") or []]
    out: list[Chunk] = []
    seen: set[str] = set()
    unmapped = 0
    for d in item.get("distractors") or []:
        cs = unit_chunks(d, chunks, unit_texts, ecfg.get("distractor_min_overlap", 0.3), ecfg.get("distractor_chunks_per_unit", 2))
        if not cs:
            unmapped += 1
        for c in cs:
            if c.chunk_id in seen:
                continue
            seen.add(c.chunk_id)
            t = norm_alnum(c.text)
            if any(q and q in t for q in quotes) or (keys and all(key_found(k, c.text) for k in keys)):
                continue
            out.append(c)
        if len(out) >= top_k:
            break
    return out[:top_k], unmapped


def _map(fn, xs: list, workers: int) -> list:
    """fn over xs: the first alone, so a dead server or a bad key fails once instead of once per item."""
    if not xs:
        return []
    first = fn(xs[0])
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return [first] + list(pool.map(fn, xs[1:]))


def evaluate(items: list[dict], rag: RAG, judge, full: dict, out_dir: Path, unit_texts: dict[str, str] | None = None,
             mode: str | None = None) -> dict:
    rcfg, ecfg = full["rag"], full["rag"]["eval"]
    patterns, speakers = ecfg["abstain_patterns"], full["generate"]["attribution"]["speakers"]
    unit_texts = unit_texts or {}
    app_cfg = rag.cfg
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    unmapped: dict[str, int] = {}

    def answer(it: dict) -> dict:
        fixed = None
        if it.get("distractors") and not it.get("answerable", True):
            fixed, unmapped[it["uid"]] = distractor_passages(it, rag.retriever.chunks, unit_texts, ecfg, app_cfg["retrieval"]["top_k"])
        try:
            return rag.answer(it["question"], tag=f"rag:{it['uid']}", passages=fixed, mode=mode)
        except RagLLMError as e:
            if it is items[0]:
                raise
            return {"error": str(e)[:300]}

    print(f"answering {len(items)} questions with {rag.llm.model} ({mode or app_cfg['retrieval']['mode']}, top {app_cfg['retrieval']['top_k']}"
          f"{', reranked' if rag.retriever.reranker else ''})")
    answers = _map(answer, items, app_cfg["llm"].get("max_workers", 4))

    def grade(pair: tuple[dict, dict]) -> dict | None:
        it, a = pair
        if judge is None or a.get("error"):
            return None
        try:
            return judge.complete_json(build_judge(it, a["answer"], a["passages"], rag.topic, rcfg["judge"]["max_chars_per_passage"]),
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
            if it["uid"] in unmapped:
                row["distractors_unmapped"] = unmapped[it["uid"]]
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = build_report(rows, items, rag, judge, full, time.time() - t0, mode)
    (out_dir / "eval_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def make_row(it: dict, a: dict, judge: dict | None, patterns: list[str], speakers: dict) -> dict:
    ref = {k: it[k] for k in ("answer", "answer_keys", "answer_aliases", "keypoints", "comparison") if it.get(k)}
    row = {"uid": it["uid"], "dataset": it["dataset"], "item_id": it["item_id"], "type": it["type"], "subtype": it.get("subtype"),
           "answerable": it.get("answerable", True), "question": it["question"], "reference": ref,
           "evidence": [{"unit_id": e.get("unit_id"), "doc_id": e.get("doc_id"), "pages": e.get("pages")} for e in it.get("evidence") or []]}
    if a.get("error"):
        return {**row, "error": a["error"], "score": 0.0, "category": "error"}
    scored = score_item(it, a["answer"], a["passages"], judge, patterns, speakers)
    return {**row, "response": a["answer"], "context": a.get("context", "retrieved"), **scored, "judge": judge,
            "passages": [{"chunk_id": c.chunk_id, "document_title": c.document_title, "pages": [c.page_start, c.page_end], "kind": c.kind}
                         for c in a["passages"]],
            "retrieved": [{**h, "in_context": h.get("in_context")} for h in a["retrieved"]],
            "context_chars": a["context_chars"], "usage": a["usage"], "latency_s": a["latency_s"]}


def build_report(rows: list[dict], items: list[dict], rag: RAG, judge, full: dict, elapsed: float, mode: str | None) -> dict:
    rcfg, ecfg = full["rag"], full["rag"]["eval"]
    app_cfg = rag.cfg
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
    window = app_cfg["llm"].get("context_window", 8192) - app_cfg["llm"].get("max_tokens", 512)
    if prompt_tokens and max(prompt_tokens) > window:
        warnings.append(f"a prompt of {max(prompt_tokens)} tokens leaves less than max_tokens in a {app_cfg['llm'].get('context_window')}-token window")
    if empties := sum(1 for r in rows if r.get("response") == ""):
        warnings.append(f"{empties} replies were empty after a retry (scored as declines; see `usage.finish_reason`): the answering model "
                        f"may be spending max_tokens on reasoning")
    fixed = [r for r in rows if r.get("context") == "fixed"]
    if empty := sum(1 for r in fixed if not r.get("passages")):
        warnings.append(f"{empty} unanswerable items had no distractor chunk left after dropping those holding the evidence")
    if unmapped := sum(r.get("distractors_unmapped", 0) for r in rows):
        warnings.append(f"{unmapped} stored distractors mapped to no chunk of the application (see `distractors_unmapped`)")
    overall = summarize(rows, n_boot, seed)
    return {
        "datasets": sorted({it["dataset"] for it in items}),
        "items": len(rows),
        "rag": {"model": rag.llm.model, "base_url": app_cfg["llm"]["base_url"], "temperature": app_cfg["llm"].get("temperature"),
                "embedding_model": app_cfg["embedding"]["model"] if rag.retriever.embedder else None,
                "reranker": rag.retriever.reranker.model if rag.retriever.reranker else None,
                "retrieval": {**app_cfg["retrieval"], "mode": mode or app_cfg["retrieval"]["mode"], "rerank": rag.retriever.reranker is not None},
                "chunking": app_cfg["chunking"], "collection": rag.retriever.collection, "chunks": len(rag.retriever.chunks),
                "prompt_version": PROMPT_VERSION, "config": app_cfg.get("_path")},
        "judge_model": judge.model if judge is not None else None,
        "judge_version": JUDGE_VERSION,
        "fixed_context_items": len(fixed),   # unanswerable items read with their stored distractors, not a search
        "overall": overall,
        "answerable": summarize([r for r in rows if r["answerable"]], n_boot, seed),
        "unanswerable": summarize([r for r in rows if not r["answerable"]], n_boot, seed),
        "per_type": {t: summarize(by_type[t], n_boot, seed) for t in TYPES if by_type.get(t)},
        "agreement": agreement(rows),
        "evidence_matching": {"page_only": overall["evidence_page_only"], "quote_split": overall["evidence_quote_split"],
                              "distractors_unmapped": sum(r.get("distractors_unmapped", 0) for r in rows)},
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
    print(f"  RAG: {r['model']}, {r['retrieval']['mode']} top {r['retrieval']['top_k']}{' + rerank' if r['retrieval'].get('rerank') else ''} "
          f"over {r['chunks']} chunks (prompt v{r['prompt_version']}); judge: {report['judge_model'] or 'none (keys only)'}")
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
    em = report["evidence_matching"]
    print(f"  evidence matching: {em['page_only']} page-only (not credited), {em['quote_split']} split across adjacent passages, "
          f"{em['distractors_unmapped']} distractors unmapped")
    ag = report["agreement"]
    if ag["items"]:
        print(f"  judge vs keys: agree on {ag['agree']:.2f} of {ag['items']} items; keys only: {ag['keys_only'] or '-'}; judge only: {ag['judge_only'] or '-'}")
    u, llm = report["usage"], report["llm"]
    print(f"  RAG calls {llm['rag'].get('calls', 0)} ({llm['rag'].get('cache_hits', 0)} cached), prompt tokens max {u['prompt_tokens_max']} "
          f"mean {u['prompt_tokens_mean']}, latency mean {str(u['latency_s_mean']) + 's' if u['latency_s_mean'] is not None else '- (all cached)'}"
          + (f"; judge calls {llm['judge'].get('calls', 0)} ({llm['judge'].get('cache_hits', 0)} cached)" if llm["judge"] else ""))
    for w in report["warnings"]:
        print(f"  WARNING: {w}")


# ----- CLI -----

def _common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("datasets", type=Path, nargs="+", help="dataset directories (or dataset.jsonl files) from `medrageval generate`")
    ap.add_argument("--config", type=Path, help="benchmark config (judge, scoring); default configs/default.yaml")
    ap.add_argument("--rag-config", type=Path, help="the application's config; default configs/rag.yaml")
    ap.add_argument("--interim", type=Path, help="benchmark parse output with elements.jsonl, for mapping distractors (default: the datasets' provenance)")
    ap.add_argument("--model", help="answering model id on the vLLM server (default: the application's llm.model)")
    ap.add_argument("--base-url", help="the answering server (default: the application's llm.base_url)")
    ap.add_argument("--judge-model", help="OpenRouter model id for the judge (default: rag.judge.model, else generate.model)")
    ap.add_argument("--no-judge", action="store_true", help="score on answer keys only; no OpenRouter calls")
    ap.add_argument("--limit", type=int, help="only the first N items (smoke runs)")
    ap.add_argument("--no-cache", action="store_true", help="ignore and do not write the response caches")


def add_arguments(ap: argparse.ArgumentParser) -> None:
    _common(ap)
    ap.add_argument("--out", type=Path, help="run directory (default: data/runs/<first dataset>)")
    ap.add_argument("--mode", choices=MODES, help="retrieval mode (default: the application's retrieval.mode)")
    ap.add_argument("--top-k", type=int, help="passages shown to the model")
    ap.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder")


def add_ablate_arguments(ap: argparse.ArgumentParser) -> None:
    _common(ap)
    ap.add_argument("--out-root", type=Path, default=Path("data/runs/ablation"))
    ap.add_argument("--modes", default=",".join(ABLATION_MODES), help=f"comma list from {ABLATION_MODES}")
    ap.add_argument("--chunk-tokens", help="comma list of target chunk sizes, e.g. 256,400,512; each is ingested as its own collection")


def _app(args: argparse.Namespace, chunk_tokens: int | None = None) -> App:
    app = App(args.rag_config, use_cache=not args.no_cache)
    if args.model:
        app.cfg["llm"]["model"] = args.model
    if args.base_url:
        app.cfg["llm"]["base_url"] = args.base_url
    if getattr(args, "top_k", None):
        app.cfg["retrieval"]["top_k"] = args.top_k
    if chunk_tokens:
        app.cfg["chunking"]["target_tokens"] = chunk_tokens
        app.cfg["chunking"]["max_tokens"] = max(chunk_tokens + 64, round(chunk_tokens * 1.28))
    return app


def _judge(args: argparse.Namespace, full: dict):
    if args.no_judge:
        return None
    if args.judge_model:
        full["rag"]["judge"]["model"] = args.judge_model
    try:
        return make_llm(full["rag"]["judge"], full["generate"], use_cache=not args.no_cache)
    except RuntimeError as e:
        raise SystemExit(f"error: {e} (the judge runs on OpenRouter; --no-judge scores on keys only)")


def _run(items: list[dict], app: App, judge, full: dict, out: Path, unit_texts: dict, mode: str, rerank: bool) -> dict:
    try:
        rag = app.rag(mode, rerank)
        print(f"{rag.retriever.collection}: {len(rag.retriever.chunks)} chunks")
        return evaluate(items, rag, judge, full, out, unit_texts, mode)
    except (RagLLMError, RuntimeError) as e:
        raise SystemExit(f"error: {e}\n(check the answering server at {app.cfg['llm']['base_url']}, the database, and that `medrag ingest` has run)")
    except LLMError as e:
        raise SystemExit(f"error: {e} (the judge on OpenRouter)")


def run_cli(args: argparse.Namespace) -> None:
    full = copy.deepcopy(load_config(args.config))
    items = load_items(args.datasets, args.limit)
    if not items:
        raise SystemExit("error: no items in the given datasets")
    unit_texts = load_unit_texts(interim_dir(items, args.interim))
    judge = _judge(args, full)   # before any answering: a missing key fails here, not after the RAG pass
    app = _app(args)
    mode = args.mode or app.cfg["retrieval"]["mode"]
    rerank = False if args.no_rerank else app.cfg["retrieval"].get("rerank", False)
    out = args.out or Path("data/runs") / args.datasets[0].name
    report = _run(items, app, judge, full, out, unit_texts, mode, rerank)
    print_report(report)
    print(f"outputs in {out}/")


def run_ablate_cli(args: argparse.Namespace) -> None:
    """Every retrieval mode (and chunk size) on the same items, each as its own run directory, with a summary table."""
    full = copy.deepcopy(load_config(args.config))
    items = load_items(args.datasets, args.limit)
    if not items:
        raise SystemExit("error: no items in the given datasets")
    unit_texts = load_unit_texts(interim_dir(items, args.interim))
    judge = _judge(args, full)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in ABLATION_MODES:
            raise SystemExit(f"error: unknown mode {m!r}; choose from {ABLATION_MODES}")
    sizes = [int(x) for x in args.chunk_tokens.split(",")] if args.chunk_tokens else [None]
    summary = []
    for size in sizes:
        app = _app(args, size)
        if size:
            from medrag.ingest import ingest
            ingest(app)   # cached when the collection exists
        for m in modes:
            label = f"{'t' + str(size) + '-' if size else ''}{m}"
            mode, rerank = (m.split("+")[0], "+rerank" in m)
            app._retriever = None   # a fresh retriever per mode (embedder / reranker as the mode needs)
            report = _run(items, app, judge, full, args.out_root / label, unit_texts, mode, rerank)
            o = report["overall"]
            summary.append({"run": label, "chunk_tokens": size, "mode": mode, "rerank": rerank, **{k: o[k] for k in
                            ("items", "score", "ci95", "hit", "all_evidence", "evidence_recall", "mrr", "key_recall", "grounded")},
                            "categories": o["categories"]})
            print(f"  {label}: score {_f(o['score'])}  hit {_f(o['hit'])}  all-ev {_f(o['all_evidence'])}  MRR {_f(o['mrr'])}")
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "ablation.json").write_text(json.dumps({"datasets": sorted({it['dataset'] for it in items}), "runs": summary}, indent=2))
    print(f"\n  {'run':22s} {'score':>5s}  {'hit':>4s}  {'all-ev':>6s}  {'recall':>6s}  {'MRR':>4s}  {'keys':>4s}")
    for s in summary:
        print(f"  {s['run']:22s} {_f(s['score']):>5s}  {_f(s['hit']):>4s}  {_f(s['all_evidence']):>6s}  {_f(s['evidence_recall']):>6s}  {_f(s['mrr']):>4s}  {_f(s['key_recall']):>4s}")
    print(f"summary in {args.out_root / 'ablation.json'}")
