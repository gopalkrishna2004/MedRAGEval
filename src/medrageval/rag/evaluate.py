"""Run the RAG application over evaluation datasets and score every answer.

The application (`medrag`) is built from its own config and asked each
question through its Python interface, exactly as a user would. Two passes:
first the answering model on every question (the first one alone, so a server
that is down stops the run before the rest are sent); then the judge on
OpenRouter grades each answer against the item and the passages shown. Both
are cached, so rescoring after a judge change makes no answering calls.

Unanswerable items are read over stand-in passages instead of a search
(`standin_passages`): the application's chunks that hold the withheld quotes
are found, the application's own retrieval is asked for the chunks nearest to
each of them, the gold chunks are removed, and the judge screens the rest
against the question and the withheld answer in one call; the first `top_k`
that it finds do not answer are the context. The reader is thereby screened on
exactly the text it is shown, so no string rule stands between the screening
and the context. An item with no passage left (or no judge to screen with) is
run but not scored (`excluded`): an empty context would make its decline a
free point.

Writes responses.jsonl (one row per item: the reply, the passages, the
candidate pool, retrieval, keys, judge, outcome) and eval_report.json (per type and overall, with 95%
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
from medrageval.generate.prompts import withhold_check
from medrageval.rag.judge import JUDGE_VERSION, build_judge
from medrageval.rag.score import _numbers, agreement, norm_alnum, quote_split_pair, score_item, summarize

ABLATION_MODES = ("bm25", "dense", "hybrid", "hybrid+rerank")


def load_items(dataset_dirs: list[Path], limit: int | None = None, types: list[str] | None = None) -> list[dict]:
    """Items of one or more datasets, optionally of the given types only. With several datasets, ids are prefixed by the
    dataset directory, since each run numbers from 001."""
    items = []
    for d in dataset_dirs:
        path = d / "dataset.jsonl" if d.is_dir() else d
        name = (path.parent if path.name == "dataset.jsonl" else path).name
        for line in path.open():
            if line.strip():
                it = json.loads(line)
                it["dataset"] = name
                it["uid"] = f"{name}/{it['item_id']}" if len(dataset_dirs) > 1 else it["item_id"]
                if not types or it["type"] in types:
                    items.append(it)
    return items[:limit] if limit else items


def gold_chunks(item: dict, chunks: list[Chunk]) -> list[Chunk]:
    """The application's chunks that hold a withheld quote, on any page (a table copied into another document counts),
    plus both halves of a quote split across two adjacent chunks."""
    evidence = (item.get("withheld") or {}).get("evidence") or []
    out: dict[str, Chunk] = {}
    for e in evidence:
        q = norm_alnum(e.get("quote") or "")
        if not q:
            continue
        holders = [c for c in chunks if q in norm_alnum(c.text)]
        out.update((c.chunk_id, c) for c in holders)
        if not holders:   # the application's chunking may cut through the quote
            pair = quote_split_pair(e, chunks)
            if pair:
                out.update((chunks[i].chunk_id, chunks[i]) for i in pair)
    return list(out.values())


def render_chunk(c: Chunk, max_chars: int) -> str:
    """A chunk for the screening judge: its id in brackets, where it is from, and its body cut at a line boundary."""
    section = " > ".join(" ".join(h.split())[:80] for h in c.heading_path[-3:]) or "-"
    body = c.body
    if len(body) > max_chars:
        cut = body.rfind("\n", 0, max_chars)
        body = body[: cut if cut > max_chars // 2 else max_chars] + "\n[...]"
    return f"[{c.chunk_id}] Document: {c.document_title or '-'} | Section: {section} | Pages: {c.page_start}-{c.page_end}\n{body}"


def withheld_figures(keys: list[str]) -> set[float]:
    """The figures a stand-in must not state, when the withheld keys carry specific numbers: a decimal, a number of 100 or
    more, or two or more numbers together. A lone small integer ("phase 3 trial", "10 RCTs") or a key without numbers
    ("company") sets none, since those digits mark nothing."""
    nums: set[float] = set()
    for k in keys:
        nums |= _numbers(k)
    if len(nums) >= 2 or any(n != int(n) or n >= 100 for n in nums):
        return nums
    return set()


def standin_passages(item: dict, rag: RAG, screener, ecfg: dict, top_k: int, mode: str | None = None) -> dict:
    """The context an unanswerable item is read over, from the application's own chunks.

    The chunks nearest to each gold chunk by the application's retrieval (`standin_search_k` each, the gold chunks
    removed; the withheld quotes are the queries when no chunk holds them) are screened by the judge against the
    question and the withheld answer in one call; the first `top_k` it finds do not answer are the passages. A chunk that
    states every withheld figure (`withheld_figures`) is dropped whatever the judge said ("figures"): the judge has passed
    a value sitting in row 30 of a long table.
    Returns {"passages", "gold", "screening", "excluded"}: `excluded` is the reason when no passage is left."""
    withheld = item.get("withheld") or {}
    gold = gold_chunks(item, rag.retriever.chunks)
    gold_ids = {c.chunk_id for c in gold}
    queries = [c.text for c in gold] or [e["quote"] for e in withheld.get("evidence") or [] if e.get("quote")] or [item["question"]]
    k = ecfg.get("standin_search_k", 10)
    ranked = [[h.chunk for h in rag.retriever.search(q, mode, k, rerank=False)] for q in queries]
    pool: dict[str, Chunk] = {}
    for i in range(k):   # interleaved, so every gold chunk's neighbours are represented when the pool is cut
        for r in ranked:
            if i < len(r) and r[i].chunk_id not in gold_ids:
                pool.setdefault(r[i].chunk_id, r[i])
    candidates = list(pool.values())[:ecfg.get("standin_candidates", 12)]
    result = {"passages": [], "gold": [c.chunk_id for c in gold], "screening": [], "excluded": None}
    if not candidates:
        result["excluded"] = "no candidate chunk near the withheld evidence"
        return result
    def screen(chunks: list[Chunk], tag: str) -> dict[str, dict]:
        shown = "\n\n".join(render_chunk(c, ecfg.get("standin_max_chars", 6000)) for c in chunks)
        check = screener.complete_json(withhold_check(shown, item["question"], str(withheld.get("answer") or ""), rag.topic, "chunk_id", every_entry=True), tag)
        return {str(r.get("chunk_id")): r for r in (check.get("results") or []) if isinstance(r, dict)}

    verdicts = screen(candidates, f"screen:{item['uid']}")
    if missing := [c for c in candidates if c.chunk_id not in verdicts]:   # the judge has listed only the passages it found answering
        verdicts.update(screen(missing, f"screen2:{item['uid']}"))
    figures = withheld_figures([str(k) for k in withheld.get("answer_keys") or []])
    for c in candidates:
        r = verdicts.get(c.chunk_id, {})
        verdict = str(r.get("answers") or "unscreened")
        if figures and figures <= _numbers(c.text):
            verdict = "figures"
        result["screening"].append({"chunk_id": c.chunk_id, "answers": verdict, "answer_if_any": r.get("answer_if_any")})
        if verdict == "no" and len(result["passages"]) < top_k:
            result["passages"].append(c)
    if not result["passages"]:
        unscreened = sum(s["answers"] == "unscreened" for s in result["screening"])
        result["excluded"] = ("the judge gave no verdict on any candidate chunk near the withheld evidence" if unscreened == len(candidates) else
                              f"every candidate chunk near the withheld evidence was screened as answering ({unscreened} left unscreened)")
    return result


def _map(fn, xs: list, workers: int) -> list:
    """fn over xs: the first alone, so a dead server or a bad key fails once instead of once per item."""
    if not xs:
        return []
    first = fn(xs[0])
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return [first] + list(pool.map(fn, xs[1:]))


def evaluate(items: list[dict], rag: RAG, judge, full: dict, out_dir: Path, mode: str | None = None) -> dict:
    rcfg, ecfg = full["rag"], full["rag"]["eval"]
    patterns, speakers = ecfg["abstain_patterns"], full["generate"]["attribution"]["speakers"]
    app_cfg = rag.cfg
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    standins: dict[str, dict] = {}

    def answer(it: dict) -> dict:
        fixed = None
        if it.get("withheld") and not it.get("answerable", True):   # read over stand-ins, not a search
            if judge is None:
                return {"excluded": "stand-in passages need the judge to screen them (run without --no-judge)"}
            try:
                s = standins[it["uid"]] = standin_passages(it, rag, judge, ecfg, app_cfg["retrieval"]["top_k"], mode)
            except LLMJsonError as e:
                return {"excluded": f"screening gave no verdict: {str(e)[:200]}"}
            except LLMError as e:
                if it is items[0]:
                    raise
                return {"error": str(e)[:300]}
            if s["excluded"]:   # an empty context makes a decline certain: a free point, so the item is not scored
                return {"excluded": s["excluded"]}
            fixed = s["passages"]
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
        if judge is None or a.get("error") or a.get("excluded"):
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
            if s := standins.get(it["uid"]):
                row["gold_chunks"], row["screening"] = s["gold"], s["screening"]
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
    if a.get("excluded"):
        return {**row, "context": "fixed", "excluded": a["excluded"], "score": None, "category": "excluded", "passages": []}
    candidates = a.get("candidates")
    scored = score_item(it, a["answer"], a["passages"], judge, patterns, speakers, candidates)
    return {**row, "response": a["answer"], "context": a.get("context", "retrieved"), **scored, "judge": judge,
            "passages": [{"chunk_id": c.chunk_id, "document_title": c.document_title, "pages": [c.page_start, c.page_end], "kind": c.kind}
                         for c in a["passages"]],
            "retrieved": [{**h, "in_context": h.get("in_context")} for h in a["retrieved"]],
            "candidates": [c.chunk_id for c in candidates] if candidates is not None else None,   # the fused pool, before reranking
            "context_chars": a["context_chars"], "usage": a["usage"], "latency_s": a["latency_s"]}


def build_report(rows: list[dict], items: list[dict], rag: RAG, judge, full: dict, elapsed: float, mode: str | None) -> dict:
    rcfg, ecfg = full["rag"], full["rag"]["eval"]
    app_cfg = rag.cfg
    n_boot, seed = ecfg["bootstrap"], ecfg["seed"]
    excluded = [r["uid"] for r in rows if r.get("excluded")]
    all_rows, rows = rows, [r for r in rows if not r.get("excluded")]   # summaries are over the items actually scored
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
    fixed = [r for r in all_rows if r.get("context") == "fixed"]
    if excluded:
        warnings.append(f"{len(excluded)} unanswerable items not scored (see `excluded` in responses.jsonl): {', '.join(excluded)}")
    screened = [r for r in all_rows if r.get("screening")]
    if no_gold := [r["uid"] for r in screened if not r.get("gold_chunks")]:
        warnings.append(f"{len(no_gold)} unanswerable items had no chunk holding a withheld quote; the quotes were the search queries: {', '.join(no_gold)}")
    screening = {"items": len(screened),
                 "candidates": sum(len(r["screening"]) for r in screened),
                 "answers": sum(1 for r in screened for s in r["screening"] if s["answers"] in ("yes", "partly")),
                 "figures": sum(1 for r in screened for s in r["screening"] if s["answers"] == "figures"),   # dropped by the figure check alone
                 "unscreened": sum(1 for r in screened for s in r["screening"] if s["answers"] == "unscreened"),
                 "passages_mean": round(sum(len(r["passages"]) for r in fixed if r.get("passages")) / len(fixed), 2) if fixed else None}
    overall = summarize(rows, n_boot, seed)
    return {
        "datasets": sorted({it["dataset"] for it in items}),
        "items": len(rows),
        "excluded": excluded,   # items run but not scored (see `excluded` in responses.jsonl)
        "rag": {"model": rag.llm.model, "base_url": app_cfg["llm"]["base_url"], "temperature": app_cfg["llm"].get("temperature"),
                "embedding_model": app_cfg["embedding"]["model"] if rag.retriever.embedder else None,
                "reranker": rag.retriever.reranker.model if rag.retriever.reranker else None,
                "retrieval": {**app_cfg["retrieval"], "mode": mode or app_cfg["retrieval"]["mode"], "rerank": rag.retriever.reranker is not None},
                "chunking": app_cfg["chunking"], "collection": rag.retriever.collection, "chunks": len(rag.retriever.chunks),
                "prompt_version": PROMPT_VERSION, "config": app_cfg.get("_path")},
        "judge_model": judge.model if judge is not None else None,
        "judge_version": JUDGE_VERSION,
        "fixed_context_items": len(fixed),   # unanswerable items read over screened stand-in passages, not a search
        "standin_screening": screening,       # candidates near the withheld evidence, and how many the judge found answering
        "overall": overall,
        "answerable": summarize([r for r in rows if r["answerable"]], n_boot, seed),
        "unanswerable": summarize([r for r in rows if not r["answerable"]], n_boot, seed),
        "per_type": {t: summarize(by_type[t], n_boot, seed) for t in TYPES if by_type.get(t)},
        "agreement": agreement(rows),
        "evidence_matching": {"page_only": overall["evidence_page_only"], "quote_split": overall["evidence_quote_split"]},
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
    k, n = r["retrieval"]["top_k"], r["retrieval"]["candidates"]
    print(f"  {'':14s} {'n':>3s}  {'score':>5s}  {'95% CI':13s}  {'hit':>4s}  {'all-ev':>6s}  {'MRR':>4s}  {'gold_prec@' + str(k):>12s}  "
          f"{'cand@' + str(n):>7s}  {'keys':>4s}  {'ground':>6s}  {'faith':>5s}")

    def line(name: str, s: dict) -> None:
        ci = f"[{s['ci95'][0]:.2f}, {s['ci95'][1]:.2f}]" if s["ci95"] else "-"
        print(f"  {name:14s} {s['items']:3d}  {_f(s['score']):>5s}  {ci:13s}  {_f(s['hit']):>4s}  {_f(s['all_evidence']):>6s}  "
              f"{_f(s['mrr']):>4s}  {_f(s['gold_prec']):>12s}  {_f(s['candidate_recall']):>7s}  {_f(s['key_recall']):>4s}  "
              f"{_f(s['grounded']):>6s}  {_f(s['faithfulness']):>5s}")

    for t, s in report["per_type"].items():
        line(t, s)
    for name in ("answerable", "unanswerable", "overall"):
        if report[name]["items"]:
            line(name, report[name])
    o = report["overall"]
    if o["keypoint_completeness"] is not None:
        print(f"  summaries: keypoints covered {o['keypoint_completeness']:.2f}, contradicted {o['keypoint_hallucination']:.2f}")
    print("  outcomes: " + ", ".join(f"{k} {v}" for k, v in sorted(o["categories"].items(), key=lambda kv: -kv[1])))
    em, sc = report["evidence_matching"], report["standin_screening"]
    print(f"  evidence matching: {em['page_only']} page-only (not credited), {em['quote_split']} split across adjacent passages")
    if sc["items"]:
        print(f"  stand-ins: {sc['items']} unanswerable items, {sc['candidates']} candidate chunks screened, {sc['answers']} dropped as answering, "
              f"{sc['figures']} for stating the withheld figures, {sc['unscreened']} unscreened; {sc['passages_mean']} passages shown on average")
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
    ap.add_argument("--model", help="answering model id on the vLLM server (default: the application's llm.model)")
    ap.add_argument("--base-url", help="the answering server (default: the application's llm.base_url)")
    ap.add_argument("--judge-model", help="OpenRouter model id for the judge (default: rag.judge.model, else generate.model)")
    ap.add_argument("--no-judge", action="store_true", help="score on answer keys only; no OpenRouter calls")
    ap.add_argument("--limit", type=int, help="only the first N items (smoke runs)")
    ap.add_argument("--types", help=f"comma list of item types to run, from {', '.join(TYPES)} (default: all)")
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


def _types(args: argparse.Namespace) -> list[str] | None:
    types = [t.strip() for t in args.types.split(",") if t.strip()] if args.types else None
    for t in types or []:
        if t not in TYPES:
            raise SystemExit(f"error: unknown type {t!r}; choose from {TYPES}")
    return types


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


def _run(items: list[dict], app: App, judge, full: dict, out: Path, mode: str, rerank: bool) -> dict:
    try:
        rag = app.rag(mode, rerank)
        print(f"{rag.retriever.collection}: {len(rag.retriever.chunks)} chunks")
        return evaluate(items, rag, judge, full, out, mode)
    except (RagLLMError, RuntimeError) as e:
        raise SystemExit(f"error: {e}\n(check the answering server at {app.cfg['llm']['base_url']}, the database, and that `medrag ingest` has run)")
    except LLMError as e:
        raise SystemExit(f"error: {e} (the judge on OpenRouter)")


def run_cli(args: argparse.Namespace) -> None:
    full = copy.deepcopy(load_config(args.config))
    items = load_items(args.datasets, args.limit, _types(args))
    if not items:
        raise SystemExit("error: no items in the given datasets")
    judge = _judge(args, full)   # before any answering: a missing key fails here, not after the RAG pass
    app = _app(args)
    mode = args.mode or app.cfg["retrieval"]["mode"]
    rerank = False if args.no_rerank else app.cfg["retrieval"].get("rerank", False)
    out = args.out or Path("data/runs") / args.datasets[0].name
    report = _run(items, app, judge, full, out, mode, rerank)
    print_report(report)
    print(f"outputs in {out}/")


def run_ablate_cli(args: argparse.Namespace) -> None:
    """Every retrieval mode (and chunk size) on the same items, each as its own run directory, with a summary table."""
    full = copy.deepcopy(load_config(args.config))
    items = load_items(args.datasets, args.limit, _types(args))
    if not items:
        raise SystemExit("error: no items in the given datasets")
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
            report = _run(items, app, judge, full, args.out_root / label, mode, rerank)
            o = report["overall"]
            summary.append({"run": label, "chunk_tokens": size, "mode": mode, "rerank": rerank, **{k: o[k] for k in
                            ("items", "score", "ci95", "hit", "all_evidence", "evidence_recall", "mrr", "gold_prec",
                             "candidate_recall", "key_recall", "grounded", "faithfulness")},
                            "categories": o["categories"]})
            print(f"  {label}: score {_f(o['score'])}  hit {_f(o['hit'])}  all-ev {_f(o['all_evidence'])}  MRR {_f(o['mrr'])}  "
                  f"gold_prec {_f(o['gold_prec'])}  cand {_f(o['candidate_recall'])}  faith {_f(o['faithfulness'])}")
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "ablation.json").write_text(json.dumps({"datasets": sorted({it['dataset'] for it in items}), "runs": summary}, indent=2))
    print(f"\n  {'run':22s} {'score':>5s}  {'hit':>4s}  {'all-ev':>6s}  {'recall':>6s}  {'MRR':>4s}  {'gold_prec':>9s}  {'cand':>4s}  {'keys':>4s}  {'faith':>5s}")
    for s in summary:
        print(f"  {s['run']:22s} {_f(s['score']):>5s}  {_f(s['hit']):>4s}  {_f(s['all_evidence']):>6s}  {_f(s['evidence_recall']):>6s}  "
              f"{_f(s['mrr']):>4s}  {_f(s['gold_prec']):>9s}  {_f(s['candidate_recall']):>4s}  {_f(s['key_recall']):>4s}  {_f(s['faithfulness']):>5s}")
    print(f"summary in {args.out_root / 'ablation.json'}")
