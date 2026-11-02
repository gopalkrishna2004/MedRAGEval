"""Scoring one RAG answer against its dataset item, and summarising a run.

Three views of each answer:
- retrieval: did the passages the model was shown hold the item's gold
  evidence? An evidence entry is found when a passage holds its unit (or,
  for a piece of a split unit or another row group of its table, holds its
  quote). `hit` also accepts a copy of the fact in another document
  (`also_in`); `all_evidence` is what multi-hop, cross-document and summary
  items need.
- keys: deterministic. Every answer key in the reply (a number may differ in
  thousands separators or trailing zeros: "£52,681" is "52681", "0.50" is
  "0.5"), the party for attribution, a decline for unanswerable items.
- judge: the frontier judge's verdict, keypoint coverage for summaries
  (RAGEval's completeness / hallucination / irrelevance) and groundedness.

The judge's verdict is the score when there is one; the keys are the score
otherwise, and their agreement with the judge is reported as a check on both.
A wrong answer is put down to retrieval when the evidence was not shown, and
to the reader when it was.
"""

import re
from collections import Counter
from random import Random

from medrageval.facts.factsheet import numbers_in
from medrageval.generate.corpus import tokenize
from medrageval.generate.validate import quote_in
from medrageval.rag.chunks import Chunk
from medrageval.text import contains_quote

VERDICT_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0, "abstained": 0.0}


def _quoted(ev: dict, chunk: Chunk, default: bool) -> bool:
    return quote_in(ev["quote"], chunk.text) if ev.get("quote") else default


def evidence_in(ev: dict, chunk: Chunk) -> bool:
    if ev.get("unit_id") in chunk.unit_ids:
        return chunk.part is None or _quoted(ev, chunk, True)
    if ev.get("element_id") in chunk.element_ids:   # another row group of the same table: only if the quoted row is in it
        return _quoted(ev, chunk, False)
    return False


def retrieval_metrics(item: dict, passages: list[Chunk]) -> dict | None:
    """Gold evidence among the passages shown, in rank order. None for unanswerable items and items without evidence."""
    ev = list({(e["unit_id"], e.get("quote")): e for e in item.get("evidence") or [] if e.get("unit_id")}.values())
    if not item.get("answerable", True) or not ev:
        return None
    found = [any(evidence_in(e, c) for c in passages) for e in ev]
    copies = set(item.get("also_in") or [])
    first = next((rank for rank, c in enumerate(passages, start=1)
                  if any(evidence_in(e, c) for e in ev) or copies & set(c.unit_ids)), None)
    return {"evidence": len(ev), "evidence_found": sum(found), "evidence_recall": round(sum(found) / len(ev), 3),
            "all_evidence": all(found), "hit": first is not None, "first_rank": first, "rr": round(1 / first, 3) if first else 0.0}


def _numbers(text: str) -> set[float]:
    out = set()
    for n in numbers_in(text):
        try:
            out.add(float(n))
        except ValueError:
            pass
    return out


def _stem(t: str) -> str:
    return t[:5] if t.isalpha() else t


def key_found(key: str, text: str) -> bool:
    """The key, verbatim up to punctuation and case; or all of its numbers; or, for a phrase, all its words up to their endings."""
    if contains_quote(key, text):
        return True
    nums = _numbers(key)
    if nums:
        return nums <= _numbers(text)
    words = {_stem(t) for t in tokenize(key) if len(t) >= 3}
    return bool(words) and words <= {_stem(t) for t in tokenize(text)}


def is_abstention(text: str, patterns: list[str]) -> bool:
    return not text.strip() or any(re.search(p, text, re.I) for p in patterns)


def _named(alias: str, text: str) -> bool:
    return bool(re.search(rf"\b{re.escape(alias)}\b", text, re.I))


def key_scores(item: dict, response: str, patterns: list[str], speakers: dict[str, list[str]]) -> dict:
    """Deterministic scores. `correct` is None where keys cannot decide (summaries)."""
    declined = is_abstention(response, patterns)
    if not item.get("answerable", True):
        return {"abstained": declined, "key_recall": None, "correct": declined}
    if item["type"] == "attribution":
        gold = item.get("answer_aliases") or item.get("answer_keys") or []
        named = [a for a in gold if _named(a, response)]
        gold_set = {a.lower() for a in gold}
        others = sorted(sp for sp, aliases in speakers.items()
                        if not gold_set & {a.lower() for a in [sp, sp.removeprefix("the ")] + aliases}
                        and any(_named(a, response) for a in [sp.removeprefix("the ")] + aliases))
        return {"abstained": declined and not named, "found": named[:1], "other_speakers": others,
                "key_recall": float(bool(named)), "correct": bool(named)}
    keys = [str(k) for k in item.get("answer_keys") or []]
    if item["type"] == "summary" or not keys:
        return {"abstained": declined, "key_recall": None, "correct": None}
    found = [k for k in keys if key_found(k, response)]
    return {"abstained": declined and not found, "found": found, "missing": [k for k in keys if k not in found],
            "key_recall": round(len(found) / len(keys), 3), "correct": len(found) == len(keys)}


def judge_score(item: dict, judge: dict | None) -> float | None:
    if not judge:
        return None
    verdict = str(judge.get("verdict") or "").lower()
    if not item.get("answerable", True):
        return 1.0 if verdict in ("correct", "abstained") else 0.0
    return VERDICT_SCORE.get(verdict)


def keypoint_stats(item: dict, judge: dict | None) -> dict | None:
    """RAGEval's keypoint metrics: shares of keypoints covered, contradicted and missing. A keypoint the judge skipped is missing."""
    kps = item.get("keypoints") or []
    if not kps or not judge:
        return None
    status = {}
    for k in judge.get("keypoints") or []:
        if isinstance(k, dict) and str(k.get("n", "")).isdigit():
            status[int(k["n"])] = str(k.get("status") or "missing")
    st = Counter(status.get(n, "missing") for n in range(1, len(kps) + 1))
    n = len(kps)
    return {"keypoints": n, "completeness": round(st["covered"] / n, 3), "hallucination": round(st["contradicted"] / n, 3),
            "irrelevance": round((n - st["covered"] - st["contradicted"]) / n, 3)}


def category(item: dict, retrieval: dict | None, declined: bool, score: float | None) -> str | None:
    """Why an answer scored what it did: retrieval when the evidence was not shown, the reader when it was."""
    if score is None:
        return None
    if not item.get("answerable", True):
        return "correct_abstention" if score >= 1 else "hallucinated_answer"
    if score >= 1:
        return "correct"
    if retrieval is not None and not retrieval["all_evidence"]:
        return "retrieval_miss"
    return "false_abstention" if declined else "reader_error"


def score_item(item: dict, response: str, passages: list[Chunk], judge: dict | None, patterns: list[str], speakers: dict) -> dict:
    retrieval = retrieval_metrics(item, passages)
    keys = key_scores(item, response, patterns, speakers)
    js = judge_score(item, judge)
    if js is not None:
        score, declined = js, str(judge.get("verdict") or "").lower() == "abstained" or (not item.get("answerable", True) and keys["abstained"])
    else:
        score = None if keys["correct"] is None else float(keys["correct"])
        declined = keys["abstained"]
    return {"retrieval": retrieval, "keys": keys, "judge_score": js, "keypoints": keypoint_stats(item, judge),
            "score": score, "abstained": declined, "category": category(item, retrieval, declined, score)}


def bootstrap_ci(values: list[float], n: int, seed: int) -> list[float] | None:
    """Percentile 95% interval of the mean, resampling items."""
    if len(values) < 2:
        return None
    rng = Random(seed)
    k = len(values)
    means = sorted(sum(rng.choices(values, k=k)) / k for _ in range(n))
    return [round(means[int(0.025 * n)], 3), round(means[min(n - 1, int(0.975 * n))], 3)]


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 3) if xs else None


def summarize(rows: list[dict], n_boot: int, seed: int) -> dict:
    scores = [r["score"] for r in rows if r.get("score") is not None]
    ret = [r["retrieval"] for r in rows if r.get("retrieval")]
    judged = [r["judge"] for r in rows if r.get("judge")]
    kps = [r["keypoints"] for r in rows if r.get("keypoints")]
    return {
        "items": len(rows),
        "score": _mean(scores),
        "ci95": bootstrap_ci(scores, n_boot, seed),
        "hit": _mean([float(x["hit"]) for x in ret]),
        "all_evidence": _mean([float(x["all_evidence"]) for x in ret]),
        "evidence_recall": _mean([x["evidence_recall"] for x in ret]),
        "mrr": _mean([x["rr"] for x in ret]),
        "key_recall": _mean([r["keys"]["key_recall"] for r in rows if r.get("keys") and r["keys"].get("key_recall") is not None]),
        "grounded": _mean([float(j["grounded"]) for j in judged if isinstance(j.get("grounded"), bool)]),
        "abstained": _mean([float(r["abstained"]) for r in rows if r.get("abstained") is not None]),
        "keypoint_completeness": _mean([k["completeness"] for k in kps]),
        "keypoint_hallucination": _mean([k["hallucination"] for k in kps]),
        "verdicts": dict(Counter(str(j.get("verdict")) for j in judged)),
        "categories": dict(Counter(r["category"] for r in rows if r.get("category"))),
        "errors": sum(bool(r.get("error")) for r in rows),
    }


def agreement(rows: list[dict]) -> dict:
    """Keys against judge on items both can decide. Disagreements are where a key or a verdict is worth a look."""
    pairs = [(r["uid"], bool(r["keys"]["correct"]), r["judge_score"] >= 1) for r in rows
             if r.get("judge_score") is not None and r.get("keys") and r["keys"].get("correct") is not None]
    return {"items": len(pairs), "agree": _mean([float(k == j) for _, k, j in pairs]),
            "keys_only": [uid for uid, k, j in pairs if k and not j],     # keys found, judge says wrong: subject or direction wrong, or a loose key
            "judge_only": [uid for uid, k, j in pairs if j and not k]}    # judge says right, keys missing: paraphrase, or a key too strict
