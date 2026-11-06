"""Scoring one RAG answer against its dataset item, and summarising a run.

The RAG under test chunks the PDF its own way, so its passages share no ids
with the benchmark. Gold evidence is matched by page and text: a passage holds
an evidence entry when its page span overlaps the evidence pages and the
normalised quote (letters and digits only) is inside the passage text. A
passage that merely overlaps the pages is recorded as `page_only` and never
counts as a hit; a quote found across two passages adjacent in the document
counts, and is recorded as `quote_split`.

Three views of each answer:
- retrieval: `hit`, `all_evidence`, `evidence_recall`, MRR and `gold_prec`
  over the passages shown; `candidate_recall` over the fused pool the reranker
  chose from, so a miss is put down to the pool or to the ordering.
- keys: deterministic. Every answer key in the reply (a number may differ in
  thousands separators or trailing zeros: "£52,681" is "52681", "0.50" is
  "0.5"), the party for attribution, a decline for unanswerable items.
- judge: the judge's verdict, keypoint coverage for summaries
  (completeness / hallucination / irrelevance), groundedness and faithfulness
  (the share of the reply's claims the passages support).

The judge's verdict is the score when there is one; the keys are the score
otherwise, and their agreement with the judge is reported as a check on both.
A wrong answer is put down to retrieval when the evidence was not shown, and
to the reader when it was.
"""

import re
from collections import Counter
from random import Random

from medrag.chunk import Chunk

from medrageval.facts.factsheet import numbers_in
from medrageval.generate.corpus import tokenize
from medrageval.text import contains_quote

VERDICT_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0, "abstained": 0.0}
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_DIGITS = re.compile(r"\D")
# The benchmark prints a redacted value as [REDACTED]; the PDF (and so the application) prints the redactor's placeholder,
# a run of x's or asterisks. Both vanish before matching, so a table row matches whichever way the gap is written.
_REDACTION = re.compile(r"\[redacted\]|(?<![a-z0-9])[x*]{3,}(?![a-z0-9])")


def norm_alnum(text: str) -> str:
    return _NON_ALNUM.sub("", _REDACTION.sub(" ", (text or "").lower()))


def pages_overlap(pages, chunk: Chunk) -> bool:
    if not pages:
        return True
    p0, p1 = (pages[0], pages[-1]) if isinstance(pages, (list, tuple)) else (pages, pages)
    return chunk.page_start <= p1 and chunk.page_end >= p0


def quote_match(ev: dict, chunk: Chunk) -> str | None:
    """'quote' when the passage is on the evidence pages and holds its quote; 'page' when only the pages overlap; None otherwise."""
    if not pages_overlap(ev.get("pages"), chunk):
        return None
    q = norm_alnum(ev.get("quote") or "")
    if q and q in norm_alnum(chunk.text):
        return "quote"
    return "page" if ev.get("pages") else None


def _adjacent(a: Chunk, b: Chunk) -> bool:
    try:
        return int(_DIGITS.sub("", a.chunk_id)) + 1 == int(_DIGITS.sub("", b.chunk_id))
    except ValueError:
        return False


def quote_split_pair(ev: dict, passages: list[Chunk]) -> tuple[int, int] | None:
    """Indexes of an adjacent pair whose joined text holds the quote, when neither does alone."""
    q = norm_alnum(ev.get("quote") or "")
    if not q:
        return None
    by_id = sorted(range(len(passages)), key=lambda i: passages[i].chunk_id)
    for i, j in zip(by_id, by_id[1:]):
        a, b = passages[i], passages[j]
        if _adjacent(a, b) and (pages_overlap(ev.get("pages"), a) or pages_overlap(ev.get("pages"), b)) \
                and q in norm_alnum(a.text) + norm_alnum(b.body):   # b's header line would sit between the two halves
            return (i, j)
    return None


def quote_split_rank(ev: dict, passages: list[Chunk]) -> int | None:
    """Rank of the first passage of an adjacent pair whose joined text holds the quote, when neither does alone."""
    pair = quote_split_pair(ev, passages)
    return min(pair) + 1 if pair else None


def gold_evidence(item: dict) -> list[dict]:
    """The item's distinct evidence entries that can be matched (a quote or pages); [] for unanswerable items."""
    if not item.get("answerable", True):
        return []
    return list({(e.get("unit_id"), e.get("quote")): e for e in item.get("evidence") or [] if e.get("quote") or e.get("pages")}.values())


def locate(ev: list[dict], passages: list[Chunk]) -> tuple[list[int | None], set[int], int, int]:
    """(rank per evidence entry, indexes of the passages that hold evidence, page-only count, split count)."""
    ranks: list[int | None] = []
    holding: set[int] = set()
    page_only = split = 0
    for e in ev:
        idx = next((i for i, c in enumerate(passages) if quote_match(e, c) == "quote"), None)
        if idx is not None:
            holding.add(idx)
            ranks.append(idx + 1)
            continue
        pair = quote_split_pair(e, passages)
        if pair:
            holding.update(pair)
            split += 1
            ranks.append(min(pair) + 1)
            continue
        page_only += any(quote_match(e, c) == "page" for c in passages)
        ranks.append(None)
    return ranks, holding, page_only, split


def retrieval_metrics(item: dict, passages: list[Chunk], candidates: list[Chunk] | None = None) -> dict | None:
    """Gold evidence among the passages shown, in rank order, and among the candidate pool when given.
    None for unanswerable items and items without evidence. `gold_prec` is the share of the passages shown
    that hold gold evidence; its ceiling is the evidence count over the passages shown, since a passage that is
    merely on topic counts as irrelevant."""
    ev = gold_evidence(item)
    if not ev:
        return None
    ranks, holding, page_only, split = locate(ev, passages)
    found = [r is not None for r in ranks]
    first = min((r for r in ranks if r is not None), default=None)
    out = {"evidence": len(ev), "evidence_found": sum(found), "evidence_recall": round(sum(found) / len(ev), 3),
           "all_evidence": all(found), "hit": first is not None, "first_rank": first, "rr": round(1 / first, 3) if first else 0.0,
           "shown": len(passages), "relevant_shown": len(holding),
           "gold_prec": round(len(holding) / len(passages), 3) if passages else 0.0,
           "page_only": page_only, "quote_split": split}
    if candidates is not None:
        c_ranks = locate(ev, candidates)[0]
        c_found = sum(r is not None for r in c_ranks)
        out.update({"candidates": len(candidates), "candidate_found": c_found, "candidate_recall": round(c_found / len(ev), 3),
                    "candidate_all_evidence": c_found == len(ev)})
    return out


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


def judge_score(item: dict, judge: dict | None, declined: bool = False) -> float | None:
    """The judge's verdict as a score. An unanswerable item scores 1 when the judge says "abstained", or says "correct" of a
    reply that declines (`declined`: the abstain patterns matched). A reply that states an answer the judge finds supported
    is not an abstention, whatever the judge calls it."""
    if not judge:
        return None
    verdict = str(judge.get("verdict") or "").lower()
    if not item.get("answerable", True):
        return 1.0 if verdict == "abstained" or (verdict == "correct" and declined) else 0.0
    return VERDICT_SCORE.get(verdict)


def faithfulness(judge: dict | None) -> float | None:
    """Share of the reply's factual claims the judge found supported by the passages shown. None without claims
    (a decline, or a judge verdict without a claim list)."""
    claims = [c for c in (judge or {}).get("claims") or [] if isinstance(c, dict)]
    if not claims:
        return None
    return round(sum(bool(c.get("supported")) for c in claims) / len(claims), 3)


def keypoint_stats(item: dict, judge: dict | None) -> dict | None:
    """Keypoint metrics: shares of keypoints covered, contradicted and missing. A keypoint the judge skipped is missing."""
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


def score_item(item: dict, response: str, passages: list[Chunk], judge: dict | None, patterns: list[str], speakers: dict,
               candidates: list[Chunk] | None = None) -> dict:
    retrieval = retrieval_metrics(item, passages, candidates)
    keys = key_scores(item, response, patterns, speakers)
    js = judge_score(item, judge, keys["abstained"])
    if js is not None:
        score = js
        # for an unanswerable item `abstained` is a clean decline: it declined and the judge found no answer stated;
        # keys.abstained keeps the raw pattern match, so "declined, then answered anyway" is visible
        declined = score >= 1 if not item.get("answerable", True) else str(judge.get("verdict") or "").lower() == "abstained"
    else:
        score = None if keys["correct"] is None else float(keys["correct"])
        declined = keys["abstained"]
    return {"retrieval": retrieval, "keys": keys, "judge_score": js, "keypoints": keypoint_stats(item, judge),
            "faithfulness": faithfulness(judge), "score": score, "abstained": declined, "category": category(item, retrieval, declined, score)}


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
        "gold_prec": _mean([x["gold_prec"] for x in ret]),           # share of the passages shown that hold gold evidence
        "candidate_recall": _mean([x["candidate_recall"] for x in ret if "candidate_recall" in x]),   # evidence in the pool before reranking
        "candidate_all_evidence": _mean([float(x["candidate_all_evidence"]) for x in ret if "candidate_all_evidence" in x]),
        "evidence_page_only": sum(x.get("page_only", 0) for x in ret),     # evidence pages shown without the quote: not credited
        "evidence_quote_split": sum(x.get("quote_split", 0) for x in ret),  # quote found across two adjacent passages: credited
        "key_recall": _mean([r["keys"]["key_recall"] for r in rows if r.get("keys") and r["keys"].get("key_recall") is not None]),
        "grounded": _mean([float(j["grounded"]) for j in judged if isinstance(j.get("grounded"), bool)]),
        "faithfulness": _mean([r["faithfulness"] for r in rows if r.get("faithfulness") is not None]),
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
