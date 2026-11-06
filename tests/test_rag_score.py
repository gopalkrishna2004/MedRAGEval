import pytest
from medrag.chunk import Chunk

from medrageval.config import load_config
from medrageval.rag.score import (agreement, bootstrap_ci, category, faithfulness, holds, is_abstention, judge_score, key_found,
                                  key_scores, keypoint_stats, norm_alnum, quote_match, retrieval_metrics, score_item, summarize)

FULL = load_config()
PATTERNS = FULL["rag"]["eval"]["abstain_patterns"]
SPEAKERS = FULL["generate"]["attribution"]["speakers"]


def chunk(cid, pages, body, title="Company evidence submission"):
    p0, p1 = pages if isinstance(pages, tuple) else (pages, pages)
    text = f"{title} | B.2 Clinical effectiveness | p. {p0}\n{body}"
    return Chunk(cid, title, ["B.2 Clinical effectiveness"], p0, p1, "text", body, text, len(text) // 4)


@pytest.mark.parametrize("key, text, ok", [
    ("£52,681", "The total costs were £52681 [2].", True),
    ("30.44 days", "The model used a cycle length of 30.44 days.", True),
    ("0.50", "The hazard ratio was 0.5 (95% CI 0.38 to 0.65).", True),
    ("0.38 to 0.65", "HR 0.50; 95% CI 0.38 to 0.65", True),
    ("28.8%", "The response rate was 26.8%.", False),
    ("Exclude subsequent treatment costs", "The scenario excludes subsequent treatment costs.", True),
    ("piecewise NMA", "It used the network meta-analysis.", False),
    ("lognormal distribution", "The company chose the log-normal distribution.", True),      # a hyphen is spelled either way
    ("log-normal", "A lognormal curve was fitted.", True),
    ("NMA-estimated HRs", "The HRs were estimated in the NMA.", True),                     # or as separate words
    ("patient organisation", "The patient organization submitted a response.", True),     # s and z endings share a stem
    ("5-10 mg", "Doses of 510 mg were given.", False),                                     # a number range keeps its dash
])
def test_key_found(key, text, ok):
    assert key_found(key, text) is ok


@pytest.mark.parametrize("text, declined", [
    ("Not found in the documents.", True),
    ("The list price is redacted in the submission.", True),
    ("The passages do not report the hazard ratio at DCO2.", True),
    ("", True),
    ("The company's base-case ICER was £24,567 per QALY gained [1].", False),
])
def test_is_abstention(text, declined):
    assert is_abstention(text, PATTERNS) is declined


ATTR = {"type": "attribution", "answerable": True, "answer_keys": ["company"], "answer_aliases": ["company", "manufacturer", "applicant", "sponsor"]}


def test_attribution_keys_name_the_party_and_record_others():
    right = key_scores(ATTR, "The company argued that ESMO treatments were not relevant comparators [1].", PATTERNS, SPEAKERS)
    assert right["correct"] and right["other_speakers"] == []
    wrong = key_scores(ATTR, "The EAG argued that they were not relevant [2].", PATTERNS, SPEAKERS)
    assert not wrong["correct"] and wrong["other_speakers"] == ["the EAG"]


def test_numerical_keys_and_decline():
    item = {"type": "numerical", "answerable": True, "answer_keys": ["7.3 months"]}
    assert key_scores(item, "Median PFS was 7.3 months [1].", PATTERNS, SPEAKERS)["correct"]
    k = key_scores(item, "Not found in the documents.", PATTERNS, SPEAKERS)
    assert k["abstained"] and not k["correct"] and k["key_recall"] == 0.0
    assert not key_scores(item, "Median PFS was 7.3 months; OS was not reported.", PATTERNS, SPEAKERS)["abstained"]


def test_unanswerable_is_correct_when_declined():
    item = {"type": "unanswerable", "answerable": False, "answer_keys": []}
    assert key_scores(item, "The price is [REDACTED] in the submission, so it is not available.", PATTERNS, SPEAKERS)["correct"]
    assert not key_scores(item, "The list price is £5,000 per pack.", PATTERNS, SPEAKERS)["correct"]


EV = {"unit_id": "d02-e0008", "pages": [120, 120], "quote": "The company's base-case ICER was £24,567 per QALY gained."}


def test_quote_match_needs_the_pages_and_the_normalised_quote():
    assert quote_match(EV, chunk("c1", 120, "Results. The company’s base-case ICER was £24 567 per QALY gained [see Table 47].")) == "quote"
    assert quote_match(EV, chunk("c1", (119, 121), "the company's base-case icer was £24,567 per qaly gained")) == "quote"
    assert quote_match(EV, chunk("c1", 120, "Table 47 shows the results of the base case.")) == "page"      # pages yes, quote no
    assert quote_match(EV, chunk("c1", 300, "The company's base-case ICER was £24,567 per QALY gained.")) is None   # same words, wrong pages


def test_redaction_placeholders_match_the_benchmarks_redacted_marker():
    ev = {"unit_id": "u", "pages": [126, 126], "quote": "Everolimus + exemestane | £25,714 | 1.96 | 1.45 | [REDACTED] | 1.30 | 0.94 | [REDACTED] | [REDACTED]"}
    row = "| Everolimus + exemestane | £25,714 | 1.96 | 1.45 | xxxxxxx | 1.30 | 0.94 | ******* | xxxxxxx |"
    assert quote_match(ev, chunk("c1", 126, row)) == "quote"
    assert quote_match(ev, chunk("c1", 126, row.replace("1.30", "1.31"))) == "page"
    short = {"unit_id": "u", "pages": [45, 45], "quote": "At the time of DCO1 (August 15th, 2022), there were [REDACTED] OS events"}
    assert quote_match(short, chunk("c1", 45, "At the time of DCO1 (August 15th, 2022), there were xx OS events (xxxxx data maturity).")) == "quote"


HEADER = "Company evidence submission: Capivasertib with fulvestrant for treating HR-positive, HER2negative advanced breast cancer [ID6370]"


def test_a_running_header_inside_the_sentence_does_not_break_the_quote():
    ev = {"unit_id": "u", "pages": [70, 71], "quote": "The trial was a robust phase 3 trial, at low risk of bias and provided valid results that are "
                                                      "generalisable to patients anticipated to receive capivasertib plus fulvestrant in clinical practice."}
    body = (f"The trial was a robust phase 3 trial, at low risk of bias and provided valid results that are generalisable to patients\n\n"
            f"{HEADER}\n\nanticipated to receive capivasertib plus fulvestrant in clinical practice. The trial demonstrated a benefit.")
    assert quote_match(ev, chunk("c1", (70, 71), body)) == "quote"
    assert holds(norm_alnum(ev["quote"]), norm_alnum(body.replace(HEADER, HEADER * 4))) is False   # a long gap is a different passage
    assert holds(norm_alnum(ev["quote"]), norm_alnum(body.replace("clinical practice", "routine care"))) is False
    assert holds("", "anything") is False


TWO = {"type": "multihop", "answerable": True, "also_in": ["d05-e0009"],
       "evidence": [EV, {"unit_id": "d03-e0030", "pages": [537, 537], "quote": "EAG base-case ICER of £31,200 per QALY gained"}]}


def test_retrieval_credits_quotes_only_and_reports_page_only_matches():
    passages = [chunk("c00010", 120, "Table 47 shows the results of the base case."),
                chunk("c00020", 120, "The company's base-case ICER was £24,567 per QALY gained [Table 47].")]
    m = retrieval_metrics(TWO, passages)
    assert m["evidence_recall"] == 0.5 and not m["all_evidence"]
    assert m["hit"] and m["first_rank"] == 2 and m["rr"] == 0.5     # the page-only passage at rank 1 is not credited
    assert m["page_only"] == 0 and m["quote_split"] == 0           # the quote was found elsewhere, so no page-only tally for it
    miss = retrieval_metrics(TWO, passages[:1])
    assert not miss["hit"] and miss["page_only"] == 1
    both = retrieval_metrics(TWO, passages + [chunk("c00300", 537, "The EAG base-case ICER of £31,200 per QALY gained is above the threshold.", "EAG report")])
    assert both["all_evidence"] and both["evidence_recall"] == 1.0
    assert retrieval_metrics({**TWO, "answerable": False}, passages) is None


def test_a_quote_split_across_two_adjacent_passages_counts():
    a = chunk("c00041", 120, "In the deterministic analysis the company's base-case ICER")
    b = chunk("c00042", 120, "was £24,567 per QALY gained, below the £30,000 threshold.")
    m = retrieval_metrics({"answerable": True, "evidence": [EV]}, [b, a])
    assert m["hit"] and m["first_rank"] == 1 and m["quote_split"] == 1
    far = chunk("c00090", 120, "was £24,567 per QALY gained, below the £30,000 threshold.")
    assert not retrieval_metrics({"answerable": True, "evidence": [EV]}, [a, far])["hit"]   # not adjacent in the document
    assert m["relevant_shown"] == 2 and m["gold_prec"] == 1.0                      # both halves hold the evidence


def test_gold_prec_counts_the_passages_that_hold_evidence():
    gold = chunk("c00020", 120, "The company's base-case ICER was £24,567 per QALY gained [Table 47].")
    noise = [chunk(f"c0010{i}", 120 + i, f"Filler passage {i} about the decision problem.") for i in range(5)]
    m = retrieval_metrics(TWO, noise[:2] + [gold] + noise[2:])
    assert m["shown"] == 6 and m["relevant_shown"] == 1 and m["gold_prec"] == round(1 / 6, 3)
    assert retrieval_metrics(TWO, noise)["gold_prec"] == 0.0


def test_candidate_recall_is_over_the_pool_before_reranking():
    gold = chunk("c00020", 120, "The company's base-case ICER was £24,567 per QALY gained [Table 47].")
    eag = chunk("c00300", 537, "The EAG base-case ICER of £31,200 per QALY gained is above the threshold.", "EAG report")
    noise = chunk("c00101", 121, "Filler passage about the decision problem.")
    m = retrieval_metrics(TWO, [noise], candidates=[noise, gold, eag])
    assert not m["hit"] and m["evidence_recall"] == 0.0                                   # nothing shown to the model...
    assert m["candidates"] == 3 and m["candidate_found"] == 2 and m["candidate_recall"] == 1.0 and m["candidate_all_evidence"]   # ...but the pool had it
    assert "candidate_recall" not in retrieval_metrics(TWO, [gold])                        # a fixed context has no pool
    half = retrieval_metrics(TWO, [gold], candidates=[gold, noise])
    assert half["candidate_recall"] == 0.5 and not half["candidate_all_evidence"]


def test_faithfulness_is_the_supported_share_of_claims():
    assert faithfulness({"claims": [{"claim": "ICER £24,567", "supported": True}, {"claim": "below £30,000", "supported": True},
                                    {"claim": "in 2023", "supported": False}]}) == round(2 / 3, 3)
    assert faithfulness({"claims": []}) is None and faithfulness(None) is None
    assert faithfulness({"grounded": True}) is None            # an older verdict without a claim list
    rows = [{"faithfulness": 1.0, "retrieval": {"hit": True, "all_evidence": True, "evidence_recall": 1.0, "rr": 1.0, "gold_prec": 0.5,
                                                "candidate_recall": 1.0, "candidate_all_evidence": True}},
            {"faithfulness": 0.5, "retrieval": {"hit": False, "all_evidence": False, "evidence_recall": 0.0, "rr": 0.0, "gold_prec": 0.0}},
            {"faithfulness": None, "retrieval": None}]
    o = summarize(rows, 10, 1)
    assert o["faithfulness"] == 0.75 and o["gold_prec"] == 0.25 and o["candidate_recall"] == 1.0 and o["candidate_all_evidence"] == 1.0


@pytest.mark.parametrize("answerable, all_ev, declined, score, expected", [
    (True, True, False, 1.0, "correct"),
    (True, False, False, 0.0, "retrieval_miss"),
    (True, False, True, 0.0, "retrieval_miss"),
    (True, True, True, 0.0, "false_abstention"),
    (True, True, False, 0.5, "reader_error"),
    (False, None, True, 1.0, "correct_abstention"),
    (False, None, False, 0.0, "hallucinated_answer"),
])
def test_category(answerable, all_ev, declined, score, expected):
    item = {"answerable": answerable}
    retrieval = {"all_evidence": all_ev} if all_ev is not None else None
    assert category(item, retrieval, declined, score) == expected


def test_judge_score_and_keypoints():
    item = {"type": "summary", "answerable": True, "keypoints": [{"point": "a"}, {"point": "b"}, {"point": "c"}, {"point": "d"}]}
    judge = {"verdict": "partial", "keypoints": [{"n": 1, "status": "covered"}, {"n": 2, "status": "covered"}, {"n": 3, "status": "contradicted"}]}
    assert judge_score(item, judge) == 0.5
    assert keypoint_stats(item, judge) == {"keypoints": 4, "completeness": 0.5, "hallucination": 0.25, "irrelevance": 0.25}
    assert judge_score({"answerable": False}, {"verdict": "abstained"}) == 1.0
    assert judge_score({"answerable": False}, {"verdict": "incorrect"}) == 0.0
    assert judge_score({"answerable": False}, {"verdict": "correct"}) == 0.0                   # "correct" of a stated answer
    assert judge_score({"answerable": False}, {"verdict": "correct"}, declined=True) == 1.0   # "correct" of a decline


def test_unanswerable_needs_a_decline_as_well_as_the_judge():
    item = {"type": "unanswerable", "answerable": False, "answer_keys": [], "evidence": []}
    ok = {"verdict": "correct", "claims": [], "reason": "supported by passage 1"}
    r = score_item(item, "The company called it a robust phase 3 trial [1].", [], ok, PATTERNS, SPEAKERS)
    assert r["score"] == 0.0 and r["category"] == "hallucinated_answer" and r["abstained"] is False
    r = score_item(item, "Not found in the documents.", [], ok, PATTERNS, SPEAKERS)
    assert r["score"] == 1.0 and r["category"] == "correct_abstention" and r["abstained"] is True
    r = score_item(item, "The figure is 1.79 [1]. Not found in the documents.", [], {"verdict": "incorrect"}, PATTERNS, SPEAKERS)
    assert r["score"] == 0.0 and r["abstained"] is False and r["keys"]["abstained"] is True   # declined, then answered anyway
    r = score_item(item, "The documents are silent on this.", [], {"verdict": "abstained"}, PATTERNS, SPEAKERS)
    assert r["score"] == 1.0 and r["abstained"] is True                                        # an off-pattern decline the judge recognised


def test_score_item_prefers_the_judge_and_agreement_lists_disagreements():
    item = {"type": "numerical", "answerable": True, "answer_keys": ["7.3 months"], "evidence": [{"unit_id": "u", "pages": [21, 21], "quote": "Median PFS 7.3"}]}
    passages = [chunk("c1", 21, "Median PFS 7.3 months versus 3.1 months")]
    judge = {"verdict": "incorrect", "grounded": True, "claims": [{"claim": "PFS 7.3 months", "supported": True}]}
    s = score_item(item, "Median PFS was 7.3 months [1].", passages, judge, PATTERNS, SPEAKERS, candidates=passages)
    assert s["score"] == 0.0 and s["keys"]["correct"] and s["category"] == "reader_error" and s["faithfulness"] == 1.0
    assert s["retrieval"]["candidate_recall"] == 1.0
    rows = [{"uid": "a", "keys": {"correct": True}, "judge_score": 0.0}, {"uid": "b", "keys": {"correct": True}, "judge_score": 1.0},
            {"uid": "c", "keys": {"correct": None}, "judge_score": 1.0}]
    assert agreement(rows) == {"items": 2, "agree": 0.5, "keys_only": ["a"], "judge_only": []}


def test_bootstrap_ci_is_deterministic_and_within_range():
    vals = [1.0, 0.0, 1.0, 0.5, 1.0, 0.0]
    ci = bootstrap_ci(vals, 200, 1)
    assert ci == bootstrap_ci(vals, 200, 1) and 0.0 <= ci[0] <= sum(vals) / len(vals) <= ci[1] <= 1.0
    assert bootstrap_ci([1.0], 200, 1) is None
