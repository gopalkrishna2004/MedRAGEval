import pytest

from medrageval.config import load_config
from medrageval.rag.chunks import Chunk
from medrageval.rag.score import (agreement, bootstrap_ci, category, is_abstention, judge_score, key_found, key_scores, keypoint_stats,
                                  retrieval_metrics, score_item)

FULL = load_config()
PATTERNS = FULL["rag"]["eval"]["abstain_patterns"]
SPEAKERS = FULL["generate"]["attribution"]["speakers"]


def chunk(cid, unit_ids, text, part=None):
    return Chunk(cid, "d02", "company_submission", [], 1, 1, unit_ids, list(dict.fromkeys(u.split("#")[0] for u in unit_ids)), text, part)


@pytest.mark.parametrize("key, text, ok", [
    ("£52,681", "The total costs were £52681 [2].", True),
    ("30.44 days", "The model used a cycle length of 30.44 days.", True),
    ("0.50", "The hazard ratio was 0.5 (95% CI 0.38 to 0.65).", True),
    ("0.38 to 0.65", "HR 0.50; 95% CI 0.38 to 0.65", True),
    ("28.8%", "The response rate was 26.8%.", False),
    ("Exclude subsequent treatment costs", "The scenario excludes subsequent treatment costs.", True),
    ("piecewise NMA", "It used the network meta-analysis.", False),
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
    # a reply that gives the key and also says something is not reported has answered, not declined
    assert not key_scores(item, "Median PFS was 7.3 months; OS was not reported.", PATTERNS, SPEAKERS)["abstained"]


def test_unanswerable_is_correct_when_declined():
    item = {"type": "unanswerable", "answerable": False, "answer_keys": []}
    assert key_scores(item, "The price is [REDACTED] in the submission, so it is not available.", PATTERNS, SPEAKERS)["correct"]
    assert not key_scores(item, "The list price is £5,000 per pack.", PATTERNS, SPEAKERS)["correct"]


TWO_EVIDENCE = {"type": "multihop", "answerable": True, "also_in": ["d05-e0009"],
                "evidence": [{"unit_id": "d02-e0008", "element_id": "d02-e0008", "quote": "£24,567 per QALY gained"},
                             {"unit_id": "d03-e0030", "element_id": "d03-e0030", "quote": "£31,200 per QALY gained"}]}


def test_retrieval_counts_evidence_and_copies_in_rank_order():
    passages = [chunk("c1", ["d05-e0009"], "a copy of the fact"), chunk("c2", ["d02-e0008"], "ICER £24,567 per QALY gained")]
    m = retrieval_metrics(TWO_EVIDENCE, passages)
    assert m["evidence_recall"] == 0.5 and not m["all_evidence"]
    assert m["hit"] and m["first_rank"] == 1 and m["rr"] == 1.0     # the copy counts for hit, not for recall
    both = retrieval_metrics(TWO_EVIDENCE, passages + [chunk("c3", ["d03-e0030"], "EAG ICER £31,200 per QALY gained")])
    assert both["all_evidence"] and both["evidence_recall"] == 1.0
    assert retrieval_metrics({**TWO_EVIDENCE, "answerable": False}, passages) is None


def test_another_row_group_of_the_table_counts_only_with_the_quoted_row():
    item = {"answerable": True, "evidence": [{"unit_id": "d04-e0001#r01", "element_id": "d04-e0001", "quote": "Answer 30: capivasertib"}]}
    assert not retrieval_metrics(item, [chunk("c1", ["d04-e0001#r00"], "Answer 3: capivasertib")])["hit"]
    assert retrieval_metrics(item, [chunk("c1", ["d04-e0001#r02"], "Answer 30: capivasertib plus fulvestrant")])["hit"]
    # a piece of a split unit holds the evidence only if the quote is in that piece
    assert not retrieval_metrics(item, [chunk("c1", ["d04-e0001#r01"], "other text", part=1)])["hit"]


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
    retrieval = None if all_ev is None else {"all_evidence": all_ev}
    assert category({"answerable": answerable}, retrieval, declined, score) == expected


def test_judge_score_and_keypoints():
    assert judge_score({"answerable": True}, {"verdict": "partial"}) == 0.5
    assert judge_score({"answerable": False}, {"verdict": "abstained"}) == 1.0
    assert judge_score({"answerable": True}, None) is None
    item = {"keypoints": [{"point": "a"}, {"point": "b"}, {"point": "c"}]}
    kp = keypoint_stats(item, {"keypoints": [{"n": 1, "status": "covered"}, {"n": 2, "status": "contradicted"}]})
    assert kp == {"keypoints": 3, "completeness": 0.333, "hallucination": 0.333, "irrelevance": 0.333}


def test_score_item_prefers_the_judge_and_falls_back_to_keys():
    item = {"type": "numerical", "answerable": True, "answer_keys": ["7.3 months"],
            "evidence": [{"unit_id": "d02-e0003", "element_id": "d02-e0003", "quote": "Median PFS (months) | 7.3 | 3.1"}]}
    passages = [chunk("c1", ["d02-e0003"], "Median PFS (months) | 7.3 | 3.1")]
    judged = score_item(item, "7.3 months [1]", passages, {"verdict": "incorrect", "grounded": True}, PATTERNS, SPEAKERS)
    assert judged["score"] == 0.0 and judged["category"] == "reader_error" and judged["keys"]["correct"]
    keys_only = score_item(item, "7.3 months [1]", passages, None, PATTERNS, SPEAKERS)
    assert keys_only["score"] == 1.0 and keys_only["category"] == "correct"


def test_bootstrap_is_deterministic():
    xs = [1.0, 0.0, 1.0, 1.0, 0.5, 0.0]
    assert bootstrap_ci(xs, 500, 7) == bootstrap_ci(xs, 500, 7)
    lo, hi = bootstrap_ci(xs, 500, 7)
    assert 0.0 <= lo < sum(xs) / len(xs) < hi <= 1.0
    assert bootstrap_ci([1.0, 1.0], 100, 1) == [1.0, 1.0] and bootstrap_ci([1.0], 100, 1) is None


def test_agreement_lists_the_disagreements():
    rows = [{"uid": "a", "keys": {"correct": True}, "judge_score": 1.0},
            {"uid": "b", "keys": {"correct": True}, "judge_score": 0.0},
            {"uid": "c", "keys": {"correct": False}, "judge_score": 1.0},
            {"uid": "d", "keys": {"correct": None}, "judge_score": 1.0}]
    assert agreement(rows) == {"items": 3, "agree": 0.333, "keys_only": ["b"], "judge_only": ["c"]}
