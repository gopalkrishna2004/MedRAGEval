import copy
import json

import pytest
from synthetic import build_corpus, config, write_interim

from medrageval.config import load_config
from medrageval.generate.llm import FakeLLM, LLMError
from medrageval.rag.evaluate import evaluate, load_items, resolve_interim
from medrageval.rag.index import build_index
from medrageval.rag.pipeline import RAG

PFS = {"item_id": "numerical-001", "type": "numerical", "subtype": "efficacy", "answerable": True,
       "question": "What was the median PFS by investigator assessment with capivasertib plus fulvestrant, in months?",
       "answer": "Median PFS was 7.3 months.", "answer_keys": ["7.3 months"], "also_in": [],
       "evidence": [{"unit_id": "d02-e0003", "element_id": "d02-e0003", "doc_id": "d02", "pages": [21, 21], "quote": "Median PFS (months) | 7.3 | 3.1"}]}
# Unanswerable: the PFS question again, its evidence withheld; the stored distractors are the context. The evidence unit itself is listed
# among them to check it is dropped.
WITHHELD = {"item_id": "unanswerable-001", "type": "unanswerable", "subtype": "withheld", "answerable": False, "source_item_id": "numerical-001",
            "question": PFS["question"], "answer": None, "answer_keys": [], "also_in": [], "evidence": [],
            "withheld": {"answer": PFS["answer"], "answer_keys": PFS["answer_keys"], "evidence": PFS["evidence"]},
            "distractors": [{"unit_id": "d02-e0004", "doc_id": "d02", "pages": [40, 40]}, {"unit_id": "d02-e0001", "doc_id": "d02", "pages": [10, 10]},
                            {"unit_id": "d02-e0003", "doc_id": "d02", "pages": [21, 21]}]}
# An unanswerable item from an older dataset, without distractors: answered over live retrieval like the rest.
PRICE = {"item_id": "unanswerable-002", "type": "unanswerable", "subtype": "redacted", "answerable": False,
         "question": "What is the list price of a pack of 64 capivasertib tablets?", "answer": None, "answer_keys": [], "also_in": [],
         "evidence": [{"unit_id": "d02-e0004", "element_id": "d02-e0004", "doc_id": "d02", "pages": [40, 40], "quote": "£[REDACTED] per pack"}]}


def write_dataset(path, items, interim):
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "dataset.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps({**it, "provenance": {"interim_dir": str(interim)}}) + "\n")
    return path


def full_config(tmp_path):
    full = copy.deepcopy(load_config())
    full["generate"] = config()
    full["rag"]["retrieval"]["mode"] = "bm25"
    full["rag"]["index_dir"] = str(tmp_path / "rag")
    full["rag"]["eval"]["bootstrap"] = 100
    return full


def question(messages):
    return messages[-1]["content"].rsplit("Question: ", 1)[-1]


def small_model(messages, tag):
    """Reads what it is shown: the PFS figure when a passage holds it, a decline otherwise."""
    passages = messages[-1]["content"].rsplit("Question: ", 1)[0]
    return "Median PFS was 7.3 months [1]." if "median" in question(messages) and "7.3" in passages else "Not found in the documents."


def passing_judge(messages, tag):
    return {"verdict": "correct", "keypoints": [], "grounded": True, "unsupported_claims": [], "reason": "ok"}


def make_rag(full, llm):
    index, _ = build_index(build_corpus(full["generate"]), full["rag"], None)
    return RAG(index, llm, full["rag"])


def test_eval_end_to_end(tmp_path):
    full = full_config(tmp_path)
    interim = write_interim(tmp_path / "interim")
    items = load_items([write_dataset(tmp_path / "ds", [PFS, WITHHELD, PRICE], interim)])
    assert resolve_interim(items, None) == interim
    judge = FakeLLM(passing_judge)
    rag = make_rag(full, FakeLLM(small_model))
    report = evaluate(items, rag, judge, full, tmp_path / "run")

    rows = [json.loads(line) for line in (tmp_path / "run" / "responses.jsonl").open()]
    assert [r["uid"] for r in rows] == ["numerical-001", "unanswerable-001", "unanswerable-002"]
    pfs, withheld, price = rows
    assert pfs["score"] == 1.0 and pfs["category"] == "correct" and pfs["keys"]["correct"] and pfs["context"] == "retrieved"
    assert pfs["retrieval"]["hit"] and pfs["retrieval"]["first_rank"] == 1
    assert "Median PFS (months) | 7.3 | 3.1" in judge.calls[0][0][1]["content"]   # the judge sees the passages the model saw
    # the same question read over the stored distractors: the chunk holding the withheld evidence is dropped, so the model declines
    assert withheld["context"] == "fixed" and withheld["category"] == "correct_abstention" and withheld["retrieval"] is None
    shown = [rag.index.by_id[h["chunk_id"]] for h in withheld["retrieved"]]
    assert [c.unit_ids[0] for c in shown] == ["d02-e0004", "d02-e0001"] and all(h["in_context"] and h["score"] is None for h in withheld["retrieved"])
    assert "7.3" not in judge.calls[1][0][1]["content"].split("Passages shown")[1]
    assert price["context"] == "retrieved" and price["category"] == "correct_abstention" and price["retrieved"][0]["score"]

    assert report["items"] == 3 and set(report["per_type"]) == {"numerical", "unanswerable"} and report["fixed_context_items"] == 1
    assert report["overall"]["score"] == 1.0 and report["agreement"] == {"items": 3, "agree": 1.0, "keys_only": [], "judge_only": []}
    assert report["judge_model"] == "fake" and report["rag"]["retrieval"]["mode"] == "bm25" and not report["warnings"]
    assert json.loads((tmp_path / "run" / "eval_report.json").read_text())["items"] == 3


def test_without_a_judge_items_are_scored_on_keys(tmp_path):
    full = full_config(tmp_path)
    items = load_items([write_dataset(tmp_path / "ds", [PFS, PRICE], tmp_path / "interim")])
    report = evaluate(items, make_rag(full, FakeLLM(small_model)), None, full, tmp_path / "run")
    assert report["overall"]["score"] == 1.0 and report["judge_model"] is None and report["agreement"]["items"] == 0


def test_first_failure_stops_the_run_later_ones_are_recorded(tmp_path):
    full = full_config(tmp_path)
    items = load_items([write_dataset(tmp_path / "ds", [PFS, PRICE], tmp_path / "interim")])

    def down(messages, tag):
        raise LLMError("connection refused")
    with pytest.raises(LLMError):
        evaluate(items, make_rag(full, FakeLLM(down)), None, full, tmp_path / "run")

    def flaky(messages, tag):
        if "median" in question(messages):
            return small_model(messages, tag)
        raise LLMError("timeout")
    report = evaluate(items, make_rag(full, FakeLLM(flaky)), None, full, tmp_path / "run2")
    assert report["overall"]["errors"] == 1 and report["warnings"][0].startswith("1 items failed")


def test_several_datasets_get_prefixed_ids(tmp_path):
    a = write_dataset(tmp_path / "smoke_a", [PFS], tmp_path / "interim")
    b = write_dataset(tmp_path / "smoke_b", [PFS], tmp_path / "other")
    items = load_items([a, b])
    assert [it["uid"] for it in items] == ["smoke_a/numerical-001", "smoke_b/numerical-001"]
    with pytest.raises(SystemExit):
        resolve_interim(items, None)   # generated from two different parse outputs
