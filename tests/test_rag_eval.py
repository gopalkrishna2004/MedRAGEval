"""The evaluator over a tiny in-memory RAG application."""

import copy
import json
import threading
from collections import Counter

import pytest
from medrag.chunk import Chunk
from medrag.generate import RAG
from medrag.llm import LLMError as RagLLMError
from medrag.retrieve import Retriever
from medrag.store import MemoryStore
from synthetic import config

from medrageval.config import load_config
from medrageval.generate.llm import FakeLLM
from medrageval.rag.evaluate import distractor_passages, evaluate, interim_dir, load_items, load_unit_texts, unit_chunks

APP_CFG = {"topic": "a test appraisal", "chunking": {"target_tokens": 400, "max_tokens": 512, "max_table_tokens": 1500},
           "embedding": {"model": "none", "dim": 4}, "store": {"backend": "memory"},
           "retrieval": {"mode": "bm25", "top_k": 3, "candidates": 10, "rrf_k": 60, "rerank": False, "max_context_chars": 12000},
           "llm": {"model": "fake-reader", "base_url": "http://fake", "temperature": 0.0, "max_tokens": 512, "context_window": 8192, "max_workers": 2},
           "_path": "test"}


def mk(cid, pages, body, title="Company submission", kind="text"):
    p0, p1 = pages
    text = f"{title} | Results | pp. {p0}-{p1}\n{body}"
    return Chunk(cid, title, ["Results"], p0, p1, kind, body, text, len(text) // 4)


CHUNKS = [
    mk("c00001", (10, 10), "Capivasertib is an AKT inhibitor given with fulvestrant. The decision problem covers the pathway-altered population."),
    mk("c00002", (21, 21), "Table 12. Efficacy results\n| Outcome | Capivasertib | Placebo |\n|---|---|---|\n| Median PFS (months) | 7.3 | 3.1 |", kind="table"),
    mk("c00003", (40, 40), "The list price of capivasertib is £[REDACTED] per pack of 64 tablets. A patient access scheme applies."),
    mk("c00004", (41, 41), "Dosing is 400 mg twice daily for four days on and three days off, in 28-day cycles."),
]

PFS = {"item_id": "numerical-001", "type": "numerical", "subtype": "efficacy", "answerable": True,
       "question": "What was the median PFS by investigator assessment with capivasertib plus fulvestrant, in months?",
       "answer": "Median PFS was 7.3 months.", "answer_keys": ["7.3 months"], "also_in": [],
       "evidence": [{"unit_id": "d02-e0003", "element_id": "d02-e0003", "doc_id": "d02", "pages": [21, 21], "quote": "Median PFS (months) | 7.3 | 3.1"}]}
# The PFS question again with its evidence withheld; the stored distractors are the context. The evidence unit itself is listed among
# them (as the benchmark would never do, but a page-mapped chunk might) to check it is dropped.
WITHHELD = {"item_id": "unanswerable-001", "type": "unanswerable", "subtype": "withheld", "answerable": False, "source_item_id": "numerical-001",
            "question": PFS["question"], "answer": None, "answer_keys": [], "also_in": [], "evidence": [],
            "withheld": {"answer": PFS["answer"], "answer_keys": PFS["answer_keys"], "evidence": PFS["evidence"]},
            "distractors": [{"unit_id": "d02-e0004", "doc_id": "d02", "pages": [40, 40], "quote": "", "caption": None},
                            {"unit_id": "d02-e0001", "doc_id": "d02", "pages": [10, 10], "quote": "", "caption": None},
                            {"unit_id": "d02-e0003", "doc_id": "d02", "pages": [21, 21], "quote": "", "caption": "Table 12. Efficacy results"}]}
# An unanswerable item without distractors (an older dataset): answered over live retrieval like the rest.
PRICE = {"item_id": "unanswerable-002", "type": "unanswerable", "subtype": "redacted", "answerable": False,
         "question": "What is the list price of a pack of 64 capivasertib tablets?", "answer": None, "answer_keys": [], "also_in": [],
         "evidence": [{"unit_id": "d02-e0004", "element_id": "d02-e0004", "doc_id": "d02", "pages": [40, 40], "quote": "£[REDACTED] per pack"}]}

UNIT_TEXTS = {"d02-e0004": "The list price of capivasertib is £[REDACTED] per pack of 64 tablets. A patient access scheme applies.",
              "d02-e0001": "Capivasertib is an AKT inhibitor given with fulvestrant.",
              "d02-e0003": "Table 12. Efficacy results | Outcome | Capivasertib | Placebo | Median PFS (months) | 7.3 | 3.1"}


class FakeReader:
    """Answers from a callable (messages, tag) -> text, counting calls like the real client."""

    def __init__(self, fn):
        self.fn, self.model, self.stats, self._lock = fn, "fake-reader", Counter(), threading.Lock()

    def complete(self, messages, tag=""):
        text = self.fn(messages, tag)
        with self._lock:
            self.stats["calls"] += 1
        return text, {"prompt_tokens": sum(len(m["content"]) // 4 for m in messages), "completion_tokens": len(text) // 4}


def make_rag(fn, chunks=CHUNKS):
    store = MemoryStore(dim=4)
    store.replace_collection("test", {}, chunks, None)
    return RAG(Retriever(store, "test", APP_CFG["retrieval"]), FakeReader(fn), copy.deepcopy(APP_CFG))


def write_dataset(path, items, interim):
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "dataset.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps({**it, "provenance": {"interim_dir": str(interim)}}) + "\n")
    return path


def full_config():
    full = copy.deepcopy(load_config())
    full["generate"] = config()
    full["rag"]["eval"]["bootstrap"] = 100
    return full


def question(messages):
    return messages[-1]["content"].rsplit("Question: ", 1)[-1]


def small_model(messages, tag):
    """Reads what it is shown: the PFS figure when a passage holds it, a decline otherwise."""
    passages = messages[-1]["content"].rsplit("Question: ", 1)[0]
    return "Median PFS was 7.3 months [1]." if "median" in question(messages) and "7.3" in passages else "Not found in the documents."


def passing_judge(messages, tag):
    declined = "Not found in the documents." in messages[-1]["content"].split("System answer:")[1].split("Passages shown")[0]
    claims = [] if declined else [{"claim": "median PFS 7.3 months", "supported": True}]
    return {"verdict": "correct", "keypoints": [], "claims": claims, "grounded": None if declined else True, "reason": "ok"}


def test_unit_chunks_map_by_text_overlap_then_caption_then_page():
    d = {"unit_id": "d02-e0004", "pages": [40, 40]}
    assert [c.chunk_id for c in unit_chunks(d, CHUNKS, UNIT_TEXTS, 0.3, 2)] == ["c00003"]
    assert [c.chunk_id for c in unit_chunks({"unit_id": "x", "pages": [21, 21], "caption": "Table 12. Efficacy results"}, CHUNKS, {}, 0.3, 2)] == ["c00002"]
    assert [c.chunk_id for c in unit_chunks({"unit_id": "x", "pages": [40, 41]}, CHUNKS, {}, 0.3, 2)] == ["c00003", "c00004"]
    assert unit_chunks({"unit_id": "x", "pages": [99, 99]}, CHUNKS, {}, 0.3, 2) == []
    # a unit whose text is known but shares nothing with the page's chunks maps to nothing rather than to a wrong chunk
    assert unit_chunks({"unit_id": "d02-e0001", "pages": [40, 40]}, CHUNKS, UNIT_TEXTS, 0.3, 2) == []


def test_distractor_passages_drop_the_withheld_evidence():
    passages, unmapped = distractor_passages(WITHHELD, CHUNKS, UNIT_TEXTS, full_config()["rag"]["eval"], 6)
    assert [c.chunk_id for c in passages] == ["c00003", "c00001"] and unmapped == 0
    assert all("7.3" not in c.text for c in passages)
    _, unmapped = distractor_passages({**WITHHELD, "distractors": [{"unit_id": "x", "pages": [500, 500]}]}, CHUNKS, {}, full_config()["rag"]["eval"], 6)
    assert unmapped == 1


def test_eval_end_to_end(tmp_path):
    full = full_config()
    items = load_items([write_dataset(tmp_path / "ds", [PFS, WITHHELD, PRICE], tmp_path / "interim")])
    assert interim_dir(items, None) is None and load_unit_texts(None) == {}
    judge = FakeLLM(passing_judge)
    rag = make_rag(small_model)
    report = evaluate(items, rag, judge, full, tmp_path / "run", UNIT_TEXTS)

    rows = [json.loads(line) for line in (tmp_path / "run" / "responses.jsonl").open()]
    assert [r["uid"] for r in rows] == ["numerical-001", "unanswerable-001", "unanswerable-002"]
    pfs, withheld, price = rows
    assert pfs["score"] == 1.0 and pfs["category"] == "correct" and pfs["keys"]["correct"] and pfs["context"] == "retrieved"
    assert pfs["retrieval"]["hit"] and pfs["retrieval"]["first_rank"] == 1 and pfs["faithfulness"] == 1.0
    assert pfs["retrieval"]["context_precision"] == round(1 / 3, 3) and pfs["retrieval"]["candidate_recall"] == 1.0
    assert pfs["candidates"][0] == "c00002" and len(pfs["candidates"]) <= 10            # the fused pool, in fused order
    assert "Median PFS (months) | 7.3 | 3.1" in judge.calls[0][0][1]["content"]   # the judge sees the passages the model saw
    assert withheld["context"] == "fixed" and withheld["category"] == "correct_abstention" and withheld["retrieval"] is None
    assert withheld["candidates"] is None and withheld["faithfulness"] is None            # no pool and no claims for a decline
    assert [p["chunk_id"] for p in withheld["passages"]] == ["c00003", "c00001"]
    assert "7.3" not in judge.calls[1][0][1]["content"].split("Passages shown")[1]
    assert price["context"] == "retrieved" and price["category"] == "correct_abstention" and price["retrieved"][0]["score"] is not None

    assert report["items"] == 3 and set(report["per_type"]) == {"numerical", "unanswerable"} and report["fixed_context_items"] == 1
    assert report["overall"]["score"] == 1.0 and report["agreement"] == {"items": 3, "agree": 1.0, "keys_only": [], "judge_only": []}
    o = report["overall"]
    assert o["context_precision"] == round(1 / 3, 3) and o["candidate_recall"] == 1.0 and o["faithfulness"] == 1.0
    assert report["judge_model"] == "fake" and report["rag"]["retrieval"]["mode"] == "bm25" and report["rag"]["chunks"] == 4
    assert report["evidence_matching"] == {"page_only": 0, "quote_split": 0, "distractors_unmapped": 0} and not report["warnings"]
    assert json.loads((tmp_path / "run" / "eval_report.json").read_text())["items"] == 3


def test_without_a_judge_items_are_scored_on_keys(tmp_path):
    items = load_items([write_dataset(tmp_path / "ds", [PFS, PRICE], tmp_path / "interim")])
    report = evaluate(items, make_rag(small_model), None, full_config(), tmp_path / "run")
    assert report["overall"]["score"] == 1.0 and report["judge_model"] is None and report["agreement"]["items"] == 0


def test_first_failure_stops_the_run_later_ones_are_recorded(tmp_path):
    items = load_items([write_dataset(tmp_path / "ds", [PFS, PRICE], tmp_path / "interim")])

    def down(messages, tag):
        raise RagLLMError("connection refused")
    with pytest.raises(RagLLMError):
        evaluate(items, make_rag(down), None, full_config(), tmp_path / "run")

    def flaky(messages, tag):
        if "median" in question(messages):
            return small_model(messages, tag)
        raise RagLLMError("timeout")
    report = evaluate(items, make_rag(flaky), None, full_config(), tmp_path / "run2")
    assert report["overall"]["errors"] == 1 and report["warnings"][0].startswith("1 items failed")


def test_several_datasets_get_prefixed_ids(tmp_path):
    a = write_dataset(tmp_path / "smoke_a", [PFS], tmp_path / "interim")
    b = write_dataset(tmp_path / "smoke_b", [PFS], tmp_path / "other")
    items = load_items([a, b])
    assert [it["uid"] for it in items] == ["smoke_a/numerical-001", "smoke_b/numerical-001"]
    assert interim_dir(items, None) is None and interim_dir(items, tmp_path) == tmp_path
