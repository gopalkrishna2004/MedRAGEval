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
from medrageval.rag.evaluate import evaluate, gold_chunks, load_items, render_chunk, standin_passages, withheld_figures

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
# The PFS question again with its evidence withheld. The stored distractors are the generator's record; the evaluator builds its
# own stand-ins from the application's chunks, so the gold chunk c00002 must never be shown.
WITHHELD = {"item_id": "unanswerable-001", "type": "unanswerable", "subtype": "withheld", "answerable": False, "source_item_id": "numerical-001",
            "question": PFS["question"], "answer": None, "answer_keys": [], "also_in": [], "evidence": [],
            "withheld": {"answer": PFS["answer"], "answer_keys": PFS["answer_keys"], "evidence": PFS["evidence"]},
            "distractors": [{"unit_id": "d02-e0004", "doc_id": "d02", "pages": [40, 40], "quote": "", "caption": None},
                            {"unit_id": "d02-e0001", "doc_id": "d02", "pages": [10, 10], "quote": "", "caption": None},
                            {"unit_id": "d02-e0003", "doc_id": "d02", "pages": [21, 21], "quote": "", "caption": "Table 12. Efficacy results"}]}
# An unanswerable item without withheld evidence (an older dataset): answered over live retrieval like the rest.
PRICE = {"item_id": "unanswerable-002", "type": "unanswerable", "subtype": "redacted", "answerable": False,
         "question": "What is the list price of a pack of 64 capivasertib tablets?", "answer": None, "answer_keys": [], "also_in": [],
         "evidence": [{"unit_id": "d02-e0004", "element_id": "d02-e0004", "doc_id": "d02", "pages": [40, 40], "quote": "£[REDACTED] per pack"}]}



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
    if tag.startswith("screen:"):   # the screening call: every candidate chunk is a clean stand-in
        return {"results": [{"chunk_id": cid, "answers": "no", "answer_if_any": None} for cid in shown_ids(messages)]}
    declined = "Not found in the documents." in messages[-1]["content"].split("System answer:")[1].split("Passages shown")[0]
    claims = [] if declined else [{"claim": "median PFS 7.3 months", "supported": True}]
    return {"verdict": "correct", "keypoints": [], "claims": claims, "grounded": None if declined else True, "reason": "ok"}


def shown_ids(messages):
    """The chunk ids a screening prompt shows, in order."""
    import re
    return re.findall(r"^\[(c\d+)\]", messages[-1]["content"], re.M)


def test_gold_chunks_hold_the_withheld_quote_anywhere():
    copy_elsewhere = mk("c00009", (300, 300), "The EAG reproduced the table: Median PFS (months) | 7.3 | 3.1.", title="EAG report")
    assert [c.chunk_id for c in gold_chunks(WITHHELD, CHUNKS + [copy_elsewhere])] == ["c00002", "c00009"]
    assert gold_chunks(PRICE, CHUNKS) == []   # no withheld evidence


def test_render_chunk_heads_with_the_id_and_cuts_long_bodies():
    text = render_chunk(CHUNKS[1], 2000)
    assert text.startswith("[c00002] Document: Company submission | Section: Results | Pages: 21-21\n") and "7.3" in text
    cut = render_chunk(mk("c00010", (1, 1), "line one\n" * 50), 100)
    assert cut.endswith("[...]") and len(cut) < 300


def test_standin_passages_search_near_the_gold_screen_and_drop_the_gold():
    rag = make_rag(small_model)
    screener = FakeLLM(passing_judge)
    s = standin_passages({**WITHHELD, "uid": "u1"}, rag, screener, full_config()["rag"]["eval"], 3)
    assert s["gold"] == ["c00002"] and s["excluded"] is None
    assert "c00002" not in [c.chunk_id for c in s["passages"]] and len(s["passages"]) == 3
    assert all("7.3" not in c.text for c in s["passages"])
    messages, tag = screener.calls[0]
    assert tag == "screen:u1" and "7.3 months" in messages[-1]["content"] and "[c00002]" not in messages[-1]["content"]
    assert [x["chunk_id"] for x in s["screening"]] == [c.chunk_id for c in s["passages"]]


def test_standin_passages_keep_only_what_the_judge_finds_clean():
    rag = make_rag(small_model)

    def strict(messages, tag):   # c00001 is said to answer; the rest are clean
        return {"results": [{"chunk_id": cid, "answers": "yes" if cid == "c00001" else "no"} for cid in shown_ids(messages)]}
    s = standin_passages({**WITHHELD, "uid": "u1"}, rag, FakeLLM(strict), full_config()["rag"]["eval"], 3)
    assert "c00001" not in [c.chunk_id for c in s["passages"]] and len(s["passages"]) == 2
    assert {x["answers"] for x in s["screening"]} == {"yes", "no"}

    def all_leak(messages, tag):
        return {"results": [{"chunk_id": cid, "answers": "partly"} for cid in shown_ids(messages)]}
    s = standin_passages({**WITHHELD, "uid": "u1"}, rag, FakeLLM(all_leak), full_config()["rag"]["eval"], 3)
    assert s["passages"] == [] and s["excluded"].startswith("every candidate chunk")


def test_withheld_figures_only_from_specific_numbers():
    assert withheld_figures(["7.3 months"]) == {7.3}
    assert withheld_figures(["153 patients (98.7%)", "134 patients (100%)"]) == {153.0, 98.7, 134.0, 100.0}
    assert withheld_figures(["£52,681"]) == {52681.0}
    assert withheld_figures(["robust phase 3 trial"]) == set() and withheld_figures(["10 RCTs"]) == set()
    assert withheld_figures(["company"]) == set() and withheld_figures([]) == set()


def test_a_chunk_stating_every_withheld_figure_is_dropped_whatever_the_judge_said():
    restated = mk("c00011", (300, 300), "The EAG notes the FAKTION median PFS of 7.3 months in its own table.", title="EAG report")
    rag = make_rag(small_model, CHUNKS + [restated])
    s = standin_passages({**WITHHELD, "uid": "u1"}, rag, FakeLLM(passing_judge), full_config()["rag"]["eval"], 6)
    verdicts = {x["chunk_id"]: x["answers"] for x in s["screening"]}
    assert verdicts["c00011"] == "figures" and "c00011" not in [c.chunk_id for c in s["passages"]]
    assert all(v == "no" for cid, v in verdicts.items() if cid != "c00011") and len(s["passages"]) == 3


def test_standin_passages_ask_again_about_chunks_the_judge_skipped():
    rag = make_rag(small_model)

    def skips_the_clean_ones(messages, tag):   # first call: only the one it finds answering; second call: the rest, as asked
        ids = shown_ids(messages)
        if tag.startswith("screen2:"):
            return {"results": [{"chunk_id": cid, "answers": "no"} for cid in ids]}
        return {"results": [{"chunk_id": ids[0], "answers": "yes"}]}
    screener = FakeLLM(skips_the_clean_ones)
    s = standin_passages({**WITHHELD, "uid": "u1"}, rag, screener, full_config()["rag"]["eval"], 3)
    assert [c[1] for c in screener.calls] == ["screen:u1", "screen2:u1"]
    assert len(shown_ids(screener.calls[1][0])) == 2 and len(s["passages"]) == 2
    assert [x["answers"] for x in s["screening"]] == ["yes", "no", "no"]
    assert "every passage" in screener.calls[0][0][0]["content"]

    def silent(messages, tag):
        return {"results": []}
    s = standin_passages({**WITHHELD, "uid": "u1"}, rag, FakeLLM(silent), full_config()["rag"]["eval"], 3)
    assert s["excluded"].startswith("the judge gave no verdict") and all(x["answers"] == "unscreened" for x in s["screening"])


def test_unanswerable_with_no_stand_in_left_is_not_scored(tmp_path):
    items = load_items([write_dataset(tmp_path / "ds", [PFS, WITHHELD], tmp_path / "interim")])
    rag = make_rag(small_model)

    def judge_fn(messages, tag):
        if tag.startswith("screen:"):
            return {"results": [{"chunk_id": cid, "answers": "yes"} for cid in shown_ids(messages)]}
        return passing_judge(messages, tag)
    report = evaluate(items, rag, FakeLLM(judge_fn), full_config(), tmp_path / "run")
    rows = [json.loads(line) for line in (tmp_path / "run" / "responses.jsonl").open()]
    assert rows[1]["category"] == "excluded" and rows[1]["score"] is None and rows[1]["excluded"].startswith("every candidate")
    assert rows[1]["gold_chunks"] == ["c00002"] and all(x["answers"] == "yes" for x in rows[1]["screening"])
    assert rag.llm.stats["calls"] == 1                                   # the excluded item is never asked
    assert report["items"] == 1 and report["excluded"] == ["unanswerable-001"] and report["unanswerable"]["items"] == 0
    assert report["overall"]["categories"] == {"correct": 1} and "unanswerable-001" in report["warnings"][0]
    assert report["standin_screening"]["answers"] == 3


def test_without_a_judge_withheld_items_are_excluded(tmp_path):
    items = load_items([write_dataset(tmp_path / "ds", [PFS, WITHHELD, PRICE], tmp_path / "interim")])
    report = evaluate(items, make_rag(small_model), None, full_config(), tmp_path / "run")
    assert report["excluded"] == ["unanswerable-001"] and report["items"] == 2 and report["overall"]["score"] == 1.0


def test_eval_end_to_end(tmp_path):
    full = full_config()
    items = load_items([write_dataset(tmp_path / "ds", [PFS, WITHHELD, PRICE], tmp_path / "interim")])
    judge = FakeLLM(passing_judge)
    rag = make_rag(small_model)
    report = evaluate(items, rag, judge, full, tmp_path / "run")

    rows = [json.loads(line) for line in (tmp_path / "run" / "responses.jsonl").open()]
    assert [r["uid"] for r in rows] == ["numerical-001", "unanswerable-001", "unanswerable-002"]
    pfs, withheld, price = rows
    assert pfs["score"] == 1.0 and pfs["category"] == "correct" and pfs["keys"]["correct"] and pfs["context"] == "retrieved"
    assert pfs["retrieval"]["hit"] and pfs["retrieval"]["first_rank"] == 1 and pfs["faithfulness"] == 1.0
    assert pfs["retrieval"]["gold_prec"] == round(1 / 3, 3) and pfs["retrieval"]["candidate_recall"] == 1.0
    assert pfs["candidates"][0] == "c00002" and len(pfs["candidates"]) <= 10            # the fused pool, in fused order
    grading = [c for c in judge.calls if c[1].startswith("judge:")]
    assert "Median PFS (months) | 7.3 | 3.1" in grading[0][0][1]["content"]   # the judge sees the passages the model saw
    assert withheld["context"] == "fixed" and withheld["category"] == "correct_abstention" and withheld["retrieval"] is None
    assert withheld["candidates"] is None and withheld["faithfulness"] is None            # no pool and no claims for a decline
    assert len(withheld["passages"]) == 3 and "c00002" not in [p["chunk_id"] for p in withheld["passages"]]
    assert withheld["gold_chunks"] == ["c00002"] and len(withheld["screening"]) == 3
    assert "7.3" not in grading[1][0][1]["content"].split("Passages shown")[1]
    assert [c[1] for c in judge.calls if c[1].startswith("screen:")] == ["screen:unanswerable-001"]   # one screening call per item
    assert price["context"] == "retrieved" and price["category"] == "correct_abstention" and price["retrieved"][0]["score"] is not None

    assert report["items"] == 3 and set(report["per_type"]) == {"numerical", "unanswerable"} and report["fixed_context_items"] == 1
    assert report["overall"]["score"] == 1.0 and report["agreement"] == {"items": 3, "agree": 1.0, "keys_only": [], "judge_only": []}
    o = report["overall"]
    assert o["gold_prec"] == round(1 / 3, 3) and o["candidate_recall"] == 1.0 and o["faithfulness"] == 1.0
    assert report["judge_model"] == "fake" and report["rag"]["retrieval"]["mode"] == "bm25" and report["rag"]["chunks"] == 4
    assert report["evidence_matching"] == {"page_only": 0, "quote_split": 0} and not report["warnings"]
    assert report["standin_screening"] == {"items": 1, "candidates": 3, "answers": 0, "figures": 0, "unscreened": 0, "passages_mean": 3.0}
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
    assert [it["item_id"] for it in load_items([write_dataset(tmp_path / "c", [PFS, WITHHELD, PRICE], tmp_path / "i")], types=["unanswerable"])] \
        == ["unanswerable-001", "unanswerable-002"]


def test_messy_setting_asks_the_twin_and_grades_against_the_clean_question(tmp_path):
    twin = {"text": "hey quick one median pfs w/ cap + fulv investigator assessed, months??", "severity": "medium", "habits": ["informal", "shorthand"],
            "persona": "a busy oncologist between clinics", "rewrite": "hey quick one, median PFS w/ cap + fulv, investigator assessed, months?", "surface": ["lowercase"]}
    ds = write_dataset(tmp_path / "ds", [{**PFS, "question_messy": twin}, {**WITHHELD, "question_messy": twin}, PRICE], tmp_path / "interim")
    asked = []

    def recording(messages, tag):
        asked.append(question(messages))
        return small_model(messages, tag)

    items = load_items([ds], question="messy")
    assert [it["question_asked"] for it in items] == [twin["text"], twin["text"], PRICE["question"]]   # no twin: the clean question
    judge = FakeLLM(passing_judge)
    report = evaluate(items, make_rag(recording), judge, full_config(), tmp_path / "run")
    assert asked[:2] == [twin["text"], twin["text"]] and asked[2] == PRICE["question"]
    rows = [json.loads(line) for line in (tmp_path / "run" / "responses.jsonl").open()]
    assert rows[0]["question"] == PFS["question"] and rows[0]["question_asked"] == twin["text"] and rows[0]["question_setting"] == "messy"
    assert rows[0]["score"] == 1.0   # the small model still finds "median" in the twin
    for msgs, tag in judge.calls:   # the judge and the screening work from the clean question
        assert twin["text"] not in msgs[1]["content"] and (PFS["question"] in msgs[1]["content"] or tag.startswith("judge:unanswerable-002"))
    assert report["question_setting"] == "messy" and report["messy_missing"] == ["unanswerable-002"]
    assert any("no messy twin" in w for w in report["warnings"])

    clean = load_items([ds])
    assert all(it["question_asked"] == it["question"] and it["question_setting"] == "clean" for it in clean)
    with pytest.raises(ValueError):
        load_items([ds], question="sloppy")


def test_run_report_leaves_out_the_answering_server_address(tmp_path):
    items = load_items([write_dataset(tmp_path / "ds", [PFS], tmp_path / "interim")])
    evaluate(items, make_rag(small_model), None, full_config(), tmp_path / "run")
    assert "http://fake" not in (tmp_path / "run" / "eval_report.json").read_text()


def test_reader_errors_leave_out_the_server_address():
    # The error text is saved on the item's row, so it must not name the machine.
    import httpx
    from medrag.llm import ChatLLM
    from openai import APIConnectionError
    llm = ChatLLM("m", "http://10.0.0.1:9/v1", "k", max_retries=0)

    def refuse(**_):
        raise APIConnectionError(request=httpx.Request("POST", "http://10.0.0.1:9/v1/chat/completions"))
    llm.client.chat.completions.create = refuse
    with pytest.raises(RagLLMError) as e:
        llm.complete([{"role": "user", "content": "q"}])
    assert "10.0.0.1" not in str(e.value)
