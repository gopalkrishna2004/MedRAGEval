import json
import re

import pytest
from synthetic import config, write_interim

from medrageval.generate.generate import generate_dataset
from medrageval.generate.prompts import PROMPT_VERSION
from medrageval.generate.llm import FakeLLM


def scripted(messages, tag):
    """A generator that quotes the first sentence of each excerpt and a judge that passes everything."""
    user = messages[-1]["content"]
    if tag.startswith("judge:"):
        return {"pass": True, "answer_supported": True, "self_contained": True, "type_fits": True, "answerable_from_single_excerpt": {}, "issues": []}
    if tag.startswith("check:"):
        n = len(re.findall(r"^\d+\. ", user, re.M))
        return {"results": [{"n": i + 1, "answered": "no", "passage_ids": [], "answer_if_any": None} for i in range(n)]}
    excerpts = re.findall(r"^\[([^\]]+)\] Document[^\n]*\n(.*?)(?=\n\nExcerpt|\Z)", user, re.M | re.S)
    if "UNANSWERABLE questions" in user:
        return {"questions": [{"question": "What was the median follow-up for overall survival in the CAPItello-291 capivasertib arm?", "missing_fact": "follow-up"},
                              {"question": "Tell me about this excerpt above?", "missing_fact": "-"}]}
    uid, body = excerpts[0]
    words = body.split()
    quote = " ".join(words[:12])
    key = " ".join(words[:3])
    if "UNANSWERABLE question" in user:
        return {"question": f"What is the {' '.join(words[1:3])} of capivasertib per pack, as the company submission would state it?",
                "answer": None, "answer_keys": [], "quotes": [{"unit_id": uid, "quote": quote}]}
    return {"question": f"What does the appraisal say about {' '.join(words[3:8])} in the CAPItello-291 population?",
            "answer": f"It says {key}.", "answer_keys": [key], "quotes": [{"unit_id": uid, "quote": quote}]}


def test_end_to_end_writes_dataset_rejects_and_report(tmp_path):
    interim = write_interim(tmp_path / "interim")
    out = tmp_path / "out"
    llm = FakeLLM(scripted)
    report = generate_dataset(interim, out, config(), llm, llm, targets={"factual": 2, "unanswerable": 2}, types=["factual", "unanswerable"])

    items = [json.loads(l) for l in (out / "dataset.jsonl").open()]
    assert [i["item_id"] for i in items] == ["factual-001", "factual-002", "unanswerable-001", "unanswerable-002"]
    f = items[0]
    assert f["answerable"] and f["evidence"][0]["doc_role"] in ("company_submission", "eag_report", "clarification")
    assert f["provenance"]["prompt_version"] == PROMPT_VERSION and f["provenance"]["model"] == "fake"
    assert f["validation"]["programmatic"]["quote_found"] and f["validation"]["judge"] is None   # factual is not judged
    u = [i for i in items if i["type"] == "unanswerable"]
    assert {i["subtype"] for i in u} == {"redacted", "not_covered"} and all(i["answer"] is None for i in u)
    nc = next(i for i in u if i["subtype"] == "not_covered")
    assert nc["search"]["verdict"] == "no" and nc["anchor"][0]["unit_id"]

    rejects = [json.loads(l) for l in (out / "rejects.jsonl").open()]
    assert any("deictic_question" in r["reasons"] for r in rejects)     # the second not-covered question
    assert all(r["item_id"] is None and r["stage"] for r in rejects)

    assert report["items"] == 4 and report["per_type"]["factual"]["accepted"] == 2
    assert report["per_type"]["unanswerable"]["subtypes"] == {"redacted": 1, "not_covered": 1}
    assert report["corpus"]["roles"]["d03"] == "eag_report" and report["llm"]["generator"]["calls"] > 0
    assert sum(1 for m, t in llm.calls if t.startswith("check:")) == 1      # one coverage call per anchor, not per question
    assert json.loads((out / "generate_report.json").read_text())["items"] == 4


def test_model_may_skip_evidence_and_the_reason_is_recorded(tmp_path):
    def picky(messages, tag):
        if len(llm.calls) == 1:   # declines the first excerpt it is given
            return {"skip": "generic statement, nothing specific to ask"}
        return scripted(messages, tag)
    interim = write_interim(tmp_path / "interim")
    llm = FakeLLM(picky)
    report = generate_dataset(interim, tmp_path / "out", config(), llm, llm, targets={"factual": 2}, types=["factual"])
    rejects = [json.loads(l) for l in (tmp_path / "out" / "rejects.jsonl").open()]
    skipped = [r for r in rejects if r["stage"] == "skipped"]
    assert len(skipped) == 1 and skipped[0]["reasons"] == ["skip:generic statement, nothing specific to ask"] and skipped[0]["item_id"] is None
    assert report["per_type"]["factual"]["accepted"] == 2 and report["per_type"]["factual"]["rejected_by_reason"]["skip"] == 1
    items = [json.loads(l) for l in (tmp_path / "out" / "dataset.jsonl").open()]
    assert all(0 <= i["validation"]["metrics"]["question_overlap"] <= 1 for i in items)


def test_partly_answered_questions_are_rejected(tmp_path):
    def hedging(messages, tag):
        out = scripted(messages, tag)
        if tag.startswith("check:"):
            out["results"][0]["answered"] = "partly"
            out["results"][0]["answer_if_any"] = "the trial is described as UK-based"
        return out
    interim = write_interim(tmp_path / "interim")
    llm = FakeLLM(hedging)
    cfg = config()
    cfg["unanswerable"]["questions_per_anchor"] = 1
    report = generate_dataset(interim, tmp_path / "out", cfg, llm, llm, targets={"unanswerable": 2}, types=["unanswerable"])
    assert report["per_type"]["unanswerable"]["accepted"] == 1 and report["per_type"]["unanswerable"]["subtypes"] == {"redacted": 1}
    rejects = [json.loads(l) for l in (tmp_path / "out" / "rejects.jsonl").open()]
    assert any(r["stage"] == "unanswerable_check" and r["reasons"] == ["answered:partly"] and r["search"]["verdict"] == "partly" for r in rejects)
    items = [json.loads(l) for l in (tmp_path / "out" / "dataset.jsonl").open()]
    assert items and all(i["subtype"] == "redacted" for i in items)


def test_shortfall_raises_after_writing(tmp_path):
    interim = write_interim(tmp_path / "interim")
    out = tmp_path / "out"
    llm = FakeLLM(scripted)
    with pytest.raises(RuntimeError, match="short of target"):
        generate_dataset(interim, out, config(), llm, llm, targets={"factual": 50}, types=["factual"])
    report = json.loads((out / "generate_report.json").read_text())
    assert report["per_type"]["factual"]["accepted"] >= 1 and report["warnings"]
    assert sum(1 for _ in (out / "dataset.jsonl").open()) == report["items"]


def test_type_subset_and_missing_input(tmp_path):
    llm = FakeLLM(scripted)
    with pytest.raises(FileNotFoundError):
        generate_dataset(tmp_path / "nowhere", tmp_path / "out", config(), llm, llm)
