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
    if tag.startswith("alone:"):
        return {"answer": None, "reason": "the passage alone does not say"}
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


@pytest.mark.parametrize("accept", [True, False])
def test_partly_answered_questions_follow_accept_partly(tmp_path, accept):
    def hedging(messages, tag):
        out = scripted(messages, tag)
        if tag.startswith("check:"):
            out["results"][0]["answered"] = "partly"
            out["results"][0]["answer_if_any"] = "the DCO1 value is given, not DCO2"
        return out
    interim = write_interim(tmp_path / "interim")
    llm = FakeLLM(hedging)
    cfg = config()
    cfg["unanswerable"]["questions_per_anchor"] = 1
    cfg["unanswerable"]["accept_partly"] = accept
    report = generate_dataset(interim, tmp_path / "out", cfg, llm, llm, targets={"unanswerable": 2}, types=["unanswerable"])
    items = [json.loads(l) for l in (tmp_path / "out" / "dataset.jsonl").open()]
    rejects = [json.loads(l) for l in (tmp_path / "out" / "rejects.jsonl").open()]
    if accept:    # a near miss: the bundle has a related fact, not the one asked
        assert report["per_type"]["unanswerable"]["subtypes"] == {"redacted": 1, "not_covered": 1}
        assert next(i for i in items if i["subtype"] == "not_covered")["search"]["verdict"] == "partly"
    else:
        assert report["per_type"]["unanswerable"]["subtypes"] == {"redacted": 1}
        assert any(r["stage"] == "unanswerable_check" and r["reasons"] == ["answered:partly"] for r in rejects)


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


def bridged(alone_answer):
    """A multi-hop item quoting both excerpts of the "see Table 7" pair; the table, given alone, answers with `alone_answer`."""
    def respond(messages, tag):
        if tag.startswith("alone:"):
            return {"answer": alone_answer if tag.endswith("d02-e0003") else None, "reason": "-"}
        if tag.startswith("judge:"):
            return scripted(messages, tag)
        return {"question": "In the biomarker-selected group, what was the median time to progression or death on capivasertib with fulvestrant, as judged by local investigators?",
                "answer": "It was 7.3 months.", "answer_keys": ["7.3"], "hop_a": "points to the results table", "hop_b": "gives the median",
                "quotes": [{"unit_id": "d02-e0002", "quote": "Progression-free survival results by investigator assessment in the PI3K/AKT pathway-altered population"},
                           {"unit_id": "d02-e0003", "quote": "Median PFS (months) | 7.3 | 3.1"}]}
    return respond


@pytest.mark.parametrize("alone", ["Median PFS was 7.3 months.", None])
def test_multihop_answerable_from_one_excerpt_is_rejected(tmp_path, alone):
    interim, out = write_interim(tmp_path / "interim"), tmp_path / "out"
    llm = FakeLLM(bridged(alone))
    if alone:   # every other candidate fails on quotes from units it does not have, so the type ends short
        with pytest.raises(RuntimeError, match="short of target"):
            generate_dataset(interim, out, config(), llm, llm, targets={"multihop": 1}, types=["multihop"])
        rejects = [json.loads(l) for l in (out / "rejects.jsonl").open()]
        assert any(r["reasons"] == ["answerable_from:d02-e0003"] and r["validation"]["alone"]["d02-e0003"]["matches"] for r in rejects)
    else:
        generate_dataset(interim, out, config(), llm, llm, targets={"multihop": 1}, types=["multihop"])
        items = [json.loads(l) for l in (out / "dataset.jsonl").open()]
        assert len(items) == 1 and not any(r["matches"] for r in items[0]["validation"]["alone"].values())
    assert sum(tag.startswith("alone:") for _, tag in llm.calls) >= 2   # one call per excerpt
