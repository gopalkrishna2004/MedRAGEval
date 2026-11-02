import json
import re

import pytest
from synthetic import build_sheet, config, write_factsheet, write_interim

from medrageval.generate.candidates import Candidate
from medrageval.generate.generate import Run, generate_dataset
from medrageval.generate.llm import FakeLLM
from medrageval.generate.prompts import PROMPT_VERSION, rewrite_feedback

FACT = re.compile(r"^Fact (\S+): (.*?) \(from the (.*?)\)\n((?:  .*(?:\n|$))*)", re.M)


def facts_in(user: str) -> list[dict]:
    out = []
    for fid, describe, doc, body in FACT.findall(user):
        lines = dict(line.strip().split(": ", 1) for line in body.splitlines() if ": " in line)
        value = lines.get("value") or (lines.get("values") or lines.get("positions") or "").split("; ")[0].split(": ", 1)[-1]
        full = lines.get("value") or lines.get("values") or lines.get("positions") or ""
        out.append({"id": fid, "describe": describe, "doc": doc, "value": value, "full": full, "qualifiers": lines.get("qualifiers", "")})
    return out


def must_terms(user: str) -> list[str]:
    m = re.search(r"must name (.*?), in these or equivalent words", user)
    return re.findall(r'\("([^"]+)"', m.group(1)) if m else []   # the first option of each set


def excerpts(user: str) -> list[tuple[str, str]]:
    return re.findall(r"^\[([^\]]+)\] Document[^\n]*\n(.*?)(?=\n\n(?:Context|Excerpt|Nearby)|\Z)", user, re.M | re.S)


def subject(fact: dict, user: str) -> str:
    quals = [q.split(": ", 1)[1] for q in fact["qualifiers"].split("; ") if ": " in q]
    return ", ".join(dict.fromkeys(must_terms(user) + quals)) or "capivasertib"


def scripted(messages, tag):
    """A generator that writes each item from the fact block and quotes the start of each excerpt; a judge that passes everything."""
    user = next(m["content"] for m in messages if m["role"] == "user")   # a rewrite appends turns; the task is the first one
    if tag.startswith("judge:"):
        return {"pass": True, "answer_supported": True, "self_contained": True, "type_fits": True, "realistic": True,
                "answerable_from_single_excerpt": {}, "issues": []}
    if tag.startswith("alone:"):
        return {"answer": None, "reason": "the passage alone does not say"}
    if tag.startswith("check:"):
        n = len(re.findall(r"^\d+\. ", user, re.M))
        return {"results": [{"n": i + 1, "answered": "no", "passage_ids": [], "answer_if_any": None} for i in range(n)]}
    if "UNANSWERABLE questions" in user:
        return {"questions": [{"question": "What was the median follow-up for overall survival in the CAPItello-291 capivasertib arm at a third data cut?", "missing_fact": "follow-up"},
                              {"question": "Tell me about this excerpt above?", "missing_fact": "-"}]}
    facts, exs = facts_in(user), excerpts(user)
    quotes = [{"unit_id": uid, "quote": " ".join(body.split()[:12])} for uid, body in exs]
    if "UNANSWERABLE question" in user:
        uid, body = next((u, b) for u, b in exs if "[REDACTED]" in b)
        line = next(line for line in body.splitlines() if "[REDACTED]" in line)
        return {"question": "What is the list price per pack of capivasertib tablets that the company submission gives?", "answer": None, "answer_keys": [],
                "quotes": [{"unit_id": uid, "quote": " ".join(line.split()[:12])}]}
    f = facts[0]
    if "SUMMARY question" in user:
        return {"question": f"What did the {f['doc']} report about the efficacy of capivasertib plus fulvestrant in its trial?",
                "answer": " ".join(f"{x['value']}." for x in facts), "answer_keys": [], "quotes": [],
                "keypoints": [{"point": f"{x['full']} ({x['describe']})", "fact_id": x["id"]} for x in facts]}
    key = " ".join(f["value"].split()[:2]).rstrip(",;")
    return {"question": f"According to the {f['doc']}, what is recorded about {' '.join(f['describe'].split()[:8])} for {subject(f, user)}?",
            "answer": f"It records {f['value']}.", "answer_keys": [key], "quotes": quotes[:1]}


def test_end_to_end_writes_dataset_rejects_and_report(tmp_path):
    interim = write_factsheet(tmp_path / "interim")
    out = tmp_path / "out"
    llm = FakeLLM(scripted)
    report = generate_dataset(interim, out, config(), llm, llm, targets={"factual": 2, "unanswerable": 2}, types=["factual", "unanswerable"])

    items = [json.loads(l) for l in (out / "dataset.jsonl").open()]
    assert [i["item_id"] for i in items] == ["factual-001", "factual-002", "unanswerable-001", "unanswerable-002"]
    f = items[0]
    assert f["answerable"] and f["facts"] and f["group"] in ("trial_design", "economic_model", "stakeholder_views")
    assert f["provenance"]["prompt_version"] == PROMPT_VERSION and f["provenance"]["model"] == "fake"
    assert f["validation"]["programmatic"]["quote_found"] and f["validation"]["judge"]["realistic"]   # every answerable type is judged
    u = [i for i in items if i["type"] == "unanswerable"]
    assert {i["subtype"] for i in u} == {"redacted", "not_covered"} and all(i["answer"] is None for i in u)
    nc = next(i for i in u if i["subtype"] == "not_covered")
    assert nc["search"]["verdict"] == "no" and nc["anchor"][0]["unit_id"] and nc["facts"]

    rejects = [json.loads(l) for l in (out / "rejects.jsonl").open()]
    assert any("deictic_question" in r["reasons"] for r in rejects)     # the second not-covered question
    assert all(r["item_id"] is None and r["stage"] for r in rejects)

    assert report["items"] == 4 and report["per_type"]["factual"]["accepted"] == 2
    assert report["per_type"]["unanswerable"]["subtypes"] == {"redacted": 1, "not_covered": 1}
    assert report["facts"]["facts"] == 16 and report["items_per_group"]
    assert sum(1 for m, t in llm.calls if t.startswith("check:")) == 1      # one coverage call per anchor, not per question


def test_summary_items_carry_keypoints_and_the_facts_quotes(tmp_path):
    interim = write_factsheet(tmp_path / "interim")
    llm = FakeLLM(scripted)
    generate_dataset(interim, tmp_path / "out", config(), llm, llm, targets={"summary": 1}, types=["summary"])
    item = json.loads((tmp_path / "out" / "dataset.jsonl").read_text().splitlines()[0])
    assert item["type"] == "summary" and item["answer_keys"] == [] and len(item["keypoints"]) == len(item["facts"]) >= 3
    assert {e["unit_id"] for e in item["evidence"]} >= {"d02-e0003"}    # the system attaches each fact's own quote


def test_model_may_skip_facts_and_the_reason_is_recorded(tmp_path):
    def picky(messages, tag):
        if len(llm.calls) == 1:   # declines the first fact it is given
            return {"skip": "generic statement, nothing specific to ask"}
        return scripted(messages, tag)
    interim = write_factsheet(tmp_path / "interim")
    llm = FakeLLM(picky)
    report = generate_dataset(interim, tmp_path / "out", config(), llm, llm, targets={"factual": 2}, types=["factual"])
    rejects = [json.loads(l) for l in (tmp_path / "out" / "rejects.jsonl").open()]
    skipped = [r for r in rejects if r["stage"] == "skipped"]
    assert len(skipped) == 1 and skipped[0]["reasons"] == ["skip:generic statement, nothing specific to ask"] and skipped[0]["item_id"] is None
    assert report["per_type"]["factual"]["accepted"] == 2 and report["per_type"]["factual"]["rejected_by_reason"]["skip"] == 1


@pytest.mark.parametrize("accept", [True, False])
def test_partly_answered_questions_follow_accept_partly(tmp_path, accept):
    def hedging(messages, tag):
        out = scripted(messages, tag)
        if tag.startswith("check:"):
            out["results"][0]["answered"] = "partly"
            out["results"][0]["answer_if_any"] = "the DCO1 value is given, not the third data cut"
        return out
    interim = write_factsheet(tmp_path / "interim")
    llm = FakeLLM(hedging)
    cfg = config()
    cfg["unanswerable"]["questions_per_anchor"] = 1
    cfg["unanswerable"]["accept_partly"] = accept
    cfg["rewrite_rejected"] = False
    report = generate_dataset(interim, tmp_path / "out", cfg, llm, llm, targets={"unanswerable": 2}, types=["unanswerable"]) if accept else None
    if accept:    # a near miss: the bundle has a related fact, not the one asked
        assert report["per_type"]["unanswerable"]["subtypes"] == {"redacted": 1, "not_covered": 1}
    else:
        with pytest.raises(RuntimeError, match="short of target"):
            generate_dataset(interim, tmp_path / "out", cfg, llm, llm, targets={"unanswerable": 4}, types=["unanswerable"])
        rejects = [json.loads(l) for l in (tmp_path / "out" / "rejects.jsonl").open()]
        assert any(r["stage"] == "unanswerable_check" and r["reasons"] == ["answered:partly"] for r in rejects)


def test_shortfall_raises_after_writing(tmp_path):
    interim = write_factsheet(tmp_path / "interim")
    out = tmp_path / "out"
    llm = FakeLLM(scripted)
    with pytest.raises(RuntimeError, match="short of target"):
        generate_dataset(interim, out, config(), llm, llm, targets={"factual": 50}, types=["factual"])
    report = json.loads((out / "generate_report.json").read_text())
    assert report["per_type"]["factual"]["accepted"] >= 1 and report["warnings"]
    assert sum(1 for _ in (out / "dataset.jsonl").open()) == report["items"]


def test_missing_parse_output_or_fact_sheet_fails_loudly(tmp_path):
    llm = FakeLLM(scripted)
    with pytest.raises(FileNotFoundError):
        generate_dataset(tmp_path / "nowhere", tmp_path / "out", config(), llm, llm)
    interim = write_interim(tmp_path / "interim")
    with pytest.raises(FileNotFoundError, match="medrageval extract"):
        generate_dataset(interim, tmp_path / "out", config(), llm, llm)


def bridged(alone_answers: bool):
    """Multi-hop items reaching a result through "the primary endpoint of CAPItello-291"; given alone, the result's excerpt answers when alone_answers."""
    keys = {}

    def respond(messages, tag):
        user = next(m["content"] for m in messages if m["role"] == "user")
        if tag.startswith("alone:"):
            cid, uid = tag.split(":", 1)[1].rsplit(":", 1)
            first = cid.split(":", 1)[1].split("+")[0]
            return {"answer": " and ".join(keys.get(cid, [])) if alone_answers and not uid.startswith(first[:3] + "-e0040") else None, "reason": "-"}
        if not tag.startswith(("multihop:")):
            return scripted(messages, tag)
        facts, exs = facts_in(user), excerpts(user)
        b = facts[1]
        keys[tag.removesuffix(":rewrite")] = [b["value"]]
        return {"question": f"For the primary endpoint of CAPItello-291 ({', '.join(must_terms(user)) or 'company submission'}), what result did capivasertib plus fulvestrant reach?",
                "answer": f"It reached {b['value']}.", "answer_keys": [b["value"]], "hop_a": "names the endpoint", "hop_b": "gives the result",
                "quotes": [{"unit_id": uid, "quote": " ".join(body.split()[:12])} for uid, body in exs]}
    return respond


@pytest.mark.parametrize("alone", [True, False])
def test_multihop_answerable_from_one_excerpt_is_rejected(tmp_path, alone):
    interim, out = write_factsheet(tmp_path / "interim"), tmp_path / "out"
    llm = FakeLLM(bridged(alone))
    if alone:
        with pytest.raises(RuntimeError, match="short of target"):
            generate_dataset(interim, out, config(), llm, llm, targets={"multihop": 1}, types=["multihop"])
        rejects = [json.loads(l) for l in (out / "rejects.jsonl").open()]
        assert any(r["reasons"][0].startswith("answerable_from:") and any(a["matches"] for a in r["validation"]["alone"].values()) for r in rejects)
    else:
        generate_dataset(interim, out, config(), llm, llm, targets={"multihop": 1}, types=["multihop"])
        items = [json.loads(l) for l in (out / "dataset.jsonl").open()]
        assert len(items) == 1 and items[0]["hops"]["bridge"] == "the primary endpoint of CAPItello-291"
        assert not any(r["matches"] for r in items[0]["validation"]["alone"].values())
    assert sum(tag.startswith("alone:") for _, tag in llm.calls) >= 2   # one call per excerpt


def test_rejected_item_is_rewritten_once_with_the_reasons(tmp_path):
    def sloppy(messages, tag):
        out = scripted(messages, tag)
        if not tag.endswith(":rewrite") and out.get("answer"):   # the first version has an unusable key
            out = {**out, "answer_keys": ["a key that is far too long for scoring"]}
        return out
    interim, out = write_factsheet(tmp_path / "interim"), tmp_path / "out"
    llm = FakeLLM(sloppy)
    report = generate_dataset(interim, out, config(), llm, llm, targets={"factual": 1}, types=["factual"])
    items = [json.loads(l) for l in (out / "dataset.jsonl").open()]
    assert len(items) == 1 and items[0]["first_attempt"]["item"]["answer_keys"] == ["a key that is far too long for scoring"]
    assert any(r.startswith("answer_key_too_long:") for r in items[0]["first_attempt"]["reasons"])
    rewrites = report["per_type"]["factual"]["rewrites"]
    assert rewrites["accepted"] == 1 and rewrites["tried"] >= 1
    retry = next(m for m, t in llm.calls if t.endswith(":rewrite"))
    assert retry[-2]["role"] == "assistant" and "far too long" in retry[-2]["content"]   # the model sees its own rejected item
    assert retry[-1]["content"].startswith("That item was rejected:") and "is too long" in retry[-1]["content"]


def test_only_failures_the_wording_can_fix_are_rewritten(tmp_path):
    llm = FakeLLM(scripted)
    corpus, sheet = build_sheet()
    run = Run(corpus, sheet, config(), llm, llm, tmp_path, 1)
    fixable = [{"stage": "programmatic", "reasons": ["answer_key_too_long:x", "question_echoes_quote:0.84"]}]
    assert run._rewritable(fixable)
    assert not run._rewritable([{"stage": "programmatic", "reasons": ["claim_restated_in:d02-e0001"]}])   # the evidence's fault
    assert not run._rewritable([{"stage": "skipped", "reasons": ["skip:generic"]}])
    assert not run._rewritable([{"stage": "generation", "reasons": ["bad_json:x"]}])
    assert not run._rewritable([{"stage": "accepted", "reasons": []}, {"stage": "unanswerable_check", "reasons": ["answered:yes"]}])
    run.cfg["rewrite_rejected"] = False
    assert not run._rewritable(fixable)
    run.close()


def test_rewrite_feedback_explains_each_failure():
    corpus, _ = build_sheet()
    c = Candidate("multihop", "bridge", [corpus.units["d02-e0040"], corpus.units["d02-e0003"]], {})
    results = [{"stage": "judge", "item": {"question": "q"}, "reasons": ["judge_failed", "judge:realistic", "answerable_from:d02-e0003"],
                "judge": {"issues": ["The table alone gives 7.3 months."]}, "alone": {"d02-e0003": {"answer": "7.3 months", "matches": True}}}]
    lines = rewrite_feedback(results, c)
    assert any(l.startswith('Given only excerpt d02-e0003, a reader answered "7.3 months"') for l in lines)
    assert any("clinician, health economist or committee member" in l for l in lines)
    assert "Reviewer: The table alone gives 7.3 months." in lines
    lines = rewrite_feedback([{"stage": "programmatic", "item": {}, "reasons": ["question_lacks_discriminator:DCO2", "table_language:Table 7", "made_up_code"]}], c)
    assert lines == ['The question must name "DCO2" (or equivalent words), so it means only this fact.',
                     'The question mentions a table, figure, column header, row or column ("Table 7"): ask about the subject itself.', "made_up_code"]


def test_judge_sees_the_context_the_writer_saw():
    from medrageval.generate.prompts import build_judge
    corpus, sheet = build_sheet()
    c = Candidate("numerical", "efficacy", sheet.units(sheet.facts["d02-efficacy.endpoint_result-001"]), {}, [sheet.facts["d02-efficacy.endpoint_result-001"]])
    item = {"question": "q", "answer": "a", "answer_keys": ["7.3 months"], "quotes": [{"unit_id": "d02-e0003", "quote": "Median PFS (months) | 7.3 | 3.1"}]}
    user = build_judge(item, c, corpus, config())[1]["content"]
    assert "Context before [d02-e0003]:\nProgression-free survival results by investigator assessment" in user
