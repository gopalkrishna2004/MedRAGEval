import json
import re

import pytest
from synthetic import build_sheet, config, facts_config, write_factsheet, write_interim

from medrageval.facts.schema import SchemaError, resolve_path
from medrageval.generate.candidates import Candidate
from medrageval.generate.generate import Run, generate_dataset, load_sheet
from medrageval.generate.llm import FakeLLM, LLMJsonError
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


def screened(user: str) -> list[str]:
    return re.findall(r"^\[([^\]]+)\] Document", user, re.M)


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
    if tag.startswith("withhold:"):   # the screening judge: no candidate distractor answers
        return {"results": [{"unit_id": uid, "answers": "no", "answer_if_any": None} for uid in screened(user)]}
    if tag.startswith("messy:"):   # the user-style rewrite: the clean question, lower case and chatty
        clean = re.search(r"^Clean question: (.*)$", user, re.M).group(1)
        return {"question": "hey quick one, " + clean.lower()}
    facts, exs = facts_in(user), excerpts(user)
    quotes = [{"unit_id": uid, "quote": " ".join(body.split()[:12])} for uid, body in exs]
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
    for it in items:   # every item carries its messy twin; an unanswerable one copies its source's
        tw = it["question_messy"]
        assert tw["text"] and tw["severity"] in ("light", "medium", "heavy") and tw["rewrite"].startswith("hey quick one, ")
        assert "error" not in tw
    src = {i["item_id"]: i for i in items}
    for u in (i for i in items if i["type"] == "unanswerable"):
        assert u["question_messy"] == src[u["source_item_id"]]["question_messy"]
    assert report["messy"]["rewritten"] == 2 and report["messy"]["copied_from_source"] == 2 and report["messy"]["errors"] == []
    assert sum(1 for _, tag in llm.calls if tag.startswith("messy:")) == 2
    f = items[0]
    assert f["answerable"] and f["facts"] and f["group"] in ("trial_design", "economic_model", "stakeholder_views")
    assert f["provenance"]["prompt_version"] == PROMPT_VERSION and f["provenance"]["model"] == "fake"
    assert f["validation"]["programmatic"]["quote_found"] and f["validation"]["judge"]["realistic"]   # every answerable type is judged
    # Unanswerable items: the accepted factual items asked again, their evidence withheld and screened distractors stored
    u = [i for i in items if i["type"] == "unanswerable"]
    by_id = {i["item_id"]: i for i in items}
    assert len(u) == 2 and all(i["subtype"] == "withheld" and i["answer"] is None and i["answer_keys"] == [] and not i["answerable"] for i in u)
    for w in u:
        src = by_id[w["source_item_id"]]
        assert src["type"] == "factual" and w["question"] == src["question"] and w["facts"] == src["facts"] and w["group"] == src["group"]
        assert w["withheld"] == {"answer": src["answer"], "answer_keys": src["answer_keys"], "evidence": src["evidence"]} and w["evidence"] == []
        withheld_units = {e["unit_id"] for e in src["evidence"]}
        assert len(w["distractors"]) >= config()["unanswerable"]["min_distractors"] and not withheld_units & {d["unit_id"] for d in w["distractors"]}
        assert all(d["doc_label"] and d["pages"] for d in w["distractors"])
        clean = [x["unit_id"] for x in w["screening"] if x["answers"] == "no"]   # the judge cleared every passage; only the key rule can flag one
        assert clean[:config()["unanswerable"]["distractors"]] == [d["unit_id"] for d in w["distractors"]]
        assert all(x["answers"] == "no" or (x["answers"] == "yes" and x["keys_present"]) for x in w["screening"])
    assert {w["source_item_id"] for w in u} == {"factual-001", "factual-002"}

    rejects = [json.loads(l) for l in (out / "rejects.jsonl").open()]
    assert all(r["item_id"] is None and r["stage"] for r in rejects)

    assert report["items"] == 4 and report["per_type"]["factual"]["accepted"] == 2
    assert report["per_type"]["unanswerable"]["subtypes"] == {"withheld": 2}
    assert report["facts"]["facts"] == 20 and report["items_per_group"] and set(report["items_per_document"]) >= {"d02"}
    mix = report["group_mix"]   # answerable items per group against the weights' share of the answerable targets
    assert sum(m["answerable"] for m in mix.values()) == 2 and abs(sum(m["target"] for m in mix.values()) - 2) < 0.5
    assert all(m["items"] >= m["answerable"] for m in mix.values()) and sum(m["items"] for m in mix.values()) == 4
    assert sum(1 for m, t in llm.calls if t.startswith("withhold:")) == 2      # one screening call per item, every candidate passage in it


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


def test_passages_that_answer_are_never_distractors(tmp_path):
    def leaky(messages, tag):
        out = scripted(messages, tag)
        if tag.startswith("withhold:"):   # the judge finds the first passage states the answer and the second part of it
            out["results"][0].update(answers="yes", answer_if_any="states it")
            out["results"][1]["answers"] = "partly"
        return out
    interim = write_factsheet(tmp_path / "interim")
    llm = FakeLLM(leaky)
    generate_dataset(interim, tmp_path / "out", config(), llm, llm, targets={"factual": 1, "unanswerable": 1}, types=["factual", "unanswerable"])
    w = next(json.loads(l) for l in (tmp_path / "out" / "dataset.jsonl").open() if '"unanswerable"' in l)
    first, second = w["screening"][:2]
    assert (first["answers"], first["answer_if_any"], second["answers"]) == ("yes", "states it", "partly")
    flagged = {x["unit_id"] for x in w["screening"] if x["answers"] != "no"}   # the judge's two, plus any the key rule caught
    assert {first["unit_id"], second["unit_id"]} <= flagged and not flagged & {d["unit_id"] for d in w["distractors"]}
    assert w["validation"]["judge"]["results"][0]["answers"] == "yes"   # the screening call is kept for audit


def test_withhold_drops_the_evidence_its_copies_and_passages_holding_the_keys(tmp_path):
    corpus, sheet = build_sheet()
    llm = FakeLLM(scripted)   # the judge clears every passage: the key rule has to catch the copy on its own
    run = Run(corpus, sheet, config(), llm, llm, tmp_path, 1)
    pfs = corpus.units["d02-e0003"]
    hint = {"question": "What was the median PFS with capivasertib plus fulvestrant in CAPItello-291?", "answer": "7.3 months", "answer_keys": ["7.3"]}
    res = run._withhold(Candidate("unanswerable", "withheld", [pfs], hint, []))
    kept = [d.unit_id for d in res["item"]["distractors"]]
    assert res["stage"] == "accepted" and "d02-e0003" not in kept and "d03-e0004" not in kept    # the EAG's copy of the table prints 7.3
    copy = next(x for x in res["item"]["screening"] if x["unit_id"] == "d03-e0004")
    assert copy["keys_present"] and copy["answers"] == "yes"
    res = run._withhold(Candidate("unanswerable", "withheld", [pfs], {**hint, "also_in": ["d03-e0004"]}, []))
    assert not any(x["unit_id"] == "d03-e0004" for x in res["item"]["screening"])   # a known copy never enters the pool
    run.close()


def test_unanswerable_needs_clean_distractors_and_answerable_sources(tmp_path):
    def everything_answers(messages, tag):
        out = scripted(messages, tag)
        if tag.startswith("withhold:"):
            for r in out["results"]:
                r["answers"] = "yes"
        return out
    interim = write_factsheet(tmp_path / "interim")
    llm = FakeLLM(everything_answers)
    with pytest.raises(RuntimeError, match="short of target"):
        generate_dataset(interim, tmp_path / "out", config(), llm, llm, targets={"factual": 1, "unanswerable": 1}, types=["factual", "unanswerable"])
    rejects = [json.loads(l) for l in (tmp_path / "out" / "rejects.jsonl").open()]
    assert any(r["stage"] == "withhold" and r["reasons"] == ["too_few_distractors:0"] and r["distractors"] == [] for r in rejects)
    assert not any(t.endswith(":rewrite") for _, t in llm.calls)   # nothing to reword
    llm = FakeLLM(scripted)
    with pytest.raises(RuntimeError, match="derived from accepted items"):   # without answerable items in the run there is nothing to withhold
        generate_dataset(interim, tmp_path / "out2", config(), llm, llm, targets={"unanswerable": 1}, types=["unanswerable"])


def test_shortfall_raises_after_writing(tmp_path):
    interim = write_factsheet(tmp_path / "interim")
    out = tmp_path / "out"
    llm = FakeLLM(scripted)
    with pytest.raises(RuntimeError, match="short of target"):
        generate_dataset(interim, out, config(), llm, llm, targets={"factual": 50}, types=["factual"])
    report = json.loads((out / "generate_report.json").read_text())
    assert report["per_type"]["factual"]["accepted"] >= 1 and report["warnings"]
    assert sum(1 for _ in (out / "dataset.jsonl").open()) == report["items"]


def losing(prefix: str, losses: float = float("inf")):
    """The scripted judge, but the first `losses` calls whose tag starts with prefix return nothing usable."""
    lost = []

    def respond(messages, tag):
        if tag.startswith(prefix) and len(lost) < losses:
            lost.append(tag)
            raise LLMJsonError(f"{tag}: model returned no JSON object: ''")
        return scripted(messages, tag)
    return respond, lost


def test_lost_verdict_costs_the_candidate_not_the_run(tmp_path):
    interim = write_factsheet(tmp_path / "interim")
    out = tmp_path / "out"
    respond, lost = losing("judge:", losses=1)
    report = generate_dataset(interim, out, config(), FakeLLM(scripted), FakeLLM(respond), targets={"factual": 2}, types=["factual"])
    st = report["per_type"]["factual"]
    assert st["accepted"] == 2 and st["rejected_by_reason"] == {"no_verdict": 1} and st["rewrites"]["tried"] == 0   # nothing to reword
    assert len(lost) == 1 and "no_verdict" in (out / "rejects.jsonl").read_text()


def test_lost_screening_costs_the_unanswerable_candidate(tmp_path):
    interim = write_factsheet(tmp_path / "interim")
    respond, lost = losing("withhold:", losses=1)
    report = generate_dataset(interim, tmp_path / "out", config(), FakeLLM(scripted), FakeLLM(respond),
                              targets={"factual": 3, "unanswerable": 1}, types=["factual", "unanswerable"])
    st = report["per_type"]["unanswerable"]
    assert st["accepted"] == 1 and st["rejected_by_reason"] == {"no_verdict": 1} and len(lost) == 1


def test_judge_that_never_answers_stops_the_run_after_writing(tmp_path):
    interim = write_factsheet(tmp_path / "interim")
    out = tmp_path / "out"
    respond, lost = losing("judge:")
    cfg = config()
    cfg["abort_after_consecutive_failures"] = 2
    with pytest.raises(RuntimeError, match="no verdict"):
        generate_dataset(interim, out, cfg, FakeLLM(scripted), FakeLLM(respond), targets={"factual": 50}, types=["factual"])
    assert 2 <= len(lost) < 50 * cfg["attempts_per_item"]["default"]   # a wrong model or limit should not spend the whole budget
    assert json.loads((out / "generate_report.json").read_text())["per_type"]["factual"]["rejected_by_reason"]["no_verdict"] >= 2


def test_missing_parse_output_or_fact_sheet_fails_loudly(tmp_path):
    llm = FakeLLM(scripted)
    with pytest.raises(FileNotFoundError):
        generate_dataset(tmp_path / "nowhere", tmp_path / "out", config(), llm, llm)
    interim = write_interim(tmp_path / "interim")
    with pytest.raises(FileNotFoundError, match="medrageval extract"):
        generate_dataset(interim, tmp_path / "out", config(), llm, llm)


def test_the_fact_sheet_loads_under_the_schema_it_was_extracted_with(tmp_path):
    interim = write_factsheet(tmp_path / "interim")
    fcfg = facts_config()
    # no extraction report: the config's schema
    assert load_sheet(interim, config(), fcfg)[1].schema.name == "NICE single technology appraisal"
    # the report names a drafted copy: generation uses it, not the config's
    nice = resolve_path(fcfg["schema"]).read_text()
    drafted = tmp_path / "drafted.yaml"
    drafted.write_text(nice.replace("name: NICE single technology appraisal", "name: drafted family"))
    (interim / "factsheet_report.json").write_text(json.dumps({"schema": {"name": "drafted family", "path": str(drafted)}}))
    assert load_sheet(interim, config(), fcfg)[1].schema.name == "drafted family"
    # --schema overrides the report
    assert load_sheet(interim, config(), fcfg, schema_path=fcfg["schema"])[1].schema.name == "NICE single technology appraisal"
    # a recorded schema that is gone fails loudly instead of falling back to another schema
    drafted.unlink()
    with pytest.raises(SchemaError, match="pass --schema"):
        load_sheet(interim, config(), fcfg)


def bridged(alone_answers: bool):
    """Multi-hop items reaching a value through the task's bridge; given alone, the value's excerpt answers when alone_answers."""
    keys, sources = {}, {}

    def respond(messages, tag):
        user = next(m["content"] for m in messages if m["role"] == "user")
        if tag.startswith("alone:"):
            cid, uid = tag.split(":", 1)[1].rsplit(":", 1)
            if not alone_answers or uid == sources.get(cid):
                return {"answer": None, "identified_by": None, "reason": "-"}
            # the table's caption says "PIK3CA": the one word of the description the passage holds
            return {"answer": " and ".join(keys.get(cid, [])), "identified_by": "PIK3CA-tested population" if "PIK3CA" in user else "Alpelisib", "reason": "-"}
        if not tag.startswith(("multihop:")):
            return scripted(messages, tag)
        facts, exs = facts_in(user), excerpts(user)
        cid = tag.removesuffix(":rewrite")
        sources[cid] = exs[0][0]   # the bridge fact's excerpt comes first
        terms = ", ".join(must_terms(user)) or "company submission"
        if "describes the" in user:   # a described bridge: the comparator's value in a table with a row per technology
            value = re.search(r"gives a value for that \w+: (.+?)\. Ask", user).group(1)
            question = f"What total QALYs ({terms}) did the company estimate for the comparator it considers relevant only for PIK3CA-mutated disease?"
            hops = {"hop_a": "identifies the comparator", "hop_b": "gives its QALYs"}
        else:
            value = facts[1]["value"]
            question = f"For the primary endpoint of CAPItello-291 ({terms}), what result did capivasertib plus fulvestrant reach?"
            hops = {"hop_a": "names the endpoint", "hop_b": "gives the result"}
        keys[cid] = [value]
        return {"question": question, "answer": f"It was {value}.", "answer_keys": [value], **hops,
                "quotes": [{"unit_id": uid, "quote": " ".join(body.split()[:12])} for uid, body in exs]}
    return respond


@pytest.mark.parametrize("alone", [True, False])
def test_multihop_answerable_from_one_excerpt_is_rejected(tmp_path, alone):
    interim, out = write_factsheet(tmp_path / "interim"), tmp_path / "out"
    llm = FakeLLM(bridged(alone))
    if alone:
        cfg = config()
        cfg["attempts_per_item"] = {**cfg["attempts_per_item"], "multihop": 1}   # the first pair only: its passage holds a word of the description
        with pytest.raises(RuntimeError, match="short of target"):
            generate_dataset(interim, out, cfg, llm, llm, targets={"multihop": 1}, types=["multihop"])
        rejects = [json.loads(l) for l in (out / "rejects.jsonl").open()]
        assert any(r["reasons"][0].startswith("answerable_from:") and any(a["matches"] and a["grounded"] for a in r["validation"]["alone"].values()) for r in rejects)
        assert not any(t.endswith(":rewrite") for _, t in llm.calls)   # the pair's fault, not the wording's: no rewrite
    else:
        generate_dataset(interim, out, config(), llm, llm, targets={"multihop": 1}, types=["multihop"])
        items = [json.loads(l) for l in (out / "dataset.jsonl").open()]
        assert len(items) == 1 and items[0]["subtype"] == "described" and items[0]["hops"]["mode"] == "qualifier"   # a rival in the passage itself comes first
        assert items[0]["hops"]["bridge"] == "the comparator described as follows: relevant only for PIK3CA-mutated disease"
        assert items[0]["hops"]["rival"] == {"tier": "unit", "id": "d05-cost_effectiveness.total_outcomes-004", "label": "everolimus plus exemestane"}
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
    assert not run._rewritable([{"stage": "no_verdict", "reasons": ["no_verdict:x"]}])
    assert not run._rewritable([{"stage": "withhold", "reasons": ["too_few_distractors:2"]}])   # no wording to fix: the pool was dirty
    run.cfg["rewrite_rejected"] = False
    assert not run._rewritable(fixable)
    run.close()


def test_rewrite_feedback_explains_each_failure():
    corpus, sheet = build_sheet()
    c = Candidate("multihop", "bridge", [corpus.units["d02-e0040"], corpus.units["d02-e0003"]], {})
    results = [{"stage": "judge", "item": {"question": "q"}, "reasons": ["judge_failed", "judge:realistic", "answerable_from:d02-e0003"],
                "judge": {"issues": ["The table alone gives 7.3 months."]}, "alone": {"d02-e0003": {"answer": "7.3 months", "matches": True}}}]
    lines = rewrite_feedback(results, c, sheet.schema.framing)
    assert any(l.startswith('Given only excerpt d02-e0003, a reader answered "7.3 months"') for l in lines)
    assert any("clinician, health economist or committee member would ask: ask about what matters in the appraisal" in l for l in lines)
    assert "Reviewer: The table alone gives 7.3 months." in lines
    # another family's schema without framing: no appraisal wording
    assert not any("appraisal" in l or "committee" in l for l in rewrite_feedback(results, c))
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


def test_each_type_continues_the_group_rotation_from_earlier_types(tmp_path, monkeypatch):
    from collections import Counter

    from medrageval.generate import generate as gen
    seen = []
    real = gen.SELECTORS["numerical"]

    def recording(corpus, sheet, cfg, rng, taken=None):
        seen.append(Counter(taken))
        return real(corpus, sheet, cfg, rng, taken)
    monkeypatch.setitem(gen.SELECTORS, "numerical", recording)
    interim = write_factsheet(tmp_path / "interim")
    llm = FakeLLM(scripted)
    generate_dataset(interim, tmp_path / "out", config(), llm, llm, targets={"factual": 2, "numerical": 1}, types=["factual", "numerical"])
    items = [json.loads(l) for l in (tmp_path / "out" / "dataset.jsonl").open()]
    assert seen == [Counter(i["group"] for i in items if i["type"] == "factual")]   # the numerical selector started from the factual items' groups


def test_target_flag_sets_one_type():
    from medrageval.generate.generate import parse_targets
    assert parse_targets(["factual=3", "unanswerable=2"]) == {"factual": 3, "unanswerable": 2}
    assert parse_targets([]) == {}
    with pytest.raises(SystemExit):
        parse_targets(["riddle=3"])
    with pytest.raises(SystemExit):
        parse_targets(["factual=many"])
