import json

import pytest
from synthetic import FACT_BUNDLE, build_corpus, config, facts_config, schema, write_interim

from medrageval.facts.extract import Batch, extract_factsheet, in_sections, plan_batches, verify
from medrageval.generate.llm import FakeLLM

CFG, FCFG, SCHEMA = config(), facts_config(), schema()


def batch(corpus, group: str, *unit_ids: str) -> Batch:
    return Batch(group, unit_ids[0][:3], 0, [corpus.units[u] for u in unit_ids])


@pytest.fixture(scope="module")
def corpus():
    return build_corpus(CFG, FACT_BUNDLE)


def check(corpus, group, uid, raw):
    return verify({"unit_id": uid, **raw}, batch(corpus, group, uid), SCHEMA.groups[group], corpus, FCFG)


def test_verbatim_fact_is_kept_with_its_qualifiers(corpus):
    fact, v = check(corpus, "efficacy", "d02-e0003", {
        "slot": "endpoint_result", "quote": "Median PFS (months) | 7.3 | 3.1",
        "qualifiers": {"trial": "CAPItello-291", "endpoint": "PFS", "colour": "blue", "arm": "x"},
        "values": {"capivasertib plus fulvestrant": "7.3 months", "placebo plus fulvestrant": "3.1 months"}})
    assert v.ok and fact["slot"] == "efficacy.endpoint_result" and fact["role"] == "company_submission"
    assert fact["qualifiers"] == {"trial": "CAPItello-291", "endpoint": "PFS", "arm": "x"}   # unknown keys dropped; the `per` key is kept
    assert fact["values"]["placebo plus fulvestrant"] == "3.1 months" and not fact["redacted"]


@pytest.mark.parametrize("raw, reason", [
    ({"slot": "endpoint_result", "quote": "Median PFS was 7.3 months", "values": {"a": "7.3 months"}}, "quote_not_found"),
    ({"slot": "endpoint_result", "quote": "Median PFS (months) | 7.3 | 3.1", "values": {"a": "7.4 months"}}, "value_not_in_quote"),
    ({"slot": "endpoint_result", "quote": "Median PFS (months) | 7.3 | 3.1", "values": {"a": "longer"}}, "value_without_number"),
    ({"slot": "endpoint_result", "quote": "Median PFS (months) | 7.3 | 3.1", "values": {"a": "[REDACTED]"}}, "redacted_value_not_in_quote"),
    ({"slot": "endpoint_result", "quote": "Median PFS (months) | 7.3 | 3.1"}, "no_value"),
    ({"slot": "median_pfs", "quote": "Median PFS (months) | 7.3 | 3.1", "value": "7.3"}, "unknown_slot"),
])
def test_unverifiable_facts_are_rejected(corpus, raw, reason):
    fact, v = check(corpus, "efficacy", "d02-e0003", raw)
    assert fact is None and v.reasons[0].startswith(reason)


def test_unit_must_be_in_the_batch(corpus):
    raw = {"slot": "endpoint_result", "unit_id": "d02-e0011", "quote": "26.0 months", "value": "26.0 months"}
    fact, v = verify(raw, batch(corpus, "efficacy", "d02-e0003"), SCHEMA.groups["efficacy"], corpus, FCFG)
    assert fact is None and v.reasons == ["unit_unknown:d02-e0011"]


def test_redacted_text_and_stance_facts(corpus):
    fact, v = check(corpus, "economic_model", "d02-e0004", {"slot": "cost_input", "quote": "The list price of capivasertib is £[REDACTED] per pack of 64 tablets",
                                                             "value": "[REDACTED]", "qualifiers": {"item": "list price"}})
    assert v.ok and fact["redacted"]
    fact, v = check(corpus, "trial_design", "d02-e0001", {"slot": "design", "quote": "CAPItello-291 was a phase 3, randomised, double-blind, placebo-controlled trial",
                                                           "value": "an open-label crossover study"})
    assert fact is None and v.reasons == ["value_not_grounded"]
    quote = "The EAG considers that the use of time to treatment discontinuation data from CAPItello-291 to model treatment duration is inappropriate"
    fact, v = check(corpus, "key_issues", "d03-e0031", {"slot": "issue", "quote": quote,
                                                         "positions": {"eag": "time to treatment discontinuation data are inappropriate", "committee": "agrees"}})
    assert v.ok and fact["positions"] == {"EAG": "time to treatment discontinuation data are inappropriate"}   # party names normalised, unknown parties dropped


def test_batches_follow_roles_sections_and_size(corpus):
    batches = plan_batches(corpus, SCHEMA, CFG, FCFG)
    roles = {b.group: {corpus.docs[b.doc_id].role for x in batches if x.group == b.group} for b in batches}
    for g, rs in roles.items():
        assert rs <= set(SCHEMA.groups[g].roles)
    eff = [u.unit_id for b in batches if b.group == "efficacy" and b.doc_id == "d02" for u in b.units]
    assert "d02-e0003" in eff and "d02-e0004" not in eff   # the results table, not the list price
    small = {**FCFG, "batch": {**FCFG["batch"], "max_batch_chars": 600}}
    assert all(sum(min(len(u.plain), 2500) + 120 for u in b.units) <= 600 or len(b.units) == 1 for b in plan_batches(corpus, SCHEMA, CFG, small))
    assert in_sections(corpus.units["d02-e0003"], ("effectiveness results",))
    assert not in_sections(corpus.units["d02-e0001"], ("effectiveness results",))   # only its own section and the parent count


def scripted_extraction(messages, tag):
    """Facts for the efficacy and cost-effectiveness batches; nothing elsewhere."""
    user = messages[-1]["content"]
    facts = []
    if tag.startswith("extract:efficacy:d02") and "[d02-e0003]" in user:
        facts = [{"slot": "endpoint_result", "unit_id": "d02-e0003", "quote": "Median PFS (months) | 7.3 | 3.1",
                  "qualifiers": {"trial": "CAPItello-291", "endpoint": "PFS"}, "values": {"capivasertib plus fulvestrant": "7.3 months", "placebo plus fulvestrant": "3.1 months"}},
                 {"slot": "endpoint_result", "unit_id": "d02-e0003", "quote": "Median PFS (months) | 7.3 | 3.1",   # the same fact twice: merged
                  "qualifiers": {"trial": "CAPItello-291", "endpoint": "PFS"}, "values": {"capivasertib plus fulvestrant": "7.3 months", "placebo plus fulvestrant": "3.1 months"}},
                 {"slot": "endpoint_result", "unit_id": "d02-e0003", "quote": "Median PFS (months) | 7.3 | 3.1", "values": {"a": "9.9 months"}}]
    if tag.startswith("extract:cost_effectiveness:d02") and "[d02-e0008]" in user:
        facts = [{"slot": "icer", "unit_id": "d02-e0008", "quote": "The company's base-case ICER for capivasertib plus fulvestrant versus everolimus plus exemestane was £24,567 per QALY gained",
                  "qualifiers": {"owner": "company", "analysis": "base case"}, "values": {"everolimus plus exemestane": "£24,567 per QALY gained"}}]
    return {"facts": facts}


def test_extraction_writes_verified_facts_with_copies(tmp_path):
    interim = write_interim(tmp_path / "interim", FACT_BUNDLE)
    llm = FakeLLM(scripted_extraction)
    report = extract_factsheet(interim, CFG, FCFG, llm, SCHEMA)
    facts = [json.loads(l) for l in (interim / "factsheet.jsonl").open()]
    assert [f["fact_id"] for f in facts] == ["d02-efficacy.endpoint_result-001", "d02-cost_effectiveness.icer-001"]
    assert set(facts[1]["also_in"]) == {"d03-e0003", "d03-e0001"}   # the EAG reprints the company's ICER, and cites it in its critique
    rejects = [json.loads(l) for l in (interim / "factsheet_rejects.jsonl").open()]
    assert [r["reasons"][0].split(":")[0] for r in rejects] == ["value_not_in_quote"]
    assert report["facts"] == 2 and report["per_group"]["efficacy"]["facts_raw"] == 3 and report["per_group"]["efficacy"]["rejected"] == {"value_not_in_quote": 1}
    assert "stakeholder_views: no facts" in report["warnings"]
    assert sum(1 for _, t in llm.calls) == report["batches"]
