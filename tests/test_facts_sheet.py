import copy

import pytest
from synthetic import build_sheet, config, facts_config, fact_list, schema

from medrageval.facts.factsheet import Fact, FactSheet, numbers_in, term_mentioned


def test_numbers_and_term_mentions():
    assert numbers_in("£24,567 per QALY; 0.50 (95% CI 0.38 to 0.65)") == {"24567", "0.50", "95", "0.38", "0.65"}
    assert term_mentioned("PI3K/AKT pathway-altered population", "In PI3K/AKT-altered patients of CAPItello-291 at DCO1, what ...?")
    assert term_mentioned("Scenario 1 – generalised gamma for OS", "in scenario 1, with a generalised gamma curve for OS")
    assert not term_mentioned("Scenario 1", "In scenario 2, what were the total life years?")   # identifiers are not paraphrased
    assert not term_mentioned("DCO1", "at the second data cut")
    assert term_mentioned("OS", "What was median overall survival?") and term_mentioned("PFS", "its progression-free survival")
    assert not term_mentioned("OS", "What was the dose?") and term_mentioned("events", "had a progression event")


def test_siblings_say_what_a_question_must_name():
    _, sheet = build_sheet()
    f = sheet.facts
    # the DCO1 median against the DCO2 median in the clarification response: the data cut or the document tells them apart
    assert ["DCO1", "company submission"] in sheet.must_mention(f["d02-efficacy.endpoint_result-001"])
    assert sheet.must_mention(f["d05-efficacy.endpoint_result-001"]) == [["DCO2", "clarification"]]
    assert sheet.must_mention(f["d02-efficacy.endpoint_result-003"]) == [["OS"]]
    assert sheet.must_mention(f["d03-cost_effectiveness.icer-001"]) == [["EAG"]]


def test_a_name_inside_a_siblings_name_does_not_tell_them_apart():
    """TA1063's two "Scenario 1" tables: naming "Scenario 1" names both, so only the longer one can be asked about."""
    _, sheet = build_sheet()
    plain, gamma = sheet.facts["d05-cost_effectiveness.total_outcomes-001"], sheet.facts["d05-cost_effectiveness.total_outcomes-002"]
    assert sheet.ambiguous(plain) and sheet.must_mention(plain) is None
    assert sheet.must_mention(gamma) == [["Scenario 1 – generalised gamma for OS"]]


def test_text_facts_in_one_document_complement_each_other():
    corpus, sheet = build_sheet()
    facts = fact_list(corpus)
    other = copy.deepcopy(next(x for x in facts if x.fact_id == "d02-trial_design.design-001"))
    other.fact_id, other.value = "d02-trial_design.design-002", "a registrational trial run in 19 countries"
    sheet = FactSheet(facts + [other], schema(), corpus, config(), facts_config())
    assert sheet.conflicts(other) == [] and not sheet.ambiguous(other)


def test_copies_redactions_and_respondents():
    _, sheet = build_sheet()
    company = sheet.facts["d02-cost_effectiveness.icer-001"]
    assert company.also_in == ["d03-e0003"] and sheet.facts["d02-economic_model.cost_input-001"].redacted
    assert sheet.facts["d04-stakeholder_views.view-001"].speaker == "Royal Marsden NHS Foundation Trust"
    assert not sheet.askable(sheet.facts["d04-stakeholder_views.respondent-001"])
    assert sheet.reported(sheet.facts["d02-efficacy.endpoint_result-001"])["data_cut"] == ["DCO1", "DCO2"]
    assert sheet.stats()["ambiguous"] == 1


def test_stale_or_foreign_fact_sheets_fail_loudly(tmp_path):
    corpus, _ = build_sheet()
    facts = fact_list(corpus)
    stale = copy.deepcopy(facts[0])
    stale.evidence = [{"unit_id": "d02-e0999", "quote": "x"}]
    with pytest.raises(ValueError, match="run it again"):
        FactSheet(facts + [stale], schema(), corpus, config(), facts_config())
    foreign = Fact.from_dict({**facts[0].to_dict(), "slot": "pricing.list_price"})
    with pytest.raises(ValueError, match="slots the schema does not have"):
        FactSheet([foreign], schema(), corpus, config(), facts_config())
    with pytest.raises(FileNotFoundError, match="medrageval extract"):
        FactSheet.load(tmp_path / "factsheet.jsonl", schema(), corpus, config(), facts_config())


def test_fact_round_trip():
    corpus, _ = build_sheet()
    f = fact_list(corpus)[0]
    assert Fact.from_dict(f.to_dict()) == f and "speaker" not in f.to_dict()


def test_rows_of_one_table_are_told_apart_by_their_arm():
    corpus, _ = build_sheet()
    facts = fact_list(corpus)
    capi = next(x for x in facts if x.fact_id == "d05-cost_effectiveness.total_outcomes-002")
    alpe = copy.deepcopy(capi)
    alpe.fact_id, alpe.values = "d05-cost_effectiveness.total_outcomes-003", {"alpelisib plus fulvestrant": "2.50"}
    sheet = FactSheet(facts + [alpe], schema(), corpus, config(), facts_config())
    assert ["alpelisib plus fulvestrant"] in sheet.must_mention(alpe)
