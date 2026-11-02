from collections import Counter
from random import Random

import pytest
from synthetic import build_sheet, config

from medrageval.generate.candidates import SELECTORS, candidate_counts, labels_match, link_match, meaningful, weighted

CFG = config()


@pytest.fixture(scope="module")
def sheet():
    return build_sheet(CFG)


def cands(sheet, name):
    corpus, s = sheet
    return list(SELECTORS[name](corpus, s, CFG, Random(1)))


def ids(cs):
    return [tuple(f.fact_id for f in c.facts) for c in cs]


def test_factual_uses_askable_text_facts(sheet):
    got = {c.facts[0].fact_id for c in cands(sheet, "factual")}
    assert "d02-trial_design.design-001" in got and "d04-stakeholder_views.view-001" in got
    assert "d04-stakeholder_views.respondent-001" not in got   # metadata, never asked about


def test_numerical_skips_redacted_and_indistinguishable_facts(sheet):
    corpus, s = sheet
    uncapped = {**CFG, "limits": {**CFG["limits"], "max_per_section": 10}}   # the seeded per-section cap is not under test here
    cs = list(SELECTORS["numerical"](corpus, s, uncapped, Random(1)))
    got = {c.facts[0].fact_id for c in cs}
    assert "d05-cost_effectiveness.total_outcomes-001" not in got   # plain "Scenario 1": two right answers
    assert "d05-cost_effectiveness.total_outcomes-002" in got and "d02-economic_model.cost_input-001" not in got
    hr = next(c for c in cs if c.facts[0].fact_id == "d02-efficacy.treatment_effect-001")
    assert hr.hint["value"] == "0.50 (95% CI 0.38 to 0.65)" and "0.50" in hr.hint["numbers"]
    gamma = next(c for c in cs if c.facts[0].fact_id == "d05-cost_effectiveness.total_outcomes-002")
    assert gamma.hint["must_mention"] == [["Scenario 1 – generalised gamma for OS"]]


def test_comparisons_need_a_difference_worth_asking_about(sheet):
    got = {c.facts[0].fact_id: c.hint for c in cands(sheet, "comparative")}
    assert "d02-efficacy.endpoint_result-001" in got and "d02-efficacy.endpoint_result-003" in got   # 7.3 vs 3.1; 26.0 vs 22.3
    assert "d02-efficacy.endpoint_result-002" not in got                                             # 78.1% vs 80.6%: 3% apart
    assert sorted(got["d02-efficacy.endpoint_result-001"]["values"]) == ["3.1 months", "7.3 months"]
    assert meaningful(3.25, 3.26, 0.1) is False and meaningful(7.3, 3.1, 0.1) and not meaningful(None, 1, 0.1)


def test_multihop_follows_schema_links_and_hides_the_bridge(sheet):
    cs = cands(sheet, "multihop")
    pairs = ids(cs)
    assert ("d02-trial_design.primary_endpoint-001", "d02-efficacy.endpoint_result-001") in pairs
    assert not any(p[0] == "d03-economic_model.extrapolation-001" for p in pairs)   # the EAG's curve is not the company's scenario
    c = next(c for c in cs if c.facts[1].fact_id == "d02-efficacy.endpoint_result-001")
    assert c.subtype == "bridge" and c.hint["bridge"] == "the primary endpoint of CAPItello-291" and c.hint["via_value"] == "PFS"
    assert c.hint["value"] == "capivasertib plus fulvestrant: 7.3 months; placebo plus fulvestrant: 3.1 months"
    assert c.hint["rival"] == {"tier": "section", "id": "d02-efficacy.endpoint_result-003", "label": "OS"}   # the OS median, in the same section
    assert all("PFS" not in s for s in c.hint["must_mention"])   # named through the bridge, so not required
    assert [u.unit_id for u in c.units] == ["d02-e0040", "d02-e0003"]
    assert link_match("generalised gamma distribution", "Scenario 1 – generalised gamma for OS") and not labels_match("generalised gamma distribution", "Scenario 1 – generalised gamma for OS")
    # the EAG's copy of the PFS table has the same rival in no shared section, and the DCO2 table has none: no candidates
    assert not any(p[1] in ("d03-efficacy.endpoint_result-001", "d05-efficacy.endpoint_result-001") for p in pairs)


def test_multihop_describes_a_thing_through_a_qualifier_link(sheet):
    cs = cands(sheet, "multihop")
    described = [c for c in cs if c.subtype == "described"]
    assert {tuple(f.fact_id for f in c.facts) for c in described} == {
        ("d02-decision_problem.comparator_choice-001", "d05-cost_effectiveness.total_outcomes-004"),
        ("d02-decision_problem.comparator_choice-002", "d05-cost_effectiveness.total_outcomes-003")}
    c = next(c for c in described if c.facts[0].fact_id == "d02-decision_problem.comparator_choice-002")
    assert c.hint["mode"] == "qualifier" and c.hint["via"] == "technology" and c.hint["via_value"] == "alpelisib plus fulvestrant"
    assert c.hint["value"] == "1.78 QALYs" and c.hint["bridge"] == "the comparator described as follows: relevant only for PIK3CA-mutated disease"
    assert c.hint["rival"]["tier"] == "unit" and c.hint["rival"]["label"] == "everolimus plus exemestane"   # the table's other row
    assert all("alpelisib" not in t.lower() for s in c.hint["must_mention"] for t in s)
    assert cs.index(c) < cs.index(next(c for c in cs if c.subtype == "bridge"))   # a rival in the passage itself comes first
    # the company's ICER is given for one comparator only: nothing in that passage to tell apart
    assert not any(c.facts[1].slot == "cost_effectiveness.icer" for c in cs)


def test_crossdoc_pairs_two_parties_on_one_subject(sheet):
    cs = cands(sheet, "crossdoc")
    assert ids(cs) == [("d02-cost_effectiveness.icer-001", "d03-cost_effectiveness.icer-001")] or \
        ids(cs) == [("d03-cost_effectiveness.icer-001", "d02-cost_effectiveness.icer-001")]
    assert set(cs[0].hint["docs"]) == {"company submission", "EAG"}
    # the DCO1 and DCO2 medians are in two documents too, but they are two data cuts, not two accounts


def test_attribution_names_the_respondent(sheet):
    cs = cands(sheet, "attribution")
    by_fact = {c.facts[0].fact_id: c for c in cs}
    stance = by_fact["d03-key_issues.issue-001"]
    assert stance.subtype == "stance" and stance.hint["speaker"] == "the EAG"
    form = by_fact["d04-stakeholder_views.view-001"]
    assert form.subtype == "form" and form.hint["speaker"] == "the clinical experts" and form.hint["speaker_name"] == "Royal Marsden NHS Foundation Trust"


def test_summary_groups_facts_of_one_document(sheet):
    cs = cands(sheet, "summary")
    eff = next(c for c in cs if c.hint["group"] == "efficacy")
    assert {f.doc_id for f in eff.facts} == {"d02"} and 3 <= len(eff.facts) <= CFG["summary"]["max_facts"]
    assert eff.hint["doc"] == "Company evidence submission"


def test_unanswerable_selectors(sheet):
    red = cands(sheet, "unanswerable_redacted")
    assert ids(red) == [("d02-economic_model.cost_input-001",)] and "list price" in red[0].hint["labels"][0]
    nc = cands(sheet, "unanswerable_not_covered")
    assert nc and all(c.facts[0].kind == "number" and len(c.units) == 1 for c in nc)
    assert len({c.units[0].section_id for c in nc if c.units[0].section_id == "d02-s002"}) <= CFG["limits"]["max_per_section"]
    pfs = next(c for c in nc if c.facts[0].slot == "efficacy.endpoint_result")   # what the bundle reports, for the near miss to avoid
    assert pfs.hint["reported"]["data_cut"] == ["DCO1", "DCO2"] and set(pfs.hint["reported"]["endpoint"]) == {"PFS", "OS"}


def test_weighted_order_follows_group_weights():
    items = [("a", i) for i in range(30)] + [("b", i) for i in range(30)] + [("c", i) for i in range(2)]
    order = list(weighted(items, lambda x: x[0], {"a": 0.6, "b": 0.3, "c": 0.1, }, Random(0), lambda x: x, None))
    first = Counter(g for g, _ in order[:10])
    assert first["a"] == 6 and first["b"] == 3 and first["c"] == 1
    assert len(order) == 62   # groups that run out drop out; nothing is lost
    assert list(weighted(items, lambda x: x[0], {"a": 1, "b": 0}, Random(0), lambda x: x, None)) == \
        list(weighted([i for i in items if i[0] == "a"], lambda x: x[0], {"a": 1}, Random(0), lambda x: x, None))


def test_selection_is_reproducible_and_counted(sheet):
    corpus, s = sheet
    assert ids(cands(sheet, "factual")) == ids(cands(sheet, "factual"))
    counts = candidate_counts(corpus, s, CFG, 1)
    assert set(counts) == set(SELECTORS) and counts["crossdoc"]["total"] == 1


def test_identifiers_must_agree_across_a_bridge():
    assert link_match("investigator-assessed progression-free survival (PFS)", "PFS")
    assert not link_match("investigator-assessed progression-free survival (PFS)", "PFS2 (second progression-free survival)")


def test_comparisons_only_across_comparable_qualifiers(sheet):
    import copy

    from medrageval.facts.factsheet import FactSheet
    from synthetic import facts_config, fact_list, schema
    corpus, _ = sheet
    facts = fact_list(corpus)
    hr = next(f for f in facts if f.fact_id == "d02-efficacy.treatment_effect-001")
    by_pop, by_endpoint = copy.deepcopy(hr), copy.deepcopy(hr)
    by_pop.fact_id, by_pop.value, by_pop.qualifiers["population"] = "d02-efficacy.treatment_effect-002", "0.79 (95% CI 0.61 to 1.02)", "overall population"
    by_endpoint.fact_id, by_endpoint.value, by_endpoint.qualifiers["endpoint"] = "d02-efficacy.treatment_effect-003", "0.30 (95% CI 0.2 to 0.5)", "ORR"
    s = FactSheet(facts + [by_pop, by_endpoint], schema(), corpus, CFG, facts_config())
    pairs = {tuple(sorted(f.fact_id for f in c.facts)) for c in SELECTORS["comparative"](corpus, s, CFG, Random(1)) if c.subtype == "facts"}
    assert ("d02-efficacy.treatment_effect-001", "d02-efficacy.treatment_effect-002") in pairs        # one HR in two populations
    assert ("d02-efficacy.treatment_effect-001", "d02-efficacy.treatment_effect-003") not in pairs    # an HR for PFS against one for ORR
