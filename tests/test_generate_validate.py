from random import Random

import pytest
from synthetic import build_sheet, config

from medrageval.generate.candidates import SELECTORS, Candidate
from medrageval.generate.validate import is_duplicate, question_overlap, validate

CFG = config()
CORPUS, SHEET = build_sheet(CFG)
F = SHEET.facts


def cand(type_, *fact_ids, subtype=None, **hint):
    facts = [F[f] for f in fact_ids]
    units = list({u.unit_id: u for f in facts for u in SHEET.units(f)}.values())
    return Candidate(type_, subtype, units, {"must_mention": SHEET.must_mention(facts[0]) or [], **hint}, facts)


def selected(name, fact_id):
    return next(c for c in SELECTORS[name](CORPUS, SHEET, CFG, Random(1)) if fact_id in [f.fact_id for f in c.facts])


def item(question, answer, keys, quotes, **extra):
    return {"question": question, "answer": answer, "answer_keys": keys,
            "quotes": [{"unit_id": u, "quote": q} for u, q in quotes], **extra}


DESIGN_QUOTE = ("d02-e0001", "CAPItello-291 was a phase 3, randomised, double-blind, placebo-controlled trial of capivasertib plus fulvestrant")
GOOD_FACTUAL = item("How was the pivotal CAPItello-291 study of capivasertib with fulvestrant designed?",
                    "It was a phase 3, randomised, double-blind, placebo-controlled trial.", ["phase 3", "double-blind"], [DESIGN_QUOTE])


def test_good_factual_item_passes():
    v = validate(GOOD_FACTUAL, cand("factual", "d02-trial_design.design-001"), CORPUS, CFG, [])
    assert v.ok and v.reasons == []


@pytest.mark.parametrize("question, reason", [
    ("What design does Table 3 give for the CAPItello-291 study of capivasertib?", "table_language:Table 3"),
    ("What design is in the CAPItello-291 row for capivasertib with fulvestrant?", "table_language:row"),
    ("For Capivasertib + fulvestrant (N=155) in CAPItello-291, what was the trial design?", "table_language:(N=1"),
    ("How was the trial in this table designed for capivasertib with fulvestrant?", "deictic_question"),
])
def test_questions_about_the_layout_are_rejected(question, reason):
    v = validate({**GOOD_FACTUAL, "question": question}, cand("factual", "d02-trial_design.design-001"), CORPUS, CFG, [])
    assert reason in v.reasons


def test_question_must_name_what_sets_the_fact_apart():
    c = selected("numerical", "d05-efficacy.endpoint_result-001")   # DCO2 median, a sibling of the DCO1 median
    assert c.hint["must_mention"] == [["DCO2", "clarification"]]
    arm, value = c.hint["label"], c.hint["value"]   # the selector picks one arm of the fact
    good = item(f"At the second data cut (DCO2), what median progression-free survival did {arm} achieve in CAPItello-291?",
                f"{value}.", [value], [("d05-e0033", "median PFS was 7.2 months with capivasertib plus fulvestrant versus 3.6 months with placebo plus fulvestrant")])
    assert validate(good, c, CORPUS, CFG, []).ok
    vague = {**good, "question": f"What median progression-free survival did {arm} achieve in the CAPItello-291 trial?"}
    assert "question_lacks_discriminator:DCO2" in validate(vague, c, CORPUS, CFG, []).reasons
    by_doc = {**good, "question": f"In the company's clarification response, what median progression-free survival did {arm} achieve in CAPItello-291?"}
    assert validate(by_doc, c, CORPUS, CFG, []).ok   # naming the document is enough


def test_keys_may_take_the_facts_natural_form():
    """The table prints "121 (78.1)"; the fact says "78.1%", which is what a correct answer says."""
    c = cand("numerical", "d02-efficacy.endpoint_result-002", numbers=["78.1"], value="78.1%")
    it = item("What share of patients on capivasertib plus fulvestrant in CAPItello-291 had a progression event or died by DCO1?",
              "78.1% did.", ["78.1%"], [("d02-e0003", "Events, n (%) | 121 (78.1) | 108 (80.6)")])
    v = validate(it, c, CORPUS, CFG, [])
    assert v.ok, v.reasons
    assert "answer_key_unsupported:79.0%" in validate({**it, "answer_keys": ["79.0%"]}, c, CORPUS, CFG, []).reasons


@pytest.mark.parametrize("keys, reason", [
    (["12"], "answer_key_generic:12"),
    (["Weak"], "answer_key_generic:Weak"),
    (["a phrase that is much too long to score"], "answer_key_too_long:a phrase that is much too long to score"),
    (["[REDACTED]"], "answer_key_garbled"),
    ([], "answer_keys_missing"),
])
def test_keys_must_be_able_to_score_an_answer(keys, reason):
    v = validate({**GOOD_FACTUAL, "answer_keys": keys}, cand("factual", "d02-trial_design.design-001"), CORPUS, CFG, [])
    assert reason in v.reasons


@pytest.mark.parametrize("change, reason", [
    (lambda i: i["quotes"][0].update(quote="CAPItello-291 was an open-label trial"), "quote_not_found:d02-e0001"),
    (lambda i: i["quotes"][0].update(unit_id="d02-e0002"), "quote_unit_unknown:d02-e0002"),
    (lambda i: i.update(answer_keys=["double-blind"], question="Was CAPItello-291, the capivasertib with fulvestrant study, double-blind?"), "answer_key_in_question:double-blind"),
    (lambda i: i.update(question="Design?"), "question_length:1"),
    (lambda i: i.update(answer=""), "no_answer"),
    (lambda i: i.update(quotes=[]), "no_quotes"),
])
def test_factual_rejections(change, reason):
    it = {**GOOD_FACTUAL, "quotes": [dict(q) for q in GOOD_FACTUAL["quotes"]]}
    change(it)
    assert reason in validate(it, cand("factual", "d02-trial_design.design-001"), CORPUS, CFG, []).reasons


def test_question_must_not_echo_the_quote_or_repeat_another():
    quote = "The company's base-case ICER for capivasertib plus fulvestrant versus everolimus plus exemestane was £24,567 per QALY gained, with incremental QALYs"
    share, shared = question_overlap("What was the company's base-case ICER for capivasertib plus fulvestrant versus everolimus plus exemestane, with incremental QALYs gained?", quote)
    assert share > CFG["validate"]["max_question_overlap"] and shared >= CFG["validate"]["echo_min_shared"]
    assert is_duplicate("How was the CAPItello-291 study designed?", ["How was the CAPItello-291 study designed?"], 0.7)
    assert "duplicate_question" in validate(GOOD_FACTUAL, cand("factual", "d02-trial_design.design-001"), CORPUS, CFG, [GOOD_FACTUAL["question"]]).reasons


def comparative_item(**over):
    base = item("In CAPItello-291, did capivasertib plus fulvestrant or placebo plus fulvestrant give the longer median overall survival?",
                "Capivasertib plus fulvestrant was longer, at 26.0 months against 22.3 months.", ["26.0 months", "22.3 months"],
                [("d02-e0011", "median overall survival was 26.0 months with capivasertib plus fulvestrant versus 22.3 months with placebo plus fulvestrant")],
                measure="median OS", label_a="capivasertib plus fulvestrant", value_a="26.0 months", label_b="placebo plus fulvestrant", value_b="22.3 months", higher="a")
    return {**base, **over}


def test_comparative_checks_values_direction_and_difference():
    c = selected("comparative", "d02-efficacy.endpoint_result-003")
    assert validate(comparative_item(), c, CORPUS, CFG, []).ok
    assert any(r.startswith("direction_inconsistent") for r in validate(comparative_item(higher="b"), c, CORPUS, CFG, []).reasons)
    tie = validate(comparative_item(value_a="26.0 months", value_b="26.0 months", higher="equal"), c, CORPUS, CFG, []).reasons
    assert any(r.startswith("comparison_not_meaningful") for r in tie)
    assert "comparative_values_not_in_facts" in validate(comparative_item(value_b="21.0 months"), c, CORPUS, CFG, []).reasons


def test_multihop_must_not_name_the_bridge():
    c = selected("multihop", "d02-trial_design.primary_endpoint-001")
    quotes = [(c.units[0].unit_id, "The primary endpoint of CAPItello-291 was progression-free survival (PFS) by investigator assessment"),
              (c.units[1].unit_id, next(e["quote"] for e in c.facts[1].evidence))]
    names = " and ".join(s[0] for s in c.hint["must_mention"])
    value = next(iter(c.facts[1].values.values()))
    good = item(f"For the primary endpoint of CAPItello-291, reported as {names}, what result did capivasertib plus fulvestrant reach?",
                f"{value}.", [value], quotes, hop_a="names the endpoint", hop_b="gives the result")
    v = validate(good, c, CORPUS, CFG, [])
    assert v.ok, v.reasons
    named = {**good, "question": f"What progression-free survival result ({names}) did capivasertib plus fulvestrant reach in CAPItello-291?"}
    assert "bridge_named_in_question:endpoint" in validate(named, c, CORPUS, CFG, []).reasons
    one = {**good, "quotes": good["quotes"][1:]}
    assert "quotes_from_one_unit" in validate(one, c, CORPUS, CFG, []).reasons


def test_described_multihop_must_not_name_the_comparator():
    c = selected("multihop", "d02-decision_problem.comparator_choice-002")
    quotes = [("d02-e0041", "alpelisib plus fulvestrant is relevant only for PIK3CA-mutated disease"), ("d05-e0034", "Alpelisib + fulvestrant | 51,365 | 2.41 | 1.78")]
    good = item("In the company's updated base case, what total QALYs were estimated for the comparator it considers relevant only for PIK3CA-mutated disease?",
                "1.78 QALYs.", ["1.78 QALYs"], quotes, hop_a="identifies the comparator", hop_b="gives its QALYs")
    v = validate(good, c, CORPUS, CFG, [])
    assert v.ok, v.reasons
    named = {**good, "question": "In the company's updated base case, what total QALYs were estimated for alpelisib, the comparator relevant only for PIK3CA-mutated disease?"}
    assert "bridge_named_in_question:technology" in validate(named, c, CORPUS, CFG, []).reasons
    # "fulvestrant" is in the bundle's topic and in every regimen: saying it names nothing
    ful = {**good, "question": "In the company's updated base case, what total QALYs were estimated for the fulvestrant-based comparator it considers relevant only for PIK3CA-mutated disease?"}
    assert validate(ful, c, CORPUS, CFG, []).ok


def test_crossdoc_needs_both_documents():
    c = selected("crossdoc", "d02-cost_effectiveness.icer-001")
    quotes = [("d02-e0008", "The company's base-case ICER for capivasertib plus fulvestrant versus everolimus plus exemestane was £24,567 per QALY gained"),
              ("d03-e0030", "The EAG's base-case ICER for capivasertib plus fulvestrant versus everolimus plus exemestane is £31,200 per QALY gained")]
    it = item("How does the EAG's base-case cost-effectiveness estimate against everolimus plus exemestane differ from the one in the company submission?",
              "The company estimated £24,567 per QALY gained; the EAG estimated £31,200 per QALY gained.", ["£24,567", "£31,200"], quotes)
    assert validate(it, c, CORPUS, CFG, []).ok
    assert "quotes_from_one_document" in validate({**it, "quotes": it["quotes"][:1]}, c, CORPUS, CFG, []).reasons


def test_attribution_answer_is_the_named_respondent():
    c = selected("attribution", "d04-stakeholder_views.view-001")
    quote = "capivasertib plus fulvestrant would be used after an aromatase inhibitor in patients whose disease has a PI3K/AKT pathway alteration"
    it = item("Who expected capivasertib with fulvestrant to be used after an aromatase inhibitor once a PI3K/AKT pathway alteration is found?",
              "Royal Marsden NHS Foundation Trust", ["Royal Marsden NHS Foundation Trust"], [(c.units[0].unit_id, quote)])
    v = validate(it, c, CORPUS, CFG, [])
    assert v.checks["answer_is_speaker"] and v.checks["speaker_not_in_question"]
    assert validate({**it, "answer": "the clinical experts"}, c, CORPUS, CFG, []).checks["answer_is_speaker"]
    assert not validate({**it, "answer": "the company"}, c, CORPUS, CFG, []).checks["answer_is_speaker"]
    assert "speaker_named_in_question" in validate({**it, "question": "Which clinical experts expected capivasertib to be used after an aromatase inhibitor?"}, c, CORPUS, CFG, []).reasons


def test_summary_keypoints_cite_and_state_their_facts():
    c = selected("summary", "d02-efficacy.endpoint_result-001")
    facts = c.facts
    quotes = [{"unit_id": e["unit_id"], "quote": e["quote"]} for f in facts for e in f.evidence]
    kps = [{"point": f"{f.value_text()} for {SHEET.describe(f)}", "fact_id": f.fact_id} for f in facts]
    it = {"question": "What efficacy results did the company submission report for capivasertib plus fulvestrant in CAPItello-291?",
          "answer": " ".join(k["point"] + "." for k in kps), "answer_keys": [], "keypoints": kps, "quotes": quotes}
    v = validate(it, c, CORPUS, CFG, [])
    assert v.ok, v.reasons
    bad = {**it, "keypoints": kps[:1] + [{"point": "Median PFS was 9.9 months with capivasertib", "fact_id": facts[1].fact_id},
                                       {"point": "x", "fact_id": "d99-nothing-001"}]}
    reasons = validate(bad, c, CORPUS, CFG, []).reasons
    assert f"keypoint_unsupported:{facts[1].fact_id}" in reasons and "keypoint_unknown_fact:d99-nothing-001" in reasons


def test_unanswerable_redacted_requires_marker_and_no_answer():
    c = selected("unanswerable_redacted", "d02-economic_model.cost_input-001")
    good = item("What is the list price per pack of capivasertib tablets?", None, [],
                [("d02-e0004", "The list price of capivasertib is £[REDACTED] per pack of 64 tablets")])
    assert validate(good, c, CORPUS, CFG, []).ok
    assert "unanswerable_has_answer" in validate({**good, "answer": "£5,000"}, c, CORPUS, CFG, []).reasons
    no_marker = {**good, "quotes": [{"unit_id": "d02-e0004", "quote": "per pack of 64 tablets (excluding VAT), and the company estimates"}]}
    assert "quote_lacks_redacted_marker" in validate(no_marker, c, CORPUS, CFG, []).reasons


@pytest.mark.parametrize("question, rejected", [
    ("What assumption did the EAG make about treatment waning in CAPItello-291, and how did it affect the ICER?", True),
    ("Which tool did the company use for the severity modifier and what estimate did it give for capivasertib?", True),
    ("How was the pivotal CAPItello-291 study of capivasertib with fulvestrant designed?", False),
])
def test_one_question_asks_one_thing(question, rejected):
    v = validate({**GOOD_FACTUAL, "question": question}, cand("factual", "d02-trial_design.design-001"), CORPUS, CFG, [])
    assert any(r.startswith("compound_question") for r in v.reasons) == rejected


def test_keys_are_values_not_labelled_values():
    v = validate({**GOOD_FACTUAL, "answer_keys": ["Placebo plus fulvestrant: 0"]}, cand("factual", "d02-trial_design.design-001"), CORPUS, CFG, [])
    assert "answer_key_has_label:Placebo plus fulvestrant: 0" in v.reasons


def test_small_counts_are_not_compared():
    from medrageval.generate.candidates import small_count, too_small
    assert small_count("1 (1%)", 5) and small_count("0", 5) and too_small("1 (1%)", "0", CFG)
    assert not small_count("3.1 months", 5) and not small_count("3%", 5) and not small_count("121 (78.1)", 5)
    c = selected("comparative", "d02-efficacy.endpoint_result-003")
    tiny = comparative_item(value_a="3 (4%)", value_b="0", answer="Capivasertib plus fulvestrant was higher, 3 (4%) against 0.",
                            quotes=[{"unit_id": "d02-e0011", "quote": "median overall survival was 26.0 months with capivasertib plus fulvestrant versus 22.3 months with placebo plus fulvestrant"}])
    assert any(r.startswith("comparison_too_small") for r in validate(tiny, c, CORPUS, CFG, []).reasons)
