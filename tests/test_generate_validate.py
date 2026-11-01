import pytest
from synthetic import DOCUMENTS, ELEMENTS, EXTRA_ELEMENTS, MORE_DOCUMENTS, MORE_ELEMENTS, build_corpus, config

from medrageval.generate.candidates import Candidate
from medrageval.generate.validate import is_duplicate, validate

CFG = config()
CORPUS = build_corpus(CFG)
U = CORPUS.units


def cand(type_, *unit_ids, subtype=None, **hint):
    return Candidate(type_, subtype, [U[u] for u in unit_ids], hint)


def item(question, answer, keys, quotes, **extra):
    return {"question": question, "answer": answer, "answer_keys": keys,
            "quotes": [{"unit_id": u, "quote": q} for u, q in quotes], **extra}


GOOD_FACTUAL = item("What kind of trial was CAPItello-291 and which patients did it enrol?",
                    "A phase 3, randomised, double-blind, placebo-controlled trial of capivasertib plus fulvestrant.",
                    ["phase 3", "double-blind"],
                    [("d02-e0001", "CAPItello-291 was a phase 3, randomised, double-blind, placebo-controlled trial of capivasertib plus fulvestrant")])


def test_good_factual_item_passes():
    v = validate(GOOD_FACTUAL, cand("factual", "d02-e0001"), CORPUS, CFG, [])
    assert v.ok and v.reasons == []


def test_question_must_not_echo_the_quote():
    echo = {**GOOD_FACTUAL, "question": "Was CAPItello-291 a phase 3, randomised, double-blind, placebo-controlled trial of capivasertib plus fulvestrant in adults?"}
    v = validate(echo, cand("factual", "d02-e0001"), CORPUS, CFG, [])
    assert any(r.startswith("question_echoes_quote:") for r in v.reasons)
    assert validate(GOOD_FACTUAL, cand("factual", "d02-e0001"), CORPUS, CFG, []).checks["question_not_echo"]
    # attribution has to state the claim, so it is exempt
    c = cand("attribution", "d03-e0001", subtype="stance", speaker="the EAG", claim="The EAG considers that the company's base-case ICER underestimates uncertainty")
    it = item("Who considers that the company's base-case ICER of £24,567 per QALY gained underestimates uncertainty because the proportional hazards assumption was not met?",
              "the EAG", ["EAG"], [("d03-e0001", "The EAG considers that the company's base-case ICER of £24,567 per QALY gained")])
    assert "question_not_echo" not in validate(it, c, CORPUS, CFG, []).checks


def test_answer_keys_are_short():
    long = {**GOOD_FACTUAL, "answer_keys": ["phase 3", "randomised, double-blind, placebo-controlled trial of capivasertib"]}
    assert any(r.startswith("answer_key_too_long:") for r in validate(long, cand("factual", "d02-e0001"), CORPUS, CFG, []).reasons)


def test_garbled_answer_key_rejected():
    v = validate({**GOOD_FACTUAL, "answer_keys": ["phase 3", "yéárs"]}, cand("factual", "d02-e0001"), CORPUS, CFG, [])
    assert "answer_key_garbled" in v.reasons


def test_twin_table_question_must_say_which_table():
    corpus = build_corpus(CFG, ELEMENTS + EXTRA_ELEMENTS)
    c = Candidate("numerical", "table", [corpus.units["d02-e0021"]], {"numbers": ["608.6"], "must_mention": ["fulvestrant arm"]})
    it = item("What AIC did the exponential model have for time to discontinuation of capivasertib in CAPItello-291?", "608.6", ["608.6"], [("d02-e0021", "Exponential | 608.6 | 611.3")])
    assert "question_lacks_discriminator:fulvestrant arm" in validate(it, c, corpus, CFG, []).reasons
    it["question"] = "What AIC did the exponential model have for time to discontinuation in the fulvestrant arm of CAPItello-291?"
    assert validate(it, c, corpus, CFG, []).ok


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"quotes": [{"unit_id": "d02-e0001", "quote": "a phase 2 randomised trial of capivasertib"}]}, "quote_not_found:d02-e0001"),
        ({"quotes": [{"unit_id": "d02-e0002", "quote": "a phase 3, randomised, double-blind"}]}, "quote_unit_unknown:d02-e0002"),
        ({"answer_keys": ["open-label"]}, "answer_key_not_in_quotes:open-label"),
        ({"question": "Was CAPItello-291 a double-blind trial of capivasertib plus fulvestrant in advanced cancer?"}, "answer_key_in_question:double-blind"),
        ({"question": "What kind of trial is described in this excerpt about capivasertib plus fulvestrant?"}, "deictic_question"),
        ({"question": "Which trial?"}, "question_length:2"),
        ({"answer": ""}, "no_answer"),
        ({"quotes": []}, "no_quotes"),
    ],
)
def test_factual_rejections(change, reason):
    v = validate({**GOOD_FACTUAL, **change}, cand("factual", "d02-e0001"), CORPUS, CFG, [])
    assert not v.ok and reason in v.reasons


def test_duplicate_questions_rejected():
    seen = ["What kind of trial was CAPItello-291 and which patients did it enrol?"]
    assert is_duplicate("What kind of trial was CAPItello-291, and which patients did it enrol?", seen, CFG["dedupe_jaccard"])
    assert not is_duplicate("What was the list price of capivasertib per pack?", seen, CFG["dedupe_jaccard"])
    v = validate(GOOD_FACTUAL, cand("factual", "d02-e0001"), CORPUS, CFG, seen)
    assert "duplicate_question" in v.reasons


def test_numerical_needs_a_number_from_the_evidence():
    c = cand("numerical", "d02-e0008", subtype="prose", numbers=["24567", "0.61", "14986"])
    quote = "The company's base-case ICER for capivasertib plus fulvestrant versus everolimus plus exemestane was £24,567 per QALY gained"
    ok = item("What was the company's base-case ICER for capivasertib plus fulvestrant versus everolimus plus exemestane?",
              "£24,567 per QALY gained.", ["£24,567"], [("d02-e0008", quote)])
    assert validate(ok, c, CORPUS, CFG, []).ok
    bad = {**ok, "answer_keys": ["per QALY gained"]}
    assert "answer_key_without_number" in validate(bad, c, CORPUS, CFG, []).reasons
    wrong = {**ok, "answer_keys": ["24,567"], "quotes": [{"unit_id": "d02-e0008", "quote": "incremental QALYs of 0.61 and incremental costs of £14,986"}]}
    assert "answer_key_not_in_quotes:24,567" in validate(wrong, c, CORPUS, CFG, []).reasons


COMP = cand("comparative", "d02-e0003", subtype="table", mode="columns",
            labels=["Capivasertib + fulvestrant (N=155)", "Placebo + fulvestrant (N=134)"],
            header="Progression or death | Capivasertib + fulvestrant (N=155) | Placebo + fulvestrant (N=134)", rows=["Median PFS (months) | 7.3 | 3.1"])
GOOD_COMP = item("In the PI3K/AKT pathway-altered population of CAPItello-291, was median PFS longer with capivasertib plus fulvestrant or with placebo plus fulvestrant?",
                 "Median PFS was longer with capivasertib plus fulvestrant: 7.3 months versus 3.1 months.", ["7.3", "3.1"],
                 [("d02-e0003", "Progression or death | Capivasertib + fulvestrant (N=155) | Placebo + fulvestrant (N=134)"), ("d02-e0003", "Median PFS (months) | 7.3 | 3.1")],
                 measure="Median PFS (months)", label_a="Capivasertib + fulvestrant (N=155)", value_a="7.3", label_b="Placebo + fulvestrant (N=134)", value_b="3.1", higher="a")


def test_comparative_passes_and_catches_direction_and_wording():
    assert validate(GOOD_COMP, COMP, CORPUS, CFG, []).ok
    assert "direction_inconsistent:b!=a" in validate({**GOOD_COMP, "higher": "b"}, COMP, CORPUS, CFG, []).reasons
    flat = {**GOOD_COMP, "answer": "Capivasertib plus fulvestrant: 7.3 months; placebo plus fulvestrant: 3.1 months."}
    assert "answer_lacks_direction" in validate(flat, COMP, CORPUS, CFG, []).reasons
    missing = {**GOOD_COMP, "value_b": "3.2"}
    assert "comparative_values_not_in_quotes" in validate(missing, COMP, CORPUS, CFG, []).reasons


def test_multihop_question_must_draw_on_both_excerpts():
    c = cand("multihop", "d02-e0002", "d02-e0003", subtype="xref", link="Table 7")
    quotes = [("d02-e0002", "are presented in Table 7, with a 50% reduction in the risk of progression or death"),
              ("d02-e0003", "Median PFS (months) | 7.3 | 3.1")]
    good = item("For the PI3K/AKT pathway-altered population in which capivasertib halved the risk of progression, what was the median PFS in months on placebo plus fulvestrant?",
                "3.1 months", ["3.1"], quotes)
    assert validate(good, c, CORPUS, CFG, []).ok
    one = item("What was the median PFS in months with placebo plus fulvestrant in the pathway-altered population?", "3.1 months", ["3.1"], quotes[1:])
    assert "quotes_from_one_unit" in validate(one, c, CORPUS, CFG, []).reasons
    named = {**good, "question": "In Table 7, for the population in which capivasertib halved the risk of progression, what was the median PFS on placebo plus fulvestrant?"}
    assert "table_named_in_question:Table 7" in validate(named, c, CORPUS, CFG, []).reasons
    e = cand("multihop", "d02-e0001", "d02-e0011", subtype="entity", link="CAPItello-291")
    bridged = item("In CAPItello-291, what was the median overall survival with capivasertib plus fulvestrant?", "26.0 months", ["26.0"],
                   [("d02-e0001", "a phase 3, randomised, double-blind, placebo-controlled trial"), ("d02-e0011", "median overall survival was 26.0 months")])
    assert "bridge_named_in_question:CAPItello-291" in validate(bridged, e, CORPUS, CFG, []).reasons
    positional = {**bridged, "question": "For the second trial named in the company's survival modelling, what was the median overall survival with capivasertib plus fulvestrant?"}
    assert "bridge_identified_by_position" in validate(positional, e, CORPUS, CFG, []).reasons


def test_crossdoc_needs_two_documents_and_no_copies():
    c = cand("crossdoc", "d03-e0001", "d02-e0008", subtype="numbers", link="24567")
    good = item("What did the EAG conclude about the company's base-case ICER of £24,567 per QALY for capivasertib plus fulvestrant?",
                "That it underestimates uncertainty because proportional hazards was not met.", ["underestimates uncertainty"],
                [("d03-e0001", "underestimates uncertainty, because the proportional hazards assumption was not met"),
                 ("d02-e0008", "base-case ICER for capivasertib plus fulvestrant versus everolimus plus exemestane was £24,567 per QALY gained")])
    assert validate(good, c, CORPUS, CFG, []).ok
    one_doc = {**good, "quotes": [good["quotes"][0]]}
    assert "quotes_from_one_document" in validate(one_doc, c, CORPUS, CFG, []).reasons


def test_attribution_speaker_rules_and_exclusivity():
    c = cand("attribution", "d03-e0001", subtype="stance", speaker="the EAG", claim="underestimates uncertainty")
    quote = "The EAG considers that the company's base-case ICER of £24,567 per QALY gained for capivasertib plus fulvestrant versus everolimus plus exemestane underestimates uncertainty"
    good = item("Who considered that the base-case ICER of £24,567 per QALY for capivasertib plus fulvestrant understates the uncertainty in the analysis?",
                "the EAG", ["EAG"], [("d03-e0001", quote)])
    v = validate(good, c, CORPUS, CFG, [])
    assert v.ok, v.reasons
    assert "speaker_named_in_question" in validate({**good, "question": "Who, according to the EAG report, found the base-case ICER of £24,567 per QALY uncertain?"}, c, CORPUS, CFG, []).reasons
    assert "answer_not_speaker:the company" in validate({**good, "answer": "the company"}, c, CORPUS, CFG, []).reasons
    # the same claim, unattributed, in the company's own document: not exclusive
    restated = item("Who concluded that the proportional hazards assumption not being met underestimates uncertainty in the base-case ICER?", "the EAG", ["EAG"],
                    [("d03-e0001", "The EAG considers that the company's base-case ICER of £24,567 per QALY gained for capivasertib plus fulvestrant versus everolimus plus exemestane underestimates uncertainty, because the proportional hazards assumption was not met")])
    cfg = config()
    cfg["attribution"]["restated_containment"] = 0.4
    v = validate(restated, c, CORPUS, cfg, [])
    assert any(r.startswith("claim_restated_in:d02-e0009") for r in v.reasons), v.reasons


def test_unanswerable_redacted_requires_marker_and_no_answer():
    c = cand("unanswerable", "d02-e0004", subtype="redacted", labels=["£"])
    good = item("What is the list price of capivasertib per pack of 64 tablets, excluding VAT?", None, [],
                [("d02-e0004", "The list price of capivasertib is £[REDACTED] per pack of 64 tablets (excluding VAT)")])
    assert validate(good, c, CORPUS, CFG, []).ok
    assert "quote_lacks_redacted_marker" in validate({**good, "quotes": [{"unit_id": "d02-e0004", "quote": "per pack of 64 tablets (excluding VAT)"}]}, c, CORPUS, CFG, []).reasons
    assert "unanswerable_has_answer" in validate({**good, "answer": "£1,200"}, c, CORPUS, CFG, []).reasons


def test_multihop_comparison_takes_one_value_from_each_table():
    more = build_corpus(CFG, ELEMENTS + MORE_ELEMENTS, DOCUMENTS + MORE_DOCUMENTS)
    c = Candidate("multihop", "comparison", [more.units["d02-e0003"], more.units["d05-e0020"]],
                  {"versions": ["company submission", "clarification"], "values": ["7.3", "7.2"], "must_mention": ["company submission", "clarification"]})
    good = item("Was median PFS with capivasertib plus fulvestrant longer in the company submission or in the clarification response's later data cut?",
                "The company submission's 7.3 months was higher than the clarification response's 7.2 months.", ["7.3", "7.2"],
                [("d02-e0003", "Median PFS (months) | 7.3 | 3.1"), ("d05-e0020", "Median PFS (months) | 7.2 | 3.6")],
                measure="Median PFS", label_a="company submission", value_a="7.3", label_b="clarification", value_b="7.2", higher="a")
    v = validate(good, c, more, CFG, [])
    assert v.ok, v.reasons
    one_table = {**good, "answer": "The company submission's 7.3 months was higher than 3.1 months.", "answer_keys": ["7.3", "3.1"], "value_b": "3.1",
                 "quotes": [{"unit_id": "d02-e0003", "quote": "Median PFS (months) | 7.3 | 3.1"}, {"unit_id": "d05-e0020", "quote": "Events, n (%) | 121 (78.1) | 108 (80.6)"}]}
    assert "comparison_values_not_one_per_table" in validate(one_table, c, more, CFG, []).reasons
    unnamed = {**good, "question": "Was median PFS with capivasertib plus fulvestrant longer at the first data cut or at the later one?"}
    assert any(r.startswith("question_lacks_discriminator:") for r in validate(unnamed, c, more, CFG, []).reasons)
