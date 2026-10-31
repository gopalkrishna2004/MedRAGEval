from random import Random

import pytest
from synthetic import build_corpus, config

from medrageval.generate.candidates import SELECTORS, arm_columns, clean_prose, numeric_hits, spread, table_comparison
from medrageval.text import table_rows

CFG = config()
CFG["limits"]["max_per_section"] = 5     # the synthetic bundle has few sections; do not hide candidates behind the coverage cap
CORPUS = build_corpus(CFG)


def run(name, seed=1):
    return list(SELECTORS[name](CORPUS, CFG, Random(seed)))


def test_factual_uses_complete_clean_paragraphs_only():
    ids = {c.units[0].unit_id for c in run("factual")}
    assert "d02-e0001" in ids and "d03-e0001" in ids
    assert "d02-e0004" not in ids          # redacted
    assert "d02-e0005" not in ids          # reference list (noise)
    assert "d04-e0001" not in ids          # the form table
    assert not clean_prose(CORPUS.units["d02-e0004"], CFG)


def test_numerical_prose_and_table_candidates():
    cands = {c.units[0].unit_id: c for c in run("numerical")}
    assert {"24567", "0.61"} <= set(cands["d02-e0008"].hint["numbers"])
    assert cands["d02-e0003"].subtype == "table" and "7.3" in cands["d02-e0003"].hint["numbers"]
    assert numeric_hits("a cost of £1,200 over 6 months and 13% of patients", CFG) == [("1200", "£"), ("6", "months"), ("13", "%")]


def test_comparative_table_arm_columns_and_prose():
    rows = table_rows(CORPUS.units["d02-e0003"].text)
    assert arm_columns(rows, CFG) == (0, [1, 2], ["Capivasertib + fulvestrant (N=155)", "Placebo + fulvestrant (N=134)"])
    hint = table_comparison(rows, CFG, Random(0))
    assert hint["mode"] == "columns" and hint["rows"][0] == "Median PFS (months) | 7.3 | 3.1"
    cands = run("comparative")
    assert cands[0].subtype == "table" and cands[0].units[0].unit_id == "d02-e0003"
    prose = [c for c in cands if c.subtype == "prose"]
    assert any(c.units[0].unit_id == "d03-e0004" and {"7.3", "3.1"} <= set(c.hint["numbers"]) for c in prose)


def test_comparative_rows_mode_for_treatment_rows():
    rows = [["Technologies", "Total costs (£)", "Total QALYs"], ["Capivasertib + fulvestrant", "£61,210", "2.85"],
            ["Alpelisib + fulvestrant", "£52,944", "2.41"], ["Everolimus + exemestane", "£25,714", "1.45"],
            ["Fulvestrant", "[REDACTED]", "1.20"]]
    hint = table_comparison(rows, CFG, Random(0))
    assert hint["mode"] == "rows" and hint["measure"] == "Total costs (£)"
    assert hint["labels"] == ["Capivasertib + fulvestrant", "Alpelisib + fulvestrant", "Everolimus + exemestane"]   # the redacted row is skipped
    assert hint["rows"][0] == "Capivasertib + fulvestrant | £61,210 | 2.85"


def test_multihop_pairs_cross_reference_and_trial_name():
    cands = run("multihop")
    assert cands[0].subtype == "entity" and cands[0].hint["link"] == "CAPItello-291"   # bridges come first
    xref = [c for c in cands if c.subtype == "xref"]
    assert [u.unit_id for u in xref[0].units] == ["d02-e0002", "d02-e0003"] and xref[0].hint["link"] == "Table 7"
    assert all(c.units[0].doc_id == c.units[1].doc_id for c in cands)


def test_crossdoc_strategies_and_copy_rejection():
    cands = run("crossdoc")
    by_sub = {}
    for c in cands:
        by_sub.setdefault(c.subtype, []).append(tuple(u.unit_id for u in c.units))
    assert ("d03-e0004", "d02-e0003") in by_sub["reference"]          # "CS Table 7"
    assert ("d03-e0001", "d02-e0008") in by_sub["numbers"]            # shared £24,567 and 0.61
    assert all(a.startswith("d03") and b.startswith("d02") for pairs in by_sub.values() for a, b in pairs)
    assert ("d03-e0003", "d02-e0008") not in [p for pairs in by_sub.values() for p in pairs]   # verbatim copy


def test_attribution_speakers_from_stance_form_and_response():
    cands = run("attribution")
    by_unit = {c.units[0].unit_id: c for c in cands}
    assert by_unit["d03-e0001"].hint["speaker"] == "the EAG" and by_unit["d03-e0001"].subtype == "stance"
    assert by_unit["d03-e0002"].hint["speaker"] == "the company" and "discontinuation" in by_unit["d03-e0002"].hint["claim"]
    forms = [c for c in cands if c.subtype == "form"]
    assert forms and all(c.hint["speaker"] == "the clinical experts" and c.units[0].element_id == "d04-e0001" for c in forms)
    assert by_unit["d05-e0002"].subtype == "response" and by_unit["d05-e0002"].hint["speaker"] == "the company"
    assert "d05-e0001" not in by_unit        # the EAG's question, not the company's response
    speakers = [c.hint["speaker"] for c in cands[:3]]
    assert len(set(speakers)) == 3           # round-robin over speakers


def test_unanswerable_redacted_labels_and_not_covered_anchors():
    red = run("unanswerable_redacted")
    assert [c.units[0].unit_id for c in red] == ["d02-e0004"]
    assert red[0].hint["labels"] == ["£"] or "£" in red[0].hint["labels"]
    nc = run("unanswerable_not_covered")
    assert all(c.subtype == "not_covered" and c.units[0].redactions == 0 for c in nc)
    assert nc[0].hint["topic"].startswith("Capivasertib with fulvestrant")


@pytest.mark.parametrize("name", list(SELECTORS))
def test_selectors_are_deterministic_for_a_seed(name):
    a = [c.candidate_id for c in run(name, 7)]
    assert a == [c.candidate_id for c in run(name, 7)]


def test_spread_round_robins_over_buckets_with_a_cap():
    items = [("a", 1), ("a", 2), ("a", 3), ("b", 1), ("c", 1)]
    out = list(spread(items, lambda x: x[0], Random(0), cap=2))
    assert len(out) == 4 and {x[0] for x in out[:3]} == {"a", "b", "c"}
