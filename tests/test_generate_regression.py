"""Extraction planning and fact-based selection on the real parse outputs, without an LLM.

Run with: uv run pytest -m regression tests/test_generate_regression.py
Needs data/interim/<name>/ from `medrageval parse`; each bundle is skipped when it is missing.
The fact-sheet tests also need factsheet.jsonl from `medrageval extract`, and are skipped without it.
"""

from pathlib import Path
from random import Random

import pytest

from medrageval.config import load_config
from medrageval.facts.extract import plan_batches
from medrageval.facts.factsheet import FactSheet
from medrageval.facts.schema import known_roles, load_schema
from medrageval.generate.candidates import SELECTORS
from medrageval.generate.corpus import load_corpus

pytestmark = pytest.mark.regression
ROOT = Path(__file__).resolve().parents[1]
FULL = load_config()
CFG, FCFG = FULL["generate"], FULL["facts"]
SCHEMA = load_schema(FCFG["schema"], known_roles(CFG))

EXPECTED = {
    "ta1063": {
        "roles": {"d01": "cover", "d02": "company_submission", "d03": "patient_summary", "d04": "clarification", "d05": "patient_org",
                  "d08": "clinical_expert", "d09": "patient_expert", "d12": "eag_report", "d13": "factual_accuracy"},
        "topic": "Capivasertib with fulvestrant",
        "figure_as_table": "d02-e0149",                           # Figure 10 (forest plot) came out of OCR as a table with "<65 yéárs"
        # units the extraction must read for a group (checked by hand against the PDF)
        "reads": {"efficacy": ["d02-e0109"],                      # CS Table 7: median PFS 7.3 vs 3.1 months (p. 42)
                  "cost_effectiveness": ["d02-e0419"]},           # CS Table 44: deterministic pairwise base case (p. 126)
    },
    "document2": {
        "roles": {"d01": "cover", "d02": "company_submission", "d03": "patient_summary", "d04": "clarification", "d06": "professional_org",
                  "d08": "eag_report", "d09": "factual_accuracy", "d10": "clinical_expert", "d12": "patient_expert"},
        "topic": "Sparsentan for treating primary IgA nephropathy",
        "reads": {},
    },
}

# Facts checked by hand against the PDF: (slot, doc, numbers the fact's value must hold, words its qualifiers must hold).
VERIFIED = {
    "ta1063": [
        ("efficacy.endpoint_result", "d02", {"7.3", "3.1"}, {"pfs"}),            # CS Table 7, p. 42: median PFS by investigator, DCO1
        ("cost_effectiveness.total_outcomes", "d02", {"3.25"}, set()),           # CS Table 44, p. 126: capivasertib total LYG
    ],
    "document2": [],
}


def _corpus(name):
    path = ROOT / "data/interim" / name
    if not (path / "elements.jsonl").exists():
        pytest.skip(f"{path} is missing; run `medrageval parse` first")
    return load_corpus(path, CFG)


@pytest.fixture(scope="module", params=list(EXPECTED))
def bundle(request):
    return request.param, _corpus(request.param)


@pytest.fixture(scope="module")
def sheet(bundle):
    name, c = bundle
    path = ROOT / "data/interim" / name / "factsheet.jsonl"
    if not path.exists():
        pytest.skip(f"{path} is missing; run `medrageval extract` first")
    return name, c, FactSheet.load(path, SCHEMA, c, CFG, FCFG)


def test_roles_and_topic(bundle):
    name, c = bundle
    exp = EXPECTED[name]
    assert {d: c.docs[d].role for d in exp["roles"]} == exp["roles"]
    assert c.topic.startswith(exp["topic"])


def test_noise_removes_reference_lists_and_figures(bundle):
    name, c = bundle
    assert c.noise["section"] >= 100 and c.noise["figure"] >= 60
    assert not any("References" in u.section_path[-1:] for u in c.units.values())
    if "figure_as_table" in EXPECTED[name]:
        assert EXPECTED[name]["figure_as_table"] not in c.units


def test_every_group_is_read_from_its_roles(bundle):
    name, c = bundle
    batches = plan_batches(c, SCHEMA, CFG, FCFG)
    groups = {b.group for b in batches}
    assert groups == {g for g, grp in SCHEMA.groups.items() if grp.weight > 0}, "a group with no document of its roles"
    for b in batches:
        assert c.docs[b.doc_id].role in SCHEMA.groups[b.group].roles
        assert not any(u.source == "ocr" for u in b.units)
        assert len(b.units) <= FCFG["batch"]["max_units"]
    for group, uids in EXPECTED[name]["reads"].items():
        read = {u.element_id for b in batches if b.group == group for u in b.units}
        assert set(uids) <= read, (group, set(uids) - read)


def test_verified_facts_were_extracted(sheet):
    name, c, s = sheet
    for slot, doc, numbers, words in VERIFIED[name]:
        hits = [f for f in s.by_slot.get(slot, []) if f.doc_id == doc and numbers <= f.numbers()
                and words <= {w.lower() for v in f.qualifiers.values() for w in v.replace("-", " ").split()}]
        assert hits, (slot, doc, numbers, words)


def test_fact_sheet_quotes_are_verbatim_and_mostly_distinguishable(sheet):
    from medrageval.generate.validate import quote_in
    name, c, s = sheet
    assert len(s) >= 200
    for f in s.facts.values():
        for e in f.evidence:
            assert quote_in(e["quote"], c.units[e["unit_id"]].plain), f.fact_id
    askable = [f for f in s.facts.values() if s.askable(f)]
    assert sum(s.ambiguous(f) for f in askable) <= 0.25 * len(askable)


def test_every_type_has_candidates_and_selection_is_reproducible(sheet):
    name, c, s = sheet
    for sel in SELECTORS:
        first = [x.candidate_id for x in SELECTORS[sel](c, s, CFG, Random(CFG["seed"]))]
        again = [x.candidate_id for x in SELECTORS[sel](c, s, CFG, Random(CFG["seed"]))]
        assert first == again, sel
        assert first, f"{sel}: no candidates"
        assert not any(u.source == "ocr" for x in SELECTORS[sel](c, s, CFG, Random(CFG["seed"])) for u in x.units), sel


def test_candidates_never_use_an_indistinguishable_fact(sheet):
    name, c, s = sheet
    for sel in ("factual", "numerical", "comparative", "multihop", "crossdoc", "summary"):
        for x in SELECTORS[sel](c, s, CFG, Random(CFG["seed"])):
            assert not any(s.ambiguous(f) for f in x.facts), (sel, x.candidate_id)
