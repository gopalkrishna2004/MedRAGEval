"""Evidence selection on the real parse outputs, without an LLM.

Run with: uv run pytest -m regression tests/test_generate_regression.py
Needs data/interim/<name>/ from `medrageval parse`; each bundle is skipped when it is missing.
"""

from pathlib import Path
from random import Random

import pytest

from medrageval.config import load_config
from medrageval.generate.candidates import SELECTORS, candidate_counts
from medrageval.generate.corpus import load_corpus

pytestmark = pytest.mark.regression
ROOT = Path(__file__).resolve().parents[1]
CFG = load_config()["generate"]

EXPECTED = {
    "ta1063": {
        "roles": {"d01": "cover", "d02": "company_submission", "d03": "patient_summary", "d04": "clarification", "d05": "patient_org",
                  "d08": "clinical_expert", "d09": "patient_expert", "d12": "eag_report", "d13": "factual_accuracy"},
        "topic": "Capivasertib with fulvestrant",
        "entities": ["CAPItello-291", "SOLAR-1", "BOLERO-2"],
        "min_candidates": {"factual": 300, "numerical": 100, "comparative": 60, "multihop": 60, "crossdoc": 35,
                           "attribution": 50, "unanswerable_redacted": 30, "unanswerable_not_covered": 25},
        "crossdoc_pair": ("d12", "d02"),
    },
    "document2": {
        "roles": {"d01": "cover", "d02": "company_submission", "d03": "patient_summary", "d04": "clarification", "d06": "professional_org",
                  "d08": "eag_report", "d09": "factual_accuracy", "d10": "clinical_expert", "d12": "patient_expert"},
        "topic": "Sparsentan for treating primary IgA nephropathy",
        "entities": ["PROTECT", "NefIgArd"],
        "min_candidates": {"factual": 400, "numerical": 150, "comparative": 100, "multihop": 60, "crossdoc": 60,
                           "attribution": 100, "unanswerable_redacted": 15, "unanswerable_not_covered": 30},
        "crossdoc_pair": ("d08", "d02"),
    },
}


def _corpus(name):
    path = ROOT / "data/interim" / name
    if not (path / "elements.jsonl").exists():
        pytest.skip(f"{path} is missing; run `medrageval parse` first")
    return load_corpus(path, CFG)


@pytest.fixture(scope="module", params=list(EXPECTED))
def bundle(request):
    return request.param, _corpus(request.param)


def test_roles_topic_and_entities(bundle):
    name, c = bundle
    exp = EXPECTED[name]
    assert {d: c.docs[d].role for d in exp["roles"]} == exp["roles"]
    assert c.topic.startswith(exp["topic"])
    assert set(exp["entities"]) <= set(c.entity_names.values())


def test_noise_removes_reference_lists_and_figures(bundle):
    name, c = bundle
    assert c.noise["section"] >= 100 and c.noise["figure"] >= 60
    assert not any("References" in u.section_path[-1:] for u in c.units.values())


def test_every_type_has_candidates_to_spare(bundle):
    name, c = bundle
    counts = candidate_counts(c, CFG, CFG["seed"])
    for sel, minimum in EXPECTED[name]["min_candidates"].items():
        assert counts[sel]["total"] >= minimum, (sel, counts[sel]["total"])


def test_crossdoc_pairs_reviewer_with_company(bundle):
    name, c = bundle
    a, b = EXPECTED[name]["crossdoc_pair"]
    pairs = [(x.units[0].doc_id, x.units[1].doc_id) for x in SELECTORS["crossdoc"](c, CFG, Random(CFG["seed"]))]
    assert (a, b) in pairs and all(p[0] != p[1] for p in pairs)


def test_attribution_has_more_than_one_speaker(bundle):
    name, c = bundle
    speakers = {x.hint["speaker"] for x in SELECTORS["attribution"](c, CFG, Random(CFG["seed"]))}
    assert "the EAG" in speakers and len(speakers) >= 3


def test_selection_is_reproducible(bundle):
    name, c = bundle
    for sel in SELECTORS:
        first = [x.candidate_id for x in SELECTORS[sel](c, CFG, Random(CFG["seed"]))][:30]
        again = [x.candidate_id for x in SELECTORS[sel](c, CFG, Random(CFG["seed"]))][:30]
        assert first == again, sel
