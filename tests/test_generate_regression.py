"""Evidence selection on the real parse outputs, without an LLM.

Run with: uv run pytest -m regression tests/test_generate_regression.py
Needs data/interim/<name>/ from `medrageval parse`; each bundle is skipped when it is missing.
"""

from pathlib import Path
from random import Random

import pytest

from medrageval.config import load_config
from medrageval.generate.candidates import SELECTORS, candidate_counts, stated_elsewhere, table_comparison
from medrageval.generate.corpus import load_corpus
from medrageval.text import table_rows

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
        # Tables 35 and 36 of the clarification response repeat the AIC/BIC rows for the two components of the regimen
        # Tables 35 and 36 of the clarification response repeat the AIC/BIC rows for the two components of the regimen; the
        # response also reprints the submission's goodness-of-fit table (Table 21 -> 39) and base-case table (44 -> 27, 28) with new values
        "twins": {"d04-e0361": ["TTD", "capivasertib arm"], "d04-e0374": ["TTD", "fulvestrant arm"], "d04-e0295": ["Scenario 1"],
                  "d02-e0343": ["company submission"], "d04-e0423": ["CAPItello-291"], "d02-e0449": ["company submission"], "d12-e0163": ["EAG"]},
        "copies": {"d12-e0298": ["d04-e0182"], "d04-e0182": ["d12-e0298"]},   # EAG Table 3.12 reprints clarification Table 18 unchanged
        "not_a_bridge": ["d12-e0393"],                            # lists the five NMA trials and describes none
        "two_party_stance": "d12-e0428",                          # "The EAG agrees with the company that ..."
        "redacted_but_stated": {"d02-e0249": "d02-e0441#r00"},   # Table 14 redacts body surface area; Table 42 prints 1.82 / 1.70
        "figure_as_table": "d02-e0160",                           # Figure 10 (forest plot) came out of OCR as a table with "<65 yéárs"
        "grouped_table": ("d12-e0327", "CAPItello-291 (PI3K/AKT altered population) | Placebo + fulvestrant 500 | 134"),
    },
    "document2": {
        "roles": {"d01": "cover", "d02": "company_submission", "d03": "patient_summary", "d04": "clarification", "d06": "professional_org",
                  "d08": "eag_report", "d09": "factual_accuracy", "d10": "clinical_expert", "d12": "patient_expert"},
        "topic": "Sparsentan for treating primary IgA nephropathy",
        "entities": ["PROTECT", "NefIgArd"],
        "min_candidates": {"factual": 400, "numerical": 150, "comparative": 100, "multihop": 60, "crossdoc": 60,
                           "attribution": 100, "unanswerable_redacted": 15, "unanswerable_not_covered": 30},
        "crossdoc_pair": ("d08", "d02"),
        # base-case results at list and PAS price (company and clarification), transition matrices per arm and period (EAG)
        "twins": {"d02-e0685": ["List", "company submission"], "d02-e0687": ["PAS", "company submission"], "d04-e0371": ["List", "clarification"],
                  "d04-e0376": ["clarification"], "d08-e0391": ["Weeks 0-12", "group", "sparsentan"], "d08-e0399": ["Weeks 12-108", "irbesartan"]},
        "copies": {},
        "not_a_bridge": [],
        "two_party_stance": None,
        "redacted_but_stated": {"d05-e0009": "d10-e0015"},       # a redacted job title on one form, printed on another
        "italic_caption": ("d04", "table", "14", "d04-e0371"),   # "_Table 14: ..._" is still a caption
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
    cands = list(SELECTORS["attribution"](c, CFG, Random(CFG["seed"])))
    speakers = {x.hint["speaker"] for x in cands}
    assert "the EAG" in speakers and len(speakers) >= 3
    if EXPECTED[name]["two_party_stance"]:
        assert EXPECTED[name]["two_party_stance"] not in {x.units[0].unit_id for x in cands}
    contested = [x for x in cands if "contested_by" in x.hint]
    assert len(contested) >= 5 and next(x for x in cands if x.hint["speaker"] == "the EAG") in contested


def test_bridges_describe_their_entity(bundle):
    name, c = bundle
    entity = [x for x in SELECTORS["multihop"](c, CFG, Random(CFG["seed"])) if x.subtype == "entity"]
    assert len(entity) >= 25
    for x in entity:
        assert x.hint["link"].lower()[:5] in x.hint["bridge"].lower() and x.hint["bridge"] in x.units[0].plain
    assert not set(EXPECTED[name]["not_a_bridge"]) & {x.units[0].unit_id for x in entity}


def test_lay_summary_is_used_last(bundle):
    name, c = bundle
    for sel in ("factual", "numerical", "unanswerable_not_covered"):
        roles = [c.docs[x.units[0].doc_id].role for x in SELECTORS[sel](c, CFG, Random(CFG["seed"]))]
        assert "patient_summary" in roles and roles.index("patient_summary") >= 0.8 * len(roles), sel


def test_twin_tables_get_their_discriminating_words(bundle):
    name, c = bundle
    for uid, phrases in EXPECTED[name]["twins"].items():
        assert c.discriminator(c.units[uid]) == phrases, uid
    assert all(all(p for p in c.discriminator(c.units[uid])) for uid in c.twins), "a twin with nothing to tell it apart"
    assert c.copies == EXPECTED[name]["copies"]
    assert not any(cp in c.twins.get(uid, []) for uid, cps in c.copies.items() for cp in cps)


def test_redacted_values_printed_elsewhere_are_skipped(bundle):
    name, c = bundle
    for uid, where in EXPECTED[name]["redacted_but_stated"].items():
        assert stated_elsewhere(c, c.units[uid], CFG) == where
        assert uid not in {x.units[0].unit_id for x in SELECTORS["unanswerable_redacted"](c, CFG, Random(CFG["seed"]))}


def test_ocr_and_figures_are_not_evidence(bundle):
    name, c = bundle
    if "figure_as_table" in EXPECTED[name]:
        assert EXPECTED[name]["figure_as_table"] not in c.units
    ocr = {uid for uid, u in c.units.items() if u.source == "ocr"}
    assert ocr, "no OCR units at all; the check is vacuous"
    for sel in SELECTORS:
        assert not any(u.unit_id in ocr for x in SELECTORS[sel](c, CFG, Random(CFG["seed"])) for u in x.units), sel


def test_grouped_table_rows_carry_their_group(bundle):
    name, c = bundle
    if "grouped_table" in EXPECTED[name]:
        uid, line = EXPECTED[name]["grouped_table"]
        assert any(l.startswith(line) for l in c.units[uid].plain.splitlines())
        labels = table_comparison(table_rows(c.units[uid].text), CFG, Random(CFG["seed"]))["labels"]
        assert len({lb.split(" ")[0] for lb in labels}) == 1, labels   # one study per comparison
    if "italic_caption" in EXPECTED[name]:
        doc, kind, number, element = EXPECTED[name]["italic_caption"]
        assert c.captions[(doc, kind, number)] == element


def test_selection_is_reproducible(bundle):
    name, c = bundle
    for sel in SELECTORS:
        first = [x.candidate_id for x in SELECTORS[sel](c, CFG, Random(CFG["seed"]))][:30]
        again = [x.candidate_id for x in SELECTORS[sel](c, CFG, Random(CFG["seed"]))][:30]
        assert first == again, sel
