import copy
import json
import re

import pytest
import yaml
from synthetic import config, facts_config

from medrageval.facts.schema import SchemaError, known_roles, load_schema, parse_schema, resolve_path

ROLES = known_roles(config())


def curated() -> dict:
    with open(resolve_path(facts_config()["schema"])) as f:
        return yaml.safe_load(f)


def test_curated_schema_loads_and_is_disease_agnostic():
    s = load_schema(facts_config()["schema"], ROLES)
    assert len(s.groups) >= 8 and abs(sum(s.weights().values()) - 1) < 1e-9
    assert s.slot("efficacy.endpoint_result").per == "arm" and "trial" in s.slot("efficacy.endpoint_result").qualifiers
    assert s.slot("key_issues.issue").kind == "stance" and s.slot("key_issues.issue").parties == ("company", "EAG")
    assert not s.slot("stakeholder_views.respondent").ask
    assert {(ln.source, ln.target, ln.via) for ln in s.links} >= {("trial_design.primary_endpoint", "efficacy.endpoint_result", "endpoint")}
    described = next(ln for ln in s.links if ln.target == "cost_effectiveness.total_outcomes" and ln.match == "qualifier")
    assert described.source_via == "comparator" and described.via == s.slot("cost_effectiveness.total_outcomes").per   # a per label, not a qualifier
    # groups, slots and links fit any appraisal; only the framing's examples are TA1063's, the prompts' own words
    text = yaml.safe_dump({k: v for k, v in curated().items() if k != "framing"}).lower()
    assert "capivasertib" not in text and "sparsentan" not in text and "capitello" not in text
    assert s.framing.subject == "one health technology appraisal" and s.framing.owners == ("company", "EAG") and len(s.framing.personas) == 6


@pytest.mark.parametrize("change, message", [
    (lambda d: d["groups"]["efficacy"].update(roles=["press_release"]), "unknown roles"),
    (lambda d: d["groups"]["efficacy"]["slots"]["endpoint_result"].update(kind="table"), "kind must be one of"),
    (lambda d: d["groups"]["key_issues"]["slots"]["issue"].pop("parties"), "stance slot lists its parties"),
    (lambda d: d["groups"]["efficacy"]["slots"]["treatment_effect"].update(per="arm", kind="text"), "only for number slots"),
    (lambda d: d["links"].append({"from": "efficacy.nothing", "to": "efficacy.endpoint_result", "via": "endpoint", "describe": "x"}), "unknown slot"),
    (lambda d: d["links"].append({"from": "trial_design.design", "to": "efficacy.endpoint_result", "via": "colour", "describe": "x"}), "no qualifier"),
    (lambda d: d["links"].append({"from": "trial_design.design", "to": "efficacy.endpoint_result", "via": "endpoint", "describe": "x", "match": "guess"}), "match must be"),
    (lambda d: d["links"].append({"from": "trial_design.design", "to": "efficacy.endpoint_result", "via": "endpoint", "describe": "x", "match": "qualifier"}), "source_via"),
    (lambda d: d["groups"]["safety"].update(sections=["(unclosed"]), "bad section pattern"),
    (lambda d: d.update(groups={}), "non-empty 'groups'"),
    (lambda d: d["framing"].update(audience="x"), "framing: unknown keys"),
    (lambda d: d["framing"].update(owners="company"), "framing.owners: expected a list"),
    (lambda d: d["framing"]["examples"].update(made_up="x"), "framing.examples: unknown keys"),
    (lambda d: d["framing"].update(subject=" "), "framing.subject: expected a non-empty string"),
])
def test_malformed_schemas_fail_loudly(change, message):
    data = copy.deepcopy(curated())
    change(data)
    with pytest.raises(SchemaError, match=message):
        parse_schema(data, ROLES)


def test_missing_schema_file(tmp_path):
    with pytest.raises(SchemaError, match="not found"):
        load_schema(tmp_path / "nope.yaml", ROLES)


def test_draft_shows_each_roles_outline_and_parses(tmp_path):
    from synthetic import FACT_BUNDLE, write_interim

    from medrageval.facts.draft import draft_schema
    from medrageval.generate.llm import FakeLLM
    proposed = curated()
    proposed["groups"]["efficacy"]["weight"] = "0.20"   # an LLM writes numbers as strings
    llm = FakeLLM(lambda messages, tag: proposed)
    data = draft_schema([write_interim(tmp_path / "interim", FACT_BUNDLE)], config(), facts_config(), llm)
    assert parse_schema(data, ROLES).groups["efficacy"].weight == 0.2
    user = llm.calls[0][0][1]["content"]
    assert "## Role: eag_report" in user and "5.2 EAG critique of the economic model" in user and "## Role: cover" not in user


def test_a_family_the_role_patterns_do_not_know_is_drafted_and_extracted_as_other(tmp_path):
    from synthetic import PAPER_SCHEMA, write_paper_interim

    from medrageval.facts.draft import draft_schema
    from medrageval.facts.extract import plan_batches
    from medrageval.generate.corpus import load_corpus
    from medrageval.generate.llm import FakeLLM
    interim = write_paper_interim(tmp_path / "paper")
    corpus = load_corpus(interim, config())
    assert [d.role for d in corpus.docs.values()] == ["other"]
    llm = FakeLLM(lambda messages, tag: PAPER_SCHEMA)
    schema = parse_schema(draft_schema([interim], config(), facts_config(), llm), ROLES)
    system, user = (m["content"] for m in llm.calls[0][0])
    assert "Use only these roles: other." in system and "## Role: other" in user and "Results" in user
    assert {b.group for b in plan_batches(corpus, schema, config(), facts_config())} == {"efficacy", "safety"}


def test_draft_with_nothing_but_cover_pages_stops_before_the_model(tmp_path):
    from synthetic import PAPER_ELEMENTS, PAPER_SECTIONS, write_interim

    from medrageval.facts.draft import draft_schema
    from medrageval.generate.llm import FakeLLM
    docs = [{"doc_id": d, "start_page": i, "end_page": i, "n_pages": 1, "signals": [], "running_text": None, "headings": ["Contents"]}
            for i, d in enumerate(["p01", "p02"], start=1)]
    els = PAPER_ELEMENTS + [dict(PAPER_ELEMENTS[5], doc_id="p02", element_id="p02-e0001", section_id=None)]
    llm = FakeLLM(lambda messages, tag: {})
    with pytest.raises(ValueError, match="nothing to draft from"):
        draft_schema([write_interim(tmp_path / "covers", els, PAPER_SECTIONS, docs)], config(), facts_config(), llm)
    assert llm.calls == []


APPRAISAL_WORDS = ("appraisal", "committee", "EAG", "ICER", "QALY", "capivasertib", "fulvestrant", "capitello", "everolimus", "company's")


def test_a_schema_without_framing_gets_prompts_with_no_appraisal_wording(tmp_path):
    from synthetic import PAPER_SCHEMA, write_paper_interim

    from medrageval.facts.extract import plan_batches
    from medrageval.facts.factsheet import FactSheet
    from medrageval.facts.prompts import build_extraction
    from medrageval.generate import prompts as P
    from medrageval.generate.candidates import Candidate
    from medrageval.generate.corpus import load_corpus
    cfg, fcfg = config(), facts_config()
    corpus = load_corpus(write_paper_interim(tmp_path / "paper"), cfg)
    schema = parse_schema(PAPER_SCHEMA, ROLES)
    fr, sheet = schema.framing, FactSheet([], schema, corpus, cfg, fcfg)
    unit = next(iter(corpus.units.values()))
    item = {"question": "q", "answer": "a", "answer_keys": ["1"], "quotes": []}
    msgs = [build_extraction(schema.groups[b.group], b.units, corpus, b.doc_id, 5, 2000, fr) for b in plan_batches(corpus, schema, cfg, fcfg)]
    for type_, sub in [("factual", None), ("numerical", None), ("comparative", None), ("attribution", None), ("summary", None),
                       ("multihop", "bridge"), ("multihop", "described")]:
        cand = Candidate(type_, sub, [unit], {"must_mention": [["Drug X"]]})
        msgs += [P.build_generation(cand, corpus, cfg, sheet), P.build_judge(item, cand, corpus, cfg, fr)]
    msgs += [P.build_alone_check("q", unit, corpus, cfg, fr), P.build_withhold_check("q", "a", [unit], corpus, framing=fr),
             P.withhold_check("[c1] x", "q", "a", corpus.topic, "chunk_id", True, fr)]
    msgs += [P.build_messy("q", persona, [h], fr) for persona in cfg["messy"]["personas"] for h in P.HABITS]
    feedback = P.rewrite_feedback([{"stage": "judge", "reasons": [r for r in P.FEEDBACK if r.startswith("judge")] + ["answer_key_has_label:x"]}],
                                  Candidate("factual", None, [unit]), fr)
    text = json.dumps(msgs) + "\n".join(feedback)
    assert corpus.topic in text and "one medical topic" in text
    assert [w for w in APPRAISAL_WORDS if re.search(rf"\b{re.escape(w)}\b", text, re.I)] == []
    assert "an appraisal committee member preparing for a meeting" in curated()["framing"]["personas"]   # the NICE schema's own personas
    assert "a busy oncologist between clinics" not in text
