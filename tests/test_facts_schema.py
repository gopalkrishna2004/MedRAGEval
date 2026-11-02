import copy

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
    text = yaml.safe_dump(curated()).lower()
    assert "capivasertib" not in text and "sparsentan" not in text and "capitello" not in text


@pytest.mark.parametrize("change, message", [
    (lambda d: d["groups"]["efficacy"].update(roles=["press_release"]), "unknown roles"),
    (lambda d: d["groups"]["efficacy"]["slots"]["endpoint_result"].update(kind="table"), "kind must be one of"),
    (lambda d: d["groups"]["key_issues"]["slots"]["issue"].pop("parties"), "stance slot lists its parties"),
    (lambda d: d["groups"]["efficacy"]["slots"]["treatment_effect"].update(per="arm", kind="text"), "only for number slots"),
    (lambda d: d["links"].append({"from": "efficacy.nothing", "to": "efficacy.endpoint_result", "via": "endpoint", "describe": "x"}), "unknown slot"),
    (lambda d: d["links"].append({"from": "trial_design.design", "to": "efficacy.endpoint_result", "via": "colour", "describe": "x"}), "no qualifier"),
    (lambda d: d["groups"]["safety"].update(sections=["(unclosed"]), "bad section pattern"),
    (lambda d: d.update(groups={}), "non-empty 'groups'"),
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
