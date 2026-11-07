import json
from random import Random

from synthetic import config

from medrageval.generate.llm import FakeLLM, LLMJsonError
from medrageval.generate.messy import messy_pass, pick, retwin, rewrite, surface_noise, typo
from medrageval.generate.prompts import HABITS, build_messy

Q = "What was the median progression-free survival with capivasertib plus fulvestrant in CAPItello-291, in months?"


def mcfg(**over):
    return {**config()["messy"], **over}


def item(item_id="factual-001", **over):
    return {"item_id": item_id, "type": "factual", "question": Q, **over}


def test_pick_is_seeded_and_keyword_query_stands_alone():
    m = mcfg()
    a, b = pick(Random("s"), m), pick(Random("s"), m)
    assert a == b
    assert a[0] in m["severity_mix"] and a[1] in m["personas"] and 1 <= len(a[2]) <= 3
    seen = set()
    for i in range(200):
        sev, _, habits = pick(Random(i), m)
        seen.add(sev)
        assert len(habits) <= m["habits_per_severity"][sev]
        if "keyword_query" in habits:
            assert habits[0] == "keyword_query" and set(habits[1:]) <= {"misspell_names"}
    assert seen == {"light", "medium", "heavy"}


def test_typo_changes_one_letter_and_leaves_short_words():
    rng = Random(1)
    assert typo("a", rng) == "a" and typo("24", rng) == "24"   # nothing to swap: under two letters
    for _ in range(50):
        w = typo("fulvestrant", rng)
        assert w != "fulvestrant" and abs(len(w) - len("fulvestrant")) <= 1


def test_surface_noise_never_touches_numbers_and_is_reproducible():
    rates = {"typo_rate": 1.0, "lowercase": 1.0, "drop_punctuation": 1.0}
    text, ops = surface_noise("Median PFS at 24 months was 7.3 months, vs 3.1?", Random(3), rates, 4)
    assert "24" in text and "7.3" in text and "3.1" in text and "?" not in text and "," not in text and text == text.lower()
    kept, _ = surface_noise("ICER of £52,681 per QALY, at 0.50.", Random(2), {"typo_rate": 0.0, "lowercase": 0.0, "drop_punctuation": 1.0}, 4)
    assert kept == "ICER of £52,681 per QALY at 0.50"
    assert "lowercase" in ops and "drop_punctuation" in ops and any(op.startswith("typo:") for op in ops)
    assert surface_noise(Q, Random(7), rates, 4) == surface_noise(Q, Random(7), rates, 4)
    quiet = {"typo_rate": 0.0, "lowercase": 0.0, "drop_punctuation": 0.0}
    assert surface_noise(Q, Random(7), quiet, 4) == (Q, [])


def test_textspeak_and_missing_spaces_leave_numbers_alone():
    rates = {"typo_rate": 0.0, "textspeak": 1.0, "join_words": 0.0, "lowercase": 0.0, "drop_punctuation": 0.0}
    text, ops = surface_noise("Please compare treatment with fulvestrant, about 7.3 months", Random(1), rates, 4)
    assert text == "pls compare tx w/ fulvestrant, abt 7.3 mo"   # the number stays; the words around it do not
    assert all(op.startswith("textspeak:") for op in ops) and "textspeak:Please>pls" in ops
    text, _ = surface_noise("for the pts to see", Random(1), {**rates, "textspeak": 0.0}, 4)
    assert text == "for the pts to see"
    rates = {"typo_rate": 0.0, "textspeak": 0.0, "join_words": 1.0, "lowercase": 0.0, "drop_punctuation": 0.0}
    text, ops = surface_noise("median pfs was 7.3 months in the trial", Random(1), rates, 4)
    assert text == "medianpfs was 7.3 monthsin thetrial" and ops == ["join:median pfs", "join:months in", "join:the trial"]   # never onto "7.3"


def test_build_messy_names_the_persona_and_the_habits_only():
    msgs = build_messy(Q, "a pharmacy student writing an essay", ["shorthand", "vague"])
    user = msgs[1]["content"]
    assert f"Clean question: {Q}" in user and "a pharmacy student" in user
    assert HABITS["shorthand"] in user and HABITS["vague"] in user and HABITS["informal"] not in user


def test_rewrite_records_the_llm_text_before_the_noise():
    gen = FakeLLM(lambda messages, tag: {"question": "median pfs cap + fulv capitello 291?"})
    tw = rewrite(item(), gen, mcfg(), 1)
    assert tw["rewrite"] == "median pfs cap + fulv capitello 291?" and tw["text"] and "error" not in tw
    assert gen.calls[0][1] == "messy:factual-001"


def test_rewrite_falls_back_to_surface_noise_when_the_llm_fails():
    def broken(messages, tag):
        raise LLMJsonError("no json")
    tw = rewrite(item(), FakeLLM(broken), mcfg(), 1)
    assert tw["error"].startswith("no_json") and tw["rewrite"] == Q and tw["habits"] == []
    tw = rewrite(item(), FakeLLM(lambda m, t: {"question": ""}), mcfg(), 1)
    assert tw["error"] == "empty_rewrite" and tw["rewrite"] == Q


def test_messy_pass_copies_the_source_twin_onto_unanswerable_items():
    gen = FakeLLM(lambda messages, tag: {"question": "rewritten " + tag})
    acc = [item("factual-001"), item("numerical-001", type="numerical"),
           item("unanswerable-001", type="unanswerable", source_item_id="factual-001"),
           item("unanswerable-002", type="unanswerable", source_item_id="factual-099")]   # source not in this run: its own rewrite
    rep = messy_pass(acc, gen, mcfg(), 5, 2)
    assert sorted(tag for _, tag in gen.calls) == ["messy:factual-001", "messy:numerical-001", "messy:unanswerable-002"]
    assert acc[2]["question_messy"] == acc[0]["question_messy"] and acc[2]["question_messy"] is not acc[0]["question_messy"]
    assert acc[3]["question_messy"]["rewrite"] == "rewritten messy:unanswerable-002"
    assert rep["items"] == 4 and rep["rewritten"] == 3 and rep["copied_from_source"] == 1 and rep["errors"] == []
    assert sum(rep["severity"].values()) == 4
    json.dumps(acc)   # serialisable as written to dataset.jsonl


def test_retwin_replaces_the_twins_in_place_and_nothing_else(tmp_path):
    rows = [{**item("factual-001"), "answer": "x", "question_messy": {"text": "old", "severity": "light", "habits": [], "surface": []}},
            item("unanswerable-001", type="unanswerable", source_item_id="factual-001", withheld={"answer": "x"})]
    (tmp_path / "dataset.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "generate_report.json").write_text(json.dumps({"items": 2, "per_type": {"factual": {}}}))
    gen = FakeLLM(lambda messages, tag: {"question": "new " + tag})
    rep = retwin(tmp_path, config(), gen, seed=3)
    out = [json.loads(l) for l in (tmp_path / "dataset.jsonl").open()]
    assert [o["item_id"] for o in out] == ["factual-001", "unanswerable-001"] and out[0]["answer"] == "x" and out[1]["withheld"] == {"answer": "x"}
    assert out[0]["question_messy"]["rewrite"] == "new messy:factual-001" and out[1]["question_messy"] == out[0]["question_messy"]
    assert rep["rewritten"] == 1 and rep["copied_from_source"] == 1 and len(gen.calls) == 1
    report = json.loads((tmp_path / "generate_report.json").read_text())
    assert report["per_type"] == {"factual": {}} and report["messy"]["seed"] == 3 and report["messy"]["model"] == "fake"
    assert report["messy"]["rewritten"] == 1 and report["messy"]["retwinned_at"]
