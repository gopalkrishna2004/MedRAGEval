import argparse

import pytest
import yaml

from medrag import cli
from medrag.app import App, collection_name
from medrag.config import check_reader, default_reader, missing_key, use_pdf, use_reader
from medrageval.rag.evaluate import _app, add_ablate_arguments, add_arguments


def cfg(tmp_path):
    configured = tmp_path / "configured.pdf"
    configured.write_bytes(b"configured")
    return {"pdf": str(configured), "topic": "the configured appraisal", "chunking": {"target_tokens": 400},
            "embedding": {"model": "org/embedder"}, "llm": {"model": "m", "base_url": "http://x"}}


def other_pdf(tmp_path):
    other = tmp_path / "other.pdf"
    other.write_bytes(b"other")
    return other


def test_another_pdf_changes_the_collection_and_drops_the_topic(tmp_path):
    base = cfg(tmp_path)
    app = App(base, pdf=other_pdf(tmp_path))
    assert app.cfg["topic"] is None
    assert collection_name(app.cfg) == collection_name(base, other_pdf(tmp_path))
    assert collection_name(app.cfg) != collection_name(base)
    assert base["topic"] == "the configured appraisal"   # the caller's config is left alone


def test_the_configured_pdf_keeps_its_topic(tmp_path):
    base = cfg(tmp_path)
    assert use_pdf(base, base["pdf"])["topic"] == "the configured appraisal"


def test_no_pdf_leaves_the_config_as_it_is(tmp_path):
    base = cfg(tmp_path)
    assert App(base).cfg is base


def test_a_missing_pdf_stops_with_a_clear_error(tmp_path):
    with pytest.raises(SystemExit, match="no such PDF"):
        App(cfg(tmp_path), pdf=tmp_path / "missing.pdf")


@pytest.mark.parametrize("argv", [["ingest", "{pdf}"], ["search", "q", "--pdf", "{pdf}"], ["ask", "q", "--pdf", "{pdf}"], ["serve", "--pdf", "{pdf}"]])
def test_every_medrag_command_passes_the_pdf_on(tmp_path, monkeypatch, argv):
    seen = {}

    def fake_app(config, use_cache, pdf):
        seen["pdf"] = pdf
        raise SystemExit(0)

    monkeypatch.setattr(cli, "App", fake_app)
    monkeypatch.setattr("sys.argv", ["medrag", *[a.format(pdf="doc.pdf") for a in argv]])
    with pytest.raises(SystemExit):
        cli.main()
    assert seen["pdf"] == "doc.pdf"


@pytest.mark.parametrize("add", [add_arguments, add_ablate_arguments])
def test_eval_and_ablate_pass_the_pdf_on(tmp_path, add):
    rag_config = tmp_path / "rag.yaml"
    rag_config.write_text("pdf: configured.pdf\ntopic: t\nllm: {model: m, base_url: 'http://x'}\nchunking: {}\nretrieval: {}\n")
    ap = argparse.ArgumentParser()
    add(ap)
    pdf = other_pdf(tmp_path)
    args = ap.parse_args(["data/datasets/d", "--rag-config", str(rag_config), "--pdf", str(pdf)])
    assert _app(args).cfg["pdf"] == str(pdf)


PROVIDERS = {"local": {"label": "Local", "base_url": "http://local/v1", "base_url_env": "TEST_LOCAL_URL", "api_key_env": "TEST_LOCAL_KEY",
                       "models": ["org/Small-3B"]},
             "hosted": {"label": "Hosted", "base_url": "https://hosted/v1", "api_key_env": "TEST_HOSTED_KEY", "needs_key": True,
                        "models": ["org/small-7b", "org/other"]}}


def providers_cfg():
    return {"llm": {"provider": "local", "model": "org/Small-3B", "temperature": 0.0, "providers": PROVIDERS}}


def test_a_provider_brings_its_server_key_and_model(monkeypatch):
    monkeypatch.delenv("TEST_LOCAL_URL", raising=False)
    base = providers_cfg()
    llm = use_reader(base)["llm"]
    assert (llm["provider"], llm["model"], llm["base_url"], llm["api_key_env"]) == ("local", "org/Small-3B", "http://local/v1", "TEST_LOCAL_KEY")
    hosted = use_reader(base, "hosted")["llm"]
    assert (hosted["model"], hosted["base_url"], hosted["api_key_env"], hosted["temperature"]) == ("org/small-7b", "https://hosted/v1", "TEST_HOSTED_KEY", 0.0)
    assert use_reader(base, "hosted", "org/other")["llm"]["model"] == "org/other"
    assert "base_url" not in base["llm"]   # the caller's config is left alone
    monkeypatch.setenv("TEST_LOCAL_URL", "http://gpu-host:8003/v1")
    assert use_reader(base)["llm"]["base_url"] == "http://gpu-host:8003/v1"
    assert use_reader(base, "hosted")["llm"]["base_url"] == "https://hosted/v1"   # the variable names the local server only
    with pytest.raises(ValueError, match="no answering provider"):
        use_reader(base, "nope")


def test_a_config_without_providers_keeps_its_own_server():
    llm = use_reader({"llm": {"model": "m", "base_url": "http://x"}})["llm"]
    assert (llm["provider"], llm["model"], llm["base_url"]) == ("default", "m", "http://x")
    assert default_reader({"model": "m", "base_url": "http://x"}) == ("default", "m")


def test_a_choice_must_be_offered_and_a_hosted_provider_needs_its_key(monkeypatch):
    llm = providers_cfg()["llm"]
    check_reader(llm, "hosted", "org/other")
    with pytest.raises(ValueError, match="not org/Small-3B"):
        check_reader(llm, "hosted", "org/Small-3B")
    with pytest.raises(ValueError, match="no answering provider"):
        check_reader(llm, "nope", "m")
    monkeypatch.delenv("TEST_HOSTED_KEY", raising=False)
    monkeypatch.delenv("TEST_LOCAL_KEY", raising=False)
    assert missing_key(use_reader({"llm": llm}, "hosted")["llm"]) == "TEST_HOSTED_KEY"
    assert missing_key(use_reader({"llm": llm})["llm"]) is None   # a local server runs without one
    monkeypatch.setenv("TEST_HOSTED_KEY", "sk-test")
    assert missing_key(use_reader({"llm": llm}, "hosted")["llm"]) is None


def test_eval_answers_on_the_provider_it_is_given(tmp_path, monkeypatch):
    rag_config = tmp_path / "rag.yaml"
    rag_config.write_text(yaml.safe_dump({"pdf": "configured.pdf", "topic": "t", "chunking": {}, "retrieval": {}, **providers_cfg()}))
    ap = argparse.ArgumentParser()
    add_arguments(ap)
    args = ap.parse_args(["data/datasets/d", "--rag-config", str(rag_config), "--provider", "hosted"])
    monkeypatch.delenv("TEST_HOSTED_KEY", raising=False)
    with pytest.raises(SystemExit, match="TEST_HOSTED_KEY is not set"):
        _app(args)
    monkeypatch.setenv("TEST_HOSTED_KEY", "sk-test")
    llm = _app(args).cfg["llm"]
    assert (llm["provider"], llm["model"], llm["base_url"]) == ("hosted", "org/small-7b", "https://hosted/v1")
    assert _app(ap.parse_args(["d", "--rag-config", str(rag_config)])).cfg["llm"]["provider"] == "local"
    with pytest.raises(SystemExit, match="no answering provider"):
        _app(ap.parse_args(["d", "--rag-config", str(rag_config), "--provider", "nope"]))
