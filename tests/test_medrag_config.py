import argparse

import pytest

from medrag import cli
from medrag.app import App, collection_name
from medrag.config import use_pdf
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
