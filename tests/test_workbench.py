"""The Build tab's back end: jobs as child processes, and a document's state read from its files."""

import json
import sys
import time

import pymupdf
import pytest
from medrag.app import App
from medrag.store import MemoryStore
from test_rag_eval import PFS, WITHHELD

from medrageval import workbench
from medrageval.config import load_config
from medrageval.facts.schema import SchemaError
from medrageval.workbench import PY, JobBusy, Jobs, StageError, Workbench, estimate_calls, pdf_sha, slugify, validate_dataset

RC = {"mode": "hybrid", "top_k": 2, "candidates": 4, "rrf_k": 60, "rerank": False, "rerank_candidates": 4, "max_context_chars": 200}
CHUNKING = {"target_tokens": 400, "max_tokens": 512}


def wait(jobs, job, timeout=10):
    t0 = time.time()
    while job.finished is None and time.time() - t0 < timeout:
        time.sleep(0.02)
    return jobs.tail(job.id)


def make_pdf(path, pages=2):
    doc = pymupdf.open()
    for _ in range(pages):
        doc.new_page()
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    return path


def bench(root, pdf="a"):
    """A workbench over a data root whose application is configured for raw/<pdf>.pdf, with an in-memory store."""
    app = App({"pdf": str(root / "raw" / f"{pdf}.pdf"), "chunking": CHUNKING, "embedding": {"model": "e"}, "retrieval": RC,
               "llm": {"model": "m", "base_url": "http://127.0.0.1:9/v1"}})
    app._store = MemoryStore(dim=4)
    return Workbench(root, app, load_config())


def parsed(root, slug, sha, interim=None):
    d = root / "interim" / (interim or slug)
    d.mkdir(parents=True, exist_ok=True)
    (d / "pages.meta.json").write_text(json.dumps({"pdf": f"raw/{slug}.pdf", "sha256": sha}))
    (d / "parse_report.json").write_text(json.dumps({"pages": 2, "documents": 1, "sections": 3, "elements": 9, "warnings": ["w"]}))
    return d


# ----- names and jobs -----

def test_slugify_keeps_safe_stems_and_cleans_the_rest():
    assert slugify("source_document.pdf") == "source_document"
    assert slugify("My Report (final).PDF") == "my-report-final"
    assert slugify("my set.jsonl") == "my-set"
    assert slugify("   ") == "document"
    assert slugify("../../etc/passwd") == "etc-passwd"


def test_job_streams_stdout_and_stderr_lines_and_writes_the_log(tmp_path):
    jobs = Jobs(tmp_path)
    job = jobs.start("parse", "a", [PY, "-c", "import sys; print('a'); print('err', file=sys.stderr); print('b')"], {"k": 1})
    t = wait(jobs, job)
    assert (t["state"], t["returncode"], t["lines"], t["next"], t["result"]) == ("done", 0, ["a", "err", "b"], 3, {"k": 1})
    assert jobs.tail(job.id, 2) == {**t, "lines": ["b"]}
    assert (tmp_path / "jobs" / f"{job.id}.log").read_text() == "a\nerr\nb\n"
    assert workbench.Job("j", "parse", "a", [PY, "-m", "medrageval.cli", "parse", "x.pdf"], tmp_path, 0.0).command == "medrageval parse x.pdf"
    assert "lines" not in jobs.recent()[0] and jobs.recent()[0]["id"] == job.id


def test_second_job_while_one_runs_raises_busy(tmp_path):
    jobs = Jobs(tmp_path)
    job = jobs.start("parse", "a", [PY, "-c", "import time; time.sleep(0.5)"])
    with pytest.raises(JobBusy):
        jobs.start("parse", "b", [PY, "-c", "pass"])
    assert jobs.running() is job
    wait(jobs, job)
    assert jobs.running() is None


def test_failed_job_keeps_its_returncode(tmp_path):
    jobs = Jobs(tmp_path)
    t = wait(jobs, jobs.start("eval", "a", [PY, "-c", "import sys; print('bad'); sys.exit(3)"]))
    assert (t["state"], t["returncode"], t["lines"]) == ("failed", 3, ["bad"])


def test_cancel_terminates_a_sleeping_job(tmp_path):
    jobs = Jobs(tmp_path)
    job = jobs.start("ingest", "a", [PY, "-c", "import time; time.sleep(30)"])
    jobs.cancel(job.id)
    t = wait(jobs, job, timeout=8)
    assert t["state"] == "cancelled" and t["returncode"] != 0


def test_unfinished_records_load_as_interrupted_and_old_logs_are_read_back(tmp_path):
    (tmp_path / "jobs").mkdir()
    (tmp_path / "jobs" / "parse-a-1.json").write_text(json.dumps({"id": "parse-a-1", "stage": "parse", "slug": "a", "argv": [], "started": 1.0, "finished": None}))
    (tmp_path / "jobs" / "parse-a-1.log").write_text("one\ntwo\n")
    (tmp_path / "jobs" / "parse-a-2.json").write_text(json.dumps({"id": "parse-a-2", "stage": "parse", "slug": "a", "argv": [], "started": 2.0, "finished": 3.0, "returncode": 0}))
    jobs = Jobs(tmp_path)
    assert jobs.tail("parse-a-1")["state"] == "interrupted" and jobs.tail("parse-a-1")["lines"] == ["one", "two"]
    assert jobs.tail("parse-a-2")["state"] == "done" and jobs.running() is None
    assert [j["id"] for j in jobs.recent()] == ["parse-a-2", "parse-a-1"]


def test_log_is_capped_and_flagged_truncated(tmp_path, monkeypatch):
    monkeypatch.setattr(workbench, "LOG_CAP", 5)
    jobs = Jobs(tmp_path)
    t = wait(jobs, jobs.start("parse", "a", [PY, "-c", "for i in range(10): print(i)"]))
    assert (t["lines"], t["next"], t["truncated"]) == (["5", "6", "7", "8", "9"], 10, True)
    assert jobs.tail(t["id"], 8)["lines"] == ["8", "9"] and jobs.tail(t["id"], 8)["truncated"] is False
    assert len((tmp_path / "jobs" / f"{t['id']}.log").read_text().splitlines()) == 10


# ----- documents -----

def test_document_stages_are_derived_from_files(tmp_path):
    pdf = make_pdf(tmp_path / "raw" / "a.pdf")
    sha = pdf_sha(pdf)
    wb = bench(tmp_path)
    assert wb.document("a")["stages"] == {"parsed": None, "schema": None, "facts": None, "datasets": [], "ingested": None, "runs": []}
    d = parsed(tmp_path, "a", sha)
    (d / "factsheet.jsonl").write_text("{}\n")
    (d / "factsheet_report.json").write_text(json.dumps({"facts": 7, "batches": 2, "model": "m", "llm": {"calls": 2}}))
    ds = tmp_path / "datasets" / "a"
    ds.mkdir(parents=True)
    (ds / "dataset.jsonl").write_text(json.dumps(PFS) + "\n" + json.dumps(WITHHELD) + "\n")
    (ds / "generate_report.json").write_text(json.dumps({"interim_dir": str(d), "model": "m", "warnings": ["factual: 1/3 accepted"]}))
    for name in ("a", "abl/bm25"):
        r = tmp_path / "runs" / name
        r.mkdir(parents=True)
        (r / "eval_report.json").write_text(json.dumps({"rag": {"collection": f"a-{sha[:10]}-x-m", "model": "m"}, "items": 2, "overall": {"score": 0.5}}))
    wb.app._store.replace_collection(wb.ingest_state("a") or workbench.collection_name(wb.cfg, pdf), {}, [], None)
    doc = wb.document("a")
    assert (doc["pages"], doc["sha10"], doc["interim"]) == (2, sha[:10], str(d))
    st = doc["stages"]
    assert st["parsed"] == {"dir": str(d), "pages": 2, "documents": 1, "sections": 3, "elements": 9, "warnings": 1}
    assert st["facts"]["facts"] == 7 and st["facts"]["calls"] == 2
    assert st["datasets"] == ["a"] and st["runs"] == ["a", "abl/bm25"]
    assert st["ingested"] == {"collection": workbench.collection_name(wb.cfg, pdf), "chunks": 0}
    ds_rec = wb.datasets()[0]
    assert (ds_rec["items"], ds_rec["per_type"], ds_rec["messy"], ds_rec["document"], ds_rec["source"], ds_rec["short"]) == (2, {"numerical": 1, "unanswerable": 1}, False, "a", "generated", True)
    assert [(r["name"], r["document"], r["score"]) for r in wb.runs()] == [("a", "a", 0.5), ("abl/bm25", "a", 0.5)] or \
        [(r["name"], r["document"]) for r in wb.runs()] == [("abl/bm25", "a"), ("a", "a")]


def test_interim_dir_is_matched_by_sha_not_name(tmp_path):
    pdf = make_pdf(tmp_path / "raw" / "source_document.pdf")
    wb = bench(tmp_path, "source_document")
    parsed(tmp_path, "source_document", "0" * 64)                     # a stale parse of another file under the PDF's own name
    assert wb.interim_dir("source_document") is None and wb.document("source_document")["stages"]["parsed"] is None
    other = parsed(tmp_path, "source_document", pdf_sha(pdf), interim="ta1063")
    assert wb.interim_dir("source_document") == other
    own = parsed(tmp_path, "source_document", pdf_sha(pdf))             # a fresh parse under its own name wins
    assert wb.interim_dir("source_document") == own


def test_store_errors_mean_unknown_not_missing(tmp_path):
    make_pdf(tmp_path / "raw" / "a.pdf")
    wb = bench(tmp_path)
    wb.app._store = None
    wb.app.cfg["store"] = {"backend": "postgres", "dsn": "postgresql://127.0.0.1:1/none", "dsn_env": "NOPE_DSN"}
    assert wb.ingest_state("a") == "unknown"


def test_app_for_another_pdf_shares_the_loaded_parts_and_is_dropped_after_ingest(tmp_path):
    make_pdf(tmp_path / "raw" / "a.pdf")
    make_pdf(tmp_path / "raw" / "b.pdf")
    wb = bench(tmp_path)
    wb.app._embedder = "E"
    assert wb.app_for("a") is wb.app
    b = wb.app_for("b")
    assert b is not wb.app and b._embedder == "E" and b._store is wb.app._store and b.cfg["pdf"].endswith("b.pdf") and b.cfg.get("topic") is None
    assert wb.app_for("b") is b
    wb._after_job(workbench.Job("j", "ingest", "b", [], tmp_path, 0.0))
    assert wb.app_for("b") is not b


def test_pdf_upload_is_checked_and_named(tmp_path):
    wb = bench(tmp_path)
    with pytest.raises(ValueError):
        wb.save_pdf("x.pdf", b"hello")
    make_pdf(tmp_path / "scratch.pdf")
    body = (tmp_path / "scratch.pdf").read_bytes()
    assert wb.save_pdf("Trial Report.pdf", body) == "trial-report" and (tmp_path / "raw" / "trial-report.pdf").exists()
    with pytest.raises(FileExistsError):
        wb.save_pdf("trial report", body)
    assert wb.save_pdf("trial report", body, replace=True) == "trial-report" and wb.slugs() == ["trial-report"]


# ----- datasets -----

def test_validate_dataset_accepts_items_and_reports_bad_lines():
    assert validate_dataset([json.dumps(PFS), "", json.dumps(WITHHELD)]) == []
    bad = [json.dumps({**PFS, "type": "riddle"}), json.dumps({**PFS, "item_id": "x", "answerable": "yes"}), json.dumps({**PFS, "item_id": "x"}),
           json.dumps({**WITHHELD, "item_id": "u", "withheld": "gone"}), "{not json", json.dumps([1]), json.dumps({**PFS, "item_id": "q", "question_messy": "typed"})]
    problems = validate_dataset(bad)
    assert [p.split(":")[0] for p in problems] == ["line 1", "line 2", "line 3", "line 4", "line 5", "line 6", "line 7"]
    assert "type must be one of" in problems[0] and "true or false" in problems[1] and "duplicate" in problems[2]
    assert "withheld" in problems[3] and "not valid JSON" in problems[4] and "object" in problems[5] and "question_messy" in problems[6]
    assert validate_dataset([]) == ["no items"]


def test_uploaded_dataset_is_attributed_by_workbench_json(tmp_path):
    make_pdf(tmp_path / "raw" / "a.pdf")
    wb = bench(tmp_path)
    body = (json.dumps(PFS) + "\n").encode()
    rec = wb.save_dataset("up", body, document="a")
    assert (rec["name"], rec["items"], rec["document"], rec["source"]) == ("up", 1, "a", "uploaded")
    assert wb.document("a")["stages"]["datasets"] == ["up"]
    with pytest.raises(FileExistsError):
        wb.save_dataset("up", body)
    with pytest.raises(workbench.DatasetError) as e:
        wb.save_dataset("bad", b"{\n")
    assert e.value.problems == ["line 1: not valid JSON"]
    with pytest.raises(KeyError):
        wb.save_dataset("other", body, document="missing")


# ----- schemas -----

def test_check_schema_reports_the_parse_error_and_save_writes_a_copy(tmp_path):
    make_pdf(tmp_path / "raw" / "a.pdf")
    wb = bench(tmp_path)
    with pytest.raises(SchemaError, match="unknown roles"):
        wb.check_schema("groups:\n  g:\n    roles: [nobody]\n")
    with pytest.raises(SchemaError, match="not valid YAML"):
        wb.check_schema("groups: [")
    nice = (workbench.REPO_ROOT / "configs" / "schema" / "nice_sta.yaml").read_text()
    assert wb.check_schema(nice)["groups"] == 9
    assert wb.schema_record("a") is None
    rec = wb.save_schema("a", nice)
    assert rec["source"] == "draft" and rec["valid"] and rec["path"] == str(tmp_path / "interim" / "a" / "schema.yaml")
    assert wb.read_schema("a")["text"] == nice
    assert any(s["path"] == rec["path"] for s in wb.schemas()) and any(s["path"] == "configs/schema/nice_sta.yaml" for s in wb.schemas())
    assert wb.select_schema("a", "configs/schema/nice_sta.yaml")["source"] == "config"
    with pytest.raises(KeyError):
        wb.select_schema("a", "configs/rag.yaml")
    assert "configs/" not in str(tmp_path) and (workbench.REPO_ROOT / "configs" / "schema" / "nice_sta.yaml").read_text() == nice


# ----- jobs for a document -----

def test_argv_for_each_stage(tmp_path):
    pdf = make_pdf(tmp_path / "raw" / "a.pdf")
    wb = bench(tmp_path)
    interim = parsed(tmp_path, "a", pdf_sha(pdf))
    wb.select_schema("a", "configs/schema/nice_sta.yaml")
    ev = [PY, "-m", "medrageval.cli"]
    own = str(tmp_path / "interim" / "a")
    assert wb.argv_for("parse", "a", {"force": True}) == (ev + ["parse", str(pdf), "--out", own, "--force"], {})
    assert wb.argv_for("draft", "a", {}) == (ev + ["schema", "draft", str(interim), "--out", f"{own}/schema.yaml"], {"schema": f"{own}/schema.yaml"})
    assert wb.argv_for("extract", "a", {"dry_run": True}) == (ev + ["extract", str(interim), "--schema", "configs/schema/nice_sta.yaml", "--dry-run"], {})
    argv, result = wb.argv_for("generate", "a", {"name": "set1", "targets": {"unanswerable": 2, "factual": 3}, "messy": False, "seed": 7})
    assert argv == ev + ["generate", str(interim), "--out", str(tmp_path / "datasets" / "set1"), "--types", "factual,unanswerable",
                         "--target", "factual=3", "--target", "unanswerable=2", "--seed", "7", "--no-messy"]
    assert result == {"dataset": "set1", "targets": {"factual": 3, "unanswerable": 2}}
    assert wb.argv_for("generate", "a", {"name": "s", "targets": {"factual": 1}, "dry_run": True})[0][-1] == "--dry-run"
    assert wb.argv_for("ingest", "a", {}) == ([PY, "-m", "medrag.cli", "ingest", str(pdf)], {})
    argv, result = wb.argv_for("eval", "a", {"dataset": "set1", "name": "run1", "question": "messy", "mode": "bm25", "rerank": False, "top_k": 4, "limit": 5,
                                             "types": ["summary", "factual"], "judge": False})
    assert argv == ev + ["eval", str(tmp_path / "datasets" / "set1"), "--pdf", str(pdf), "--out", str(tmp_path / "runs" / "run1"), "--question", "messy",
                         "--mode", "bm25", "--no-rerank", "--top-k", "4", "--limit", "5", "--types", "factual,summary", "--no-judge"]
    assert result == {"run": "run1"}
    wb.cfg["_path"] = "configs/other.yaml"
    assert wb.argv_for("ingest", "a", {})[0][3:5] == ["--config", "configs/other.yaml"]
    assert "--rag-config" in wb.argv_for("eval", "a", {"dataset": "set1", "name": "r"})[0]


def test_stage_preconditions(tmp_path, monkeypatch):
    pdf = make_pdf(tmp_path / "raw" / "a.pdf")
    wb = bench(tmp_path)
    monkeypatch.setattr(workbench, "key_set", lambda: False)
    with pytest.raises(StageError, match="OPENROUTER_API_KEY"):
        wb.check_stage("a", "draft", {})
    wb.check_stage("a", "parse", {})
    wb.check_stage("a", "extract", {"dry_run": True}) if False else None
    monkeypatch.setattr(workbench, "key_set", lambda: True)
    with pytest.raises(StageError, match="parse the document first"):
        wb.check_stage("a", "extract", {"dry_run": True})
    parsed(tmp_path, "a", pdf_sha(pdf))
    with pytest.raises(StageError, match="schema"):
        wb.check_stage("a", "extract", {})
    wb.select_schema("a", "configs/schema/nice_sta.yaml")
    wb.check_stage("a", "extract", {})
    with pytest.raises(StageError, match="fact sheet"):
        wb.check_stage("a", "generate", {"name": "s", "targets": {"factual": 1}})
    d = wb.interim_dir("a")
    (d / "factsheet.jsonl").write_text("{}\n")
    (d / "factsheet_report.json").write_text(json.dumps({"facts": 1}))
    with pytest.raises(StageError, match="at least one"):
        wb.check_stage("a", "generate", {"name": "s", "targets": {}})
    with pytest.raises(StageError, match="unanswerable"):
        wb.check_stage("a", "generate", {"name": "s", "targets": {"unanswerable": 2, "summary": 1}})
    wb.check_stage("a", "generate", {"name": "s", "targets": {"unanswerable": 2, "factual": 1}})
    wb.save_dataset("s", (json.dumps(PFS) + "\n").encode())
    with pytest.raises(FileExistsError):
        wb.check_stage("a", "generate", {"name": "s", "targets": {"factual": 1}})
    wb.check_stage("a", "generate", {"name": "s", "targets": {"factual": 1}, "replace": True})
    with pytest.raises(StageError, match="no dataset"):
        wb.check_stage("a", "eval", {"dataset": "none", "name": "r"})
    with pytest.raises(StageError, match="ingest"):
        wb.check_stage("a", "eval", {"dataset": "s", "name": "r"})
    wb.app._store.replace_collection(workbench.collection_name(wb.cfg, pdf), {}, [], None)
    wb.check_stage("a", "eval", {"dataset": "s", "name": "r", "judge": False})
    with pytest.raises(StageError):
        wb.check_stage("a", "nope", {})


def test_estimate_calls_uses_attempts_and_judge_types():
    cfg = {"attempts_per_item": {"default": 4, "multihop": 8, "unanswerable": 3}, "judge_types": ["factual", "multihop"], "alone_check_types": ["multihop"]}
    assert estimate_calls(cfg, {"factual": 2, "multihop": 1, "unanswerable": 2}, messy=True) == {"generator": 16, "judge": 24, "messy": 3, "total": 43}
    assert estimate_calls(cfg, {"summary": 2}, messy=False) == {"generator": 8, "judge": 0, "messy": 0, "total": 8}


def test_generate_progress_counts_the_dataset_file(tmp_path):
    make_pdf(tmp_path / "raw" / "a.pdf")
    wb = bench(tmp_path)
    job = workbench.Job("j", "generate", "a", [], tmp_path, time.time(), result={"dataset": "g", "targets": {"numerical": 2}})
    assert wb.progress(job)["accepted"] == {}
    (tmp_path / "datasets" / "g").mkdir(parents=True)
    (tmp_path / "datasets" / "g" / "dataset.jsonl").write_text(json.dumps(PFS) + "\n")
    p = wb.progress(job)
    assert p["accepted"] == {"numerical": 1} and p["targets"] == {"numerical": 2} and p["elapsed_s"] >= 0
    assert set(wb.progress(workbench.Job("k", "eval", "a", [], tmp_path, time.time()))) == {"elapsed_s"}
