import json
import sys
import time

import numpy as np
import pymupdf
import pytest
from fastapi.testclient import TestClient
from medrag.app import App, collection_name
from medrag.chunk import Chunk
from medrag.generate import ABSTAIN
from medrag.llm import LLMError, make_llm
from medrag.store import MemoryStore
from test_rag_eval import PFS, WITHHELD

from medrageval import server
from medrageval.generate.generate import TYPES
from medrageval.server import create_app, load_run
from medrageval.workbench import Workbench, pdf_sha

RC = {"mode": "hybrid", "top_k": 2, "candidates": 4, "rrf_k": 60, "rerank": True, "rerank_candidates": 4, "max_context_chars": 200}
CHUNKING = {"target_tokens": 400, "max_tokens": 512}


class Embedder:
    model, query_prefix = "fixed", ""

    def embed(self, texts):
        return np.ones((len(texts), 4), dtype=np.float32) / 2


class ReverseReranker:
    def scores(self, query, chunks):
        return [float(i) for i in range(len(chunks))]


class FakeLLM:
    model, base_url = "fake", "http://127.0.0.1:9/v1"

    def __init__(self, reply="Median PFS was 7.3 months [1]."):
        self.reply, self.prompts = reply, []

    def complete(self, messages, tag=""):
        self.prompts.append(messages)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply, {"cached": True}


def mk(cid, body, page):
    text = f"Doc | Sec | p. {page}\n{body}"
    return Chunk(cid, "Doc", ["Trial", "Sec"], page, page + 1, "text", body, text, len(text) // 4)


CHUNKS = [mk("c1", "Median PFS was 7.3 months with capivasertib.", 1), mk("c2", "Fatigue was reported in 58% of patients.", 2),
          mk("c3", "x" * 300, 3)]


def make_pdf(path, pages=3):
    doc = pymupdf.open()
    for _ in range(pages):
        doc.new_page()
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    return path


@pytest.fixture
def root(tmp_path):
    """A data root with one PDF, raw/bundle.pdf, and a run under runs/run that was made over it."""
    r = tmp_path / "data"
    pdf = make_pdf(r / "raw" / "bundle.pdf")
    d = r / "runs" / "run"
    d.mkdir(parents=True)
    (d / "eval_report.json").write_text(json.dumps({
        "rag": {"model": "m", "base_url": "http://10.0.0.1:8003/v1", "retrieval": RC, "collection": f"bundle-{pdf_sha(pdf)[:10]}-x-m"}, "judge_model": "j",
        "overall": {"items": 1, "score": 1.0}, "answerable": {"items": 1}, "unanswerable": {"items": 0},
        "per_type": {"factual": {"items": 1, "score": 1.0}}, "warnings": []}))
    (d / "responses.jsonl").write_text(json.dumps({"question": "q", "category": "correct", "candidates": [{"chunk_id": "c1"}]}) + "\n")
    return r


def client(root, llm=None, examples=(), ingested=True, run="run"):
    pdf = root / "raw" / "bundle.pdf"
    cfg = {"pdf": str(pdf), "topic": "t", "chunking": CHUNKING, "retrieval": RC, "llm": {"model": "fake", "base_url": "http://127.0.0.1:9/v1"},
           "embedding": {"model": "e"}, "demo": {"examples": list(examples)}}
    store = MemoryStore(dim=4)
    if ingested:
        store.replace_collection(collection_name(cfg, pdf), {}, CHUNKS, Embedder().embed([c.text for c in CHUNKS]))
    app = App(cfg)
    app._store, app._embedder, app._reranker, app._llm = store, Embedder(), ReverseReranker(), llm or FakeLLM()
    return TestClient(create_app(app, root / "runs" / run, root=root))


def fake_job(monkeypatch, code):
    monkeypatch.setattr(Workbench, "argv_for", lambda self, stage, slug, opts: ([sys.executable, "-c", code], {"stage": stage}))


def finished(c, job_id, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        t = c.get(f"/api/jobs/{job_id}").json()
        if t["state"] != "running":
            return t
        time.sleep(0.02)
    raise AssertionError("job still running")


def test_page_is_served(root):
    r = client(root).get("/")
    assert r.status_code == 200 and "MedRAG" in r.text


def test_status_reports_the_corpus_and_a_dead_reader(root):
    s = client(root, examples=["Who?"]).get("/api/status").json()
    assert (s["chunks"], s["pages"], s["reader_up"], s["examples"], s["document"], s["run"], s["job"]) == (3, 3, False, ["Who?"], "bundle", "run", None)
    assert s["ingested"]["chunks"] == 3 and s["postgres_up"] is True and isinstance(s["key_set"], bool)
    assert "base_url" not in s


def test_retrieve_then_answer_cites_the_passages_the_model_saw(root):
    llm = FakeLLM()
    c = client(root, llm)
    hits = c.post("/api/retrieve", json={"question": "median PFS", "rerank": False}).json()["hits"]
    assert len(hits) == 2 and hits[0]["n"] == 1 and hits[0]["section"] == ["Trial", "Sec"] and hits[0]["pages"] == [1, 2]
    assert "rerank" not in hits[0]["ranks"]
    a = c.post("/api/answer", json={"question": "median PFS", "chunk_ids": [h["chunk_id"] for h in hits], "document": "bundle"}).json()
    assert a["answer"].endswith("[1].") and a["cached"] is True and a["abstained"] is False
    assert a["chunk_ids"] == [h["chunk_id"] for h in hits if h["n"]]
    assert CHUNKS[0].text in llm.prompts[0][1]["content"] or CHUNKS[1].text in llm.prompts[0][1]["content"]


def test_retrieve_modes_and_rerank_toggle(root):
    c = client(root)
    assert set(c.post("/api/retrieve", json={"question": "fatigue", "mode": "bm25", "rerank": False}).json()["hits"][0]["ranks"]) == {"bm25", "fused"}
    assert set(c.post("/api/retrieve", json={"question": "fatigue", "mode": "dense", "rerank": False}).json()["hits"][0]["ranks"]) == {"dense", "fused"}
    assert "rerank" in c.post("/api/retrieve", json={"question": "fatigue", "rerank": True}).json()["hits"][0]["ranks"]
    assert c.post("/api/retrieve", json={"question": "fatigue", "mode": "nope"}).status_code == 422
    assert c.post("/api/retrieve", json={"question": "   "}).status_code == 422
    assert c.post("/api/retrieve", json={"question": "fatigue", "document": "nope"}).status_code == 404


def test_a_passage_over_the_context_budget_is_left_unnumbered(root):
    hits = client(root).post("/api/retrieve", json={"question": "anything", "mode": "dense", "rerank": True}).json()["hits"]
    assert [(h["chunk_id"], h["n"]) for h in hits] == [("c3", 1), ("c2", None)]     # the reranker puts the long c3 first


def test_a_question_matching_nothing_returns_no_hits(root):
    assert client(root).post("/api/retrieve", json={"question": "zzzz", "mode": "bm25", "rerank": False}).json()["hits"] == []


def test_abstention_is_flagged(root):
    a = client(root, FakeLLM(ABSTAIN)).post("/api/answer", json={"question": "q", "chunk_ids": ["c1"]}).json()
    assert a["abstained"] is True


def test_unknown_chunk_is_a_404(root):
    assert client(root).post("/api/answer", json={"question": "q", "chunk_ids": ["nope"]}).status_code == 404


def test_reader_failure_is_a_503_without_the_server_address(root):
    r = client(root, FakeLLM(LLMError("fake at http://10.0.0.1:8003/v1: refused"))).post("/api/answer", json={"question": "q", "chunk_ids": ["c1"]})
    assert r.status_code == 503 and "10.0.0.1" not in r.text


def test_ask_on_an_uningested_document_is_409(root):
    c = client(root, ingested=False)
    s = c.get("/api/status").json()
    assert (s["chunks"], s["ingested"], s["postgres_up"]) == (0, None, True)
    assert c.post("/api/retrieve", json={"question": "fatigue"}).status_code == 409
    assert c.post("/api/answer", json={"question": "q", "chunk_ids": ["c1"]}).status_code == 409


def test_pdf_pages_render_and_stay_in_range(root):
    c = client(root)
    r = c.get("/api/documents/bundle/page/3.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png" and r.content[:4] == b"\x89PNG"
    assert c.get("/api/documents/bundle/page/0.png").status_code == 404 and c.get("/api/documents/bundle/page/4.png").status_code == 404
    assert c.get("/api/documents/other/page/1.png").status_code == 404


def test_run_keeps_results_and_drops_candidates_and_the_server_address(root):
    c = client(root)
    r = c.get("/api/runs/run")
    assert r.status_code == 200 and "10.0.0.1" not in r.text
    body = r.json()
    assert body["overall"]["score"] == 1.0 and body["model"] == "m" and body["items"] == [{"question": "q", "category": "correct"}]
    assert body["document"] == "bundle" and body["name"] == "run"
    assert [(x["name"], x["document"], x["score"]) for x in c.get("/api/runs").json()] == [("run", "bundle", 1.0)]
    assert c.get("/api/runs/nope").status_code == 404 and c.get("/api/runs/..%2F..%2Fraw").status_code == 404


def test_missing_run_is_a_404_and_asking_still_works(root, tmp_path):
    c = client(root, run="absent")
    assert c.get("/api/runs/absent").status_code == 404 and c.get("/api/status").status_code == 200
    assert c.get("/api/status").json()["run"] is None
    assert load_run(tmp_path / "absent") is None


# ----- build -----

def test_documents_lists_the_pdf_with_derived_stages(root):
    docs = client(root).get("/api/documents").json()
    assert [d["slug"] for d in docs] == ["bundle"] and docs[0]["pages"] == 3
    st = docs[0]["stages"]
    assert (st["parsed"], st["schema"], st["facts"], st["datasets"], st["runs"], st["ingested"]["chunks"]) == (None, None, None, [], ["run"], 3)
    assert client(root).get("/api/documents/bundle").json()["slug"] == "bundle"
    assert client(root).get("/api/documents/nope").status_code == 404


def test_pdf_upload_checks_magic_size_and_existing_slug(root, tmp_path, monkeypatch):
    c = client(root)
    body = make_pdf(tmp_path / "scratch.pdf", pages=2).read_bytes()
    assert c.post("/api/documents?name=New%20Doc.pdf", content=b"hello world").status_code == 415
    r = c.post("/api/documents?name=New%20Doc.pdf", content=body)
    assert r.status_code == 201 and r.json()["slug"] == "new-doc" and r.json()["pages"] == 2
    assert c.post("/api/documents?name=new-doc", content=body).status_code == 409
    assert c.post("/api/documents?name=new-doc&replace=1", content=body).status_code == 201
    assert [d["slug"] for d in c.get("/api/documents").json()] == ["bundle", "new-doc"]
    assert c.get("/api/status?document=new-doc").json()["ingested"] is None
    monkeypatch.setattr(server, "PDF_MAX", 10)
    assert c.post("/api/documents?name=big", content=body).status_code == 413


def test_dataset_upload_rejects_bad_lines_and_lists_good_ones(root):
    c = client(root)
    r = c.post("/api/datasets?name=up&document=bundle", content=(json.dumps({**PFS, "type": "riddle"}) + "\n{\n").encode())
    assert r.status_code == 422 and len(r.json()["detail"]) == 2 and "line 2: not valid JSON" in r.json()["detail"]
    good = (json.dumps(PFS) + "\n" + json.dumps(WITHHELD) + "\n").encode()
    r = c.post("/api/datasets?name=up&document=bundle", content=good)
    assert r.status_code == 201 and r.json()["items"] == 2 and r.json()["document"] == "bundle" and r.json()["source"] == "uploaded"
    assert c.post("/api/datasets?name=up", content=good).status_code == 409
    assert c.post("/api/datasets?name=up&replace=1", content=good).status_code == 201
    assert c.post("/api/datasets?name=bad%20name", content=good).status_code == 422
    assert c.post("/api/datasets?name=x&document=nope", content=good).status_code == 404
    assert [d["name"] for d in c.get("/api/datasets").json()] == ["up"]
    assert c.get("/api/documents/bundle").json()["stages"]["datasets"] == ["up"]


def test_schema_check_put_and_select(root):
    c = client(root)
    assert c.get("/api/documents/bundle/schema").status_code == 404
    r = c.post("/api/documents/bundle/schema/check", json={"text": "groups:\n  g:\n    roles: [nobody]\n"})
    assert r.status_code == 422 and "unknown roles" in r.json()["detail"]
    nice = next(s for s in c.get("/api/schemas").json() if s["path"].endswith("nice_sta.yaml"))
    assert nice["valid"] and nice["source"] == "config" and nice["groups"] == 9
    text = open(nice["path"]).read()
    assert c.post("/api/documents/bundle/schema/check", json={"text": text}).json()["groups"] == 9
    r = c.put("/api/documents/bundle/schema", json={"text": text})
    assert r.status_code == 200 and r.json()["source"] == "draft" and (root / "interim" / "bundle" / "schema.yaml").read_text() == text
    assert c.get("/api/documents/bundle/schema").json()["text"] == text
    assert c.put("/api/documents/bundle/schema", json={"text": "groups: 3"}).status_code == 422
    assert c.post("/api/documents/bundle/schema/select", json={"path": nice["path"]}).json()["source"] == "config"
    assert c.post("/api/documents/bundle/schema/select", json={"path": "configs/rag.yaml"}).status_code == 404
    assert c.get("/api/documents/bundle").json()["stages"]["schema"]["path"] == nice["path"]


def test_build_config_and_estimate(root):
    c = client(root)
    cfg = c.get("/api/build/config").json()
    assert cfg["types"] == list(TYPES) and set(cfg["models"]) == {"generator", "judge", "facts", "eval_judge", "reader"}
    est = c.post("/api/build/estimate", json={"targets": {"factual": 1}, "messy": False}).json()
    assert est["generator"] == cfg["attempts_per_item"]["default"] and est["messy"] == 0
    assert c.post("/api/build/estimate", json={"targets": {"riddle": 1}}).status_code == 422


def test_job_route_streams_a_fake_command(root, monkeypatch):
    fake_job(monkeypatch, "print('one'); print('two')")
    c = client(root)
    r = c.post("/api/documents/bundle/jobs", json={"stage": "parse", "options": {"force": True}})
    assert r.status_code == 202 and r.json()["state"] == "running" and r.json()["stage"] == "parse"
    t = finished(c, r.json()["id"])
    assert (t["returncode"], t["lines"], t["result"], t["progress"].keys()) == (0, ["one", "two"], {"stage": "parse"}, {"elapsed_s"})
    assert c.get(f"/api/jobs/{t['id']}?from=1").json()["lines"] == ["two"]
    assert [j["id"] for j in c.get("/api/jobs").json()] == [t["id"]] and "lines" not in c.get("/api/jobs").json()[0]
    assert c.get("/api/jobs/nope").status_code == 404 and c.post("/api/jobs/nope/cancel").status_code == 404
    assert c.post("/api/documents/bundle/jobs", json={"stage": "parse", "options": {"force": "maybe"}}).status_code == 422
    assert c.post("/api/documents/bundle/jobs", json={"stage": "fly"}).status_code == 422


def test_job_route_returns_409_while_busy_and_cancel_stops_it(root, monkeypatch):
    fake_job(monkeypatch, "import time; time.sleep(30)")
    c = client(root)
    job = c.post("/api/documents/bundle/jobs", json={"stage": "parse"}).json()
    assert c.post("/api/documents/bundle/jobs", json={"stage": "parse"}).status_code == 409
    assert c.get("/api/status").json()["job"] == job["id"]
    assert c.post(f"/api/jobs/{job['id']}/cancel").status_code == 200
    assert finished(c, job["id"])["state"] == "cancelled"


def test_job_route_returns_422_on_a_missing_precondition(root, monkeypatch):
    from medrageval import workbench
    monkeypatch.setattr(workbench, "key_set", lambda: True)
    c = client(root)
    r = c.post("/api/documents/bundle/jobs", json={"stage": "extract"})
    assert r.status_code == 422 and "parse the document first" in r.json()["detail"]
    r = c.post("/api/documents/bundle/jobs", json={"stage": "eval", "options": {"dataset": "none", "name": "r"}})
    assert r.status_code == 422 and "no dataset" in r.json()["detail"]
    monkeypatch.setattr(workbench, "key_set", lambda: False)
    r = c.post("/api/documents/bundle/jobs", json={"stage": "draft"})
    assert r.status_code == 422 and "OPENROUTER_API_KEY" in r.json()["detail"]
    assert c.post("/api/documents/bundle/jobs", json={"stage": "ingest"}).status_code == 202   # no key needed


def test_ingest_job_completion_drops_the_cached_app(root, monkeypatch):
    fake_job(monkeypatch, "print('stored')")
    c = client(root)
    wb = c.app.state.workbench
    c.get("/api/status")
    assert wb.app._retriever is not None
    job = c.post("/api/documents/bundle/jobs", json={"stage": "ingest"}).json()
    finished(c, job["id"])
    assert wb.app._retriever is None
    assert c.get("/api/status").json()["chunks"] == 3


def test_make_llm_reads_max_retries():
    assert make_llm({"model": "m", "base_url": "http://127.0.0.1:9/v1", "max_retries": 0}, False).client.max_retries == 0
    assert make_llm({"model": "m", "base_url": "http://127.0.0.1:9/v1"}, False).client.max_retries == 3
