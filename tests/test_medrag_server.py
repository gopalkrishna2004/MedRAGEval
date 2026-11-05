import json

import numpy as np
import pymupdf
import pytest
from fastapi.testclient import TestClient

from medrag.app import App
from medrag.chunk import Chunk
from medrag.generate import ABSTAIN
from medrag.llm import LLMError, make_llm
from medrag.retrieve import Retriever
from medrag.server import create_app, load_run
from medrag.store import MemoryStore

RC = {"mode": "hybrid", "top_k": 2, "candidates": 4, "rrf_k": 60, "rerank": True, "rerank_candidates": 4, "max_context_chars": 200}


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


@pytest.fixture
def pdf(tmp_path):
    doc = pymupdf.open()
    for _ in range(3):
        doc.new_page()
    path = tmp_path / "bundle.pdf"
    doc.save(path)
    return path


@pytest.fixture
def run_dir(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    (d / "eval_report.json").write_text(json.dumps({
        "rag": {"model": "m", "base_url": "http://10.0.0.1:8003/v1", "retrieval": RC}, "judge_model": "j",
        "overall": {"items": 1, "score": 1.0}, "answerable": {"items": 1}, "unanswerable": {"items": 0},
        "per_type": {"factual": {"items": 1, "score": 1.0}}, "warnings": []}))
    (d / "responses.jsonl").write_text(json.dumps({"question": "q", "category": "correct", "candidates": [{"chunk_id": "c1"}]}) + "\n")
    return d


def client(pdf, run_dir, llm=None, examples=()):
    store = MemoryStore(dim=4)
    store.replace_collection("t", {}, CHUNKS, Embedder().embed([c.text for c in CHUNKS]))
    app = App({"pdf": str(pdf), "topic": "t", "retrieval": RC, "llm": {"model": "fake", "base_url": "http://127.0.0.1:9/v1"},
               "embedding": {"model": "e"}, "demo": {"examples": list(examples)}})
    app._retriever, app._llm = Retriever(store, "t", RC, Embedder(), ReverseReranker()), llm or FakeLLM()
    return TestClient(create_app(app, run_dir))


def test_page_is_served(pdf, run_dir):
    r = client(pdf, run_dir).get("/")
    assert r.status_code == 200 and "MedRAG" in r.text


def test_status_reports_the_corpus_and_a_dead_reader(pdf, run_dir):
    s = client(pdf, run_dir, examples=["Who?"]).get("/api/status").json()
    assert (s["chunks"], s["pages"], s["reader_up"], s["examples"]) == (3, 3, False, ["Who?"])
    assert "base_url" not in s


def test_retrieve_then_answer_cites_the_passages_the_model_saw(pdf, run_dir):
    llm = FakeLLM()
    c = client(pdf, run_dir, llm)
    hits = c.post("/api/retrieve", json={"question": "median PFS", "rerank": False}).json()["hits"]
    assert len(hits) == 2 and hits[0]["n"] == 1 and hits[0]["section"] == ["Trial", "Sec"] and hits[0]["pages"] == [1, 2]
    assert "rerank" not in hits[0]["ranks"]
    a = c.post("/api/answer", json={"question": "median PFS", "chunk_ids": [h["chunk_id"] for h in hits]}).json()
    assert a["answer"].endswith("[1].") and a["cached"] is True and a["abstained"] is False
    assert a["chunk_ids"] == [h["chunk_id"] for h in hits if h["n"]]
    assert CHUNKS[0].text in llm.prompts[0][1]["content"] or CHUNKS[1].text in llm.prompts[0][1]["content"]


def test_retrieve_modes_and_rerank_toggle(pdf, run_dir):
    c = client(pdf, run_dir)
    assert set(c.post("/api/retrieve", json={"question": "fatigue", "mode": "bm25", "rerank": False}).json()["hits"][0]["ranks"]) == {"bm25", "fused"}
    assert set(c.post("/api/retrieve", json={"question": "fatigue", "mode": "dense", "rerank": False}).json()["hits"][0]["ranks"]) == {"dense", "fused"}
    assert "rerank" in c.post("/api/retrieve", json={"question": "fatigue", "rerank": True}).json()["hits"][0]["ranks"]
    assert c.post("/api/retrieve", json={"question": "fatigue", "mode": "nope"}).status_code == 422
    assert c.post("/api/retrieve", json={"question": "   "}).status_code == 422


def test_a_passage_over_the_context_budget_is_left_unnumbered(pdf, run_dir):
    hits = client(pdf, run_dir).post("/api/retrieve", json={"question": "anything", "mode": "dense", "rerank": True}).json()["hits"]
    assert [(h["chunk_id"], h["n"]) for h in hits] == [("c3", 1), ("c2", None)]     # the reranker puts the long c3 first


def test_a_question_matching_nothing_returns_no_hits(pdf, run_dir):
    assert client(pdf, run_dir).post("/api/retrieve", json={"question": "zzzz", "mode": "bm25", "rerank": False}).json()["hits"] == []


def test_abstention_is_flagged(pdf, run_dir):
    a = client(pdf, run_dir, FakeLLM(ABSTAIN)).post("/api/answer", json={"question": "q", "chunk_ids": ["c1"]}).json()
    assert a["abstained"] is True


def test_unknown_chunk_is_a_404(pdf, run_dir):
    assert client(pdf, run_dir).post("/api/answer", json={"question": "q", "chunk_ids": ["nope"]}).status_code == 404


def test_reader_failure_is_a_503_without_the_server_address(pdf, run_dir):
    r = client(pdf, run_dir, FakeLLM(LLMError("fake at http://10.0.0.1:8003/v1: refused"))).post("/api/answer", json={"question": "q", "chunk_ids": ["c1"]})
    assert r.status_code == 503 and "10.0.0.1" not in r.text


def test_pdf_pages_render_and_stay_in_range(pdf, run_dir):
    c = client(pdf, run_dir)
    r = c.get("/api/page/3.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png" and r.content[:4] == b"\x89PNG"
    assert c.get("/api/page/0.png").status_code == 404 and c.get("/api/page/4.png").status_code == 404


def test_run_keeps_results_and_drops_candidates_and_the_server_address(pdf, run_dir):
    r = client(pdf, run_dir).get("/api/run")
    assert r.status_code == 200 and "10.0.0.1" not in r.text
    body = r.json()
    assert body["overall"]["score"] == 1.0 and body["model"] == "m" and body["items"] == [{"question": "q", "category": "correct"}]


def test_missing_run_is_a_404_and_asking_still_works(pdf, tmp_path):
    c = client(pdf, tmp_path / "absent")
    assert c.get("/api/run").status_code == 404 and c.get("/api/status").status_code == 200
    assert load_run(tmp_path / "absent") is None


def test_make_llm_reads_max_retries():
    assert make_llm({"model": "m", "base_url": "http://127.0.0.1:9/v1", "max_retries": 0}, False).client.max_retries == 0
    assert make_llm({"model": "m", "base_url": "http://127.0.0.1:9/v1"}, False).client.max_retries == 3
