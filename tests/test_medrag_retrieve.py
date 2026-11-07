import numpy as np
import pytest
import re
import zlib

from medrag.chunk import Chunk
from medrag.lexical import tokenize
from medrag.retrieve import Retriever, rrf
from medrag.store import MemoryStore

CFG = {"mode": "hybrid", "top_k": 3, "candidates": 10, "rrf_k": 60, "rerank": False, "rerank_candidates": 10}


class HashEmbedder:
    """Bag of words hashed into 64 dimensions: similar wording, similar vector."""
    model, query_prefix = "hash-64", ""

    def embed(self, texts):
        m = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                m[i, zlib.crc32(tok.encode()) % 64] += 1
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)


class ReverseReranker:
    model = "reverse"

    def scores(self, query, chunks):
        return [float(i) for i in range(len(chunks))]     # the last candidate scores highest


def mk(cid, body, page):
    text = f"Doc | Sec | p. {page}\n{body}"
    return Chunk(cid, "Doc", ["Sec"], page, page, "text", body, text, len(text) // 4)


CHUNKS = [mk("c1", "The company base-case ICER was £24,567 per QALY gained.", 120),
          mk("c2", "The EAG base-case ICER was £52,681 per QALY gained.", 537),
          mk("c3", "Median PFS was 7.3 months with capivasertib versus 3.1 months.", 21),
          mk("c4", "Patient organisations described the burden of advanced breast cancer.", 340)]


def store():
    s = MemoryStore(dim=64)
    s.replace_collection("t", {}, CHUNKS, HashEmbedder().embed([c.text for c in CHUNKS]))
    return s


def test_rrf_fuses_two_rankings():
    assert [cid for cid, _ in rrf([["a", "b", "c"], ["b", "d"]], 60)][:2] == ["b", "a"]


def test_modes_and_ranks():
    r = Retriever(store(), "t", CFG, HashEmbedder())
    bm = r.search("EAG ICER £52,681", mode="bm25")
    assert bm[0].chunk.chunk_id == "c2" and set(bm[0].ranks) == {"bm25", "fused"}
    de = r.search("EAG ICER £52,681", mode="dense")
    assert de[0].chunk.chunk_id == "c2" and set(de[0].ranks) == {"dense", "fused"}
    hy = r.search("median PFS months", mode="hybrid", k=2)
    assert hy[0].chunk.chunk_id == "c3" and hy[0].ranks["bm25"] == 1 and hy[0].ranks["dense"] == 1 and len(hy) == 2
    with pytest.raises(ValueError):
        r.search("x", mode="nope")
    with pytest.raises(RuntimeError):
        Retriever(store(), "t", CFG).search("x", mode="dense")       # no embedder


def test_reranker_reorders_the_fused_candidates():
    r = Retriever(store(), "t", CFG, HashEmbedder(), ReverseReranker())
    plain = r.search("ICER per QALY", rerank=False)
    reranked = r.search("ICER per QALY", rerank=True)
    assert [h.chunk.chunk_id for h in reranked] != [h.chunk.chunk_id for h in plain]
    assert reranked[0].ranks["rerank"] == 1 and reranked[0].ranks["fused"] > 1


def test_retrieve_returns_the_candidate_pool_in_fused_order():
    r = Retriever(store(), "t", {**CFG, "top_k": 1, "candidates": 3}, HashEmbedder(), ReverseReranker())
    hits, pool = r.retrieve("ICER per QALY", rerank=True)
    assert len(hits) == 1 and len(pool) == 3                                   # the pool is `candidates` wide, whatever top_k is
    assert [c.chunk_id for c in pool] == [h.chunk.chunk_id for h in r.search("ICER per QALY", rerank=False, k=3)]   # fused order, unreranked
    assert hits[0].chunk.chunk_id in {c.chunk_id for c in pool}


def test_memory_store_persists(tmp_path):
    s = MemoryStore(tmp_path / "ms", dim=64)
    s.replace_collection("t", {"pdf_path": "x.pdf"}, CHUNKS, HashEmbedder().embed([c.text for c in CHUNKS]))
    again = MemoryStore(tmp_path / "ms", dim=64)
    assert again.collection("t")["n_chunks"] == 4 and again.load_chunks("t") == CHUNKS
    assert again.dense_search("t", HashEmbedder().embed(["EAG ICER £52,681"])[0], 1)[0][0] == "c2"
    with pytest.raises(RuntimeError):
        Retriever(again, "missing", CFG)


def test_rrf_weights_scale_each_ranking():
    assert rrf([["a"], ["b"]], 60, [1.0, 0.5])[0][0] == "a"
    assert rrf([["a"], ["b"]], 60, [0.5, 1.0])[0][0] == "b"


def test_dense_weight_changes_the_fused_order():
    r = Retriever(store(), "t", {**CFG, "rrf_weights": {"bm25": 1.0, "dense": 0.0}}, HashEmbedder())
    hits = r.search("ICER per QALY", mode="hybrid", k=4)
    assert [h.ranks["bm25"] for h in hits[:2]] == [1, 2]       # with dense at zero the fused order is BM25's; dense-only finds follow


def test_document_hint_boosts_the_named_document():
    other = mk("c5", "The EAG considered the uncertainty in the cost-effectiveness estimates substantial.", 540)
    other.document_title, other.text = "EAG Report", other.text.replace("Doc |", "EAG Report |")
    s = MemoryStore(dim=64)
    s.replace_collection("t", {}, CHUNKS + [other], HashEmbedder().embed([c.text for c in CHUNKS + [other]]))
    hints = [{"query": r"\bEAG\b", "title": "EAG Report"}]
    plain = Retriever(s, "t", {**CFG, "document_hints": hints, "hint_boost": 0.0}, HashEmbedder()).search("EAG ICER", mode="bm25", k=1)
    boosted = Retriever(s, "t", {**CFG, "document_hints": hints, "hint_boost": 5.0}, HashEmbedder()).search("EAG ICER", mode="bm25", k=1)
    assert plain[0].chunk.chunk_id == "c2" and boosted[0].chunk.chunk_id == "c5"
    unnamed = Retriever(s, "t", {**CFG, "document_hints": hints, "hint_boost": 5.0}, HashEmbedder()).search("company ICER £24,567", mode="bm25", k=1)
    assert unnamed[0].chunk.chunk_id == "c1"                 # no party named: no boost


def test_neighbours_follow_the_ranked_hits_under_the_same_heading():
    r = Retriever(store(), "t", {**CFG, "neighbours": 1, "max_passages": 4}, HashEmbedder())
    hits, pool = r.retrieve("median PFS months", mode="bm25", k=1)
    assert hits[0].chunk.chunk_id == "c3" and [c.chunk_id for c in pool] == ["c3"]      # neighbours are shown, not pooled
    assert [h.chunk.chunk_id for h in hits[1:]] == ["c2", "c4"] and all(h.ranks == {"neighbour_of": "c3"} for h in hits[1:])
    assert [h.rank for h in hits] == [1, 2, 3]
    capped = Retriever(store(), "t", {**CFG, "neighbours": 1, "max_passages": 2}, HashEmbedder()).search("median PFS months", mode="bm25", k=1)
    assert len(capped) == 2
    assert len(r.search("median PFS months", mode="bm25", k=1, plain=True)) == 1


class FakeLLM:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def complete(self, messages, tag=""):
        self.calls.append(messages)
        return self.reply, {}


def test_rewrite_fuses_the_restated_queries():
    llm = FakeLLM('1. "EAG base-case ICER £52,681 per QALY"\n- EAG base-case ICER £52,681 per QALY\nshort one\n')
    cfg = {**CFG, "rewrite": {"enabled": True, "max_queries": 2, "passages": 2, "weight": 0.7}}
    r = Retriever(store(), "t", cfg, HashEmbedder(), llm=llm, topic="an appraisal")
    full = r.retrieve_full("assessment group estimate", mode="bm25", k=2)       # words the chunks never use: the first pass finds nothing
    hits = full.hits
    assert full.queries == ["assessment group estimate", "EAG base-case ICER £52,681 per QALY"]   # dedupe, list marks and quotes stripped, fragments dropped
    assert hits[0].chunk.chunk_id == "c2" and hits[0].ranks["bm25@1"] == 1 and hits[0].ranks["bm25"] is None
    assert "an appraisal" in llm.calls[0][0]["content"] and "Question: assessment group estimate" in llm.calls[0][1]["content"]
    plain = r.retrieve_full("assessment group estimate", mode="bm25", k=2, plain=True)
    assert len(llm.calls) == 1 and plain.queries == ["assessment group estimate"]    # plain: no rewrite


def test_parse_queries():
    from medrag.retrieve import parse_queries
    assert parse_queries("Q one here\n\n2) Q one here\n“Q two here”\nno", "Q ONE here", 5) == ["Q two here"]


def test_document_hint_patterns_match_what_they_name():
    # A double-quoted YAML "\\b" is a backspace, which silently never matches.
    from medrag.config import load_config
    hints = {h["title"]: re.compile(h["query"], re.I) for h in load_config()["retrieval"]["document_hints"]}
    assert hints["Document B"].search("What does Document B say?")
    assert hints["EAG Report"].search("the EAG preferred")
