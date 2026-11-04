import numpy as np
import pytest
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
