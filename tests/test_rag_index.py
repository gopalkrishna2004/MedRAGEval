import copy
import zlib

import numpy as np
import pytest
from synthetic import ELEMENTS, build_corpus, config

from medrageval.config import load_config
from medrageval.generate.corpus import tokenize
from medrageval.rag.index import build_index, normalise, rrf

RAG = load_config()["rag"]


class HashEmbedder:
    """Bag of words hashed into 256 dimensions: similar wording, similar vector. Counts its calls."""

    model, query_prefix = "hash-256", ""

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        m = np.zeros((len(texts), 256), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                m[i, zlib.crc32(tok.encode()) % 256] += 1
        return normalise(m)


def test_rrf_rewards_agreement_between_rankings():
    fused = rrf([["a", "b", "c"], ["b", "c", "a"]], 60)
    assert [cid for cid, _ in fused] == ["b", "a", "c"]
    assert rrf([["x", "y"]], 60)[0] == ("x", pytest.approx(1 / 61))


def test_bm25_finds_the_icer_passage():
    index, _ = build_index(build_corpus(config()), RAG, None)
    hits = index.search("company base-case ICER capivasertib versus everolimus plus exemestane", mode="bm25", k=3)
    assert "£24,567" in index.by_id[hits[0]["chunk_id"]].text
    assert hits[0]["rank"] == 1 and set(hits[0]["ranks"]) == {"bm25"}


def test_dense_needs_an_embedder():
    index, _ = build_index(build_corpus(config()), RAG, None)
    with pytest.raises(RuntimeError):
        index.search("median PFS", mode="dense")


def test_hybrid_fuses_both_rankings(tmp_path):
    index, stats = build_index(build_corpus(config()), RAG, HashEmbedder(), tmp_path)
    hits = index.search("median progression-free survival with capivasertib plus fulvestrant", mode="hybrid", k=4)
    assert len(hits) == 4 and all(set(h["ranks"]) == {"bm25", "dense"} for h in hits)
    assert any("7.3" in index.by_id[h["chunk_id"]].text for h in hits)
    assert stats["embedding"]["dim"] == 256 and (tmp_path / "chunks.jsonl").exists()


def test_embeddings_are_cached_until_the_chunks_change(tmp_path):
    emb = HashEmbedder()
    _, first = build_index(build_corpus(config()), RAG, emb, tmp_path)
    _, second = build_index(build_corpus(config()), RAG, emb, tmp_path)
    assert emb.calls == 1 and not first["embedding"]["cached"] and second["embedding"]["cached"]
    cfg = copy.deepcopy(RAG)
    cfg["chunking"]["max_chars"] = 300                 # different chunks: embedded again
    build_index(build_corpus(config(), elements=ELEMENTS), cfg, emb, tmp_path)
    assert emb.calls == 2
