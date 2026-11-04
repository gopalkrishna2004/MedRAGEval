"""Hybrid retrieval: BM25 and dense candidates fused by reciprocal rank, then reranked.

Reciprocal rank fusion needs no score calibration between the two
retrievers. The cross-encoder then reads the query with each fused candidate
and reorders them; it is the stage that tells "the EAG's ICER" from "the
company's ICER" when both chunks mention both.
"""

from dataclasses import dataclass, field

import numpy as np

from medrag.chunk import Chunk
from medrag.lexical import BM25, tokenize

MODES = ("bm25", "dense", "hybrid")


@dataclass
class Hit:
    chunk: Chunk
    rank: int
    score: float | None
    ranks: dict = field(default_factory=dict)      # retriever -> rank (bm25, dense, fused, rerank)

    def to_json(self) -> dict:
        return {"chunk_id": self.chunk.chunk_id, "rank": self.rank, "score": self.score, "ranks": self.ranks,
                "document_title": self.chunk.document_title, "pages": [self.chunk.page_start, self.chunk.page_end]}


def rrf(rankings: list[list[str]], k: int) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


class Retriever:
    def __init__(self, store, collection: str, cfg: dict, embedder=None, reranker=None):
        self.store, self.collection, self.cfg = store, collection, cfg
        self.embedder, self.reranker = embedder, reranker
        self.chunks = store.load_chunks(collection)
        if not self.chunks:
            raise RuntimeError(f"collection {collection!r} is empty or missing: run `medrag ingest` first")
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.bm25 = BM25({c.chunk_id: tokenize(c.text) for c in self.chunks})

    def search(self, query: str, mode: str | None = None, k: int | None = None, rerank: bool | None = None) -> list[Hit]:
        return self.retrieve(query, mode, k, rerank)[0]

    def retrieve(self, query: str, mode: str | None = None, k: int | None = None, rerank: bool | None = None) -> tuple[list[Hit], list[Chunk]]:
        """(hits, candidates): the top k after reranking, and the fused candidate pool (`candidates` chunks, in fused order)
        before the reranker and the top-k cut, so a miss can be put down to the pool or to the ordering."""
        rc = self.cfg
        mode, k, n = mode or rc["mode"], k or rc["top_k"], rc["candidates"]
        rerank = rc.get("rerank", False) if rerank is None else rerank
        if mode not in MODES:
            raise ValueError(f"retrieval mode {mode!r} is not one of {MODES}")
        ranked: dict[str, list[str]] = {}
        if mode in ("bm25", "hybrid"):
            ranked["bm25"] = [cid for cid, _ in self.bm25.search(tokenize(query), n)]
        if mode in ("dense", "hybrid"):
            if self.embedder is None:
                raise RuntimeError("dense retrieval needs an embedder")
            q = self.embedder.embed([self.embedder.query_prefix + query])[0]
            ranked["dense"] = [cid for cid, _ in self.store.dense_search(self.collection, np.asarray(q, dtype=np.float32), n)]
        fused = rrf(list(ranked.values()), rc["rrf_k"])
        positions = {name: {cid: r for r, cid in enumerate(ids, start=1)} for name, ids in ranked.items()}
        order = [(cid, s) for cid, s in fused][:n]
        pool = [self.by_id[cid] for cid, _ in order]
        fused_rank = {cid: r for r, (cid, _) in enumerate(order, start=1)}
        rerank_rank: dict[str, int] = {}
        if rerank and self.reranker is not None and order:
            m = rc.get("rerank_candidates", 30)
            head = order[:m]
            scores = self.reranker.scores(query, [self.by_id[cid] for cid, _ in head])
            head = sorted(zip((cid for cid, _ in head), scores), key=lambda x: -x[1])
            rerank_rank = {cid: r for r, (cid, _) in enumerate(head, start=1)}
            order = head + order[m:]
        hits = []
        for rank, (cid, score) in enumerate(order[:k], start=1):
            ranks = {name: pos.get(cid) for name, pos in positions.items()}
            ranks["fused"] = fused_rank.get(cid)
            if rerank_rank:
                ranks["rerank"] = rerank_rank.get(cid)
            hits.append(Hit(self.by_id[cid], rank, round(float(score), 5), ranks))
        return hits, pool
