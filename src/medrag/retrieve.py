"""Hybrid retrieval: BM25 and dense candidates fused by reciprocal rank, then reranked.

Reciprocal rank fusion needs no score calibration between the two
retrievers. The cross-encoder then reads the query with each fused candidate
and reorders them; it is the stage that tells "the EAG's ICER" from "the
company's ICER" when both chunks mention both.

Four optional settings, each off by default, address the ways a bundle of
committee papers defeats plain retrieval:

- `rrf_weights`: every chunk of the bundle is about the same drug, so dense
  similarity separates sections poorly and BM25's rare tokens carry more of
  the signal; a weight below 1 for dense keeps its hits without letting its
  tail outvote a strong lexical match.
- `document_hints` + `hint_boost`: a question that names a party ("the EAG
  report", "the company submission") wants that document; the fused score of
  its chunks is raised by the boost. A boost, not a filter, because the party
  is sometimes quoted in another document.
- `neighbours`: an overview question is answered by several adjacent chunks
  of one section, of which the retrievers find one or two; each shown chunk
  brings its neighbours under the same heading, after the ranked hits.
- `rewrite`: a question that describes a thing instead of naming it ("the
  comparator the company considers relevant for PIK3CA-mutated disease") is
  retrieved twice: the first pass's top passages let the answering model
  restate it with the name, and rankings for the restated queries are fused
  with the original's at a lower weight.
"""

import re
from dataclasses import dataclass, field

import numpy as np

from medrag.chunk import Chunk
from medrag.lexical import BM25, tokenize

MODES = ("bm25", "dense", "hybrid")
_LIST_MARK = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


@dataclass
class Hit:
    chunk: Chunk
    rank: int
    score: float | None
    ranks: dict = field(default_factory=dict)      # retriever -> rank (bm25, dense, fused, rerank), or neighbour_of -> anchor chunk id

    def to_json(self) -> dict:
        return {"chunk_id": self.chunk.chunk_id, "rank": self.rank, "score": self.score, "ranks": self.ranks,
                "document_title": self.chunk.document_title, "pages": [self.chunk.page_start, self.chunk.page_end]}


@dataclass
class Retrieval:
    hits: list[Hit]
    pool: list[Chunk]
    queries: list[str]


def rrf(rankings: list[list[str]], k: int, weights: list[float] | None = None) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for i, ranking in enumerate(rankings):
        w = 1.0 if weights is None else weights[i]
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def rewrite_prompt(question: str, passages: list[Chunk], topic: str, n: int, snippet_chars: int) -> list[dict]:
    system = (f"You help search a collection of documents about {topic}. Given a question and passages found for it, write {n} "
              "alternative search queries that would find the passage answering the question.\n"
              "- Name explicitly anything the question only describes; work the name out from the passages.\n"
              "- Use the words and abbreviations the documents use.\n"
              "- One query per line; no numbering, quotes or commentary.")
    snippets = "\n\n".join(f"- {' '.join(c.text[:snippet_chars].split())}" for c in passages)
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"Question: {question}\n\nPassages:\n\n{snippets}\n\nQueries:"}]


def parse_queries(text: str, original: str, n: int) -> list[str]:
    """Up to n distinct queries, one per line, list marks and quotes stripped, the original and fragments dropped."""
    out: list[str] = []
    seen = {" ".join(original.lower().split())}
    for line in text.splitlines():
        q = _LIST_MARK.sub("", line).strip().strip('"“”').strip()
        key = " ".join(q.lower().split())
        if len(q.split()) >= 3 and key not in seen:
            seen.add(key)
            out.append(q)
    return out[:n]


class Retriever:
    def __init__(self, store, collection: str, cfg: dict, embedder=None, reranker=None, llm=None, topic: str | None = None):
        self.store, self.collection, self.cfg = store, collection, cfg
        self.embedder, self.reranker, self.llm = embedder, reranker, llm
        self.topic = topic or "the documents"
        self.chunks = store.load_chunks(collection)
        if not self.chunks:
            raise RuntimeError(f"collection {collection!r} is empty or missing: run `medrag ingest` first")
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.position = {c.chunk_id: i for i, c in enumerate(self.chunks)}     # chunks are in document order
        self.bm25 = BM25({c.chunk_id: tokenize(c.text) for c in self.chunks})
        self.hints = [(re.compile(h["query"], re.I), re.compile(h["title"], re.I)) for h in cfg.get("document_hints") or []]

    def search(self, query: str, mode: str | None = None, k: int | None = None, rerank: bool | None = None, plain: bool = False) -> list[Hit]:
        return self.retrieve(query, mode, k, rerank, plain)[0]

    def retrieve(self, query: str, mode: str | None = None, k: int | None = None, rerank: bool | None = None,
                 plain: bool = False) -> tuple[list[Hit], list[Chunk]]:
        """(hits, candidates) of `retrieve_full`."""
        r = self.retrieve_full(query, mode, k, rerank, plain)
        return r.hits, r.pool

    def rankings(self, query: str, mode: str, n: int) -> dict[str, list[str]]:
        """retriever name -> its top n chunk ids for the query."""
        ranked: dict[str, list[str]] = {}
        if mode in ("bm25", "hybrid"):
            ranked["bm25"] = [cid for cid, _ in self.bm25.search(tokenize(query), n)]
        if mode in ("dense", "hybrid"):
            if self.embedder is None:
                raise RuntimeError("dense retrieval needs an embedder")
            q = self.embedder.embed([self.embedder.query_prefix + query])[0]
            ranked["dense"] = [cid for cid, _ in self.store.dense_search(self.collection, np.asarray(q, dtype=np.float32), n)]
        return ranked

    def rewrite(self, query: str, seeds: list[Chunk]) -> list[str]:
        """Alternative queries from the answering model, given the first pass's top passages."""
        rw = self.cfg.get("rewrite") or {}
        n = rw.get("max_queries", 2)
        text, _ = self.llm.complete(rewrite_prompt(query, seeds, self.topic, n, rw.get("snippet_chars", 500)), f"rewrite:{query[:60]}")
        return parse_queries(text, query, n)

    def neighbours(self, anchor: Chunk, each_side: int) -> list[Chunk]:
        """Chunks within each_side positions of the anchor under the same document and heading, nearest first."""
        i = self.position[anchor.chunk_id]
        out = []
        for d in range(1, each_side + 1):
            for j in (i - d, i + d):
                if 0 <= j < len(self.chunks):
                    c = self.chunks[j]
                    if c.document_title == anchor.document_title and c.heading_path == anchor.heading_path:
                        out.append(c)
        return out

    def retrieve_full(self, query: str, mode: str | None = None, k: int | None = None, rerank: bool | None = None,
                      plain: bool = False) -> "Retrieval":
        """The top k after reranking (plus their neighbours, when configured); the fused candidate pool (`candidates`
        chunks, in fused order) before the reranker and the top-k cut, so a miss can be put down to the pool or to the
        ordering; and the queries run (the query, then its rewrites). `plain` skips the rewrite, the document boost and
        the neighbours: retrieval for a text that is not a user's question."""
        rc = self.cfg
        mode, k, n = mode or rc["mode"], k or rc["top_k"], rc["candidates"]
        rerank = rc.get("rerank", False) if rerank is None else rerank
        if mode not in MODES:
            raise ValueError(f"retrieval mode {mode!r} is not one of {MODES}")
        wcfg = rc.get("rrf_weights") or {}
        ranked = self.rankings(query, mode, n)
        names, rankings, weights = list(ranked), list(ranked.values()), [float(wcfg.get(nm, 1.0)) for nm in ranked]
        queries = [query]
        rw = rc.get("rewrite") or {}
        if rw.get("enabled") and self.llm is not None and not plain:
            seeds = [self.by_id[cid] for cid, _ in rrf(rankings, rc["rrf_k"], weights)[:rw.get("passages", 4)]]
            for i, q2 in enumerate(self.rewrite(query, seeds), start=1):
                queries.append(q2)
                for nm, ids in self.rankings(q2, mode, n).items():
                    names.append(f"{nm}@{i}")
                    rankings.append(ids)
                    weights.append(float(wcfg.get(nm, 1.0)) * float(rw.get("weight", 0.7)))
        fused = rrf(rankings, rc["rrf_k"], weights)
        boost = float(rc.get("hint_boost") or 0.0)
        if boost and self.hints and not plain:
            titles = [t for q_re, t in self.hints if q_re.search(query)]
            if titles:
                fused = sorted(((cid, s * (1 + boost) if any(t.search(self.by_id[cid].document_title or "") for t in titles) else s)
                                for cid, s in fused), key=lambda x: (-x[1], x[0]))
        positions = {name: {cid: r for r, cid in enumerate(ids, start=1)} for name, ids in zip(names, rankings)}
        order = fused[:n]
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
        each_side = 0 if plain else int(rc.get("neighbours") or 0)
        if each_side:
            cap = int(rc.get("max_passages") or k * (1 + 2 * each_side))
            have = {h.chunk.chunk_id for h in hits}
            for anchor in list(hits):
                for c in self.neighbours(anchor.chunk, each_side):
                    if len(hits) >= cap:
                        break
                    if c.chunk_id not in have:
                        have.add(c.chunk_id)
                        hits.append(Hit(c, len(hits) + 1, None, {"neighbour_of": anchor.chunk.chunk_id}))
        return Retrieval(hits, pool, queries)
