"""The retrieval index: BM25 and dense vectors over the chunks, fused by reciprocal rank.

BM25 is the corpus's own (`generate/corpus.py`): exact names, trial ids and
figures ("CAPItello-291", "£24,567") are what it finds and embeddings blur.
The dense side catches paraphrase ("cost per QALY gained" for "ICER"). The
two rankings are fused with reciprocal rank fusion, which needs no score
calibration between them.

Embeddings come from any OpenAI-compatible /v1/embeddings server (vLLM's
pooling runner by default). With a couple of thousand chunks a normalised
numpy matrix and a dot product is the whole vector store. Vectors are cached
under `index_dir`, keyed on the embedding model and the chunk texts, so a
rerun embeds nothing and a chunking change re-embeds everything.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Protocol

import numpy as np
from dotenv import load_dotenv

from medrageval.generate.corpus import BM25, Corpus, load_corpus, tokenize
from medrageval.generate.llm import LLMError
from medrageval.rag.chunks import Chunk, build_chunks

MODES = ("bm25", "dense", "hybrid")


class Embedder(Protocol):
    model: str
    query_prefix: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class OpenAIEmbedder:
    def __init__(self, model: str, base_url: str, api_key: str, batch_size: int = 32, query_prefix: str = "", timeout: float = 300.0):
        from openai import OpenAI
        self.model, self.batch_size, self.query_prefix = model, batch_size, query_prefix
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=5)

    def embed(self, texts: list[str]) -> np.ndarray:
        from openai import APIError
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            try:
                r = self.client.embeddings.create(model=self.model, input=texts[i:i + self.batch_size])
            except APIError as e:
                raise LLMError(f"{self.model} embeddings: {e}") from e
            out.extend(d.embedding for d in sorted(r.data, key=lambda d: d.index))
        return normalise(np.asarray(out, dtype=np.float32))


def make_embedder(section: dict) -> OpenAIEmbedder:
    load_dotenv()
    key = os.environ.get(section.get("api_key_env") or "", "").strip() or "EMPTY"
    return OpenAIEmbedder(section["model"], section["base_url"], key, section["batch_size"], section.get("query_prefix") or "",
                          section.get("timeout", 300.0))


def normalise(m: np.ndarray) -> np.ndarray:
    return m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-12)


def rrf(rankings: list[list[str]], k: int) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: score(c) = sum over rankings of 1 / (k + rank). Ties go to the earlier id."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


class Index:
    def __init__(self, chunks: list[Chunk], cfg: dict, topic: str, vectors: np.ndarray | None = None, embedder: Embedder | None = None):
        self.chunks, self.cfg, self.topic = chunks, cfg, topic
        self.by_id = {c.chunk_id: c for c in chunks}
        for c in chunks:
            c.tokens = tokenize(c.text)
        self.bm25 = BM25({c.chunk_id: c.tokens for c in chunks})
        self.vectors, self.embedder = vectors, embedder

    def search(self, query: str, mode: str | None = None, k: int | None = None) -> list[dict]:
        """Top k as {chunk_id, rank, score, ranks: {retriever: rank}}, best first."""
        rc = self.cfg["retrieval"]
        mode, k, n = mode or rc["mode"], k or rc["top_k"], rc["candidates"]
        if mode not in MODES:
            raise ValueError(f"retrieval mode {mode!r} is not one of {MODES}")
        ranked: dict[str, list[str]] = {}
        if mode in ("bm25", "hybrid"):
            ranked["bm25"] = [cid for cid, _ in self.bm25.search(tokenize(query), k=n)]
        if mode in ("dense", "hybrid"):
            if self.vectors is None or self.embedder is None:
                raise RuntimeError("dense retrieval needs embeddings: build the index with an embedder")
            q = self.embedder.embed([self.embedder.query_prefix + query])[0]
            sims = self.vectors @ q
            top = np.argsort(-sims, kind="stable")[:n]
            ranked["dense"] = [self.chunks[i].chunk_id for i in top]
        positions = {name: {cid: r for r, cid in enumerate(ids, start=1)} for name, ids in ranked.items()}
        return [{"chunk_id": cid, "rank": rank, "score": round(score, 5),
                 "ranks": {name: pos.get(cid) for name, pos in positions.items()}}
                for rank, (cid, score) in enumerate(rrf(list(ranked.values()), rc["rrf_k"])[:k], start=1)]


def _fingerprint(model: str, chunks: list[Chunk]) -> str:
    h = hashlib.sha256(model.encode())
    for c in chunks:
        h.update(b"\0" + c.text.encode())
    return h.hexdigest()


def embed_chunks(chunks: list[Chunk], embedder: Embedder, out_dir: Path) -> tuple[np.ndarray, dict]:
    """Chunk vectors, from the cache when the model and chunk texts are unchanged."""
    meta_path, vec_path = out_dir / "index.json", out_dir / "embeddings.npy"
    key = _fingerprint(embedder.model, chunks)
    if meta_path.exists() and vec_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("key") == key:
            return np.load(vec_path), {**meta, "cached": True}
    t0 = time.time()
    vectors = embedder.embed([c.text for c in chunks])
    if vectors.shape[0] != len(chunks):
        raise RuntimeError(f"{embedder.model}: {vectors.shape[0]} embeddings for {len(chunks)} chunks")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(vec_path, vectors)
    meta = {"key": key, "embedding_model": embedder.model, "chunks": len(chunks), "dim": int(vectors.shape[1]),
            "embed_seconds": round(time.time() - t0, 1)}
    meta_path.write_text(json.dumps(meta, indent=2))
    return vectors, {**meta, "cached": False}


def build_index(corpus: Corpus, cfg: dict, embedder: Embedder | None, out_dir: Path | None = None) -> tuple[Index, dict]:
    """The index over a corpus, and its stats. Without an embedder only BM25 works (dense and hybrid raise)."""
    chunks = build_chunks(corpus, cfg)
    stats = {"chunks": len(chunks), "chars": sum(len(c.text) for c in chunks),
             "max_chars": max((len(c.text) for c in chunks), default=0), "split_parts": sum(c.part is not None for c in chunks)}
    vectors = None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "chunks.jsonl").open("w") as f:
            for c in chunks:
                f.write(json.dumps(c.to_json(), ensure_ascii=False) + "\n")
    if embedder is not None:
        if out_dir is None:
            vectors = embedder.embed([c.text for c in chunks])
            stats["embedding"] = {"embedding_model": embedder.model, "cached": False}
        else:
            vectors, stats["embedding"] = embed_chunks(chunks, embedder, out_dir)
    return Index(chunks, cfg, corpus.topic, vectors, embedder), stats


def load_index(interim_dir: Path, full_cfg: dict, mode: str | None = None, embedder: Embedder | None = None) -> tuple[Index, Corpus, dict]:
    """Corpus, chunks and (unless the mode is bm25) embeddings for one parse output directory."""
    rcfg = full_cfg["rag"]
    corpus = load_corpus(interim_dir, full_cfg["generate"])
    if embedder is None and (mode or rcfg["retrieval"]["mode"]) != "bm25":
        embedder = make_embedder(rcfg["embedding"])
    index, stats = build_index(corpus, rcfg, embedder, Path(rcfg["index_dir"]) / Path(interim_dir).name)
    return index, corpus, stats
