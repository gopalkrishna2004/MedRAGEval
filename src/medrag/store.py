"""Chunk stores: Postgres with pgvector, and an in-memory store for tests.

One `collections` row per ingested (PDF, chunking, embedding model); its
chunks in `chunks` with a `vector` column under an HNSW cosine index, their
metadata in jsonb. Dense search is a SQL query; the lexical side (BM25) runs
in process over the chunks the retriever loads once per collection.
"""

import json
import os
from pathlib import Path

import numpy as np

from medrag.chunk import Chunk

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS collections (
    name            text PRIMARY KEY,
    pdf_path        text,
    pdf_sha256      text,
    chunking        jsonb,
    embedding_model text,
    dim             int,
    n_chunks        int,
    created_at      timestamptz DEFAULT now()
);
CREATE TABLE IF NOT EXISTS chunks (
    id              bigserial PRIMARY KEY,
    collection      text NOT NULL REFERENCES collections(name) ON DELETE CASCADE,
    chunk_id        text NOT NULL,
    document_title  text,
    heading_path    text[],
    page_start      int,
    page_end        int,
    kind            text,
    part            int,
    body            text NOT NULL,
    text            text NOT NULL,
    tokens          int,
    metadata        jsonb DEFAULT '{{}}'::jsonb,
    embedding       vector({dim}),
    UNIQUE (collection, chunk_id)
);
CREATE INDEX IF NOT EXISTS chunks_collection_idx ON chunks (collection);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
"""


def _row_to_chunk(r) -> Chunk:
    chunk_id, title, path, p0, p1, kind, part, body, text, tokens, meta = r
    return Chunk(chunk_id, title, list(path or []), p0, p1, kind, body, text, tokens, part, meta or {})


class PgStore:
    def __init__(self, dsn: str, dim: int = 1024, schema: str = "public"):
        import psycopg
        from pgvector.psycopg import register_vector
        self.dsn, self.dim, self.schema = dsn, dim, schema
        self.conn = psycopg.connect(dsn, autocommit=False)
        if schema != "public":
            with self.conn.cursor() as cur:
                cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
                cur.execute(f'SET search_path TO "{schema}", public')
            self.conn.commit()
        self.init_schema()           # CREATE ... IF NOT EXISTS: cheap, and the vector type must exist before it is registered
        register_vector(self.conn)

    def init_schema(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute(SCHEMA.format(dim=self.dim))
        self.conn.commit()

    def collection(self, name: str) -> dict | None:
        with self.conn.cursor() as cur:
            cur.execute("SELECT name, pdf_path, pdf_sha256, chunking, embedding_model, dim, n_chunks, created_at FROM collections WHERE name=%s", (name,))
            r = cur.fetchone()
        if not r:
            return None
        return dict(zip(("name", "pdf_path", "pdf_sha256", "chunking", "embedding_model", "dim", "n_chunks", "created_at"), r))

    def collections(self) -> list[dict]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT name, pdf_path, embedding_model, n_chunks, created_at FROM collections ORDER BY created_at")
            return [dict(zip(("name", "pdf_path", "embedding_model", "n_chunks", "created_at"), r)) for r in cur.fetchall()]

    def replace_collection(self, name: str, meta: dict, chunks: list[Chunk], vectors: np.ndarray | None) -> None:
        if vectors is not None and vectors.shape[0] != len(chunks):
            raise ValueError(f"{vectors.shape[0]} vectors for {len(chunks)} chunks")
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM collections WHERE name=%s", (name,))
            cur.execute("INSERT INTO collections (name, pdf_path, pdf_sha256, chunking, embedding_model, dim, n_chunks) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                        (name, meta.get("pdf_path"), meta.get("pdf_sha256"), json.dumps(meta.get("chunking") or {}),
                         meta.get("embedding_model"), self.dim if vectors is not None else None, len(chunks)))
            rows = [(name, c.chunk_id, c.document_title, c.heading_path, c.page_start, c.page_end, c.kind, c.part, c.body, c.text,
                     c.tokens, json.dumps(c.metadata or {}), vectors[i] if vectors is not None else None) for i, c in enumerate(chunks)]
            cur.executemany("INSERT INTO chunks (collection, chunk_id, document_title, heading_path, page_start, page_end, kind, part, body, text, tokens, metadata, embedding) "
                            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", rows)
        self.conn.commit()

    def load_chunks(self, name: str) -> list[Chunk]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT chunk_id, document_title, heading_path, page_start, page_end, kind, part, body, text, tokens, metadata "
                        "FROM chunks WHERE collection=%s ORDER BY chunk_id", (name,))
            return [_row_to_chunk(r) for r in cur.fetchall()]

    def dense_search(self, name: str, qvec: np.ndarray, n: int) -> list[tuple[str, float]]:
        """(chunk_id, cosine similarity), best first."""
        with self.conn.cursor() as cur:
            cur.execute(f"SET LOCAL hnsw.ef_search = {int(max(40, 2 * n))}")   # SET takes no bind parameters
            cur.execute("SELECT chunk_id, 1 - (embedding <=> %s) AS sim FROM chunks WHERE collection=%s AND embedding IS NOT NULL "
                        "ORDER BY embedding <=> %s LIMIT %s", (qvec, name, qvec, n))
            out = [(cid, float(sim)) for cid, sim in cur.fetchall()]
        self.conn.rollback()   # end the read transaction (SET LOCAL scope)
        return out

    def drop_schema(self) -> None:
        """Remove a non-public schema and everything in it (tests)."""
        if self.schema == "public":
            raise ValueError("refusing to drop the public schema")
        with self.conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS "{self.schema}" CASCADE')
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


class MemoryStore:
    """The same interface over dicts and a numpy matrix; optionally persisted under a directory."""

    def __init__(self, root: str | Path | None = None, dim: int = 1024):
        self.root, self.dim = Path(root) if root else None, dim
        self.meta: dict[str, dict] = {}
        self.chunks: dict[str, list[Chunk]] = {}
        self.vectors: dict[str, np.ndarray | None] = {}
        if self.root and self.root.exists():
            for d in self.root.iterdir():
                if (d / "meta.json").exists():
                    self._load(d.name)

    def init_schema(self) -> None:
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)

    def _load(self, name: str) -> None:
        d = self.root / name
        self.meta[name] = json.loads((d / "meta.json").read_text())
        self.chunks[name] = [Chunk.from_json(json.loads(l)) for l in (d / "chunks.jsonl").open() if l.strip()]
        self.vectors[name] = np.load(d / "embeddings.npy") if (d / "embeddings.npy").exists() else None

    def collection(self, name: str) -> dict | None:
        return self.meta.get(name)

    def collections(self) -> list[dict]:
        return list(self.meta.values())

    def replace_collection(self, name: str, meta: dict, chunks: list[Chunk], vectors: np.ndarray | None) -> None:
        self.meta[name] = {**meta, "name": name, "n_chunks": len(chunks)}
        self.chunks[name], self.vectors[name] = list(chunks), vectors
        if self.root:
            d = self.root / name
            d.mkdir(parents=True, exist_ok=True)
            (d / "meta.json").write_text(json.dumps(self.meta[name], default=str))
            with (d / "chunks.jsonl").open("w") as f:
                for c in chunks:
                    f.write(json.dumps(c.to_json(), ensure_ascii=False) + "\n")
            if vectors is not None:
                np.save(d / "embeddings.npy", vectors)
            elif (d / "embeddings.npy").exists():
                os.remove(d / "embeddings.npy")

    def load_chunks(self, name: str) -> list[Chunk]:
        return list(self.chunks.get(name, []))

    def dense_search(self, name: str, qvec: np.ndarray, n: int) -> list[tuple[str, float]]:
        v = self.vectors.get(name)
        if v is None:
            return []
        sims = v @ qvec
        top = np.argsort(-sims, kind="stable")[:n]
        return [(self.chunks[name][i].chunk_id, float(sims[i])) for i in top]

    def close(self) -> None:
        pass


def make_store(cfg: dict):
    s = cfg["store"]
    dim = cfg["embedding"].get("dim", 1024)
    if s.get("backend", "postgres") == "memory":
        return MemoryStore(Path(cfg["cache_dir"]) / "memory_store", dim)
    dsn = os.environ.get(s.get("dsn_env") or "", "").strip() or s["dsn"]
    return PgStore(dsn, dim)
