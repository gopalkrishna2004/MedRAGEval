"""Postgres + pgvector round trip. Skipped when no database answers at MEDRAG_TEST_DSN (default: the config's dsn)."""

import os

import numpy as np
import pytest

from medrag.chunk import Chunk
from medrag.config import load_config


@pytest.fixture(scope="module")
def pg():
    cfg = load_config()
    dsn = os.environ.get("MEDRAG_TEST_DSN") or os.environ.get(cfg["store"].get("dsn_env") or "") or cfg["store"]["dsn"]
    try:
        from medrag.store import PgStore
        store = PgStore(dsn, dim=4, schema="medrag_test")   # its own schema: 4-dim vectors, dropped afterwards
    except Exception as e:   # no server, no extension
        pytest.skip(f"no Postgres with pgvector at {dsn}: {e}")
    yield store
    store.drop_schema()
    store.close()


def mk(cid, body, page, kind="text"):
    text = f"Doc | Sec | p. {page}\n{body}"
    return Chunk(cid, "Doc", ["Sec", "Sub"], page, page, kind, body, text, len(text) // 4, metadata={"caption": "x"} if kind == "table" else {})


def test_round_trip_and_dense_search(pg):
    chunks = [mk("c1", "alpha", 1), mk("c2", "beta", 2, "table"), mk("c3", "gamma", 3)]
    vecs = np.eye(4, dtype=np.float32)[:3]
    pg.replace_collection("_test_medrag", {"pdf_path": "x.pdf", "pdf_sha256": "abc", "chunking": {"target_tokens": 1}, "embedding_model": "unit"}, chunks, vecs)
    meta = pg.collection("_test_medrag")
    assert meta["n_chunks"] == 3 and meta["embedding_model"] == "unit" and meta["chunking"] == {"target_tokens": 1}
    assert pg.load_chunks("_test_medrag") == chunks
    assert pg.dense_search("_test_medrag", np.array([0, 1, 0, 0], dtype=np.float32), 2)[0] == ("c2", pytest.approx(1.0))
    # a rebuild replaces the collection in one transaction
    pg.replace_collection("_test_medrag", {}, chunks[:1], vecs[:1])
    assert [c.chunk_id for c in pg.load_chunks("_test_medrag")] == ["c1"]
    assert any(c["name"] == "_test_medrag" for c in pg.collections())
