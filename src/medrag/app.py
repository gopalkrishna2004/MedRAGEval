"""Wiring: config -> store, embedder, reranker, retriever, LLM."""

import hashlib
import json
from pathlib import Path

from medrag.config import load_config
from medrag.embed import make_embedder
from medrag.generate import RAG
from medrag.llm import make_llm
from medrag.pdf import pdf_sha256
from medrag.retrieve import Retriever
from medrag.store import make_store


def collection_name(cfg: dict, pdf: str | Path | None = None) -> str:
    """One collection per (PDF, chunking settings, embedding model): a chunk-size sweep ingests side by side."""
    pdf = Path(pdf or cfg["pdf"])
    ch = cfg["chunking"]
    fp = hashlib.sha256(json.dumps({k: ch[k] for k in sorted(ch) if k != "tokenizer"}, sort_keys=True).encode()).hexdigest()[:8]
    model = cfg["embedding"]["model"].split("/")[-1].lower()
    return f"{pdf.stem}-{pdf_sha256(pdf)[:10]}-{fp}-{model}"


class App:
    """Lazily built components, so `medrag search --mode bm25` loads no model."""

    def __init__(self, cfg: dict | str | Path | None = None, use_cache: bool = True):
        self.cfg = cfg if isinstance(cfg, dict) else load_config(cfg)
        self.use_cache = use_cache
        self._store = self._embedder = self._reranker = self._retriever = self._llm = None

    @property
    def store(self):
        if self._store is None:
            self._store = make_store(self.cfg)
        return self._store

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = make_embedder(self.cfg["embedding"])
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None and self.cfg["retrieval"].get("rerank"):
            from medrag.rerank import Reranker
            self._reranker = Reranker(self.cfg["retrieval"]["rerank_model"], self.cfg["embedding"].get("device"))
        return self._reranker

    def retriever(self, mode: str | None = None, rerank: bool | None = None) -> Retriever:
        if self._retriever is None:
            rc = self.cfg["retrieval"]
            mode = mode or rc["mode"]
            rerank = rc.get("rerank", False) if rerank is None else rerank
            rewrite = (rc.get("rewrite") or {}).get("enabled", False)
            self._retriever = Retriever(self.store, collection_name(self.cfg), rc,
                                        self.embedder if mode != "bm25" else None, self.reranker if rerank else None,
                                        self.llm if rewrite else None, self.cfg.get("topic"))
        return self._retriever

    @property
    def llm(self):
        if self._llm is None:
            self._llm = make_llm(self.cfg["llm"], self.use_cache)
        return self._llm

    def rag(self, mode: str | None = None, rerank: bool | None = None) -> RAG:
        return RAG(self.retriever(mode, rerank), self.llm, self.cfg)
