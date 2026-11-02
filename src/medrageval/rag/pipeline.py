"""The RAG under evaluation: retrieve, fit the passages into the context budget, answer.

The top `top_k` chunks are shown in rank order as numbered passages, and
chunks past `max_context_chars` are dropped, so the prompt fits the small
model's window (vLLM rejects an overlong prompt instead of truncating it).
The answer is free text; the model and retrieval settings are whatever the
`rag` config says, so a model swap or a retrieval change is a config edit
and a rerun of the same dataset.
"""

import time

from medrageval.rag.chunks import Chunk
from medrageval.rag.index import Index
from medrageval.rag.prompts import build_answer


class RAG:
    def __init__(self, index: Index, llm, cfg: dict):
        self.index, self.llm, self.cfg = index, llm, cfg

    def fit(self, hits: list[dict]) -> list[Chunk]:
        """Chunks in rank order up to the context budget; the first is always kept."""
        budget = self.cfg["retrieval"]["max_context_chars"]
        out, used = [], 0
        for h in hits:
            c = self.index.by_id[h["chunk_id"]]
            if out and used + len(c.text) > budget:
                continue   # a smaller lower-ranked chunk may still fit
            out.append(c)
            used += len(c.text)
        return out

    def answer(self, question: str, tag: str = "") -> dict:
        t0 = time.time()
        hits = self.index.search(question)
        passages = self.fit(hits)
        shown = {c.chunk_id for c in passages}
        text, usage = self.llm.complete_text(build_answer(question, passages, self.index.topic), tag or f"rag:{question[:60]}")
        retrieved = [{**h, "in_context": h["chunk_id"] in shown, "doc_id": self.index.by_id[h["chunk_id"]].doc_id,
                      "pages": [self.index.by_id[h["chunk_id"]].page_start, self.index.by_id[h["chunk_id"]].page_end]} for h in hits]
        return {"answer": text.strip(), "retrieved": retrieved, "passages": passages,
                "context_chars": sum(len(c.text) for c in passages), "usage": usage, "latency_s": round(time.time() - t0, 2)}
