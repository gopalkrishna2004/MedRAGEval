"""The RAG under evaluation: retrieve, fit the passages into the context budget, answer.

The top `top_k` chunks are shown in rank order as numbered passages, and
chunks past `max_context_chars` are dropped, so the prompt fits the small
model's window (vLLM rejects an overlong prompt instead of truncating it).
The answer is free text; the model and retrieval settings are whatever the
`rag` config says, so a model swap or a retrieval change is a config edit
and a rerun of the same dataset.

An unanswerable item carries the passages it is to be read with: the
distractors chosen when its evidence was withheld. `distractor_chunks` turns them into chunks, dropping any chunk that
also holds the withheld evidence, and `answer` reads them in place of a
search, so the evidence is certainly absent.
"""

import time

from medrageval.rag.chunks import Chunk
from medrageval.rag.index import Index
from medrageval.rag.prompts import build_answer
from medrageval.rag.score import evidence_in


def distractor_chunks(item: dict, index: Index) -> list[Chunk]:
    """The chunks of the item's stored distractors, in stored order, minus any that holds the withheld evidence or a copy of it."""
    withheld = (item.get("withheld") or {}).get("evidence") or []
    copies = set(item.get("also_in") or [])
    out: list[Chunk] = []
    for d in item.get("distractors") or []:
        for c in index.by_unit.get(d.get("unit_id"), []):
            if any(c.chunk_id == o.chunk_id for o in out) or copies & set(c.unit_ids) or any(evidence_in(e, c) for e in withheld):
                continue
            out.append(c)
    return out


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

    def answer(self, question: str, tag: str = "", passages: list[Chunk] | None = None) -> dict:
        """Retrieve and answer; given `passages`, answer over the first top_k of them instead of searching."""
        t0 = time.time()
        if passages is None:
            hits = self.index.search(question)
            context = "retrieved"
        else:
            hits = [{"chunk_id": c.chunk_id, "rank": i, "score": None, "ranks": {}} for i, c in enumerate(passages[:self.cfg["retrieval"]["top_k"]], start=1)]
            context = "fixed"
        passages = self.fit(hits)
        shown = {c.chunk_id for c in passages}
        text, usage = self.llm.complete_text(build_answer(question, passages, self.index.topic), tag or f"rag:{question[:60]}")
        retrieved = [{**h, "in_context": h["chunk_id"] in shown, "doc_id": self.index.by_id[h["chunk_id"]].doc_id,
                      "pages": [self.index.by_id[h["chunk_id"]].page_start, self.index.by_id[h["chunk_id"]].page_end]} for h in hits]
        return {"answer": text.strip(), "retrieved": retrieved, "passages": passages, "context": context,
                "context_chars": sum(len(c.text) for c in passages), "usage": usage, "latency_s": round(time.time() - t0, 2)}
