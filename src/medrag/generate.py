"""Answering: fit the retrieved chunks into the context budget and ask the model.

One prompt for every question. It asks for the exact figure with its unit,
citations by passage number, no outside knowledge, no guessing of values
printed as [REDACTED], and one fixed sentence when the passages do not answer.
`passages` bypasses retrieval and answers over the given chunks, which the
evaluation harness uses for questions whose evidence it withholds. The reply
carries the passages shown and, for a retrieved answer, the candidate pool the
reranker chose from.
"""

import time

from medrag.chunk import Chunk
from medrag.retrieve import Hit, Retriever

PROMPT_VERSION = "3"
ABSTAIN = "Not found in the documents."


def build_prompt(question: str, passages: list[Chunk], topic: str) -> list[dict]:
    system = (f"You answer questions about {topic}, using only the numbered passages below, which come from its documents. "
              "Each passage starts with a line naming its document, section and pages.\n"
              "- Answer in one or two sentences. Give figures exactly as the passages print them, with their units.\n"
              "- If the question asks for an overview or a summary, answer in a short paragraph that covers each relevant point.\n"
              "- Say whose position, analysis or estimate it is (the company, the assessment group, the committee, a clinical or "
              "patient expert, a patient organisation) when the passage says.\n"
              "- Cite the passages you used, like [2].\n"
              "- Values printed as [REDACTED] or as a run of x's are confidential: never guess them or take a neighbouring value instead.\n"
              f"- If the passages do not answer the question, reply exactly: {ABSTAIN}\n"
              "- Do not use knowledge from outside the passages.\n"
              f"(prompt version {PROMPT_VERSION})")
    context = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(passages, start=1))
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"Passages:\n\n{context}\n\nQuestion: {question}"}]


class RAG:
    def __init__(self, retriever: Retriever, llm, cfg: dict):
        """`cfg` is the full application config."""
        self.retriever, self.llm, self.cfg = retriever, llm, cfg
        self.topic = cfg.get("topic") or "the documents"

    def fit(self, chunks: list[Chunk]) -> list[Chunk]:
        """Chunks in rank order up to the context budget; the first is always kept, a smaller later one may still fit."""
        budget = self.cfg["retrieval"]["max_context_chars"]
        out, used = [], 0
        for c in chunks:
            if out and used + len(c.text) > budget:
                continue
            out.append(c)
            used += len(c.text)
        return out

    def answer(self, question: str, tag: str = "", passages: list[Chunk] | None = None, mode: str | None = None) -> dict:
        t0 = time.time()
        if passages is None:
            hits, candidates = self.retriever.retrieve(question, mode=mode)
            context = "retrieved"
        else:
            hits = [Hit(c, i, None, {}) for i, c in enumerate(passages[:self.cfg["retrieval"]["top_k"]], start=1)]
            candidates, context = None, "fixed"
        shown = self.fit([h.chunk for h in hits])
        shown_ids = {c.chunk_id for c in shown}
        text, usage = self.llm.complete(build_prompt(question, shown, self.topic), tag or f"rag:{question[:60]}")
        return {"answer": text.strip(), "passages": shown, "context": context, "candidates": candidates,
                "retrieved": [{**h.to_json(), "in_context": h.chunk.chunk_id in shown_ids} for h in hits],
                "context_chars": sum(len(c.text) for c in shown), "usage": usage, "latency_s": round(time.time() - t0, 2)}
