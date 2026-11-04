"""BM25 over chunk texts, in process.

Postgres full-text ranking (ts_rank) has no inverse document frequency, so a
common word outranks a rare trial id. With a few thousand chunks per bundle a
BM25 over token lists loaded from the store is instant and exact, so the
lexical side lives here and Postgres keeps the chunks, metadata and vectors.
"""

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase word and number tokens; "£24,567" -> "24,567", "CAPItello-291" -> "capitello", "291"."""
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, docs: dict[str, list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids = list(docs)
        self.tf = [Counter(docs[i]) for i in self.ids]
        self.len = [len(docs[i]) for i in self.ids]
        self.avg = sum(self.len) / len(self.len) if self.len else 0.0
        df: Counter = Counter()
        for tf in self.tf:
            df.update(tf.keys())
        n = len(self.ids)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query_tokens: list[str], k: int) -> list[tuple[str, float]]:
        q = [t for t in set(query_tokens) if t in self.idf]
        if not q or not self.ids:
            return []
        scores = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * self.len[i] / self.avg) if self.avg else self.k1
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            if s > 0:
                scores.append((self.ids[i], s))
        scores.sort(key=lambda x: (-x[1], x[0]))
        return scores[:k]
