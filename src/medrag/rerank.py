"""Cross-encoder reranking of the fused candidates."""

from medrag.chunk import Chunk
from medrag.embed import pick_device


class Reranker:
    def __init__(self, model: str, device: str | None = None, max_length: int = 1024):
        from sentence_transformers import CrossEncoder
        self.model = model
        self.ce = CrossEncoder(model, device=pick_device(device), max_length=max_length)

    def scores(self, query: str, chunks: list[Chunk]) -> list[float]:
        if not chunks:
            return []
        return [float(s) for s in self.ce.predict([(query, c.text) for c in chunks], batch_size=8, show_progress_bar=False)]
