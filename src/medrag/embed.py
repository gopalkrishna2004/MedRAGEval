"""Embeddings: a local sentence-transformers model, or any OpenAI-compatible /v1/embeddings server."""

from typing import Protocol

import numpy as np

from medrag.config import env_or


class Embedder(Protocol):
    model: str
    query_prefix: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


def normalise(m: np.ndarray) -> np.ndarray:
    return (m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-12)).astype(np.float32)


def pick_device(name: str | None) -> str:
    if name and name != "auto":
        return name
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class LocalEmbedder:
    def __init__(self, model: str, device: str | None = None, batch_size: int = 16, query_prefix: str = ""):
        from sentence_transformers import SentenceTransformer
        self.model, self.batch_size, self.query_prefix = model, batch_size, query_prefix
        self.device = pick_device(device)
        self.st = SentenceTransformer(model, device=self.device)

    def embed(self, texts: list[str]) -> np.ndarray:
        v = self.st.encode(texts, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=len(texts) > 64)
        return normalise(np.asarray(v, dtype=np.float32))


class OpenAIEmbedder:
    def __init__(self, model: str, base_url: str, api_key: str, batch_size: int = 32, query_prefix: str = "", timeout: float = 300.0):
        from openai import OpenAI
        self.model, self.batch_size, self.query_prefix = model, batch_size, query_prefix
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=5)

    def embed(self, texts: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            r = self.client.embeddings.create(model=self.model, input=texts[i:i + self.batch_size])
            out.extend(d.embedding for d in sorted(r.data, key=lambda d: d.index))
        return normalise(np.asarray(out, dtype=np.float32))


def make_embedder(section: dict) -> Embedder:
    if section.get("provider", "local") == "openai":
        return OpenAIEmbedder(section["model"], section["base_url"], env_or(section, "api_key_env", fallback="EMPTY"),
                              section.get("batch_size", 32), section.get("query_prefix") or "", section.get("timeout", 300.0))
    return LocalEmbedder(section["model"], section.get("device"), section.get("batch_size", 16), section.get("query_prefix") or "")
