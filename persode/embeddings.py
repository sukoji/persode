"""Pluggable embedding backends for the memory store.

The paper uses a Chroma vector database with model embeddings. To keep the core
reproducible with zero external services, the default backend here is a
deterministic **hashing embedder** (a hashed bag-of-words projected to a fixed
dimension and L2-normalised). It is fast, offline, and good enough to
demonstrate the retrieval behaviour.

Set ``PERSODE_EMBEDDER=sentence-transformers`` (and install the package) to use
real semantic embeddings instead.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import List, Sequence

import numpy as np


class BaseEmbedder:
    dim: int

    def embed(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self.embed(t) for t in texts])


class HashingEmbedder(BaseEmbedder):
    """Deterministic offline embedder: hashed bag-of-words → unit vector.

    Uses per-token hashing (the "hashing trick") with a signed hash to reduce
    collisions, then L2-normalises. Fully deterministic across runs and
    machines, which makes experiments reproducible.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9']+", text.lower())

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        for tok in self._tokenize(text):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


class SentenceTransformerEmbedder(BaseEmbedder):
    """Wrapper over ``sentence-transformers`` (optional dependency)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> np.ndarray:
        v = self._model.encode([text], normalize_embeddings=True)[0]
        return np.asarray(v, dtype=np.float64)

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(list(texts), normalize_embeddings=True),
            dtype=np.float64,
        )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for (optionally non-normalised) vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def get_embedder(name: str | None = None, **kwargs) -> BaseEmbedder:
    """Factory. ``name`` defaults to ``$PERSODE_EMBEDDER`` or ``hashing``."""
    name = (name or os.environ.get("PERSODE_EMBEDDER", "hashing")).lower()
    if name in ("hashing", "hash", "offline"):
        return HashingEmbedder(**kwargs)
    if name in ("sentence-transformers", "st", "sbert"):
        return SentenceTransformerEmbedder(**kwargs)
    raise ValueError(f"unknown embedder: {name!r}")
