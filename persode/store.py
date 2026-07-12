"""Vector memory store and the Memory Selection Block (RAG retrieval).

This module reproduces the retrieval half of Persode's episodic memory system
(Figure 2, blocks "Chroma Vector Data Base", "Memory Selection Block" and
"Memory Indexing, Ranking"):

- memories are embedded and stored with their scoring metadata;
- retrieval *fuses* semantic similarity (the RAG part) with the Memory-Strength
  score S (the Ebbinghaus / emotional-salience part), so that responses are
  grounded in memories that are both **relevant** and **emotionally significant**;
- retrieving a memory **reinforces** it (recall frequency ↑, decay clock reset).

A pure-numpy backend is used by default; a Chroma backend can be swapped in by
anyone who installs ``chromadb`` (the fusion logic is identical either way).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .embeddings import BaseEmbedder, cosine_similarity, get_embedder
from .memory import Memory, MemoryStrengthScorer, SHORT_TERM_WINDOW_DAYS


@dataclass
class RetrievalResult:
    """One retrieved memory with the scores that got it selected."""

    memory: Memory
    similarity: float          # cosine similarity to the query
    salience: float            # decay-free weighted avg of E,R,C (retrieval-time Eq.1)
    retention: float           # full Memory-Strength S with decay (store consolidation)
    fused_score: float         # final ranking score
    is_short_term: bool        # inside the six-day recent window?

    def to_dict(self) -> dict:
        d = self.memory.to_dict()
        d.update(
            similarity=round(self.similarity, 4),
            salience=round(self.salience, 4),
            retention=round(self.retention, 4),
            fused_score=round(self.fused_score, 4),
            is_short_term=self.is_short_term,
        )
        return d


class MemoryStore:
    """In-memory episodic store with strength-aware retrieval.

    Args:
        embedder: Embedding backend (defaults to the offline hashing embedder).
        scorer: Memory-Strength scorer implementing Eq. (1).
        w_similarity: Fusion weight ``α`` on semantic similarity. The strength
            term gets ``1 - α``. ``α = 1`` is pure RAG; ``α = 0`` is pure
            salience-based recall. The default balances the two.
    """

    def __init__(
        self,
        embedder: Optional[BaseEmbedder] = None,
        scorer: Optional[MemoryStrengthScorer] = None,
        w_similarity: float = 0.5,
    ) -> None:
        self.embedder = embedder or get_embedder()
        self.scorer = scorer or MemoryStrengthScorer()
        self.w_similarity = float(w_similarity)
        self._memories: List[Memory] = []

    # -- ingestion --------------------------------------------------------
    def add(self, memory: Memory) -> Memory:
        if memory.embedding is None:
            memory.embedding = self.embedder.embed(memory.text).tolist()
        self._memories.append(memory)
        return memory

    def add_many(self, memories: Sequence[Memory]) -> None:
        for m in memories:
            self.add(m)

    def __len__(self) -> int:
        return len(self._memories)

    @property
    def memories(self) -> List[Memory]:
        return list(self._memories)

    # -- retrieval (Memory Selection Block) ------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        now: Optional[datetime] = None,
        reinforce: bool = True,
        use_query_relevance_as_context: bool = True,
    ) -> List[RetrievalResult]:
        """Retrieve the top-k memories fusing similarity with memory salience.

        The fused score is::

            fused = α · similarity + (1 - α) · salience

        where ``salience`` is Eq. (1) evaluated with ``d(Δt) = 1`` — the
        "normalized weighted average used at retrieval time" the paper describes.
        Using the decay-free salience (rather than the fully-decayed strength)
        is what lets an emotionally significant *long-term* memory resurface
        alongside the most on-topic ones, instead of being erased by its age.
        The contextual-relevance term C can optionally be supplied by the query
        similarity ("ensuring contextual relevance through effective retrieval").

        Note that when ``use_query_relevance_as_context`` is true, similarity
        also enters the salience term through C, so the *effective* weight on
        similarity is ``α + (1-α)·wC/(wE+wR+wC)`` — e.g. with equal weights and
        α = 0, similarity still contributes 1/3 of the fused score. Pass
        ``use_query_relevance_as_context=False`` for a fusion whose salience
        term is fully similarity-free (the memory's stored C is used instead).

        The full decayed strength is still reported as ``retention`` for
        transparency. When ``reinforce`` is true, every returned memory is marked
        as recalled, modelling the reinforcement of significant memories.
        """
        now = now or datetime.now(timezone.utc)
        if not self._memories:
            return []

        q = self.embedder.embed(query)
        results: List[RetrievalResult] = []
        for m in self._memories:
            emb = np.asarray(m.embedding, dtype=np.float64)
            sim = cosine_similarity(q, emb)
            context_rel = sim if use_query_relevance_as_context else None
            salience = self.scorer.base_salience(m, context_relevance=context_rel)
            retention = self.scorer.score(m, now=now, context_relevance=context_rel)
            fused = self.w_similarity * sim + (1.0 - self.w_similarity) * salience
            results.append(
                RetrievalResult(
                    memory=m,
                    similarity=sim,
                    salience=salience,
                    retention=retention,
                    fused_score=fused,
                    is_short_term=m.is_short_term(now),
                )
            )

        results.sort(key=lambda r: r.fused_score, reverse=True)
        top = results[:top_k]

        if reinforce:
            for r in top:
                r.memory.mark_recalled(now)
        return top

    def recent(self, now: Optional[datetime] = None,
               window_days: float = SHORT_TERM_WINDOW_DAYS) -> List[Memory]:
        """Short-term buffer: memories within the recent window (paper §3.2)."""
        now = now or datetime.now(timezone.utc)
        recents = [m for m in self._memories if m.is_short_term(now, window_days)]
        recents.sort(key=lambda m: m.created_at, reverse=True)
        return recents

    # -- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(
            json.dumps([m.to_dict() for m in self._memories], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def load(self, path: str | Path) -> None:
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        self._memories = [Memory.from_dict(d) for d in data]
