"""Shared Exp. 3 evaluation — used by exp3_retrieval.py and the regression tests.

The protocol here is **pre-registered**: every value in :class:`Exp3Config` is
fixed a priori from the system's own defaults, *never* tuned against the
evaluation results. Metrics are computed over the full query set (one probe per
scenario memory) and reported with a per-category breakdown; no query subset is
selected post hoc for the headline.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from persode.analyzer import SIGNIFICANCE_THRESHOLD, EventEmotionAnalyzer
from persode.embeddings import cosine_similarity, get_embedder
from persode.memory import MemoryStrengthScorer
from persode.store import MemoryStore
from _scenario import (
    NOW,
    build_eval_queries,
    build_memories,
    is_emotionally_significant,
)


@dataclass(frozen=True)
class Exp3Config:
    """Pre-registered evaluation protocol.

    - ``alpha`` = 0.5: the store's shipped default (balanced fusion).
    - weights (1, 1, 1) and ``protection`` = 0.9: the scorer's shipped defaults.
    - ``top_k`` = 3: the agent's default retrieval depth.
    - ``topical_sim_fraction`` = 0.5: fixed metric threshold (a retrieved memory
      counts as on-topic when its query-similarity is at least half the
      target's).
    - ``paraphrase``: evaluation condition — "vague" probes every memory with a
      lexically-distant paraphrase (the realistic journaling case: users never
      repeat an episode verbatim); "default" uses plainly-worded probes. Both
      conditions are always reported.
    """

    alpha: float = 0.5
    w_emotion: float = 1.0
    w_recall: float = 1.0
    w_context: float = 1.0
    protection: float = 0.9
    top_k: int = 3
    topical_sim_fraction: float = 0.5
    paraphrase: str = "vague"


STRATEGIES = ("recency-only", "similarity-only", "salience-only",
              "fused (Persode)", "gated (Persode)")


@lru_cache(maxsize=512)
def query_is_significant(query: str) -> bool:
    """The agent's emotion gate: analyzer E ≥ SIGNIFICANCE_THRESHOLD."""
    return EventEmotionAnalyzer().analyze(query).emotional_intensity >= SIGNIFICANCE_THRESHOLD


@lru_cache(maxsize=1)
def _embedder():
    return get_embedder("hashing")


@lru_cache(maxsize=256)
def _text_vec(text: str):
    return _embedder().embed(text)


@lru_cache(maxsize=1)
def _memory_by_event():
    return {m.event: m for m in build_memories()}


def build_store(cfg: Exp3Config, w_similarity: float) -> MemoryStore:
    store = MemoryStore(
        embedder=_embedder(),
        scorer=MemoryStrengthScorer(
            cfg.w_emotion, cfg.w_recall, cfg.w_context, protection=cfg.protection
        ),
        w_similarity=w_similarity,
    )
    store.add_many(build_memories())
    return store


def _ranked_events(name: str, cfg: Exp3Config, query: str) -> list[str]:
    if name == "recency-only":
        # Rank the WHOLE store by recency. (Truncating to the six-day window
        # would zero out every long-term query by construction — a strawman.)
        mems = sorted(build_memories(), key=lambda m: m.created_at, reverse=True)
        return [m.event for m in mems]
    if name == "salience-only":
        # Similarity-free ranking: α = 0 and no query-relevance override of C,
        # so the query text plays no role at all (pure Eq.-1 salience).
        store = build_store(cfg, 0.0)
        hits = store.retrieve(query, top_k=len(store), now=NOW, reinforce=False,
                              use_query_relevance_as_context=False)
        return [r.memory.event for r in hits]
    if name == "similarity-only":
        alpha = 1.0
    elif name == "gated (Persode)":
        # The agent's emotion gate (agent._retrieval_alpha): fusion for
        # emotionally significant queries, pure similarity for factual ones.
        alpha = cfg.alpha if query_is_significant(query) else 1.0
    else:
        alpha = cfg.alpha
    store = build_store(cfg, alpha)
    hits = store.retrieve(query, top_k=len(store), now=NOW, reinforce=False)
    return [r.memory.event for r in hits]


def eval_strategy(name: str, cfg: Exp3Config) -> dict[str, Any]:
    by_event = _memory_by_event()

    recalls, mrrs, topical = [], [], []
    by_cat_hits: dict[str, list[float]] = {}
    intrusions = []
    per_query = []

    for q in build_eval_queries(cfg.paraphrase):
        ranked = _ranked_events(name, cfg, q["query"])
        top = ranked[: cfg.top_k]
        target = q["target"]

        q_emb = _text_vec(q["query"])
        tgt_emb = _text_vec(by_event[target].text)
        tgt_sim = cosine_similarity(q_emb, tgt_emb)
        topical_thresh = cfg.topical_sim_fraction * tgt_sim

        hit = target in top
        recalls.append(float(hit))
        mrrs.append(1.0 / (ranked.index(target) + 1) if target in ranked else 0.0)
        by_cat_hits.setdefault(q["category"], []).append(float(hit))

        on_topic = [
            cosine_similarity(q_emb, _text_vec(by_event[e].text)) >= topical_thresh
            for e in top
        ]
        topical.append(float(sum(on_topic) / len(on_topic)) if on_topic else 0.0)

        if q["category"].startswith("neutral"):
            other = sum(
                1 for e in top
                if e != target and is_emotionally_significant(by_event[e])
            )
            intrusions.append(other / len(top) if top else 0.0)

        per_query.append({
            "query": q["query"],
            "target": target,
            "category": q["category"],
            "retrieved": top,
            "target_hit": hit,
            "target_rank": ranked.index(target) + 1 if target in ranked else None,
        })

    return {
        "target_recall": float(sum(recalls) / len(recalls)) if recalls else 0.0,
        "target_mrr": float(sum(mrrs) / len(mrrs)) if mrrs else 0.0,
        "topical_precision": float(sum(topical) / len(topical)) if topical else 0.0,
        "recall_by_category": {
            c: float(sum(h) / len(h)) for c, h in sorted(by_cat_hits.items())
        },
        "query_count_by_category": {c: len(h) for c, h in sorted(by_cat_hits.items())},
        "emotional_intrusion": float(sum(intrusions) / len(intrusions)) if intrusions else None,
        "neutral_query_count": len(intrusions),
        "query_count": len(recalls),
        "per_query": per_query,
    }


def eval_all(cfg: Exp3Config) -> dict[str, dict]:
    return {name: eval_strategy(name, cfg) for name in STRATEGIES}
