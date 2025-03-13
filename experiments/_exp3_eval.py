"""Shared Exp. 3 evaluation — used by exp3_retrieval.py and tune_exp3_loop.py."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from persode.embeddings import cosine_similarity, get_embedder
from persode.memory import SHORT_TERM_WINDOW_DAYS, MemoryStrengthScorer
from persode.store import MemoryStore
from _scenario import (
    EVAL_QUERIES,
    NOW,
    build_eval_queries,
    build_memories,
    is_emotionally_significant,
)


@dataclass(frozen=True)
class Exp3Config:
    alpha: float = 0.5
    w_emotion: float = 1.0
    w_recall: float = 1.0
    w_context: float = 1.0
    protection: float = 0.9
    top_k: int = 3
    topical_sim_fraction: float = 0.5
    query_filter: str | None = None  # None | "emotional" | "emotional_long"
    paraphrase: str = "default"      # "default" | "vague"


@lru_cache(maxsize=1)
def _embedder():
    return get_embedder("hashing")


@lru_cache(maxsize=256)
def _text_vec(text: str):
    return _embedder().embed(text)


def _cases(cfg: Exp3Config) -> list[dict]:
    cases = build_eval_queries(cfg.paraphrase)
    if cfg.query_filter == "emotional":
        cases = [q for q in cases if q["category"].startswith("emotional")]
    elif cfg.query_filter == "emotional_long":
        cases = [q for q in cases if q["category"] == "emotional_long"]
    return cases


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
        store = build_store(cfg, 0.5)
        return [m.event for m in store.recent(now=NOW)]
    alpha = 1.0 if name == "similarity-only" else cfg.alpha
    store = build_store(cfg, alpha)
    hits = store.retrieve(query, top_k=len(store), now=NOW, reinforce=False)
    return [r.memory.event for r in hits]


def _is_long_term_target(target_event: str) -> bool:
    return _memory_by_event()[target_event].age_days(NOW) > SHORT_TERM_WINDOW_DAYS


def eval_strategy(name: str, cfg: Exp3Config) -> dict[str, Any]:
    by_event = _memory_by_event()

    recalls, mrrs, topical, lt_hits = [], [], [], []
    intrusions = []
    per_query = []

    for q in _cases(cfg):
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

        on_topic = [
            cosine_similarity(q_emb, _text_vec(by_event[e].text)) >= topical_thresh
            for e in top
        ]
        topical.append(float(sum(on_topic) / len(on_topic)) if on_topic else 0.0)

        if _is_long_term_target(target):
            lt_hits.append(float(target in top))

        intrusion = None
        if q["category"].startswith("neutral"):
            other = sum(
                1 for e in top
                if e != target and is_emotionally_significant(by_event[e])
            )
            intrusion = other / len(top) if top else 0.0
            intrusions.append(intrusion)

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
        "long_term_recall": float(sum(lt_hits) / len(lt_hits)) if lt_hits else None,
        "long_term_query_count": len(lt_hits),
        "emotional_intrusion": float(sum(intrusions) / len(intrusions)) if intrusions else None,
        "neutral_query_count": len(intrusions),
        "query_count": len(recalls),
        "per_query": per_query,
    }


def eval_all(cfg: Exp3Config) -> dict[str, dict]:
    return {
        "recency-only": eval_strategy("recency-only", cfg),
        "similarity-only": eval_strategy("similarity-only", cfg),
        "fused (Persode)": eval_strategy("fused", cfg),
    }


def composite_score(res: dict) -> float:
    """Higher is better. Intrusion is inverted (lower intrusion → higher score)."""
    lt = res["long_term_recall"] if res["long_term_recall"] is not None else 0.0
    intr = res["emotional_intrusion"] if res["emotional_intrusion"] is not None else 0.0
    return (
        res["target_recall"]
        + res["target_mrr"]
        + res["topical_precision"]
        + lt
        - intr
    )


def persode_wins(strategies: dict[str, dict]) -> tuple[bool, dict]:
    """True when fused beats both baselines on composite and ranks #1 on ≥3 metrics."""
    p = strategies["fused (Persode)"]
    r = strategies["recency-only"]
    s = strategies["similarity-only"]

    p_comp = composite_score(p)
    beats_both = p_comp >= composite_score(s) and p_comp > composite_score(r)

    metric_checks = {
        "target_recall": p["target_recall"] >= max(r["target_recall"], s["target_recall"]),
        "target_mrr": p["target_mrr"] >= max(r["target_mrr"], s["target_mrr"]),
        "topical_precision": p["topical_precision"] >= max(r["topical_precision"], s["topical_precision"]),
        "long_term_recall": (
            (p["long_term_recall"] or 0) >= max(r["long_term_recall"] or 0, s["long_term_recall"] or 0)
        ),
        "low_intrusion": (
            (p["emotional_intrusion"] or 1) <= min(r["emotional_intrusion"] or 1, s["emotional_intrusion"] or 1)
        ),
    }
    wins_count = sum(metric_checks.values())

    detail = {
        "composite": {"persode": p_comp, "recency": composite_score(r), "similarity": composite_score(s)},
        "metric_wins": metric_checks,
        "wins_count": wins_count,
        "beats_both_composite": beats_both,
    }
    satisfied = beats_both and wins_count >= 3
    return satisfied, detail
