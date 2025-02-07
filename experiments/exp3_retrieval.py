"""Experiment 3 — salience-aware RAG retrieval (the Memory Selection Block).

We compare three retrieval strategies over the same scenario and query set:

    1. recency-only    — the most recent memories (a short-buffer baseline)
    2. similarity-only — pure RAG (cosine similarity, α = 1)
    3. fused (Persode) — similarity ⊕ memory salience (Eq. 1 at retrieval time)

Metrics (all derived from fixed ground truth, not from what the scorer optimises):

  * **target-recall@k** — is the query's designated target memory in the top-k?
  * **target-MRR**      — mean reciprocal rank of the target (0 if not retrieved)
  * **topical-precision@k** — fraction of top-k whose embedding similarity to the
    query is ≥ 50 % of the target's similarity (penalises off-topic padding)
  * **long-term recall@k** — among queries whose target is older than the six-day
    short-term window, fraction where that target appears in top-k
  * **emotional-intrusion@k** — among *neutral-target* queries only, fraction of
    top-k slots occupied by high-E memories other than the target (salience bleed)

Outputs:
    results/exp3_retrieval.png
    results/exp3_retrieval.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _style as style  # noqa: E402
from persode.embeddings import cosine_similarity, get_embedder  # noqa: E402
from persode.memory import SHORT_TERM_WINDOW_DAYS, MemoryStrengthScorer  # noqa: E402
from persode.store import MemoryStore  # noqa: E402
from _scenario import (  # noqa: E402
    EVAL_QUERIES,
    EMOTION_THRESHOLD,
    NOW,
    build_memories,
    is_emotionally_significant,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

TOP_K = 3
TOPICAL_SIM_FRACTION = 0.5  # retrieved m is "on-topic" if sim(q,m) ≥ this × sim(q,target)


@dataclass(frozen=True)
class QueryCase:
    query: str
    target: str
    category: str


def _cases() -> list[QueryCase]:
    return [QueryCase(**q) for q in EVAL_QUERIES]


def _memory_by_event() -> dict[str, object]:
    return {m.event: m for m in build_memories()}


def build_store(w_similarity: float) -> MemoryStore:
    store = MemoryStore(
        embedder=get_embedder("hashing"),
        scorer=MemoryStrengthScorer(1, 1, 1),
        w_similarity=w_similarity,
    )
    store.add_many(build_memories())
    return store


def _ranked_events(name: str, w_similarity: float | None, query: str) -> list[str]:
    if name == "recency-only":
        store = build_store(0.5)
        return [m.event for m in store.recent(now=NOW)]
    store = build_store(w_similarity)
    hits = store.retrieve(query, top_k=len(store), now=NOW, reinforce=False)
    return [r.memory.event for r in hits]


def retrieve_events(name: str, w_similarity: float | None, query: str) -> list[str]:
    return _ranked_events(name, w_similarity, query)[:TOP_K]


def _target_sim(embedder, query: str, target_event: str) -> float:
    by_event = _memory_by_event()
    target = by_event[target_event]
    q = embedder.embed(query)
    t = embedder.embed(target.text)
    return cosine_similarity(q, t)


def _memory_sim(embedder, query: str, event: str) -> float:
    by_event = _memory_by_event()
    q = embedder.embed(query)
    m = embedder.embed(by_event[event].text)
    return cosine_similarity(q, m)


def _is_long_term_target(target_event: str) -> bool:
    return _memory_by_event()[target_event].age_days(NOW) > SHORT_TERM_WINDOW_DAYS


def eval_strategy(name: str, w_similarity: float | None) -> dict:
    embedder = get_embedder("hashing")
    cases = _cases()

    recalls, mrrs, topical, lt_hits, lt_total = [], [], [], [], 0
    intrusions, intrusion_slots = [], 0
    per_query = []

    for case in cases:
        ranked = _ranked_events(name, w_similarity, case.query)
        top = ranked[:TOP_K]
        tgt_sim = _target_sim(embedder, case.query, case.target)
        topical_thresh = TOPICAL_SIM_FRACTION * tgt_sim

        # target recall & MRR
        hit = case.target in top
        recalls.append(float(hit))
        if case.target in ranked:
            mrrs.append(1.0 / (ranked.index(case.target) + 1))
        else:
            mrrs.append(0.0)

        # topical precision@k
        on_topic = [
            _memory_sim(embedder, case.query, e) >= topical_thresh for e in top
        ]
        tp = float(np.mean(on_topic)) if on_topic else 0.0
        topical.append(tp)

        # long-term recall (stratified)
        lt = False
        if _is_long_term_target(case.target):
            lt_total += 1
            lt = case.target in top
            lt_hits.append(float(lt))

        # emotional intrusion (neutral-target queries only)
        intrusion = 0.0
        if case.category.startswith("neutral"):
            slots = len(top)
            intrusion_slots += slots
            other_emotional = sum(
                1 for e in top
                if e != case.target and is_emotionally_significant(_memory_by_event()[e])
            )
            intrusion = other_emotional / slots if slots else 0.0
            intrusions.append(intrusion)

        per_query.append({
            "query": case.query,
            "target": case.target,
            "category": case.category,
            "retrieved": top,
            "target_hit": hit,
            "target_rank": ranked.index(case.target) + 1 if case.target in ranked else None,
            "target_mrr": round(mrrs[-1], 4),
            "topical_precision": round(tp, 4),
            "topical_threshold": round(topical_thresh, 4),
            "long_term_target": _is_long_term_target(case.target),
            "long_term_hit": lt if _is_long_term_target(case.target) else None,
            "emotional_intrusion": round(intrusion, 4) if case.category.startswith("neutral") else None,
        })

    return {
        "target_recall": float(np.mean(recalls)),
        "target_mrr": float(np.mean(mrrs)),
        "topical_precision": float(np.mean(topical)),
        "long_term_recall": float(np.mean(lt_hits)) if lt_total else None,
        "long_term_query_count": lt_total,
        "emotional_intrusion": float(np.mean(intrusions)) if intrusions else None,
        "neutral_query_count": len(intrusions),
        "per_query": per_query,
    }


def main() -> None:
    strategies = {
        "recency-only":    eval_strategy("recency-only", None),
        "similarity-only": eval_strategy("similarity-only", 1.0),
        "fused (Persode)": eval_strategy("fused", 0.5),
    }

    lt_targets = sorted(
        q["target"] for q in EVAL_QUERIES if _is_long_term_target(q["target"])
    )
    print(f"long-term targets (age > {SHORT_TERM_WINDOW_DAYS:g}d): {lt_targets}")
    print(f"queries: {len(EVAL_QUERIES)}  (E≥{EMOTION_THRESHOLD} = significant)\n")

    for name, res in strategies.items():
        print(f"### {name}")
        print(f"  target-recall@{TOP_K}     = {res['target_recall']:.2f}")
        print(f"  target-MRR              = {res['target_mrr']:.3f}")
        print(f"  topical-precision@{TOP_K} = {res['topical_precision']:.2f}")
        if res["long_term_recall"] is not None:
            print(f"  long-term recall@{TOP_K}  = {res['long_term_recall']:.2f}  "
                  f"(n={res['long_term_query_count']})")
        if res["emotional_intrusion"] is not None:
            print(f"  emotional-intrusion@{TOP_K} = {res['emotional_intrusion']:.2f}  "
                  f"(neutral queries, n={res['neutral_query_count']})")

    # ---- grouped bar chart ----
    style.apply()
    metrics = [
        (f"target-recall@{TOP_K}", "target_recall", style.BLUE),
        (f"topical-precision@{TOP_K}", "topical_precision", style.AQUA),
        (f"long-term recall@{TOP_K}", "long_term_recall", style.YELLOW),
        (f"emotional intrusion@{TOP_K} ↓", "emotional_intrusion", style.VIOLET),
    ]
    names = list(strategies.keys())
    x = np.arange(len(names))
    w = 0.19
    gap = 0.02
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, key, color) in enumerate(metrics):
        vals = []
        for n in names:
            v = strategies[n].get(key)
            vals.append(0.0 if v is None else v)
        offset = (i - 1.5) * (w + gap)
        bars = ax.bar(x + offset, vals, w, label=label, color=color)
        for b in bars:
            if b.get_height() > 0:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                        f"{b.get_height():.2f}", ha="center", fontsize=8.5,
                        color=style.INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, color=style.INK)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("score (lower is better for intrusion)")
    ax.set_title(f"Memory Selection Block — retrieval vs baselines "
                 f"({len(EVAL_QUERIES)} queries, top-{TOP_K})")
    ax.legend(loc="upper left", ncol=1, fontsize=9)
    style.style_axes(ax)
    fig.tight_layout()
    out_png = RESULTS / "exp3_retrieval.png"
    fig.savefig(out_png, bbox_inches="tight")
    print(f"\nsaved {out_png}")

    payload = {
        "reference_now": NOW.isoformat(),
        "top_k": TOP_K,
        "query_count": len(EVAL_QUERIES),
        "emotion_threshold": EMOTION_THRESHOLD,
        "topical_sim_fraction": TOPICAL_SIM_FRACTION,
        "long_term_targets": lt_targets,
        "strategies": strategies,
    }
    (RESULTS / "exp3_retrieval.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
