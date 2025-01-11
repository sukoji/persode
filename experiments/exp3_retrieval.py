"""Experiment 3 — salience-aware RAG retrieval (the Memory Selection Block).

We compare three retrieval strategies over the same scenario and query set:

    1. recency-only    — the most recent memories (a short-buffer baseline)
    2. similarity-only — pure RAG (cosine similarity, α = 1)
    3. fused (Persode) — similarity ⊕ memory salience (Eq. 1 at retrieval time)

Two metrics, reported together:

  * **sig-precision@k** — fraction of retrieved memories that are emotionally
    significant (ground-truth labelled in the scenario).
  * **long-term recall@k** — fraction of queries for which an emotionally
    significant memory *older than the six-day short-term window* is surfaced.
    A recency buffer structurally scores 0 here: it can never reach past its
    window. This is precisely the long-term store the paper argues for.

Outputs:
    results/exp3_retrieval.png
    results/exp3_retrieval.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _style as style  # noqa: E402
from persode.embeddings import get_embedder  # noqa: E402
from persode.memory import SHORT_TERM_WINDOW_DAYS, MemoryStrengthScorer  # noqa: E402
from persode.store import MemoryStore  # noqa: E402
from _scenario import EMOTIONALLY_SIGNIFICANT, NOW, build_memories  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

QUERIES = [
    "I'm feeling really happy about a big achievement today",
    "Something ruined my day and I'm upset about my clothes",
    "I miss someone I loved very much and I'm grieving",
    "I'm nervous and anxious about an upcoming exam",
]
# Topical ground truth: the one memory each query is *about*. sig-precision
# alone would reward returning any emotional memory regardless of topic, so we
# also require the topically correct memory to be retrieved (target-recall@k).
QUERY_TARGET = {
    QUERIES[0]: "graduation ceremony celebration",
    QUERIES[1]: "car splashed water ruined favorite outfit",
    QUERIES[2]: "lost beloved dog",
    QUERIES[3]: "anxious before final exam",
}
TOP_K = 3

# Emotionally significant memories that live *outside* the short-term window.
_long_term_significant = {
    m.event for m in build_memories()
    if m.event in EMOTIONALLY_SIGNIFICANT and m.age_days(NOW) > SHORT_TERM_WINDOW_DAYS
}


def significant(event: str) -> bool:
    return event in EMOTIONALLY_SIGNIFICANT


def build_store(w_similarity: float) -> MemoryStore:
    store = MemoryStore(
        embedder=get_embedder("hashing"),
        scorer=MemoryStrengthScorer(1, 1, 1),
        w_similarity=w_similarity,
    )
    store.add_many(build_memories())
    return store


def retrieve_events(name: str, w_similarity: float | None, query: str) -> list[str]:
    if name == "recency-only":
        store = build_store(0.5)
        return [m.event for m in store.recent(now=NOW)[:TOP_K]]
    store = build_store(w_similarity)
    hits = store.retrieve(query, top_k=TOP_K, now=NOW, reinforce=False)
    return [r.memory.event for r in hits]


def eval_strategy(name: str, w_similarity: float | None) -> dict:
    precisions, tgt_hits, lt_hits, per_query = [], [], [], []
    for q in QUERIES:
        events = retrieve_events(name, w_similarity, q)
        p = float(np.mean([significant(e) for e in events])) if events else 0.0
        tgt = QUERY_TARGET[q] in events
        lt = any(e in _long_term_significant for e in events)
        precisions.append(p)
        tgt_hits.append(tgt)
        lt_hits.append(lt)
        per_query.append({"query": q, "retrieved": events,
                          "precision": round(p, 3), "target_hit": tgt,
                          "long_term_hit": lt})
    return {
        "mean_precision": float(np.mean(precisions)),
        "target_recall": float(np.mean(tgt_hits)),
        "long_term_recall": float(np.mean(lt_hits)),
        "per_query": per_query,
    }


def main() -> None:
    strategies = {
        "recency-only":    eval_strategy("recency-only", None),
        "similarity-only": eval_strategy("similarity-only", 1.0),
        "fused (Persode)": eval_strategy("fused", 0.5),
    }

    print(f"long-term significant memories (age > {SHORT_TERM_WINDOW_DAYS:g}d): {sorted(_long_term_significant)}")
    for name, res in strategies.items():
        print(f"\n### {name}  sig-precision@{TOP_K}={res['mean_precision']:.2f}  "
              f"target-recall@{TOP_K}={res['target_recall']:.2f}  "
              f"long-term-recall@{TOP_K}={res['long_term_recall']:.2f}")
        for pq in res["per_query"]:
            flag = ("" + (" [TGT]" if pq["target_hit"] else "")
                    + (" [LT]" if pq["long_term_hit"] else ""))
            print(f"  Q: {pq['query'][:44]:44s} -> {pq['retrieved']}{flag}")

    # ---- grouped bar chart: three metrics ----
    style.apply()
    metrics = [
        (f"sig-precision@{TOP_K}", "mean_precision", style.BLUE),
        (f"target-recall@{TOP_K}", "target_recall", style.AQUA),
        (f"long-term recall@{TOP_K}", "long_term_recall", style.YELLOW),
    ]
    names = list(strategies.keys())
    x = np.arange(len(names))
    w = 0.26
    gap = 0.02  # ~2px surface gap between adjacent bars
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, (label, key, color) in enumerate(metrics):
        vals = [strategies[n][key] for n in names]
        bars = ax.bar(x + (i - 1) * (w + gap), vals, w, label=label, color=color)
        for b in bars:  # direct value labels (contrast relief for aqua/yellow)
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.025,
                    f"{b.get_height():.2f}", ha="center", fontsize=9.5,
                    color=style.INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10.5, color=style.INK)
    ax.set_ylim(0, 1.14)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("score")
    ax.set_title("Memory Selection Block — retrieval quality vs baselines "
                 f"({len(QUERIES)} emotional queries, top-{TOP_K})")
    ax.legend(loc="upper left", ncol=1)
    style.style_axes(ax)
    fig.tight_layout()
    out_png = RESULTS / "exp3_retrieval.png"
    fig.savefig(out_png, bbox_inches="tight")
    print(f"\nsaved {out_png}")

    payload = {"reference_now": NOW.isoformat(), "top_k": TOP_K,
               "long_term_significant": sorted(_long_term_significant),
               "strategies": strategies}
    (RESULTS / "exp3_retrieval.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
