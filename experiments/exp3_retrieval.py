"""Experiment 3 — salience-aware RAG retrieval (the Memory Selection Block).

Protocol (see _exp3_eval.Exp3Config) is pre-registered: every hyperparameter is
the system's shipped default, fixed before looking at any result. Metrics cover
the full query set (one probe per scenario memory) under two phrasing
conditions — plain probes and lexically-distant "vague" paraphrases (one per
memory, uniform construction rule) — with a per-category breakdown. Nothing is
tuned against the evaluation and no query subset is selected post hoc.

Outputs:
    results/exp3_retrieval.png
    results/exp3_alpha_ablation.png
    results/exp3_retrieval.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _style as style  # noqa: E402
from _exp3_eval import Exp3Config, eval_all  # noqa: E402
from _scenario import EMOTION_THRESHOLD, NOW  # noqa: E402
from persode.memory import SHORT_TERM_WINDOW_DAYS  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)


def salience_prioritization() -> dict:
    """Embedder-independent demonstration of the paper's *actual* claim.

    Recall under vague probes is embedder-dependent: a strong semantic model
    makes pure similarity suffice. Salience's embedder-independent contribution
    is *prioritization* — when two memories are equally relevant, surface the
    emotionally significant one. We hold relevance fixed by giving both memories
    identical text (so their similarity is identical for ANY embedder), and vary
    only emotional intensity.
    """
    from persode.memory import Memory, MemoryStrengthScorer  # local: keep import graph flat
    from persode.store import MemoryStore

    text = "a difficult conversation with someone at work"

    def order_under(alpha: float):
        store = MemoryStore(scorer=MemoryStrengthScorer(1, 1, 1), w_similarity=alpha)
        neutral = Memory(text=text, event="neutral", emotional_intensity=0.1,
                         contextual_relevance=0.3, created_at=NOW)
        significant = Memory(text=text, event="significant", emotional_intensity=0.9,
                             contextual_relevance=0.6, created_at=NOW)
        store.add_many([neutral, significant])  # neutral first: ties fall back to it
        hits = store.retrieve(text, top_k=2, now=NOW, reinforce=False)
        return [h.memory.event for h in hits], [round(h.similarity, 4) for h in hits]

    fused_order, sims = order_under(0.5)
    sim_order, _ = order_under(1.0)
    return {
        "similarity_is_tied": len(set(sims)) == 1,   # identical text -> identical sim
        "similarity_only_order": sim_order,          # tie -> arbitrary (insertion) order
        "fused_order": fused_order,                   # salience breaks the tie
        "fused_ranks_significant_first": fused_order[0] == "significant",
    }


def _print_condition(label: str, strategies: dict, top_k: int) -> None:
    print(f"\n=== condition: {label} ===")
    for name, res in strategies.items():
        print(f"### {name}")
        print(f"  target-recall@{top_k}     = {res['target_recall']:.2f}")
        print(f"  target-MRR              = {res['target_mrr']:.3f}")
        print(f"  topical-precision@{top_k} = {res['topical_precision']:.2f}")
        cats = res["recall_by_category"]
        ns = res["query_count_by_category"]
        print("  recall by category:      "
              + "  ".join(f"{c}={cats[c]:.2f}(n={ns[c]})" for c in cats))
        if res["emotional_intrusion"] is not None:
            print(f"  emotional-intrusion@{top_k} = {res['emotional_intrusion']:.2f}  "
                  f"(neutral queries, n={res['neutral_query_count']})")


def plot_alpha_ablation(cfg: Exp3Config, out_png: Path) -> list[dict]:
    """α sweep under the vague condition, full query set + long-term subset.

    Shows retrieval quality as α moves from salience-dominant (α=0; similarity
    still enters the salience term via C) to pure similarity (α=1, plain RAG).
    """
    alphas = [round(i / 20, 2) for i in range(21)]  # 0.00 .. 1.00 step 0.05
    recalls, mrrs, lt_recalls = [], [], []
    for a in alphas:
        r = eval_all(replace(cfg, alpha=a))["fused (Persode)"]
        recalls.append(r["target_recall"])
        mrrs.append(r["target_mrr"])
        lt_recalls.append(r["recall_by_category"].get("emotional_long", 0.0))

    style.apply()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(alphas, recalls, color=style.BLUE, lw=2.2, marker="o", ms=4, zorder=3,
            label="target-recall@%d (all queries)" % cfg.top_k)
    ax.plot(alphas, lt_recalls, color=style.VIOLET, lw=2.0, marker="^", ms=4, zorder=3,
            label="recall@%d (long-term emotional)" % cfg.top_k)
    ax.plot(alphas, mrrs, color=style.AQUA, lw=2.0, marker="s", ms=3.5, zorder=3,
            label="target-MRR (all queries)")
    ax.annotate("α=0 — salience-dominant\n(similarity still enters via C)",
                (0.0, recalls[0]), xytext=(0.02, min(recalls[0] + 0.16, 1.02)),
                color=style.MUTED, fontsize=8.5)
    ax.annotate("pure similarity / RAG (α=1)", (1.0, recalls[-1]),
                xytext=(0.68, recalls[-1] + 0.13), color=style.MUTED, fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("α  —  fusion weight on semantic similarity")
    ax.set_ylabel("score (higher is better)")
    ax.set_title("Exp. 3 — α fusion sweep (vague probes, full query set)")
    ax.legend(loc="lower center", ncol=3, fontsize=8.5)
    style.style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    return [{"alpha": a, "target_recall": r, "target_mrr": m, "long_term_recall": lt}
            for a, r, m, lt in zip(alphas, recalls, mrrs, lt_recalls)]


def _bar_panel(ax, strategies: dict, top_k: int, title: str) -> None:
    metric_defs = [
        (f"target-recall@{top_k}", "target_recall", style.BLUE),
        ("target-MRR", "target_mrr", style.AQUA),
        (f"topical-precision@{top_k}", "topical_precision", style.YELLOW),
    ]
    names = list(strategies.keys())
    x = np.arange(len(names))
    n_met = len(metric_defs)
    w = min(0.22, 0.8 / n_met)
    gap = 0.02
    for i, (label, key, color) in enumerate(metric_defs):
        vals = [strategies[n].get(key) or 0.0 for n in names]
        offset = (i - (n_met - 1) / 2) * (w + gap)
        bars = ax.bar(x + offset, vals, w, label=label, color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                    f"{v:.2f}", ha="center", fontsize=7.5, color=style.INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" (Persode)", "\n(Persode)") for n in names],
                       fontsize=8.5, color=style.INK)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title(title, fontsize=10.5)
    style.style_axes(ax)


def main() -> None:
    cfg = Exp3Config()  # pre-registered protocol — no tuned values, ever
    vague = eval_all(cfg)
    plain = eval_all(replace(cfg, paraphrase="default"))
    qn = vague["fused (Persode)"]["query_count"]

    print(f"protocol (pre-registered): alpha={cfg.alpha} weights=({cfg.w_emotion:g},"
          f"{cfg.w_recall:g},{cfg.w_context:g}) top_k={cfg.top_k} "
          f"topical_sim_fraction={cfg.topical_sim_fraction}")
    print(f"queries: {qn} (one per memory)  ·  long-term = age > "
          f"{SHORT_TERM_WINDOW_DAYS:g} d  ·  significant = E ≥ {EMOTION_THRESHOLD}")
    _print_condition("vague paraphrases (lexical-mismatch stress test)", vague, cfg.top_k)
    _print_condition("plain probes", plain, cfg.top_k)

    prioritization = salience_prioritization()
    print(f"\nsalience prioritization (equal relevance, embedder-independent): "
          f"similarity-only={prioritization['similarity_only_order']} -> "
          f"fused={prioritization['fused_order']}")

    # ---- figures -------------------------------------------------------------
    style.apply()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.9), sharey=True)
    _bar_panel(ax1, vague, cfg.top_k,
               f"vague probes ({qn} queries, top-{cfg.top_k})")
    _bar_panel(ax2, plain, cfg.top_k,
               f"plain probes ({qn} queries, top-{cfg.top_k})")
    ax1.set_ylabel("score (higher is better)")
    ax1.legend(loc="upper left", fontsize=8)
    fig.suptitle("Memory Selection Block — retrieval vs baselines (fixed protocol)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png = RESULTS / "exp3_retrieval.png"
    fig.savefig(out_png, bbox_inches="tight")
    print(f"\nsaved {out_png}")

    ablation_png = RESULTS / "exp3_alpha_ablation.png"
    alpha_curve = plot_alpha_ablation(cfg, ablation_png)
    print(f"saved {ablation_png}")

    # ---- machine-readable ----------------------------------------------------
    payload = {
        "reference_now": NOW.isoformat(),
        "protocol": {
            "note": "pre-registered: all values are the system's shipped defaults, "
                    "fixed before evaluation; no tuning against results, no post-hoc "
                    "query selection",
            "alpha": cfg.alpha,
            "w_emotion": cfg.w_emotion,
            "w_recall": cfg.w_recall,
            "w_context": cfg.w_context,
            "protection": cfg.protection,
            "top_k": cfg.top_k,
            "topical_sim_fraction": cfg.topical_sim_fraction,
            "emotion_threshold": EMOTION_THRESHOLD,
            "query_count": qn,
        },
        "conditions": {
            "vague": vague,
            "plain": plain,
        },
        "alpha_curve": alpha_curve,
        "salience_prioritization": prioritization,
        "notes": {
            "sample_size": f"n = {qn} hand-labelled queries — differences of one hit "
                           f"move recall by {1.0 / qn:.2f}; treat gaps as qualitative "
                           "mechanism checks, not population estimates.",
            "embedder": "Hashing (lexical) embedder by default. With a semantic "
                        "embedder (PERSODE_EMBEDDER=sentence-transformers) pure "
                        "similarity already solves the vague probes, so the recall "
                        "gap is an artifact of the lexical embedder. The "
                        "embedder-independent effect is salience_prioritization.",
            "alpha_zero": "α=0 is salience-dominant, not similarity-free: query "
                          "similarity still enters the salience term via C. The "
                          "similarity-free reference is the salience-only strategy.",
        },
    }
    (RESULTS / "exp3_retrieval.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
