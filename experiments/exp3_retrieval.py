"""Experiment 3 — salience-aware RAG retrieval (the Memory Selection Block).

Loads tuned hyperparameters from results/exp3_tuned_config.json when present
(written by tune_exp3_loop.py). Otherwise uses paper defaults (α = 0.5).

Outputs:
    results/exp3_retrieval.png
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
from _scenario import EMOTION_THRESHOLD, EVAL_QUERIES, NOW, build_memories  # noqa: E402
from persode.memory import SHORT_TERM_WINDOW_DAYS  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
CONFIG_PATH = RESULTS / "exp3_tuned_config.json"
RESULTS.mkdir(exist_ok=True)


def load_config() -> Exp3Config:
    if CONFIG_PATH.exists():
        d = json.loads(CONFIG_PATH.read_text())["tuned"]
        return Exp3Config(**d)
    return Exp3Config()


def salience_prioritization() -> dict:
    """Embedder-independent demonstration of the paper's *actual* claim.

    The retrieval table above (recall on vague probes) is embedder-dependent: a
    strong semantic model makes pure similarity suffice. Salience's embedder-
    independent contribution is *prioritization* — when two memories are equally
    relevant, surface the emotionally significant one. We hold relevance fixed by
    giving both memories identical text (so their similarity is identical for
    ANY embedder), and vary only emotional intensity.
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


def _is_long_term_target(target: str) -> bool:
    by = {m.event: m for m in build_memories()}
    return by[target].age_days(NOW) > SHORT_TERM_WINDOW_DAYS


def plot_alpha_ablation(cfg: Exp3Config, out_png: Path) -> list[dict]:
    """Fine α sweep behind the fusion-ablation figure (scoped config).

    Shows retrieval quality as α moves from pure salience (α=0) to pure similarity
    (α=1, plain RAG), making the interior plateau — where fusing both signals wins —
    visible rather than asserted.
    """
    alphas = [round(i / 20, 2) for i in range(21)]  # 0.00 .. 1.00 step 0.05
    recalls, mrrs = [], []
    for a in alphas:
        r = eval_all(replace(cfg, alpha=a))["fused (Persode)"]
        recalls.append(r["target_recall"])
        mrrs.append(r["target_mrr"])

    style.apply()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    peak = max(recalls)
    plateau = [a for a, rr in zip(alphas, recalls) if rr >= peak - 1e-9]
    if plateau:
        ax.axvspan(min(plateau), max(plateau), color=style.GRID, alpha=0.5, lw=0, zorder=0,
                   label=f"recall plateau (α ∈ [{min(plateau):g}, {max(plateau):g}])")
    ax.plot(alphas, recalls, color=style.BLUE, lw=2.2, marker="o", ms=4, zorder=3,
            label="target-recall@4")
    ax.plot(alphas, mrrs, color=style.AQUA, lw=2.0, marker="s", ms=3.5, zorder=3,
            label="target-MRR")
    ax.scatter([0.0, 1.0], [recalls[0], recalls[-1]], s=46, color=style.VIOLET,
               edgecolor=style.SURFACE, linewidth=1.4, zorder=4)
    ax.annotate("pure salience (α=0)", (0.0, recalls[0]),
                xytext=(0.03, recalls[0] + 0.13), color=style.MUTED, fontsize=9)
    ax.annotate("pure similarity / RAG (α=1)", (1.0, recalls[-1]),
                xytext=(0.55, recalls[-1] + 0.13), color=style.MUTED, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("α  —  fusion weight on semantic similarity")
    ax.set_ylabel("score (higher is better)")
    ax.set_title("Exp. 3 — α fusion ablation (scoped: long-term emotional, vague probes)")
    ax.legend(loc="lower center", ncol=3, fontsize=8.5)
    style.style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    return [{"alpha": a, "target_recall": r, "target_mrr": m}
            for a, r, m in zip(alphas, recalls, mrrs)]


def main() -> None:
    cfg = load_config()
    strategies = eval_all(cfg)
    top_k = cfg.top_k

    lt_targets = sorted(q["target"] for q in EVAL_QUERIES if _is_long_term_target(q["target"]))
    qn = strategies["fused (Persode)"]["query_count"]
    print(f"config: alpha={cfg.alpha} top_k={top_k} filter={cfg.query_filter}")
    print(f"long-term targets (age > {SHORT_TERM_WINDOW_DAYS:g}d): {lt_targets}")
    print(f"queries: {qn}  (E≥{EMOTION_THRESHOLD} = significant)\n")

    for name, res in strategies.items():
        print(f"### {name}")
        print(f"  target-recall@{top_k}     = {res['target_recall']:.2f}")
        print(f"  target-MRR              = {res['target_mrr']:.3f}")
        print(f"  topical-precision@{top_k} = {res['topical_precision']:.2f}")
        if res["long_term_recall"] is not None:
            print(f"  long-term recall@{top_k}  = {res['long_term_recall']:.2f}  "
                  f"(n={res['long_term_query_count']})")
        if res["emotional_intrusion"] is not None:
            print(f"  emotional-intrusion@{top_k} = {res['emotional_intrusion']:.2f}  "
                  f"(neutral queries, n={res['neutral_query_count']})")

    # ---- robustness: does the headline survive design choices? --------------
    # The chart above is the *scoped* claim (long-term emotional recall under
    # lexically-distant probes). These three checks exist so the scoping cannot
    # be mistaken for cherry-picking; they are written to JSON and printed here.
    def _triple(strats: dict) -> dict:
        return {
            n: {k: round(v, 3) for k, v in r.items()
                if k in ("target_recall", "target_mrr", "topical_precision")}
            for n, r in strats.items()
        }

    full_set = _triple(eval_all(replace(cfg, query_filter=None)))          # all 10 queries
    scoped_plain = _triple(eval_all(replace(cfg, paraphrase="default")))    # scoped, non-vague
    alpha_grid = [
        {"alpha": a,
         **{k: round(eval_all(replace(cfg, alpha=a))["fused (Persode)"][k], 3)
            for k in ("target_recall", "target_mrr")}}
        for a in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    prioritization = salience_prioritization()
    robustness = {
        "full_set_vague": full_set,
        "scoped_plain_probes": scoped_plain,
        "alpha_sensitivity": alpha_grid,
        "salience_prioritization": prioritization,
        "embedder_note": "The recall table uses the hashing (lexical) embedder; a semantic "
                         "embedder makes similarity-only suffice. salience_prioritization is "
                         "embedder-independent (identical text => identical similarity).",
        "why_filter": "The paper's claim is scoped to emotionally-significant long-term "
                      "memories, so metrics are reported on exactly those queries.",
        "why_vague": "With verbatim-like probes, similarity-only already recalls the "
                     "long-term target (see scoped_plain_probes); the fusion gain only "
                     "appears when the probe is lexically distant from the stored episode.",
        "tradeoff": "Over the full query mix, fusion is net-neutral vs pure RAG "
                    "(see full_set_vague): it reallocates capacity toward long-term "
                    "emotional recall at a small cost on lexically-easy queries.",
    }
    print("\n--- robustness (why the scoping is not cherry-picking) ---")
    print(f"full set (10 q, vague)   fused vs sim recall: "
          f"{full_set['fused (Persode)']['target_recall']} vs "
          f"{full_set['similarity-only']['target_recall']}")
    print(f"scoped, non-vague probes  sim recall: "
          f"{scoped_plain['similarity-only']['target_recall']}  "
          f"(already solved → vague probes are the discriminating case)")
    print("alpha sweep (scoped) recall: "
          + ", ".join(f"{g['alpha']}:{g['target_recall']}" for g in alpha_grid))
    print(f"salience prioritization (equal relevance, embedder-independent): "
          f"similarity-only={prioritization['similarity_only_order']} -> "
          f"fused={prioritization['fused_order']}")

    style.apply()
    # Chart metrics must match the README table. Under the emotional_long filter
    # every probe is a long-term target, so long-term recall is identical to
    # target-recall by construction — showing it as a separate bar would double-
    # count the same win, so it is reported in JSON only, not plotted.
    metric_defs = [
        (f"target-recall@{top_k}", "target_recall", style.BLUE),
        (f"target-MRR", "target_mrr", style.AQUA),
        (f"topical-precision@{top_k}", "topical_precision", style.YELLOW),
    ]
    metrics = [
        (label, key, color) for label, key, color in metric_defs
        if any(strategies[n].get(key) is not None for n in strategies)
    ]
    names = list(strategies.keys())
    x = np.arange(len(names))
    n_met = len(metrics)
    w = min(0.22, 0.8 / max(n_met, 1))
    gap = 0.02
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for i, (label, key, color) in enumerate(metrics):
        vals = [strategies[n].get(key) or 0.0 for n in names]
        offset = (i - (n_met - 1) / 2) * (w + gap)
        bars = ax.bar(x + offset, vals, w, label=label, color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                    f"{v:.2f}", ha="center", fontsize=8.5, color=style.INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, color=style.INK)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("score (higher is better)")
    title = f"Memory Selection Block — retrieval vs baselines ({qn} queries, top-{top_k})"
    if cfg.query_filter:
        title += f"  [{cfg.query_filter}]"
    if cfg.paraphrase == "vague":
        title += "  · vague probes"
    ax.set_title(title)
    ax.legend(loc="upper left", ncol=1, fontsize=9)
    style.style_axes(ax)
    fig.tight_layout()
    out_png = RESULTS / "exp3_retrieval.png"
    fig.savefig(out_png, bbox_inches="tight")
    print(f"\nsaved {out_png}")

    ablation_png = RESULTS / "exp3_alpha_ablation.png"
    alpha_curve = plot_alpha_ablation(cfg, ablation_png)
    robustness["alpha_curve"] = alpha_curve
    print(f"saved {ablation_png}")

    payload = {
        "reference_now": NOW.isoformat(),
        "top_k": top_k,
        "query_count": qn,
        "emotion_threshold": EMOTION_THRESHOLD,
        "topical_sim_fraction": cfg.topical_sim_fraction,
        "tuned_config": {
            "alpha": cfg.alpha,
            "w_emotion": cfg.w_emotion,
            "w_recall": cfg.w_recall,
            "w_context": cfg.w_context,
            "protection": cfg.protection,
            "query_filter": cfg.query_filter,
            "paraphrase": cfg.paraphrase,
        },
        "long_term_targets": lt_targets,
        "strategies": strategies,
        "robustness": robustness,
    }
    (RESULTS / "exp3_retrieval.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
