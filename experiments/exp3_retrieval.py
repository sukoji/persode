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


def _is_long_term_target(target: str) -> bool:
    by = {m.event: m for m in build_memories()}
    return by[target].age_days(NOW) > SHORT_TERM_WINDOW_DAYS


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
    }
    (RESULTS / "exp3_retrieval.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
