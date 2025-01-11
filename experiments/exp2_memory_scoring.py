"""Experiment 2 — Memory-Strength Scoring (Eq. 1) behaviour & weight ablation.

We rank the scenario's memories by strength S and show how the ranking changes
as we re-weight the three terms (emotional intensity E, recall frequency R,
contextual relevance C). This makes concrete the paper's claim that
emotionally salient, frequently recalled memories are prioritised for retention.

Outputs:
    results/exp2_memory_scoring.png
    results/exp2_memory_scoring.json
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
from persode.memory import MemoryStrengthScorer  # noqa: E402
from _scenario import EMOTIONALLY_SIGNIFICANT, NOW, build_memories  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)


CONFIGS = {
    "balanced (1,1,1)":       MemoryStrengthScorer(1, 1, 1),
    "emotion-heavy (3,1,1)":  MemoryStrengthScorer(3, 1, 1),
    "recall-heavy (1,3,1)":   MemoryStrengthScorer(1, 3, 1),
    "context-heavy (1,1,3)":  MemoryStrengthScorer(1, 1, 3),
}


def main() -> None:
    memories = build_memories()
    labels = [m.event[:26] for m in memories]

    results = {}
    for name, scorer in CONFIGS.items():
        scored = [(m.event, scorer.score(m, now=NOW)) for m in memories]
        results[name] = scored

    # Print the balanced ranking.
    print("=== Balanced ranking (wE=wR=wC=1) ===")
    ranked = sorted(results["balanced (1,1,1)"], key=lambda x: x[1], reverse=True)
    for i, (event, s) in enumerate(ranked, 1):
        print(f"{i:2d}. S={s:.3f}  {event}")

    # ---- dot plot: one row per memory, one dot per weight config ----
    style.apply()
    palette = {  # fixed categorical slot order
        "balanced (1,1,1)": style.BLUE,
        "emotion-heavy (3,1,1)": style.AQUA,
        "recall-heavy (1,3,1)": style.YELLOW,
        "context-heavy (1,1,3)": style.VIOLET,
    }
    order = [e for e, _ in sorted(results["balanced (1,1,1)"],
                                  key=lambda x: x[1], reverse=True)]
    y = {event: len(order) - 1 - i for i, event in enumerate(order)}

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    by_event = {e: [] for e in order}
    for name, scored in results.items():
        for event, s in scored:
            by_event[event].append(s)
    for event, vals in by_event.items():
        ax.plot([min(vals), max(vals)], [y[event]] * 2,
                color=style.BASELINE, lw=1.1, zorder=2)
    for name, scored in results.items():
        ax.scatter([s for _, s in scored], [y[e] for e, _ in scored],
                   s=58, color=palette[name], label=name,
                   edgecolor=style.SURFACE, linewidth=1.4, zorder=3)

    dog = "lost beloved dog"
    bal_dog = dict(results["balanced (1,1,1)"])[dog]
    emo_dog = dict(results["emotion-heavy (3,1,1)"])[dog]
    ax.annotate(f"×{emo_dog / bal_dog:.1f} under emotion-heavy weights",
                (emo_dog, y[dog]), xytext=(emo_dog + 0.035, y[dog] - 0.06),
                va="center", color=style.INK_2, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=style.BASELINE, lw=0.9))

    ax.set_yticks([y[e] for e in order])
    ax.set_yticklabels(order)
    for tick, event in zip(ax.get_yticklabels(), order):
        tick.set_color(style.INK if event in EMOTIONALLY_SIGNIFICANT else style.MUTED)
    ax.set_xlabel("memory strength  S  (at the reference date)")
    ax.set_xlim(0, None)
    ax.set_title("Eq. 1 weight ablation — how (wE, wR, wC) reorders the memory store")
    ax.legend(loc="lower right", ncol=1)
    style.style_axes(ax, y_only=False)
    ax.grid(axis="x")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.text(0.005, 0.008, "dark event labels = emotionally significant memories (ground truth)",
             color=style.MUTED, fontsize=8.5)
    out_png = RESULTS / "exp2_memory_scoring.png"
    fig.savefig(out_png, bbox_inches="tight")
    print(f"saved {out_png}")

    # ---- machine-readable ----
    payload = {
        "reference_now": NOW.isoformat(),
        "rankings": {
            name: [{"event": e, "S": round(s, 4)}
                   for e, s in sorted(scored, key=lambda x: x[1], reverse=True)]
            for name, scored in results.items()
        },
    }
    (RESULTS / "exp2_memory_scoring.json").write_text(json.dumps(payload, indent=2))

    # A small observation the README can cite.
    bal = {e: s for e, s in results["balanced (1,1,1)"]}
    emo = {e: s for e, s in results["emotion-heavy (3,1,1)"]}
    dog = "lost beloved dog"
    print(f"\n'{dog}': balanced S={bal[dog]:.3f} -> emotion-heavy S={emo[dog]:.3f} "
          f"(+{100*(emo[dog]-bal[dog])/bal[dog]:.0f}%)")


if __name__ == "__main__":
    main()
