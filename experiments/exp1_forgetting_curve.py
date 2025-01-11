"""Experiment 1 — Ebbinghaus forgetting curve calibration.

Reproduces the paper's claim (§3.2) that short-term memory is anchored to a
six-day window with a ~75% retention drop. We verify the exponential decay
``d(Δt) = e^(-λΔt)`` is calibrated so that d(6 days) = 0.25, and visualise how
emotionally salient memories survive the decay far longer than neutral ones.

Outputs:
    results/exp1_forgetting_curve.png
    results/exp1_forgetting_curve.json
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
from persode.memory import (  # noqa: E402
    DEFAULT_LAMBDA, SHORT_TERM_WINDOW_DAYS, Memory, MemoryStrengthScorer, ebbinghaus_decay,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)


def main() -> None:
    days = np.linspace(0, 30, 300)
    retention = np.array([ebbinghaus_decay(d) for d in days])

    # Calibration check: retention at the six-day boundary.
    r6 = ebbinghaus_decay(SHORT_TERM_WINDOW_DAYS)
    print(f"lambda = {DEFAULT_LAMBDA:.5f} / day")
    print(f"retention at 6 days = {r6:.4f}  (target 0.25, i.e. ~75% drop)")
    assert abs(r6 - 0.25) < 1e-6, "decay is not calibrated to 25% at 6 days"

    # Compare three memories with identical recency but different salience.
    scorer = MemoryStrengthScorer()
    profiles = {
        "high emotion (E=0.95)": dict(emotional_intensity=0.95, contextual_relevance=0.7, recall_count=3),
        "medium (E=0.5)":        dict(emotional_intensity=0.50, contextual_relevance=0.4, recall_count=1),
        "neutral (E=0.15)":      dict(emotional_intensity=0.15, contextual_relevance=0.25, recall_count=0),
    }
    curves = {}
    for label, kw in profiles.items():
        vals = []
        for d in days:
            m = Memory(text="x", **kw)
            # score at age d by faking creation d days before NOW
            from datetime import timedelta
            m.created_at = m.created_at - timedelta(days=float(d))
            vals.append(scorer.score(m))
        curves[label] = vals

    # ---- plot ----
    style.apply()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # Panel A — the calibrated decay curve (single series: no legend box).
    ax1.axvspan(0, SHORT_TERM_WINDOW_DAYS, color=style.GRID, alpha=0.45, lw=0, zorder=0)
    ax1.plot(days, retention, color=style.BLUE, lw=2.2, zorder=3)
    ax1.scatter([SHORT_TERM_WINDOW_DAYS], [r6], s=42, color=style.BLUE,
                edgecolor=style.SURFACE, linewidth=1.6, zorder=4)
    ax1.annotate(f"day 6 → {r6:.2f}\n(the ~75% drop)", (SHORT_TERM_WINDOW_DAYS, r6),
                 xytext=(10.5, 0.44), color=style.INK_2, fontsize=9.5,
                 arrowprops=dict(arrowstyle="-", color=style.BASELINE, lw=0.9))
    ax1.annotate("half-life 3 d", (3, 0.5), xytext=(3.35, 0.60),
                 color=style.MUTED, fontsize=9)
    ax1.scatter([3], [0.5], s=24, color=style.BLUE, edgecolor=style.SURFACE,
                linewidth=1.2, zorder=4)
    ax1.text(SHORT_TERM_WINDOW_DAYS / 2, 0.035, "short-term\nwindow",
             ha="center", color=style.MUTED, fontsize=8.5)
    ax1.set_title("Ebbinghaus decay  $d(\\Delta t)=e^{-\\lambda\\Delta t}$,  λ = ln4/6")
    ax1.set_xlabel("elapsed time Δt (days)")
    ax1.set_ylabel("retention  d(Δt)")
    ax1.set_xlim(0, 30)
    ax1.set_ylim(0, 1.02)
    style.style_axes(ax1)

    # Panel B — memory strength by salience: an ordered magnitude → ordinal ramp.
    ramp = {"high emotion (E=0.95)": style.BLUE_650,
            "medium (E=0.5)": style.BLUE_450,
            "neutral (E=0.15)": style.BLUE_250}
    ax2.axvspan(0, SHORT_TERM_WINDOW_DAYS, color=style.GRID, alpha=0.45, lw=0, zorder=0)
    for label, vals in curves.items():
        ax2.plot(days, vals, color=ramp[label], zorder=3, label=label)
    ax2.set_title("Memory strength S over time, by salience")
    ax2.set_xlabel("memory age (days)")
    ax2.set_ylabel("memory strength  S")
    ax2.set_xlim(0, 30)
    ax2.set_ylim(0, None)
    ax2.legend(loc="upper right")
    style.style_axes(ax2)

    fig.tight_layout(w_pad=4.0)
    out_png = RESULTS / "exp1_forgetting_curve.png"
    fig.savefig(out_png, bbox_inches="tight")
    print(f"saved {out_png}")

    # ---- machine-readable results ----
    half_life = np.log(2) / DEFAULT_LAMBDA
    payload = {
        "lambda_per_day": DEFAULT_LAMBDA,
        "retention_at_6_days": r6,
        "half_life_days": half_life,
        "retention_samples": {str(int(d)): ebbinghaus_decay(d) for d in [0, 1, 3, 6, 12, 30]},
        "crossover_note": "high-emotion memory strength exceeds neutral's initial strength for many days despite decay",
    }
    (RESULTS / "exp1_forgetting_curve.json").write_text(json.dumps(payload, indent=2))
    print(f"half-life ~ {half_life:.2f} days")


if __name__ == "__main__":
    main()
