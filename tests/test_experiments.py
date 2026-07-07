"""Results-regression tests.

These lock in the exact numbers the README reports for each experiment, computed
straight from the code (not from the committed JSON). If an experiment's behaviour
drifts, these fail — forcing the documented claims to be updated in lock-step, so
the README can never silently over- or under-state a result.
"""

import json
import math
import sys
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from persode.memory import (  # noqa: E402
    DEFAULT_LAMBDA, Memory, MemoryStrengthScorer, ebbinghaus_decay,
)
from _scenario import NOW, build_memories  # noqa: E402
from _exp3_eval import Exp3Config, eval_all  # noqa: E402


def _exp3_cfg() -> Exp3Config:
    d = json.loads((ROOT / "results" / "exp3_tuned_config.json").read_text())["tuned"]
    return Exp3Config(**d)


# ---- Exp. 1 — forgetting-curve calibration --------------------------------
def test_exp1_lambda_and_calibration():
    assert DEFAULT_LAMBDA == math.log(4) / 6           # lambda = ln4/6
    assert abs(ebbinghaus_decay(6) - 0.25) < 1e-9      # 25% at day 6 (~75% drop)
    assert abs(math.log(2) / DEFAULT_LAMBDA - 3.0) < 1e-6   # half-life 3 d


def test_exp1_consolidation_at_30_days():
    scorer = MemoryStrengthScorer()

    def s_at_30(E, C, R):
        m = Memory(text="x", emotional_intensity=E, contextual_relevance=C, recall_count=R)
        m.created_at = m.created_at - timedelta(days=30)
        return scorer.score(m)

    high = s_at_30(0.95, 0.7, 3)
    neutral = s_at_30(0.15, 0.25, 0)
    assert round(high, 3) == 0.044        # README Exp.1: high-salience S at 30 d
    assert round(neutral, 4) == 0.0003    # equally-old neutral memory
    assert 145 < high / neutral < 152     # "~150x"


# ---- Exp. 2 — Eq.1 weight ablation ----------------------------------------
def test_exp2_emotion_heavy_reordering():
    mems = build_memories()
    balanced = MemoryStrengthScorer(1, 1, 1)
    emotion_heavy = MemoryStrengthScorer(3, 1, 1)

    def rank(scorer):
        scored = sorted(((m.event, scorer.score(m, now=NOW)) for m in mems),
                        key=lambda x: -x[1])
        return [e for e, _ in scored]

    b = {m.event: balanced.score(m, now=NOW) for m in mems}
    e = {m.event: emotion_heavy.score(m, now=NOW) for m in mems}
    ratio = e["lost beloved dog"] / b["lost beloved dog"]
    assert 2.55 < ratio < 2.65                                   # "x2.6"
    assert rank(balanced).index("lost beloved dog") + 1 == 7     # 7th balanced
    assert rank(emotion_heavy).index("lost beloved dog") + 1 == 5  # -> 5th


# ---- Exp. 3 — salience-aware retrieval ------------------------------------
def test_exp3_scoped_headline():
    s = eval_all(_exp3_cfg())
    assert round(s["recency-only"]["target_recall"], 2) == 0.00
    assert round(s["similarity-only"]["target_recall"], 2) == 0.40
    assert round(s["fused (Persode)"]["target_recall"], 2) == 0.80
    assert round(s["fused (Persode)"]["target_mrr"], 2) == 0.56
    assert round(s["similarity-only"]["topical_precision"], 2) == 1.00
    assert round(s["fused (Persode)"]["topical_precision"], 2) == 0.95


def test_exp3_robustness_is_not_cherry_picking():
    cfg = _exp3_cfg()

    # Full query mix: fusion is net-neutral vs pure RAG (0.70 == 0.70).
    full = eval_all(replace(cfg, query_filter=None))
    assert round(full["fused (Persode)"]["target_recall"], 2) == 0.70
    assert round(full["similarity-only"]["target_recall"], 2) == 0.70

    # Plain (non-vague) probes: pure RAG already solves it -> why vague matters.
    plain = eval_all(replace(cfg, paraphrase="default"))
    assert round(plain["similarity-only"]["target_recall"], 2) == 1.00

    # alpha is a plateau, not a magic value.
    for a in (0.5, 0.75):
        assert round(eval_all(replace(cfg, alpha=a))["fused (Persode)"]["target_recall"], 2) == 0.80
    for a in (0.0, 1.0):
        assert round(eval_all(replace(cfg, alpha=a))["fused (Persode)"]["target_recall"], 2) == 0.40
