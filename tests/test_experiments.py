"""Results-regression tests.

These lock in the exact numbers the README reports for each experiment, computed
straight from the code (not from the committed JSON). If an experiment's behaviour
drifts, these fail — forcing the documented claims to be updated in lock-step, so
the README can never silently over- or under-state a result.
"""

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
from _scenario import NOW, QUERY_PARAPHRASES_VAGUE, build_memories  # noqa: E402
from _exp3_eval import Exp3Config, eval_all  # noqa: E402


# ---- Exp. 1 — forgetting-curve calibration --------------------------------
def test_exp1_lambda_and_calibration():
    assert DEFAULT_LAMBDA == math.log(4) / 6           # lambda = ln4/6
    assert abs(ebbinghaus_decay(6) - 0.25) < 1e-9      # 25% at day 6 (~75% drop)
    assert abs(math.log(2) / DEFAULT_LAMBDA - 3.0) < 1e-6   # half-life 3 d


def test_exp1_consolidation_at_30_days():
    scorer = MemoryStrengthScorer()

    def s_at_30(E, C, R):
        m = Memory(text="x", emotional_intensity=E, contextual_relevance=C, recall_count=R,
                   created_at=NOW - timedelta(days=30))
        return scorer.score(m, now=NOW)

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


# ---- Exp. 3 — salience-aware retrieval (pre-registered protocol) -----------
def test_exp3_protocol_is_fixed_defaults():
    # The protocol must stay pre-registered: shipped defaults, no tuning knobs.
    cfg = Exp3Config()
    assert (cfg.alpha, cfg.top_k, cfg.topical_sim_fraction) == (0.5, 3, 0.5)
    assert (cfg.w_emotion, cfg.w_recall, cfg.w_context, cfg.protection) == (1, 1, 1, 0.9)
    assert not hasattr(cfg, "query_filter")  # post-hoc query selection stays deleted


def test_exp3_vague_paraphrases_cover_every_memory():
    # The vague condition must probe ALL memories, not just the ones where the
    # method shines — asymmetric probe construction was a past defect.
    events = {m.event for m in build_memories()}
    assert set(QUERY_PARAPHRASES_VAGUE) == events


def test_exp3_vague_condition_headline():
    s = eval_all(Exp3Config())  # paraphrase="vague"
    assert round(s["recency-only"]["target_recall"], 2) == 0.30
    assert round(s["similarity-only"]["target_recall"], 2) == 0.30
    assert round(s["salience-only"]["target_recall"], 2) == 0.30
    assert round(s["fused (Persode)"]["target_recall"], 2) == 0.40
    assert round(s["gated (Persode)"]["target_recall"], 2) == 0.40
    # The mechanism-specific gain: long-term emotional recall under lexical mismatch.
    assert round(s["similarity-only"]["recall_by_category"]["emotional_long"], 2) == 0.40
    assert round(s["fused (Persode)"]["recall_by_category"]["emotional_long"], 2) == 0.60
    assert round(s["recency-only"]["recall_by_category"]["emotional_long"], 2) == 0.00
    # Gate recovers neutral-recent and cuts intrusion back to similarity's level —
    # at the cost of vague emotional queries the keyword analyzer cannot flag.
    assert round(s["gated (Persode)"]["recall_by_category"]["neutral_recent"], 2) == 0.33
    assert round(s["gated (Persode)"]["recall_by_category"]["emotional_long"], 2) == 0.40
    assert s["gated (Persode)"]["emotional_intrusion"] == s["similarity-only"]["emotional_intrusion"]


def test_exp3_plain_condition_and_tradeoffs():
    # Honesty guards: fusion is NOT a free win and the README must say so.
    plain = eval_all(Exp3Config(paraphrase="default"))
    # With plainly-worded probes, pure similarity solves everything; always-on
    # fusion costs recall; the emotion gate restores the perfect score.
    assert round(plain["similarity-only"]["target_recall"], 2) == 1.00
    assert round(plain["fused (Persode)"]["target_recall"], 2) == 0.80
    assert round(plain["gated (Persode)"]["target_recall"], 2) == 1.00
    # Under vague probes, fusion sacrifices neutral-recent recall...
    vague = eval_all(Exp3Config())
    assert vague["fused (Persode)"]["recall_by_category"]["neutral_recent"] \
        <= vague["similarity-only"]["recall_by_category"]["neutral_recent"]
    # ...and pushes more emotional memories into neutral queries (intrusion).
    assert vague["fused (Persode)"]["emotional_intrusion"] \
        >= vague["similarity-only"]["emotional_intrusion"]


def test_exp3_alpha_sweep_shape():
    # α ≈ 0.5 gives the long-term-emotional bump (0.60); both extremes fall back.
    for a in (0.45, 0.5, 0.6, 0.7):
        r = eval_all(replace(Exp3Config(), alpha=a))["fused (Persode)"]
        assert round(r["recall_by_category"]["emotional_long"], 2) == 0.60
    for a in (0.0, 1.0):
        r = eval_all(replace(Exp3Config(), alpha=a))["fused (Persode)"]
        assert round(r["recall_by_category"]["emotional_long"], 2) == 0.40


def test_exp3_semantic_embedder_closes_the_recall_gap():
    # Honesty guard for the disclosed embedder-dependence: with a real semantic
    # embedder, pure similarity already recalls every target (1.00), so the
    # hashing-embedder recall gain must not be sold as a universal win over RAG.
    # Skips in the offline suite (CI installs [dev], not [semantic]).
    import pytest
    pytest.importorskip("sentence_transformers")
    import _exp3_eval as E
    from persode.embeddings import get_embedder

    st = get_embedder("sentence-transformers")
    orig = E._embedder
    E._embedder = lambda: st
    E._text_vec.cache_clear()
    try:
        s = E.eval_all(Exp3Config())
    finally:
        E._embedder = orig
        E._text_vec.cache_clear()
    assert round(s["similarity-only"]["target_recall"], 2) == 1.00
    assert round(s["fused (Persode)"]["target_recall"], 2) == 1.00


def test_exp3_salience_prioritization_is_embedder_independent():
    # The paper's actual claim, isolated: with two equally-relevant memories
    # (identical text -> identical similarity for ANY embedder), fusion ranks the
    # emotionally-significant one first; pure similarity leaves them tied.
    import exp3_retrieval as X
    r = X.salience_prioritization()
    assert r["similarity_is_tied"] is True
    assert r["fused_ranks_significant_first"] is True
    assert r["similarity_only_order"] == ["neutral", "significant"]
    assert r["fused_order"] == ["significant", "neutral"]


def test_exp5_locomo_headline_and_honesty():
    # Pins the LoCoMo (public benchmark) hashing-embedder numbers the README
    # reports, and the honesty guard: on factual QA the salience prior COSTS
    # recall — fused must trail pure similarity, and that gap must stay reported.
    # Skips when the (CC BY-NC, not redistributed) data has not been downloaded.
    import pytest
    import exp5_locomo as X
    if not X.DATA_PATH.exists():
        pytest.skip("LoCoMo data not downloaded (run experiments/exp5_locomo.py)")

    res = X.evaluate("hashing")
    s = res["strategies"]
    assert s["fused (Persode)"]["query_count"] == 1535
    assert round(s["similarity-only"]["evidence_recall@5"], 3) == 0.152
    assert round(s["fused (Persode)"]["evidence_recall@5"], 3) == 0.135
    assert round(s["gated (Persode)"]["evidence_recall@5"], 3) == 0.154
    assert round(s["salience-only"]["evidence_recall@5"], 3) == 0.011
    assert round(s["recency-only"]["evidence_recall@5"], 3) == 0.002
    # Honesty guard: the factual-QA cost of ungated fusion is real and stays visible.
    assert s["fused (Persode)"]["evidence_recall@5"] < s["similarity-only"]["evidence_recall@5"]
    assert s["fused (Persode)"]["mrr"] < s["similarity-only"]["mrr"]
    # The emotion gate removes that cost (parity with pure similarity or better).
    assert s["gated (Persode)"]["evidence_recall@5"] >= s["similarity-only"]["evidence_recall@5"]
    assert s["gated (Persode)"]["evidence_recall@5"] > s["fused (Persode)"]["evidence_recall@5"]
    # The gate fires on a small emotional minority of factual-benchmark queries.
    assert 0.0 < res["gate"]["fused_fraction"] < 0.10
    # Exclusions are accounted for, never silent.
    assert res["excluded"] == {"adversarial_cat5": 446, "no_evidence": 4,
                               "unresolvable_evidence": 1}


def test_exp4_personalization_is_verified():
    # The onboarding -> visual personalization claim, quantified: every onboarding
    # attribute is injected, the two profiles differ, and they share the emotion mood.
    import exp4_visual_prompt as X
    p = X.personalization_check()
    assert p["attribute_injection"] == "24/24"
    assert p["all_attributes_injected"] is True
    assert p["all_prompts_differ"] is True
    assert p["all_share_mood"] is True
