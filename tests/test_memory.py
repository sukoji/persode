import math
from datetime import timedelta, timezone, datetime

from persode.memory import (
    DEFAULT_LAMBDA, SHORT_TERM_WINDOW_DAYS, Memory, MemoryStrengthScorer, ebbinghaus_decay,
)


def test_decay_calibrated_to_6_day_window():
    # ~75% drop => 25% retention at the six-day boundary.
    assert math.isclose(ebbinghaus_decay(SHORT_TERM_WINDOW_DAYS), 0.25, abs_tol=1e-9)


def test_decay_monotonic_and_bounded():
    prev = 1.0
    for d in range(0, 40):
        v = ebbinghaus_decay(d)
        assert 0.0 < v <= 1.0
        assert v <= prev + 1e-12
        prev = v


def test_decay_clamps_negative_time():
    assert ebbinghaus_decay(-5) == 1.0


def test_equation1_reduces_to_weighted_average_without_decay():
    # d(Δt)=1 at age 0 -> S equals the normalized weighted average.
    m = Memory(text="x", emotional_intensity=0.8, contextual_relevance=0.4, recall_count=0)
    scorer = MemoryStrengthScorer(1, 1, 1)
    now = m.created_at  # age 0
    expected = (0.8 + m.recall_frequency() + 0.4) / 3
    assert math.isclose(scorer.score(m, now=now), expected, rel_tol=1e-9)


def test_weights_need_not_sum_to_one():
    m = Memory(text="x", emotional_intensity=1.0, contextual_relevance=1.0, recall_count=100)
    now = m.created_at
    s = MemoryStrengthScorer(2, 5, 3).score(m, now=now)
    assert 0.0 <= s <= 1.0


def test_emotional_memory_outlives_neutral():
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    old = now - timedelta(days=20)
    hot = Memory(text="x", emotional_intensity=0.95, contextual_relevance=0.7, recall_count=3, created_at=old)
    cold = Memory(text="x", emotional_intensity=0.15, contextual_relevance=0.25, recall_count=0, created_at=old)
    scorer = MemoryStrengthScorer()
    assert scorer.score(hot, now=now) > scorer.score(cold, now=now)


def test_recall_reinforces_and_resets_clock():
    now = datetime(2025, 1, 10, tzinfo=timezone.utc)
    m = Memory(text="x", created_at=now - timedelta(days=10))
    assert m.age_days(now) == 10
    m.mark_recalled(now)
    assert m.recall_count == 1
    assert m.age_days(now) == 0  # decay clock reset


def test_short_term_window():
    now = datetime(2025, 1, 10, tzinfo=timezone.utc)
    recent = Memory(text="x", created_at=now - timedelta(days=3))
    old = Memory(text="x", created_at=now - timedelta(days=10))
    assert recent.is_short_term(now)
    assert not old.is_short_term(now)


def test_serialization_roundtrip():
    m = Memory(text="hello", event="e", emotion="joyful", emotional_intensity=0.7,
               hashtags=["#A"], recall_count=2)
    m2 = Memory.from_dict(m.to_dict())
    assert m2.text == m.text and m2.emotion == m.emotion
    assert m2.emotional_intensity == m.emotional_intensity
    assert m2.hashtags == m.hashtags
