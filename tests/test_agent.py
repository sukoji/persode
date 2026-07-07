"""Regression tests for EpisodicMemoryAgent.create_journal recall selection.

The "reminded me of ..." line must reference a *distinct* past episode — never
the memory just ingested nor a near-duplicate of the same event.
"""

from datetime import datetime, timezone

from persode import EpisodicMemoryAgent, MemoryStore, OnboardingPreferences
from persode.memory import Memory


def _now():
    return datetime(2025, 1, 24, tzinfo=timezone.utc)


def _agent_with(*memories):
    store = MemoryStore()
    store.add_many(memories)
    return EpisodicMemoryAgent(preferences=OnboardingPreferences(), store=store)


def test_journal_recall_excludes_same_episode():
    # The store already holds this exact event; journaling it again must not
    # "remind" the user of itself (the near-duplicate is filtered out).
    agent = _agent_with(
        Memory(text="A car splashed water and ruined my favorite outfit",
               event="car splashed water ruined favorite outfit",
               emotion="angry", emotional_intensity=0.9, created_at=_now()),
    )
    entry = agent.create_journal(
        "A car splashed water and ruined my favorite outfit!", now=_now())
    assert "reminded me of" not in entry.diary.lower()


def test_journal_recall_surfaces_distinct_memory():
    # A genuinely different past episode should be recalled.
    agent = _agent_with(
        Memory(text="I celebrated my graduation, it was a huge success",
               event="graduation ceremony celebration",
               emotion="joyful", emotional_intensity=0.9, created_at=_now()),
    )
    entry = agent.create_journal(
        "I was scolded by my mom and I feel regretful", now=_now())
    assert "reminded me of graduation ceremony celebration" in entry.diary.lower()
