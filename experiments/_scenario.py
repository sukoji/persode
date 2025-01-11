"""Shared, reproducible scenario for the experiments.

All experiments anchor to a fixed reference "now" so results are identical on
every run and machine. The memories are drawn from the paper's own vignettes
(graduation, the joyful meal, the puddle that ruined an outfit) plus neutral
filler events, spread across a range of ages and emotional intensities.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persode.memory import Memory  # noqa: E402

# Fixed reference clock (matches the paper's "December 2024" vignettes era).
NOW = datetime(2025, 1, 24, 12, 0, 0, tzinfo=timezone.utc)


def _ago(days: float) -> datetime:
    return NOW - timedelta(days=days)


def build_memories() -> list[Memory]:
    """A hand-authored episodic memory set with ground-truth emotion labels."""
    return [
        # event, emotion, E, C, age(days), recalls, hashtags
        Memory(text="I celebrated my graduation ceremony, it was a huge success!",
               event="graduation ceremony celebration", emotion="joyful",
               emotional_intensity=0.92, contextual_relevance=0.6,
               created_at=_ago(38), recall_count=2,
               hashtags=["#Graduation", "#Success", "#Celebration"]),
        Memory(text="Had a wonderful family dinner with delicious meat, so joyful.",
               event="family dinner with meat", emotion="joyful",
               emotional_intensity=0.8, contextual_relevance=0.5,
               created_at=_ago(3), recall_count=0,
               hashtags=["#Meat", "#Joyful", "#Delicious"]),
        Memory(text="A car splashed puddle water on me and ruined my favorite outfit.",
               event="car splashed water ruined favorite outfit", emotion="angry",
               emotional_intensity=0.88, contextual_relevance=0.55,
               created_at=_ago(4), recall_count=1,
               hashtags=["#FavoriteOutfit", "#Upset", "#Laundry"]),
        Memory(text="I felt so anxious before my final exam, could not sleep.",
               event="anxious before final exam", emotion="anxious",
               emotional_intensity=0.78, contextual_relevance=0.5,
               created_at=_ago(30), recall_count=0,
               hashtags=["#Exam", "#Anxious"]),
        Memory(text="I lost my beloved dog last month and cried for days.",
               event="lost beloved dog", emotion="sad",
               emotional_intensity=0.95, contextual_relevance=0.7,
               created_at=_ago(35), recall_count=3,
               hashtags=["#Dog", "#Loss", "#Grief"]),
        Memory(text="Bought groceries and cooked a simple pasta dinner.",
               event="bought groceries cooked pasta", emotion="neutral",
               emotional_intensity=0.2, contextual_relevance=0.3,
               created_at=_ago(2), recall_count=0,
               hashtags=["#Groceries", "#Pasta"]),
        Memory(text="Took the bus to work, nothing special happened.",
               event="took bus to work", emotion="neutral",
               emotional_intensity=0.15, contextual_relevance=0.25,
               created_at=_ago(1), recall_count=0,
               hashtags=["#Commute"]),
        Memory(text="Had a peaceful walk in the park at sunset, felt content.",
               event="peaceful walk park sunset", emotion="content",
               emotional_intensity=0.5, contextual_relevance=0.4,
               created_at=_ago(12), recall_count=0,
               hashtags=["#Park", "#Sunset"]),
        Memory(text="Argued with my best friend and felt frustrated afterwards.",
               event="argued with best friend", emotion="angry",
               emotional_intensity=0.75, contextual_relevance=0.5,
               created_at=_ago(20), recall_count=0,
               hashtags=["#Friend", "#Argument"]),
        Memory(text="Reorganized my closet and did the laundry on a slow afternoon.",
               event="reorganized closet did laundry", emotion="neutral",
               emotional_intensity=0.2, contextual_relevance=0.3,
               created_at=_ago(5), recall_count=0,
               hashtags=["#Laundry", "#Cleaning"]),
    ]


# Ground-truth: which events are "emotionally significant" (|valence| high, E high).
EMOTIONALLY_SIGNIFICANT = {
    "graduation ceremony celebration",
    "family dinner with meat",
    "car splashed water ruined favorite outfit",
    "anxious before final exam",
    "lost beloved dog",
    "argued with best friend",
}
