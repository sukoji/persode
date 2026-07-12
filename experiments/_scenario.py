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


# Objective significance threshold (used by Exp. 3 metrics — not hand-picked per event).
EMOTION_THRESHOLD = 0.6


def is_emotionally_significant(memory: Memory) -> bool:
    """True when emotional intensity E meets the paper's 'significant event' bar."""
    return memory.emotional_intensity >= EMOTION_THRESHOLD


# Hand labels for Exp. 2 visualisation only (same threshold, kept as a set for styling).
EMOTIONALLY_SIGNIFICANT = {
    m.event for m in build_memories() if is_emotionally_significant(m)
}


# Exp. 3 evaluation queries — one natural-language probe per scenario memory.
# Targets are fixed before any retrieval run; categories stratify reporting.
EVAL_QUERIES = [
    {
        "query": "I'm feeling really happy about a big achievement today",
        "target": "graduation ceremony celebration",
        "category": "emotional_long",
    },
    {
        "query": "I remember a joyful meal I shared with my family",
        "target": "family dinner with meat",
        "category": "emotional_recent",
    },
    {
        "query": "Something ruined my day and I'm upset about my clothes",
        "target": "car splashed water ruined favorite outfit",
        "category": "emotional_recent",
    },
    {
        "query": "I'm nervous and anxious about an upcoming exam",
        "target": "anxious before final exam",
        "category": "emotional_long",
    },
    {
        "query": "I miss someone I loved very much and I'm grieving",
        "target": "lost beloved dog",
        "category": "emotional_long",
    },
    {
        "query": "What did I cook after buying groceries?",
        "target": "bought groceries cooked pasta",
        "category": "neutral_recent",
    },
    {
        "query": "How was my commute on the bus?",
        "target": "took bus to work",
        "category": "neutral_recent",
    },
    {
        "query": "I had a peaceful walk outdoors at sunset",
        "target": "peaceful walk park sunset",
        "category": "emotional_long",
    },
    {
        "query": "I argued with a close friend and felt frustrated",
        "target": "argued with best friend",
        "category": "emotional_long",
    },
    {
        "query": "What household chores did I do on a slow afternoon?",
        "target": "reorganized closet did laundry",
        "category": "neutral_recent",
    },
]

# Vague paraphrases — one per target, written under a single uniform rule applied
# to EVERY memory (not only the ones a given method favours): re-describe the
# episode without reusing any content word from the stored memory text, so
# lexical overlap with the store is minimal across the board. This makes the
# "vague" condition a symmetric stress test of retrieval under lexical mismatch.
QUERY_PARAPHRASES_VAGUE: dict[str, str] = {
    "graduation ceremony celebration": (
        "I still feel proud when I think back to finishing school"
    ),
    "family dinner with meat": (
        "That warm evening sharing food with the people closest to me"
    ),
    "car splashed water ruined favorite outfit": (
        "Something on the street soaked and wrecked the clothes I loved most"
    ),
    "anxious before final exam": (
        "That stressful period before a big test still haunts me"
    ),
    "lost beloved dog": (
        "The emptiness after losing someone close never really left"
    ),
    "bought groceries cooked pasta": (
        "What did I end up making at home after shopping for food?"
    ),
    "took bus to work": (
        "How did my dull ride to the office go?"
    ),
    "peaceful walk park sunset": (
        "A quiet moment outdoors at dusk still calms me"
    ),
    "argued with best friend": (
        "I keep replaying a fallout with someone important to me"
    ),
    "reorganized closet did laundry": (
        "That uneventful stretch spent tidying things up at home"
    ),
}


def build_eval_queries(paraphrase: str = "default") -> list[dict]:
    """Return evaluation queries, optionally with vague long-term paraphrases."""
    queries = [dict(q) for q in EVAL_QUERIES]
    if paraphrase == "vague":
        for q in queries:
            if q["target"] in QUERY_PARAPHRASES_VAGUE:
                q["query"] = QUERY_PARAPHRASES_VAGUE[q["target"]]
    return queries
