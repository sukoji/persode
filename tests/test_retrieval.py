from datetime import datetime, timedelta, timezone

from persode.embeddings import HashingEmbedder, cosine_similarity
from persode.memory import Memory, MemoryStrengthScorer
from persode.store import MemoryStore


def _now():
    return datetime(2025, 1, 24, tzinfo=timezone.utc)


def test_hashing_embedder_deterministic():
    e = HashingEmbedder()
    assert (e.embed("hello world") == e.embed("hello world")).all()
    sim = cosine_similarity(e.embed("happy graduation day"), e.embed("graduation was happy"))
    assert sim > 0.0


def test_retrieval_returns_topk_sorted():
    store = MemoryStore(embedder=HashingEmbedder())
    store.add_many([
        Memory(text="graduation celebration success", emotion="joyful",
               emotional_intensity=0.9, created_at=_now() - timedelta(days=3)),
        Memory(text="took the bus to work", emotion="neutral",
               emotional_intensity=0.15, created_at=_now() - timedelta(days=1)),
        Memory(text="lost my dog and cried", emotion="sad",
               emotional_intensity=0.95, created_at=_now() - timedelta(days=2)),
    ])
    res = store.retrieve("I am so happy about my graduation", top_k=2, now=_now())
    assert len(res) == 2
    assert res[0].fused_score >= res[1].fused_score


def test_fusion_prefers_emotional_over_neutral_recent():
    # A very recent neutral memory vs a slightly older but emotional one.
    store = MemoryStore(embedder=HashingEmbedder(),
                        scorer=MemoryStrengthScorer(1, 1, 1), w_similarity=0.3)
    neutral = Memory(text="ordinary errands groceries", emotion="neutral",
                     emotional_intensity=0.1, created_at=_now())
    emotional = Memory(text="ordinary errands groceries", emotion="sad",
                       emotional_intensity=0.95, contextual_relevance=0.6,
                       created_at=_now() - timedelta(days=1))
    store.add_many([neutral, emotional])
    res = store.retrieve("errands groceries", top_k=1, now=_now())
    assert res[0].memory is emotional


def test_reinforcement_on_retrieval():
    store = MemoryStore(embedder=HashingEmbedder())
    m = Memory(text="graduation success", emotion="joyful", created_at=_now())
    store.add(m)
    assert m.recall_count == 0
    store.retrieve("graduation", top_k=1, now=_now(), reinforce=True)
    assert m.recall_count == 1


def test_save_load_roundtrip(tmp_path):
    store = MemoryStore(embedder=HashingEmbedder())
    store.add(Memory(text="graduation success", emotion="joyful", created_at=_now()))
    p = tmp_path / "mem.json"
    store.save(p)
    store2 = MemoryStore(embedder=HashingEmbedder())
    store2.load(p)
    assert len(store2) == 1
    assert store2.memories[0].emotion == "joyful"
