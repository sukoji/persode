"""Persode — Personalized visual journaling with an episodic memory-aware AI agent.

Reference implementation of the core components described in:

    Persode: Personalized Visual Journaling with Episodic Memory-Aware AI Agent
    Jin et al., arXiv:2508.20585 (2025)

This package reproduces the paper's *algorithmic* core so it can be studied and
tested without any paid API keys:

- ``memory``      — Ebbinghaus forgetting curve + Memory-Strength Scoring (Eq. 1)
- ``analyzer``    — Event-Emotion Analyzer (structured metadata extraction)
- ``embeddings``  — pluggable embedding backends (offline hashing / sentence-transformers)
- ``store``       — vector memory store + the Memory Selection Block (RAG retrieval)
- ``onboarding``  — onboarding preferences → chatbot style + visual style
- ``templates``   — Dual-Template framework (diary + Few-Shot visual prompt)
- ``agent``       — EpisodicMemoryAgent that wires everything together
- ``llm``         — optional GPT-4o / DALL·E 3 adapters with offline stubs
"""

from .memory import Memory, MemoryStrengthScorer, ebbinghaus_decay, DEFAULT_LAMBDA
from .analyzer import EventEmotionAnalyzer, EpisodeMetadata
from .embeddings import get_embedder, HashingEmbedder
from .store import MemoryStore, RetrievalResult
from .onboarding import OnboardingPreferences
from .templates import FewShotTemplateSystem, DiaryTemplate
from .agent import EpisodicMemoryAgent

__version__ = "0.1.0"

__all__ = [
    "Memory",
    "MemoryStrengthScorer",
    "ebbinghaus_decay",
    "DEFAULT_LAMBDA",
    "EventEmotionAnalyzer",
    "EpisodeMetadata",
    "get_embedder",
    "HashingEmbedder",
    "MemoryStore",
    "RetrievalResult",
    "OnboardingPreferences",
    "FewShotTemplateSystem",
    "DiaryTemplate",
    "EpisodicMemoryAgent",
    "__version__",
]
