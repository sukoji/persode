"""EpisodicMemoryAgent — the orchestrator that wires the components together.

It reproduces the end-to-end flow of Figure 2:

    dialogue → Event-Emotion Analyzer → Memory (scored) → Vector store
             → Memory Selection Block (RAG) → augmented response
             → Dual-Template journal (diary + visual prompt)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from .analyzer import EventEmotionAnalyzer, EpisodeMetadata
from .llm import LLMClient, OfflineLLM
from .memory import Memory, MemoryStrengthScorer
from .onboarding import OnboardingPreferences
from .store import MemoryStore, RetrievalResult
from .templates import DiaryTemplate, FewShotTemplateSystem, VisualPrompt


@dataclass
class JournalEntry:
    """The final artifact: diary text + illustration prompt + provenance."""

    date: str
    title: str
    diary: str
    visual_prompt: VisualPrompt
    episode: EpisodeMetadata
    retrieved: List[RetrievalResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "title": self.title,
            "diary": self.diary,
            "visual_prompt": self.visual_prompt.prompt,
            "episode": self.episode.to_dict(),
            "retrieved": [r.memory.event or r.memory.text for r in self.retrieved],
        }


class EpisodicMemoryAgent:
    """High-level Persode agent.

    Args:
        preferences: The user's onboarding profile.
        store: Optional pre-built memory store (else one is created).
        llm: Optional LLM client; defaults to the offline stub.
    """

    def __init__(
        self,
        preferences: Optional[OnboardingPreferences] = None,
        store: Optional[MemoryStore] = None,
        llm: Optional[LLMClient] = None,
    ) -> None:
        self.preferences = preferences or OnboardingPreferences()
        self.llm = llm or OfflineLLM()
        self.store = store or MemoryStore()
        self.analyzer = EventEmotionAnalyzer(llm=self._maybe_llm())
        self.diary_template = DiaryTemplate()
        self.visual_template = FewShotTemplateSystem()

    def _maybe_llm(self) -> Optional[LLMClient]:
        # Only pass a real LLM to the analyzer; the offline stub returns "{}"
        # for json_mode, which the analyzer correctly falls back from.
        return None if isinstance(self.llm, OfflineLLM) else self.llm

    # -- ingestion --------------------------------------------------------
    def ingest(self, utterance: str, now: Optional[datetime] = None,
               contextual_relevance: float = 0.5) -> Memory:
        """Analyze an utterance and store it as a scored episodic memory."""
        now = now or datetime.now(timezone.utc)
        meta = self.analyzer.analyze(utterance)
        memory = Memory(
            text=meta.text,
            event=meta.event,
            emotion=meta.emotion,
            hashtags=meta.hashtags,
            emotional_intensity=meta.emotional_intensity,
            contextual_relevance=contextual_relevance,
            created_at=now,
        )
        return self.store.add(memory)

    # -- conversation (RAG) ----------------------------------------------
    def respond(self, user_input: str, top_k: int = 3,
                now: Optional[datetime] = None) -> str:
        """Produce a memory-augmented, style-conditioned response."""
        retrieved = self.store.retrieve(user_input, top_k=top_k, now=now)
        memory_context = "\n".join(
            f"- {r.memory.event or r.memory.text} (emotion: {r.memory.emotion})"
            for r in retrieved
        ) or "- (no prior memories yet)"

        system = self.preferences.build_style_prompt()
        user = (
            f"Relevant memories retrieved for context:\n{memory_context}\n\n"
            f"User just said: {user_input}\n"
            f"Respond empathetically, weaving in a relevant memory if it fits."
        )
        return self.llm.complete(system=system, user=user, temperature=0.7)

    # -- journal generation (Dual-Template) ------------------------------
    def create_journal(self, utterance: str, top_k: int = 2,
                        now: Optional[datetime] = None) -> JournalEntry:
        """Full pipeline: dialogue → diary entry + visual prompt."""
        now = now or datetime.now(timezone.utc)
        memory = self.ingest(utterance, now=now)
        # Retrieve related memories (excluding the one we just added would need
        # id filtering; here reinforcement is disabled to keep generation pure).
        retrieved = self.store.retrieve(utterance, top_k=top_k, now=now, reinforce=False)
        related_events = [r.memory.event for r in retrieved if r.memory.id != memory.id]

        episode = self.analyzer.analyze(utterance)

        # Diary (text template): use the real LLM if present, else offline compose.
        if isinstance(self.llm, OfflineLLM):
            diary = self.diary_template.compose_offline(episode, self.preferences, related_events)
        else:
            prompt = self.diary_template.build_prompt(episode, self.preferences, related_events)
            diary = self.llm.complete(system=self.preferences.build_style_prompt(),
                                      user=prompt, temperature=0.7)

        # Visual (image template).
        visual = self.visual_template.build(episode, self.preferences)

        return JournalEntry(
            date=now.date().isoformat(),
            title=episode.title,
            diary=diary,
            visual_prompt=visual,
            episode=episode,
            retrieved=retrieved,
        )
