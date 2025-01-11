"""Episodic memory representation and the Memory-Strength Scoring Mechanism.

This module implements two ideas from the Persode paper (§3.2, §4.2):

1. **Ebbinghaus forgetting curve** — a time-based decay ``d(Δt)`` that models the
   steep short-term retention drop (~75% within six days) reported by Ebbinghaus.

2. **Memory-Strength Scoring (Eq. 1)** — a weighted combination of *emotional
   intensity* (E), *recall frequency* (R) and *contextual relevance* (C),
   attenuated by the decay term:

   .. math::

       S \\;=\\; d(\\Delta t)\\cdot\\frac{w_E E + w_R R + w_C C}{w_E + w_R + w_C}

   Deeper, emotionally salient memories (Craik & Lockhart's *Levels of
   Processing*) survive the decay and stay retrievable long after neutral ones
   have faded — exactly the behaviour the paper wants from long-term memory.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

SECONDS_PER_DAY = 86_400.0

# The paper anchors short-term memory to a six-day window with a ~75% retention
# drop. We calibrate the exponential decay so that d(6 days) == 0.25, i.e. only
# 25% of the initial strength survives after six days:
#     e^(-lambda * 6) = 0.25  =>  lambda = ln(4) / 6
DEFAULT_LAMBDA = math.log(4.0) / 6.0  # ≈ 0.23105 per day
SHORT_TERM_WINDOW_DAYS = 6.0


def ebbinghaus_decay(delta_days: float, lam: float = DEFAULT_LAMBDA) -> float:
    """Return the retention factor ``d(Δt) = e^(-λ·Δt)`` in ``(0, 1]``.

    Args:
        delta_days: Elapsed time since the memory was formed, in days.
            Negative values are clamped to 0 (a memory cannot be "in the future").
        lam: Decay rate per day. The default is calibrated so that six days of
            elapsed time leaves 25% retention (a ~75% drop), matching the paper.
    """
    return math.exp(-lam * max(0.0, delta_days))


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


@dataclass
class Memory:
    """A single episodic memory fragment extracted from a conversation.

    Attributes mirror the "Meta Data" panel in Figure 1 of the paper:
    an event, its emotion, hashtags, a timestamp, and the components that feed
    the Memory-Strength Scoring Mechanism.
    """

    text: str
    event: str = ""
    emotion: str = ""
    hashtags: List[str] = field(default_factory=list)

    # Scoring components, all normalised to [0, 1].
    emotional_intensity: float = 0.0        # E — how affectively charged the memory is
    contextual_relevance: float = 0.0       # C — salience / relevance at encoding time
    recall_count: int = 0                   # raw retrieval count → feeds R

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_recalled: Optional[datetime] = None

    embedding: Optional[List[float]] = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def __post_init__(self) -> None:
        self.emotional_intensity = _clip01(float(self.emotional_intensity))
        self.contextual_relevance = _clip01(float(self.contextual_relevance))
        if self.created_at.tzinfo is None:
            self.created_at = self.created_at.replace(tzinfo=timezone.utc)

    # -- time helpers -----------------------------------------------------
    def age_days(self, now: Optional[datetime] = None) -> float:
        now = now or datetime.now(timezone.utc)
        return max(0.0, (now - self.created_at).total_seconds() / SECONDS_PER_DAY)

    def is_short_term(self, now: Optional[datetime] = None,
                      window_days: float = SHORT_TERM_WINDOW_DAYS) -> bool:
        """Whether the memory falls inside the recent short-term window."""
        return self.age_days(now) <= window_days

    def recall_frequency(self, saturation: float = 5.0) -> float:
        """Normalise the raw recall count into R ∈ [0, 1].

        Uses a saturating curve ``n / (n + saturation)`` so that the first few
        recalls matter a lot (reinforcement) while additional recalls have
        diminishing returns.
        """
        n = max(0, self.recall_count)
        return n / (n + saturation)

    def mark_recalled(self, now: Optional[datetime] = None) -> None:
        """Reinforce this memory: bump recall frequency and reset its timestamp.

        Following MemoryBank / LUFY, retrieving a memory strengthens it. We also
        refresh ``created_at`` so that a re-lived memory restarts its decay clock
        (spaced repetition), which is how emotionally significant events stay
        alive far beyond the six-day window.
        """
        now = now or datetime.now(timezone.utc)
        self.recall_count += 1
        self.last_recalled = now
        self.created_at = now

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        d["last_recalled"] = self.last_recalled.isoformat() if self.last_recalled else None
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Memory":
        d = dict(d)
        d["created_at"] = _parse_dt(d.get("created_at"))
        lr = d.get("last_recalled")
        d["last_recalled"] = _parse_dt(lr) if lr else None
        return cls(**d)


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif value is None:
        dt = datetime.now(timezone.utc)
    else:
        dt = datetime.fromisoformat(str(value))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


@dataclass
class MemoryStrengthScorer:
    """Implements Equation (1) of the paper, with salience-modulated decay.

    ``S = d(Δt) · (wE·E + wR·R + wC·C) / (wE + wR + wC)``

    The weights are tunable and *need not sum to 1* (they are normalised
    internally). ``d(Δt) = 1`` recovers the plain normalised weighted average
    used at retrieval time, as noted in the paper.

    **Consolidation (the long-term store).** The paper describes a two-stage
    memory: a steep short-term forgetting curve (~75% drop in six days) *and* a
    long-term store where "significant events" are retained by emotional
    intensity, recall frequency and contextual relevance. A single fixed
    exponential would erase everything within a month, which is the opposite of
    what the paper wants. Following Craik & Lockhart's *Levels of Processing*
    ("deeper processing → longer-lasting retention"), we let salience **slow the
    decay**:

        λ_eff = λ · (1 - γ · k),   k = base_salience ∈ [0, 1]

    A neutral memory (k ≈ 0) follows the full Ebbinghaus curve; a deeply
    processed, emotionally salient memory (k → 1) is consolidated and barely
    fades. ``γ`` (``protection``) controls how strongly salience protects a
    memory; ``γ = 0`` disables consolidation and recovers a plain fixed decay.
    """

    w_emotion: float = 1.0   # wE
    w_recall: float = 1.0    # wR
    w_context: float = 1.0   # wC
    lam: float = DEFAULT_LAMBDA
    protection: float = 0.9  # γ — how much salience slows forgetting

    def __post_init__(self) -> None:
        for name in ("w_emotion", "w_recall", "w_context"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if (self.w_emotion + self.w_recall + self.w_context) <= 0:
            raise ValueError("at least one weight must be positive")
        if not (0.0 <= self.protection <= 1.0):
            raise ValueError("protection (gamma) must be in [0, 1]")

    def base_salience(self, memory: Memory,
                      context_relevance: Optional[float] = None) -> float:
        """The decay-free weighted average — Eq. (1) with ``d(Δt) = 1``.

        Args:
            context_relevance: Optionally override the memory's stored C with a
                query-time relevance (e.g. cosine similarity to the current
                dialogue). This is how the Memory Selection Block folds RAG
                similarity into the score at retrieval time.
        """
        C = memory.contextual_relevance if context_relevance is None else _clip01(context_relevance)
        numerator = (
            self.w_emotion * memory.emotional_intensity
            + self.w_recall * memory.recall_frequency()
            + self.w_context * C
        )
        denom = self.w_emotion + self.w_recall + self.w_context
        return numerator / denom

    def effective_lambda(self, memory: Memory,
                         context_relevance: Optional[float] = None) -> float:
        """Salience-slowed decay rate ``λ_eff = λ·(1 - γ·k)``."""
        k = self.base_salience(memory, context_relevance)
        return self.lam * (1.0 - self.protection * k)

    def effective_decay(self, memory: Memory, now: Optional[datetime] = None,
                        context_relevance: Optional[float] = None) -> float:
        """Consolidation-modulated retention factor ``d(Δt)`` for this memory."""
        lam_eff = self.effective_lambda(memory, context_relevance)
        return ebbinghaus_decay(memory.age_days(now), lam_eff)

    def score(self, memory: Memory, now: Optional[datetime] = None,
              context_relevance: Optional[float] = None) -> float:
        """Full memory strength ``S`` including the (modulated) decay term."""
        k = self.base_salience(memory, context_relevance)
        lam_eff = self.lam * (1.0 - self.protection * k)
        decay = ebbinghaus_decay(memory.age_days(now), lam_eff)
        return decay * k

    def rank(self, memories: Sequence[Memory], now: Optional[datetime] = None,
             ) -> List[tuple[Memory, float]]:
        """Return memories sorted by descending strength ``S``."""
        scored = [(m, self.score(m, now)) for m in memories]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored
