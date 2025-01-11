"""Event-Emotion Analyzer — turns raw dialogue into structured episode metadata.

Corresponds to the *Event-Emotion Analyzer* block in Figure 2 of the paper.
It extracts, from a user's utterance:

- an **event** (what happened),
- an **emotion** (the dominant affect),
- **hashtags** (contextual tags such as ``#FavoriteOutfit`` / ``#Laundry``),
- an **emotional intensity** score E ∈ [0, 1] used by the Memory-Strength Scorer.

The default backend is a transparent, dependency-free lexicon so that everything
is reproducible offline. If an :class:`~persode.llm.LLMClient` is supplied, the
analyzer instead asks GPT-4o to fill the same schema (paper's actual pipeline).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# A compact valence/arousal-style lexicon. ``intensity`` approximates the
# affective arousal the paper's "emotional intensity" term captures.
EMOTION_LEXICON: Dict[str, Dict[str, Any]] = {
    "joyful":       {"keywords": ["joy", "joyful", "happy", "delighted", "excited", "celebrate", "celebration", "proud", "grateful", "love", "wonderful", "amazing", "great"], "intensity": 0.85, "valence": +1},
    "content":      {"keywords": ["calm", "peaceful", "serene", "relaxed", "content", "cozy", "tranquil", "satisfied"], "intensity": 0.45, "valence": +1},
    "sad":          {"keywords": ["sad", "sorrow", "sorrowful", "cry", "crying", "tears", "heartbroken", "lonely", "miss", "grief", "grieving", "down", "disappointed", "hurt"], "intensity": 0.8, "valence": -1},
    "angry":        {"keywords": ["angry", "anger", "furious", "mad", "annoyed", "irritated", "frustrated", "frustration", "upset", "rage", "ruined", "scolded"], "intensity": 0.82, "valence": -1},
    "anxious":      {"keywords": ["anxious", "anxiety", "nervous", "worried", "worry", "scared", "afraid", "fear", "stressed", "overwhelmed"], "intensity": 0.7, "valence": -1},
    "reflective":   {"keywords": ["reflect", "reflective", "think", "thoughtful", "wonder", "regret", "regretful", "contemplative", "nostalgic", "unsure", "ashamed"], "intensity": 0.5, "valence": 0},
    "surprised":    {"keywords": ["surprised", "shocked", "unexpected", "sudden", "astonished", "startled"], "intensity": 0.6, "valence": 0},
    "neutral":      {"keywords": [], "intensity": 0.2, "valence": 0},
}

# Intensity amplifiers — exclamation, ALL CAPS, and boosters raise arousal.
_BOOSTERS = ["really", "so", "very", "extremely", "truly", "deeply", "completely", "all"]
_STOPWORDS = set("""a an the this that these those i you he she it we they me my your his her our their
of to in on at for with and or but if then so as is am are was were be been being do did does have has had
i'm i've it's don't didn't wasn't weren't just about not no yes very really so too my me up out day today""".split())


@dataclass
class EpisodeMetadata:
    """Structured output of the analyzer (one per extracted episode)."""

    text: str
    event: str
    emotion: str
    hashtags: List[str] = field(default_factory=list)
    emotional_intensity: float = 0.0
    valence: int = 0            # -1 negative / 0 neutral / +1 positive
    title: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EventEmotionAnalyzer:
    """Extracts :class:`EpisodeMetadata` from dialogue text.

    Args:
        llm: Optional LLM client. When provided, extraction is delegated to the
            model with a strict JSON schema; otherwise the offline lexicon runs.
    """

    def __init__(self, llm: Optional["LLMClient"] = None) -> None:  # noqa: F821
        self.llm = llm

    # -- public API -------------------------------------------------------
    def analyze(self, text: str) -> EpisodeMetadata:
        """Analyze a single utterance / short passage into one episode."""
        text = text.strip()
        if self.llm is not None:
            meta = self._analyze_with_llm(text)
            if meta is not None:
                return meta
        return self._analyze_lexicon(text)

    def segment(self, text: str) -> List[EpisodeMetadata]:
        """Segment a longer passage into discrete episodes (paper §4.2).

        Splits on sentence boundaries and merges adjacent sentences that share
        the same dominant emotion, so each returned episode carries a coherent
        event-emotion pair.
        """
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        episodes: List[EpisodeMetadata] = []
        for sent in sentences:
            meta = self.analyze(sent)
            if episodes and episodes[-1].emotion == meta.emotion:
                prev = episodes[-1]
                merged_text = f"{prev.text} {meta.text}".strip()
                episodes[-1] = self.analyze(merged_text)
            else:
                episodes.append(meta)
        return episodes or [self.analyze(text)]

    # -- offline lexicon backend -----------------------------------------
    def _analyze_lexicon(self, text: str) -> EpisodeMetadata:
        tokens = re.findall(r"[a-zA-Z']+", text.lower())
        token_set = set(tokens)

        # Dominant emotion = lexicon entry with the most keyword hits.
        best_emotion, best_hits = "neutral", 0
        for emotion, spec in EMOTION_LEXICON.items():
            hits = sum(1 for kw in spec["keywords"] if kw in token_set)
            if hits > best_hits:
                best_emotion, best_hits = emotion, hits

        spec = EMOTION_LEXICON[best_emotion]
        intensity = spec["intensity"] if best_hits else EMOTION_LEXICON["neutral"]["intensity"]

        # Amplify intensity for emphasis cues.
        if "!" in text:
            intensity += 0.08 * text.count("!")
        if any(b in tokens for b in _BOOSTERS):
            intensity += 0.06
        caps_words = re.findall(r"\b[A-Z]{3,}\b", text)
        if caps_words:
            intensity += 0.05
        intensity = max(0.0, min(1.0, intensity))

        event = self._extract_event(text, tokens)
        hashtags = self._extract_hashtags(tokens, best_emotion)
        title = self._make_title(event, best_emotion)

        return EpisodeMetadata(
            text=text,
            event=event,
            emotion=best_emotion,
            hashtags=hashtags,
            emotional_intensity=round(intensity, 3),
            valence=int(spec["valence"]),
            title=title,
        )

    @staticmethod
    def _extract_event(text: str, tokens: List[str]) -> str:
        """Heuristic event summary: the content words of the utterance."""
        content = [t for t in tokens if t not in _STOPWORDS and len(t) > 2]
        # Keep original order, drop duplicates.
        seen, keep = set(), []
        for t in content:
            if t not in seen:
                seen.add(t)
                keep.append(t)
        return " ".join(keep[:8]) if keep else text[:60]

    @staticmethod
    def _extract_hashtags(tokens: List[str], emotion: str) -> List[str]:
        content = [t for t in tokens if t not in _STOPWORDS and len(t) > 3
                   and t not in {kw for spec in EMOTION_LEXICON.values() for kw in spec["keywords"]}]
        tags = []
        seen = set()
        for t in content:
            if t not in seen:
                seen.add(t)
                tags.append("#" + t.capitalize())
            if len(tags) >= 3:
                break
        if emotion != "neutral":
            tags.append("#" + emotion.capitalize())
        return tags

    @staticmethod
    def _make_title(event: str, emotion: str) -> str:
        words = event.split()[:4]
        return (" ".join(w.capitalize() for w in words) or "Untitled").strip()

    # -- optional LLM backend --------------------------------------------
    _LLM_SYSTEM = (
        "You are the Event-Emotion Analyzer of a journaling app. "
        "Given a user's utterance, extract structured episodic metadata. "
        "Respond with ONLY a JSON object with keys: "
        "event (string), emotion (one of joyful, content, sad, angry, anxious, reflective, surprised, neutral), "
        "hashtags (array of 2-4 '#Tag' strings), emotional_intensity (0..1 float), "
        "valence (-1, 0 or 1), title (short string)."
    )

    def _analyze_with_llm(self, text: str) -> Optional[EpisodeMetadata]:
        try:
            raw = self.llm.complete(
                system=self._LLM_SYSTEM,
                user=text,
                temperature=0.0,
                json_mode=True,
            )
            data = json.loads(raw)
            return EpisodeMetadata(
                text=text,
                event=str(data.get("event", "")),
                emotion=str(data.get("emotion", "neutral")),
                hashtags=list(data.get("hashtags", [])),
                emotional_intensity=float(data.get("emotional_intensity", 0.0)),
                valence=int(data.get("valence", 0)),
                title=str(data.get("title", "")),
            )
        except Exception:
            # Any failure (network, parse, schema) falls back to the lexicon.
            return None
