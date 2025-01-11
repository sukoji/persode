"""Dual-Template Episodic Memory Generation Framework (paper §3.3, §4.3, §4.4).

Two complementary templates turn structured episode metadata into a journal:

- :class:`DiaryTemplate` — the **text template**: a reflective, first-person
  diary entry (produced by GPT-4o in the paper; a deterministic composer is used
  offline).
- :class:`FewShotTemplateSystem` — the **visual template**: automatically fuses
  onboarding visual preferences with event-emotion metadata into a detailed
  text-to-image prompt (fed to DALL·E 3 in the paper).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .analyzer import EpisodeMetadata
from .onboarding import OnboardingPreferences


# A short-hand mapping from emotion → visual mood cues for the image prompt.
_EMOTION_VISUALS = {
    "joyful": "bright warm lighting, uplifting joyful mood",
    "content": "soft golden light, calm and serene mood",
    "sad": "muted cool tones, melancholic reflective mood",
    "angry": "tense dramatic lighting, stormy frustrated mood",
    "anxious": "dim uneasy lighting, restless anxious mood",
    "reflective": "gentle diffused light, contemplative nostalgic mood",
    "surprised": "sharp contrast lighting, startled mood",
    "neutral": "even natural lighting, quiet everyday mood",
}


@dataclass
class VisualPrompt:
    prompt: str
    negative_prompt: str = "text, watermark, extra fingers, deformed"


class FewShotTemplateSystem:
    """Builds DALL·E-style prompts by combining preferences + metadata.

    Paper §4.3: *"combines user-defined onboarding preferences (age, fashion,
    background) with metadata extracted from user conversations ... automates the
    transformation of these inputs into detailed image prompts."*  A couple of
    few-shot exemplars (drawn straight from the paper's own examples) anchor the
    format.
    """

    FEW_SHOT_EXEMPLARS = [
        {
            "meta": "event: scolded for spending allowance; emotion: sorrowful but reflective; tags: #Allowance #Regret",
            "style": "soft anime illustration, a 15-year-old character, dyed yellow hair, casual fashion",
            "prompt": (
                "Soft anime illustration of a 15-year-old teenage girl with dyed yellow hair in "
                "casual clothing, standing with a regretful downcast expression as her mother scolds "
                "her, muted cool tones and melancholic reflective mood, cinematic composition."
            ),
        },
        {
            "meta": "event: car splashed puddle water and ruined favorite outfit; emotion: frustration and sadness; tags: #FavoriteOutfit #Laundry #Upset",
            "style": "soft anime illustration, a 20-year-old character, casual fashion, vibrant city background",
            "prompt": (
                "Soft anime illustration of a 20-year-old character in casual clothing kneeling by a "
                "washing basin, hand-washing a water-stained outfit with a contemplative sorrowful "
                "expression, muted cool tones and melancholic reflective mood, vibrant city background."
            ),
        },
    ]

    def build(self, episode: EpisodeMetadata, prefs: OnboardingPreferences) -> VisualPrompt:
        style = prefs.build_visual_style()
        mood = _EMOTION_VISUALS.get(episode.emotion, _EMOTION_VISUALS["neutral"])
        event = episode.event or episode.text
        prompt = (
            f"{style['descriptor_string']}, depicting {event}, "
            f"conveying {episode.emotion} emotion, {mood}, cinematic composition, "
            f"highly detailed."
        )
        return VisualPrompt(prompt=prompt)

    def build_llm_prompt(self, episode: EpisodeMetadata, prefs: OnboardingPreferences) -> str:
        """The few-shot prompt that would be sent to an image-prompt LLM."""
        lines = [
            "Transform the episode metadata and user style into a single vivid "
            "image-generation prompt. Follow the format of the examples.",
            "",
        ]
        for ex in self.FEW_SHOT_EXEMPLARS:
            lines += [f"METADATA: {ex['meta']}", f"STYLE: {ex['style']}",
                      f"PROMPT: {ex['prompt']}", ""]
        style = prefs.build_visual_style()
        meta = (f"event: {episode.event}; emotion: {episode.emotion}; "
                f"tags: {' '.join(episode.hashtags)}")
        lines += [f"METADATA: {meta}", f"STYLE: {style['descriptor_string']}", "PROMPT:"]
        return "\n".join(lines)


class DiaryTemplate:
    """Builds the reflective, first-person diary entry (text template).

    In the paper GPT-4o writes this from the dialogue; offline we compose a
    faithful reflective entry from the extracted metadata and retrieved memories
    so the pipeline is runnable end-to-end without an API key.
    """

    _OPENERS = {
        "joyful": "Today felt bright.",
        "content": "Today moved at a gentle pace.",
        "sad": "Today sat heavy on me.",
        "angry": "Today tested my patience.",
        "anxious": "Today kept me on edge.",
        "reflective": "Today gave me something to think about.",
        "surprised": "Today caught me off guard.",
        "neutral": "Today was an ordinary day.",
    }
    _CLOSERS = {
        1: "I want to hold on to this feeling.",
        0: "I'll let today simply be what it was.",
        -1: "I hope tomorrow feels a little lighter.",
    }
    # Noun form of each emotion label, for natural diary phrasing.
    _EMOTION_NOUNS = {
        "joyful": "joy",
        "content": "contentment",
        "sad": "sadness",
        "angry": "anger",
        "anxious": "anxiety",
        "reflective": "weight",
        "surprised": "surprise",
        "neutral": "feeling",
    }

    def build_prompt(self, episode: EpisodeMetadata, prefs: OnboardingPreferences,
                     related: Optional[List[str]] = None) -> str:
        """The instruction that would be sent to GPT-4o."""
        related = related or []
        ctx = ("\nRelated past memories:\n- " + "\n- ".join(related)) if related else ""
        return (
            f"{prefs.build_style_prompt()}\n"
            f"Write a short first-person reflective diary entry (3-5 sentences) about this "
            f"event: '{episode.event}'. Dominant emotion: {episode.emotion}. "
            f"Tags: {', '.join(episode.hashtags)}.{ctx}\n"
            f"Keep it {prefs.response_length}, emotionally honest, and tailored to the user."
        )

    def compose_offline(self, episode: EpisodeMetadata, prefs: OnboardingPreferences,
                        related: Optional[List[str]] = None) -> str:
        """Deterministic offline diary entry (no LLM required)."""
        opener = self._OPENERS.get(episode.emotion, self._OPENERS["neutral"])
        body = f"I keep coming back to {episode.event}."
        if episode.emotional_intensity >= 0.7:
            noun = self._EMOTION_NOUNS.get(episode.emotion, "feeling")
            body += f" The {noun} of it still lingers."
        recall = ""
        if related:
            recall = f" It reminded me of {related[0]}."
        closer = self._CLOSERS.get(episode.valence, self._CLOSERS[0])
        tags = " ".join(episode.hashtags)
        return f"{opener} {body}{recall} {closer}\n\n{tags}".strip()
