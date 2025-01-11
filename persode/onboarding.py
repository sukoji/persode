"""Onboarding preferences — the personalization layer (paper §4.1, Figure 3).

During onboarding the user fixes (a) visual identity — glasses, fashion,
background — and (b) chatbot personality — conversation style, response length,
empathy. These are stored once and then *dynamically referenced* both when the
agent answers (Figure 1.B "Chatbot Style Integration") and when the Few-Shot
Template System builds image prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List

# Options mirror the toggles shown in Figure 3 of the paper.
CONVERSATION_STYLES = {
    "default": "Maintains a balanced, neutral tone.",
    "friendly": "Warm and approachable, like a close friend.",
    "thoughtful": "Measured and introspective, inviting deeper reflection.",
    "emotional": "Highly emotional and expressive, mirroring the user's feelings with passion.",
}
RESPONSE_LENGTHS = {
    "detailed": "Prefers detailed and lengthy answers that fully explore the moment.",
    "concise": "Prefers short, to-the-point answers.",
}
PERSONALITIES = {
    "empathetic": "Responds with passionate empathy toward the user's feelings.",
    "listener": "Acts as a calm listener, giving the user space to talk.",
}


@dataclass
class OnboardingPreferences:
    """A user's stored onboarding profile."""

    # Demographics / persona
    name: str = "User"
    age: int = 20

    # Visual identity (feeds the text-to-image pipeline)
    glasses: bool = False
    fashion_style: str = "casual"          # casual | trendy | ...
    hair: str = "black shoulder-length hair"
    background_theme: str = "city"         # city | nature | ...
    background_style: str = "vibrant"      # minimal | vibrant | ...
    art_style: str = "soft anime illustration"

    # Chatbot personality (feeds the response prompt)
    conversation_style: str = "friendly"   # default | friendly | thoughtful | emotional
    response_length: str = "detailed"      # detailed | concise
    personality: str = "empathetic"        # empathetic | listener

    extra_visual_tags: List[str] = field(default_factory=list)

    # -- chatbot style prompt (Figure 1.B) --------------------------------
    def build_style_prompt(self) -> str:
        """Compose the chatbot's system-prompt fragment from the toggles.

        The paper describes *automatically generating and combining tailored
        prompts* from the selected options; this is that combination.
        """
        parts = [
            f"You are {self.name}'s personal journaling companion.",
            CONVERSATION_STYLES.get(self.conversation_style, CONVERSATION_STYLES["default"]),
            PERSONALITIES.get(self.personality, PERSONALITIES["empathetic"]),
            RESPONSE_LENGTHS.get(self.response_length, RESPONSE_LENGTHS["detailed"]),
        ]
        if self.conversation_style == "emotional":
            parts.append(
                "Use rich, adolescent-like emotional expressions and avoid overly "
                "complex vocabulary."
            )
        return " ".join(parts)

    # -- visual style dict (Few-Shot Template System) ---------------------
    def build_visual_style(self) -> Dict[str, Any]:
        """The stylistic block injected into every image prompt."""
        descriptors = [
            self.art_style,
            f"a {self.age}-year-old character",
            self.hair,
            "wearing glasses" if self.glasses else "no glasses",
            f"{self.fashion_style} fashion",
            f"{self.background_style} {self.background_theme} background",
        ]
        descriptors.extend(self.extra_visual_tags)
        return {
            "art_style": self.art_style,
            "descriptors": descriptors,
            "descriptor_string": ", ".join(descriptors),
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OnboardingPreferences":
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in allowed})
