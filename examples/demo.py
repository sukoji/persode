"""End-to-end Persode demo — a short reflective session, offline.

Run:  python examples/demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persode import EpisodicMemoryAgent, MemoryStore, OnboardingPreferences


def main() -> None:
    # 1) Onboarding — the user personalizes identity + chatbot personality.
    prefs = OnboardingPreferences(
        name="Mina", age=17, glasses=False, fashion_style="trendy",
        hair="dyed yellow hair", background_theme="city", background_style="vibrant",
        conversation_style="emotional", response_length="detailed", personality="empathetic",
    )
    print("Chatbot style prompt:\n ", prefs.build_style_prompt(), "\n")

    agent = EpisodicMemoryAgent(preferences=prefs, store=MemoryStore())

    # 2) A few reflective turns get analyzed and stored as episodic memories.
    turns = [
        "I celebrated my graduation today and I was overjoyed!",
        "I had a cozy dinner with my family, it felt warm and content.",
        "I bought groceries and cooked pasta, nothing special.",
        "A car splashed water on me and ruined my favorite outfit, I'm so upset!",
    ]
    for t in turns:
        m = agent.ingest(t)
        print(f"  stored: [{m.emotion:9s} E={m.emotional_intensity:.2f}] {m.event}")

    # 3) Memory-augmented conversation (RAG retrieval + style).
    print("\n--- agent response (offline stub) ---")
    print(agent.respond("I feel proud of myself lately, like when I graduated."))

    # 4) Dual-Template journal for the last emotional event.
    print("\n--- illustrated diary ---")
    entry = agent.create_journal("A car splashed water on me and ruined my favorite outfit, I'm so upset!")
    print("Title :", entry.title)
    print("Diary :", entry.diary.replace("\n", " "))
    print("Visual:", entry.visual_prompt.prompt)


if __name__ == "__main__":
    main()
