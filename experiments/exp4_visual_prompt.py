"""Experiment 4 — Dual-Template journal generation (diary + visual prompt).

Reproduces the paper's end-to-end examples (§4.3, §4.4): a dialogue turn becomes
(a) a reflective diary entry and (b) a personalised DALL·E-style image prompt via
the Few-Shot Template System. We also show that changing onboarding preferences
deterministically changes the visual prompt — the personalization the paper
emphasises.

Outputs:
    results/exp4_journals.json
    results/exp4_journals.md
"""

from __future__ import annotations

import json
import sys
from datetime import timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from persode.agent import EpisodicMemoryAgent  # noqa: E402
from persode.onboarding import OnboardingPreferences  # noqa: E402
from persode.store import MemoryStore  # noqa: E402
from _scenario import NOW, build_memories  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

# Two contrasting user profiles to demonstrate personalization.
PROFILE_A = OnboardingPreferences(
    name="Mina", age=15, glasses=False, fashion_style="trendy",
    hair="dyed yellow hair", background_theme="city", background_style="vibrant",
    conversation_style="emotional", response_length="detailed", personality="empathetic",
)
PROFILE_B = OnboardingPreferences(
    name="Jun", age=27, glasses=True, fashion_style="minimal",
    hair="short black hair", background_theme="nature", background_style="minimal",
    conversation_style="thoughtful", response_length="concise", personality="listener",
)

# The paper's two flagship vignettes.
UTTERANCES = [
    "I was scolded by my mom for spending all my allowance, I feel regretful.",
    "A car splashed puddle water on me and completely ruined my favorite outfit!",
]


def run_profile(profile: OnboardingPreferences) -> list[dict]:
    # Fresh store seeded with scenario memories so retrieval has context.
    store = MemoryStore()
    store.add_many(build_memories())
    agent = EpisodicMemoryAgent(preferences=profile, store=store)

    entries = []
    for utt in UTTERANCES:
        entry = agent.create_journal(utt, now=NOW)
        entries.append(entry.to_dict())
    return entries


def main() -> None:
    output = {
        "reference_now": NOW.isoformat(),
        "profiles": {
            "Mina (15, emotional, trendy, dyed yellow hair)": run_profile(PROFILE_A),
            "Jun (27, thoughtful, minimal, glasses)": run_profile(PROFILE_B),
        },
    }
    (RESULTS / "exp4_journals.json").write_text(json.dumps(output, indent=2, ensure_ascii=False))

    # Human-readable markdown.
    md = ["# Exp.4 — Dual-Template journal generation\n",
          f"_Reference date: {NOW.date().isoformat()}_\n"]
    for profile_name, entries in output["profiles"].items():
        md.append(f"\n## Profile: {profile_name}\n")
        for e in entries:
            md.append(f"### {e['title']}  ·  {e['date']}\n")
            md.append(f"**Diary**\n\n> {e['diary'].strip()}\n")
            md.append(f"**Visual prompt**\n\n```\n{e['visual_prompt']}\n```\n")
            md.append(f"**Extracted episode** — emotion: `{e['episode']['emotion']}`, "
                      f"intensity: `{e['episode']['emotional_intensity']}`, "
                      f"tags: {', '.join(e['episode']['hashtags'])}\n")
    (RESULTS / "exp4_journals.md").write_text("\n".join(md), encoding="utf-8")

    # Console preview.
    for profile_name, entries in output["profiles"].items():
        print(f"\n================ {profile_name} ================")
        for e in entries:
            print(f"\n[{e['title']}]  ({e['episode']['emotion']}, "
                  f"E={e['episode']['emotional_intensity']})")
            print("  DIARY :", e["diary"].replace("\n", " ")[:160])
            print("  VISUAL:", e["visual_prompt"][:160])
    print(f"\nsaved {RESULTS / 'exp4_journals.json'} and .md")


if __name__ == "__main__":
    main()
