**English** | [한국어](README.ko.md)

**English** | [한국어](README.ko.md) | [中文](README.zh-CN.md)

<div align="center">

# Persode

**Episodic memory-aware journaling agent — official implementation**

Official implementation of
[*Persode: Personalized Visual Journaling with Episodic Memory-Aware AI Agent*](https://arxiv.org/abs/2508.20585) (Jin et al., 2025)

🏆 **Best Oral Presentation — ICES 2025**

[![ICES 2025 Best Oral Presentation](https://img.shields.io/badge/ICES%202025-Best%20Oral%20Presentation-f0b400.svg?logo=awardslabs&logoColor=white)](https://arxiv.org/abs/2508.20585)
[![arXiv](https://img.shields.io/badge/arXiv-2508.20585-b31b1b.svg)](https://arxiv.org/abs/2508.20585)
[![Python](https://img.shields.io/badge/python-3.9%2B-2a78d6.svg)](pyproject.toml)
[![CI](https://github.com/sukoji/persode/actions/workflows/ci.yml/badge.svg)](https://github.com/sukoji/persode/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-52514e.svg)](LICENSE)

</div>

---

Persode is a journaling chatbot with a human-like memory model: recent events fade on an **Ebbinghaus curve**, emotionally intense ones **consolidate** into long-term storage, and retrieval **fuses semantic similarity with emotional salience** to resurface the right episode — then renders it as an illustrated diary entry (reflective text + image prompt).

This repository implements that memory core deterministically and offline. The GPT-4o / DALL·E 3 calls are replaced by transparent stubs so the memory model is unit-testable with no API key; optional adapters ([`persode/llm.py`](persode/llm.py)) enable the full LLM pipeline. The experiments below validate each algorithmic mechanism against the design; the user study is planned as future work.

## Architecture

<p align="center">
  <img src="docs/figure2_overview.png" width="92%" alt="Figure 2 from Jin et al. (2025): Persode system architecture">
</p>
<p align="center"><sub><b>Figure 2</b> from the <a href="https://arxiv.org/abs/2508.20585">paper</a>. Each block maps to a module in <code>persode/</code>; GPT-4o / DALL·E 3 are replaced offline by deterministic equivalents.</sub></p>

| Module | Paper | Role |
|---|---|---|
| [`memory.py`](persode/memory.py) | §4.2, Eq. 1 | Ebbinghaus decay `d(Δt)=e^(−λΔt)` and Memory-Strength Scoring `S = d(Δt)·(wE·E+wR·R+wC·C)/(wE+wR+wC)`, with salience-modulated consolidation |
| [`analyzer.py`](persode/analyzer.py) | §4.2 | Event-Emotion Analyzer: utterance → event, emotion, intensity E, hashtags |
| [`store.py`](persode/store.py) | §3.2 | Vector store + Memory Selection Block: retrieval fusing similarity with salience; recall reinforces a memory and resets its decay clock |
| [`onboarding.py`](persode/onboarding.py) | §3.1, §4.1 | Onboarding preferences → chatbot persona + visual identity |
| [`templates.py`](persode/templates.py) | §3.3, §4.3 | Dual-Template framework: reflective diary + few-shot visual-prompt templates |
| [`agent.py`](persode/agent.py) | Fig. 2 | `EpisodicMemoryAgent` — ingest → retrieve → respond → journal |
| [`embeddings.py`](persode/embeddings.py) | — | Pluggable embedders: offline hashing (default) or sentence-transformers |
| [`llm.py`](persode/llm.py) | §4.1, §4.3 | Optional GPT-4o / DALL·E 3 adapters with offline stubs |

## Quickstart

```bash
pip install -e .          # numpy + matplotlib
python examples/demo.py   # end-to-end session, offline
```

```python
from persode import EpisodicMemoryAgent, MemoryStore, OnboardingPreferences

prefs = OnboardingPreferences(
    name="Mina", age=17, glasses=False, fashion_style="trendy",
    hair="dyed yellow hair", background_theme="city", background_style="vibrant",
    conversation_style="emotional", response_length="detailed", personality="empathetic",
)
agent = EpisodicMemoryAgent(preferences=prefs, store=MemoryStore())

agent.ingest("I celebrated my graduation today and I was overjoyed!")
print(agent.respond("I feel proud of myself lately, like when I graduated."))

entry = agent.create_journal("A car splashed water on me and ruined my favorite outfit!")
print(entry.diary)
print(entry.visual_prompt.prompt)
```

Optional extras: `pip install -e ".[semantic]"` (sentence-transformers), `".[openai]"` (GPT-4o / DALL·E), `".[dev]"` (pytest).

## Experiments

Four deterministic scripts validate each mechanism of the system. A fixed reference clock and hand-labelled scenario ([`experiments/_scenario.py`](experiments/_scenario.py)) make every run bit-identical; figures and machine-readable JSON are written to [`results/`](results). Labels are objective (`E ≥ 0.6` = significant, `age > 6 d` = long-term).

```bash
python experiments/run_all.py
```

| # | Mechanism | Result |
|---|---|---|
| **1** | [Forgetting curve](experiments/exp1_forgetting_curve.py) | λ = ln 4⁄6 ≈ 0.231/day from the paper's 6-day / ~75 % anchor (half-life 3 d); consolidation holds an intense memory at **S ≈ 0.044** vs **≈ 0.0003** for a neutral one at 30 days. |
| **2** | [Memory-strength scoring (Eq. 1)](experiments/exp2_memory_scoring.py) | Emotion-weighted scoring raises a month-old intense memory (`lost beloved dog`, E = 0.95) to **×2.6** its balanced value, 7th → 5th in the store. |
| **3** | [Salience-aware retrieval](experiments/exp3_retrieval.py) | Under lexically-distant probes, fusion (α = 0.5) lifts long-term emotional recall@3 to **0.60** vs **0.40** for pure similarity — at a disclosed cost on neutral/plain queries (see detail below). |
| **4** | [Dual-Template generation](experiments/exp4_visual_prompt.py) | One utterance → diary + visual prompt; **24/24** onboarding attributes injected, prompts differ by profile, emotion-mood shared. |
| **5** | [Public benchmark — LoCoMo](experiments/exp5_locomo.py) | 1,535 factual QA / 5.9k turns: ungated fusion costs recall (**0.30** vs **0.35** for pure RAG, recall@5 MiniLM) — so the agent **gates fusion by query emotion**, which restores parity (**0.35**) while keeping Exp. 3's emotional-resurfacing gains. |
| **6** | [EpiRec — labeled benchmark](experiments/exp6_epirec.py) | On [EpiRec](https://github.com/sukoji/epirec) (504 probes, authored emotion labels, corpus frozen before evaluation): the gate holds parity with pure RAG (**0.84** overall recall@3), but always-on fusion loses on *every* stratum — with the keyword analyzer's E, no measurable resurfacing gain survives at scale. Open problem, honestly recorded. |

<p align="center">
  <img src="results/exp1_forgetting_curve.png" width="49%" alt="Exp 1 — forgetting curve">
  <img src="results/exp3_retrieval.png" width="49%" alt="Exp 3 — retrieval vs baselines">
</p>

### Exp. 3 — retrieval detail

The protocol is **pre-registered**: every hyperparameter is the system's shipped default (α = 0.5, weights (1, 1, 1), top-k = 3, fixed metric threshold), set before looking at any result — nothing is tuned against the evaluation and no query subset is picked post hoc. All 10 queries (one per stored memory) run under two phrasing conditions: *plain* probes and *vague* paraphrases (one per memory, uniform rule: no content word from the stored text is reused). Hashing embedder; every number is pinned by regression tests. (In the figure, *topical-precision@3* counts retrieved memories whose query-similarity is at least half the target's — a drift check on what fusion pulls in.)

| Strategy | recall@3 (vague) | · long-term emotional (n=5) | recall@3 (plain) |
|---|---:|---:|---:|
| recency-only | 0.30 | 0.00 | 0.30 |
| similarity-only (pure RAG) | 0.30 | 0.40 | **1.00** |
| salience-only (similarity-free) | 0.30 | 0.40 | 0.30 |
| fused (α = 0.5, always) | **0.40** | **0.60** | 0.80 |
| **gated (the agent: fusion iff query E ≥ 0.6)** | **0.40** | 0.40 | **1.00** |

What fusion buys — and what it costs ([`results/exp3_retrieval.json`](results/exp3_retrieval.json)):

- **The gain is scoped:** under lexical mismatch, fusion recovers long-term emotional episodes that pure similarity misses (0.60 vs 0.40) and recency can never reach (0.00).
- **It is not a free win:** on plain probes pure similarity solves all 10 queries (1.00) while fusion drops two (0.80); under vague probes fusion loses the neutral-recent targets (0.00 vs 0.33) and pushes more emotional memories into neutral queries (intrusion 0.89 vs 0.67) — salience biases retrieval toward emotional content *by design*.
- **The gate resolves the tradeoff where the analyzer can see the emotion:** the agent applies fusion only when the query itself is emotionally significant (offline analyzer E ≥ 0.6, the repo's existing significance constant). Gated retrieval keeps similarity's perfect plain-probe recall (1.00 vs fusion's 0.80), recovers the neutral-recent targets under vague probes (0.33 vs 0.00) and cuts intrusion back to similarity's level (0.67 vs 0.89). Its cost: the keyword analyzer cannot detect emotion in deliberately word-avoiding paraphrases, so two vague emotional queries slip through the gate to plain similarity (long-term emotional 0.40 vs 0.60 for always-on fusion) — with the paper's GPT-4o analyzer as the gate this coverage should improve; that remains untested here.
- **α:** the long-term emotional bump (0.60) holds for α ∈ [0.45, 0.70]; both extremes fall back to 0.40. Note α = 0 is salience-*dominant*, not similarity-free — query similarity still enters the salience term via C; the similarity-free reference is the salience-only row.
- **Embedder:** with a semantic embedder (`PERSODE_EMBEDDER=sentence-transformers`), pure RAG reaches recall 1.00 on both conditions — the recall gap above is an artifact of the lexical embedder. Salience's embedder-independent effect is *prioritization*: given two equally-relevant memories, fusion ranks the emotionally-significant one first (`salience_prioritization` in the JSON).
- **Sample size:** n = 10 hand-labelled queries; one hit moves recall by 0.10. Read the gaps as deterministic mechanism checks, not population estimates.

<p align="center"><img src="results/exp3_alpha_ablation.png" width="72%" alt="Exp 3 — α fusion ablation sweep"></p>

### Exp. 5 — public benchmark (LoCoMo)

[LoCoMo](https://github.com/snap-research/locomo) (Maharana et al., ACL 2024) provides very-long multi-session conversations (10 dialogues, 5,882 turns, real session timestamps) whose QA pairs are annotated with the exact **evidence turns** — so the Memory Selection Block can be scored as pure retrieval, with no LLM in the loop. The protocol is pre-registered like Exp. 3 (memory construction, the same four strategies at shipped defaults, metrics, and QA-inclusion rules all fixed a priori; the adversarial category is excluded because it is unanswerable by design). 1,535 QA evaluated; the CC BY-NC data is downloaded on demand, never redistributed.

```bash
python experiments/exp5_locomo.py   # downloads data on first run
```

| Strategy | recall@5 (hashing) | recall@5 (MiniLM) | MRR (MiniLM) |
|---|---:|---:|---:|
| recency-only | 0.00 | 0.00 | 0.01 |
| similarity-only (pure RAG) | 0.15 | 0.35 | 0.29 |
| salience-only (similarity-free) | 0.01 | 0.01 | 0.02 |
| fused (α = 0.5, always) | 0.13 | 0.30 | 0.26 |
| **gated (the agent)** | **0.15** | **0.35** | **0.29** |

<p align="center"><img src="results/exp5_locomo.png" width="88%" alt="Exp 5 — LoCoMo evidence retrieval"></p>

What this shows:

- **On factual QA, always-on salience is a cost:** ungated fusion trails pure similarity by ~15 % relative recall@5, consistently across both embedders, all four categories, and all 10 conversations (0.302 ± 0.055 vs 0.352 ± 0.068). LoCoMo questions ask *facts* ("When did Caroline…"), so weighting emotional salience into the ranking only displaces on-topic turns.
- **The emotion gate removes that cost:** the agent applies fusion only to emotionally significant queries (analyzer E ≥ 0.6 — 2.9 % of LoCoMo questions), and gated retrieval matches pure similarity to three decimals (0.3535 vs 0.3533 MiniLM; 0.154 vs 0.153 hashing — the flagged queries gain slightly from fusion). Together with Exp. 3 this completes the mechanism: fusion for emotional resurfacing, similarity for factual lookup, chosen per query by the system itself.
- **Provenance, disclosed:** the gated strategy was added *after* the initial run exposed the ungated cost; the gate rule itself reuses the repo's pre-existing significance constant (E ≥ 0.6) and was fixed before evaluating it — nothing was tuned against LoCoMo results, and the ungated row stays reported.
- **No ceiling anywhere:** the best configuration reaches 0.35 recall@5, in line with LoCoMo's reputation as a hard retrieval benchmark; nothing here is saturated or hand-picked.

### Exp. 6 — EpiRec: the resurfacing claim on labeled, held-out data

The remaining gap after Exp. 5 was that no public benchmark carries emotional-salience labels, so the emotional-resurfacing claim rested on Exp. 3's hand-made n = 10 scenario. [EpiRec](https://github.com/sukoji/epirec) closes it: 12 personas, 168 timestamped journal episodes with authored intensity/valence labels, 504 probes in three types — factual, reflective naming the emotion, and reflective with **no emotion words and no content words reused** (mechanically enforced). Construction was pre-registered and the corpus frozen before any Persode strategy ran; the authored labels are never inputs to retrieval — E comes from the system's own analyzer, so the whole pipeline is tested against labels it never saw.

recall@3, MiniLM embeddings ([`results/exp6_epirec.json`](results/exp6_epirec.json) has hashing + full strata):

| Strategy | factual | reflective explicit | reflective implicit | overall |
|---|---:|---:|---:|---:|
| similarity-only (pure RAG) | 1.00 | 0.88 | **0.66** | **0.84** |
| fused (α = 0.5, always) | 0.99 | 0.82 | 0.60 | 0.80 |
| **gated (the agent)** | 1.00 | 0.87 | **0.66** | **0.84** |

- **The gate does its job at scale**: parity with pure similarity on all three probe types (gate fires on 13 % of reflective probes).
- **The honest headline is negative**: always-on fusion loses on every stratum — *including* high-intensity emotional episodes (implicit-high 0.60 vs 0.63), the exact case it was designed for. Exp. 3's small-scale resurfacing gain does not replicate on independent data when E comes from the keyword analyzer: the salience prior needs a better emotion estimator (e.g., the paper's GPT-4o analyzer) to earn its keep. That is now the system's documented open problem, and EpiRec's implicit stratum (best method 0.66) is the headroom to close.

## Tests

```bash
python -m pytest    # 40 tests, no network
```

Cover decay calibration, Eq. 1 scoring and consolidation, retrieval fusion and reinforcement, RAG-grounded responses, journal recall de-duplication, analyzer extraction, template determinism, and results-regression checks that pin every number above — including honesty guards that fail if fusion's costs (plain-probe and neutral-query losses, and the LoCoMo factual-QA gap) stop being reported. Two further tests need optional extras: the semantic embedder, and the downloaded LoCoMo data.

## Implementation notes

**Specified in the paper.** Eq. 1 Memory-Strength Scoring (§4.2); Ebbinghaus decay `d(Δt)=e^(−λΔt)` (§4.2); six-day / ~75 % short-term window (§3.2); Dual-Template framework (§3.3, §4.3); onboarding → persona and visual identity (§3.1, §4.1); Event-Emotion Analyzer and the RAG Memory Selection Block (§3.2).

**Set in this code** (where the paper leaves values open). λ = ln 4⁄6 (from the 6-day / 25 % anchor); consolidation `λ_eff = λ·(1 − γ·k)`, so salient memories persist past the short-term window; retrieval fusion `α·similarity + (1−α)·salience`, α = 0.5 (note similarity also enters salience via C, so the effective similarity weight at α = 0.5 is ≈ 0.67 with equal weights); an **emotion gate** on retrieval (fusion only when the query's analyzer E ≥ 0.6, else pure similarity — motivated by Exp. 5); reinforcement on recall restarts the decay clock at `last_recalled` without rewriting the formation date (spaced repetition); offline lexicon / template / hashing stubs standing in for GPT-4o / DALL·E 3. The Exp. 3 evaluation protocol is pre-registered from these defaults — no hyperparameter or query selection against the results.

**Not included.** The user study (future work) and real image generation; the offline analyzer is keyword-based — which bounds both the emotion gate (it misses emotion in word-avoiding paraphrases) and, per Exp. 6, the salience prior itself: on EpiRec's labeled corpus, fusion driven by keyword-derived E shows no resurfacing gain over pure similarity, so the mechanism's value is currently conditional on a stronger E estimator (the paper's GPT-4o analyzer — untested here). The definitive test of the journaling experience remains the future user study.

## Citation

```bibtex
@inproceedings{jin2025persode,
  title     = {Persode: Personalized Visual Journaling with Episodic Memory-Aware AI Agent},
  author    = {Jin, Seokho and Kim, Manseo and Byun, Sungho and Kim, Hansol and
               Lee, Jungmin and Baek, Sujeong and Kim, Semi and Park, Sanghum and Park, Sung},
  booktitle = {ICES},
  year      = {2025},
  note      = {Best Oral Presentation. arXiv:2508.20585},
  eprint    = {2508.20585},
  archivePrefix = {arXiv},
  primaryClass  = {cs.HC}
}
```

## License

[MIT](LICENSE)
