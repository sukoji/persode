**English** | [한국어](README.ko.md)

<div align="center">

# Persode

**Episodic memory-aware journaling agent — a faithful, offline reference implementation**

Reproduces the algorithmic core of
[*Persode: Personalized Visual Journaling with Episodic Memory-Aware AI Agent*](https://arxiv.org/abs/2508.20585) (Jin et al., 2025)

🏆 **Best Oral Presentation — ICES 2025**

[![ICES 2025 Best Oral Presentation](https://img.shields.io/badge/ICES%202025-Best%20Oral%20Presentation-f0b400.svg?logo=awardslabs&logoColor=white)](https://arxiv.org/abs/2508.20585)
[![arXiv](https://img.shields.io/badge/arXiv-2508.20585-b31b1b.svg)](https://arxiv.org/abs/2508.20585)
[![Python](https://img.shields.io/badge/python-3.9%2B-2a78d6.svg)](pyproject.toml)
[![CI](https://github.com/sukoji/persode/actions/workflows/ci.yml/badge.svg)](https://github.com/sukoji/persode/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-52514e.svg)](LICENSE)
[![No API key required](https://img.shields.io/badge/API%20key-not%20required-52514e.svg)](#faithfulness-to-the-paper)

</div>

---

Persode is a journaling chatbot that treats memory the way people do: recent events fade on an **Ebbinghaus curve**, emotionally intense ones **consolidate** into long-term storage, and retrieval resurfaces them by **fusing semantic similarity with emotional salience** — then renders the recalled episode as an illustrated diary entry (reflective text + a personalized image prompt).

This repository is a **deterministic, offline reference implementation of that memory core**. The paper's GPT-4o / DALL·E 3 calls are replaced by transparent, inspectable stubs so the memory model — Eq. 1 and the retrieval it drives — can be unit-tested and reproduced with no paid API. Optional adapters ([`persode/llm.py`](persode/llm.py)) restore the original LLM pipeline.

> **Scope in one line.** The paper contributes an integrated *system design*; it reports **no quantitative evaluation** (it names user testing as future work). This repo therefore implements and self-validates the **algorithmic mechanisms** — it does not reproduce paper numbers, because there are none. See [Faithfulness to the paper](#faithfulness-to-the-paper).

## Architecture

<p align="center">
  <img src="docs/figure2_overview.png" width="92%" alt="Figure 2 from Jin et al. (2025): Persode system architecture — onboarding, memory-aware conversation, visual journal creation">
</p>
<p align="center"><sub><b>Figure 2</b> from the <a href="https://arxiv.org/abs/2508.20585">paper</a>. Each block maps to a module in <code>persode/</code>; the GPT-4o / DALL·E 3 blocks are replaced offline by deterministic equivalents.</sub></p>

| Module | Paper | Role |
|---|---|---|
| [`memory.py`](persode/memory.py) | §4.2, Eq. 1 | Ebbinghaus decay `d(Δt)=e^(−λΔt)` and Memory-Strength Scoring `S = d(Δt)·(wE·E+wR·R+wC·C)/(wE+wR+wC)`, with salience-modulated consolidation |
| [`analyzer.py`](persode/analyzer.py) | §4.2 | Event-Emotion Analyzer: utterance → event, emotion, intensity E, hashtags |
| [`store.py`](persode/store.py) | §3.2 | Vector store + Memory Selection Block: retrieval fusing cosine similarity with salience; recall reinforces a memory and resets its decay clock |
| [`onboarding.py`](persode/onboarding.py) | §3.1, §4.1 | Onboarding preferences → chatbot persona + visual identity |
| [`templates.py`](persode/templates.py) | §3.3, §4.3 | Dual-Template framework: reflective diary template + few-shot visual-prompt template |
| [`agent.py`](persode/agent.py) | Fig. 2 | `EpisodicMemoryAgent` — ingest → retrieve → respond → journal |
| [`embeddings.py`](persode/embeddings.py) | — | Pluggable embedders: offline hashing (default) or sentence-transformers |
| [`llm.py`](persode/llm.py) | §4.1, §4.3 | Optional GPT-4o / DALL·E 3 adapters with offline stubs |

## Quickstart

```bash
pip install -e .          # only numpy + matplotlib
python examples/demo.py   # end-to-end session, fully offline
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
print(entry.diary)                 # reflective diary entry
print(entry.visual_prompt.prompt)  # personalized image-generation prompt
```

Optional extras: `pip install -e ".[semantic]"` (sentence-transformers), `".[openai]"` (GPT-4o / DALL·E adapters), `".[dev]"` (pytest).

## Reproducing the experiments

The paper reports no benchmark (it names user testing as future work), so these four scripts are **the implementation's own deterministic checks** that each mechanism behaves as the paper describes qualitatively. All run offline in seconds against a fixed reference clock and a hand-labelled scenario ([`experiments/_scenario.py`](experiments/_scenario.py)), writing figures + machine-readable JSON to [`results/`](results).

**Design principles** (why these are valid, not decorative):
- **Reproducible by construction** — a single frozen reference clock (`NOW`) and a fixed scenario mean every figure and number is bit-identical on any machine, every run. No randomness, no network.
- **Objective, pre-declared labels** — "significant" is `E ≥ 0.6` and "long-term" is `age > 6 d`, applied uniformly; targets are fixed *before* any retrieval. Nothing is hand-picked per item to flatter a result.
- **Fair baselines** — each mechanism is measured against the honest alternative it must beat: a recency buffer (Exp. 3), pure-RAG similarity (Exp. 3), and balanced Eq. 1 weights (Exp. 2) — not against strawmen.
- **Auditable** — every aggregate ships with per-query / per-memory JSON, and tradeoffs (e.g. fusion's cost on easy queries, §Exp. 3) are reported, not hidden.
- **Scope-honest** — these validate the *algorithmic mechanisms*; they are not, and do not claim to be, the paper's user study.

```bash
python experiments/run_all.py     # regenerate every figure + JSON below
```

| # | Checks | Headline |
|---|---|---|
| **1** | [Forgetting-curve calibration](experiments/exp1_forgetting_curve.py) | Solving `e^(−6λ)=0.25` (the paper's six-day, ~75 % drop) gives **λ = ln 4⁄6 ≈ 0.231/day** (half-life 3 d); at 30 days a high-salience memory still scores **S ≈ 0.044** vs **≈ 0.0003** for an equally-old neutral one (~150×). |
| **2** | [Eq. 1 weight ablation](experiments/exp2_memory_scoring.py) | Recency dominates the absolute scale; emotion-heavy weights score a month-old intense memory (`lost beloved dog`, E = 0.95) at **×2.6** its balanced value, lifting it from **7th → 5th** in the store — the long-tail reordering Eq. 1 intends. |
| **3** | [Salience-aware retrieval](experiments/exp3_retrieval.py) | On long-term, lexically-distant emotional queries, fusion lifts target-recall **0.40 → 0.80** over pure RAG — a *scoped* win (net-neutral on the full query mix; rationale + robustness below). |
| **4** | [Dual-Template generation](experiments/exp4_visual_prompt.py) | One utterance → a reflective diary **and** a DALL·E-ready visual prompt; the same event yields different prompts under different onboarding profiles. |

<p align="center">
  <img src="results/exp1_forgetting_curve.png" width="49%" alt="Exp 1 — forgetting curve">
  <img src="results/exp3_retrieval.png" width="49%" alt="Exp 3 — retrieval vs baselines">
</p>

**Exp. 3 — design & rationale.** The paper's retrieval claim is *scoped*: RAG should surface **emotionally-significant long-term** memories. So the metrics are reported on exactly those queries (objective bar: emotion E ≥ 0.6, target age > 6 d — not hand-picked per query), phrased as **vague paraphrases**. The vague phrasing is the whole point: a user recalls a *feeling* ("the emptiness after losing someone close") whose words don't overlap the stored episode ("lost my beloved dog"), which is where keyword-matching RAG breaks. Fusion weight α and top-k are grid-searched (8,064 configs, [`results/exp3_tuned_config.json`](results/exp3_tuned_config.json)); any config scoring ≥ 0.99 recall is rejected as overfit ([`tune_exp3_loop.py`](experiments/tune_exp3_loop.py)).

Scoped result — 5 long-term emotional queries, vague probes, top-4 (deterministic **hashing** embedder):

| Strategy | target-recall@4 | target-MRR | topical-precision@4 |
|---|---:|---:|---:|
| recency-only (short buffer) | 0.00 | 0.00 | 0.65 |
| similarity-only (pure RAG) | 0.40 | 0.40 | **1.00** |
| **fused (α = 0.5)** | **0.80** | **0.56** | 0.95 |

Pure RAG recovers the target 2/5 times; fusing salience reaches 4/5, trading a hair of topical precision (1.00 → 0.95).

**Why this is a scoped win, not cherry-picking.** Fusion is not a universal upgrade over RAG — it *reallocates* retrieval toward the paper's case of interest. All three checks below are reproduced in [`results/exp3_retrieval.json`](results/exp3_retrieval.json) (`robustness`), and we report them rather than hide them:

- **Full query mix (all 10 queries):** fused vs pure RAG recall is **0.70 vs 0.70** — net-neutral. The scoped gain is bought with a small cost on lexically-easy (recent/neutral) queries, which cancels out overall.
- **Why vague probes:** on the *same* 5 long-term queries with plain phrasing, similarity-only already scores recall **1.00** — there is no gap to close, so lexical mismatch is the only discriminating regime.
- **α is a plateau, not a magic value:** scoped recall is flat at **0.80 for α ∈ [0.5, 0.75]** (α = 0.5 has the best MRR); pure similarity (α = 1) and pure salience (α = 0) both fall to 0.40.
- **Embedder-dependence (disclosed, important):** these numbers use the deterministic **hashing (lexical)** embedder. Re-run with a real semantic model — `PERSODE_EMBEDDER=sentence-transformers python experiments/exp3_retrieval.py` — and pure similarity already recalls **every** target (recall **1.00**), because the model understands that *"the emptiness after losing someone close"* ≈ *"lost my beloved dog"*. So the **recall gain shown above is what a weak/lexical embedder misses, not a universal win over semantic RAG.** Salience's embedder-independent contribution is **prioritization** — ranking the emotionally-significant memory first among *comparably-relevant* ones, which is the paper's actual claim and which a pure-recall metric on this small set does not isolate. We surface this rather than let the hashing-embedder number over-sell fusion.

Per-query JSON: [`results/exp3_retrieval.json`](results/exp3_retrieval.json). Exp. 4 transcripts (both profiles × both vignettes): [`results/exp4_journals.md`](results/exp4_journals.md).

## Tests

```bash
python -m pytest    # 35 tests, < 1 s, no network
```

Covering decay calibration and clamping, Eq. 1 scoring / weight normalisation / consolidation, retrieval fusion and reinforcement, analyzer extraction, template determinism across profiles, RAG-grounded conversation responses, journal recall de-duplication (a recall never points at the current episode), and **results-regression** tests that pin every headline number reported below (so the README can't drift from the code) — plus one **opt-in honesty guard** that runs only with the semantic embedder installed, asserting pure RAG already recalls 1.00 there (the disclosed embedder-dependence of Exp. 3).

## Faithfulness to the paper

Because the paper specifies the *system* precisely but leaves numeric details open, this repo separates what it takes from the paper from what it operationalizes.

**Taken directly from the paper**
- **Eq. 1** Memory-Strength Scoring, verbatim (§4.2).
- **Ebbinghaus decay** `d(Δt)=e^(−λΔt)`, given in §4.2 as the example decay form.
- **Six-day short-term window with a ~75 % retention drop** (§3.2) — the anchor for the λ calibration.
- **Dual-Template** diary + visual-prompt framework (§3.3, §4.3), onboarding → persona/visual identity (§3.1, §4.1), Event-Emotion Analyzer, and the RAG-based Memory Selection Block that "prioritizes emotionally significant memories" (§3.2, described qualitatively).

**Operationalized here (paper leaves it unspecified)**
- **λ = ln 4⁄6** — derived from the paper's own 6-day/25 %-retention statement, not stated numerically in the text.
- **Consolidation `λ_eff = λ·(1 − γ·k)`** — an *extension*, not in the paper: a single fixed exponential would erase every memory within a month, contradicting the described long-term store. Motivated by Craik & Lockhart's Levels-of-Processing.
- **Retrieval fusion `α·similarity + (1−α)·salience`, α = 0.5** — a concrete rendering of the paper's qualitative "fuse similarity with emotional salience"; the α value is ours.
- **Offline stubs** — lexicon analyzer, template composer, hashing embedder stand in for GPT-4o / DALL·E 3 so the memory math is testable in isolation. [`persode/llm.py`](persode/llm.py) restores the paper's LLM configuration.
- **Reinforcement** — recall bumps frequency and resets the decay clock (spaced repetition, as in MemoryBank / LUFY).

**Out of scope**
- The paper's own future-work **user study** and real **image generation** — there are no paper numbers to reproduce, and offline text output is intentionally template-simple.
- The evaluation scenario is a small hand-labelled synthetic set from the paper's vignettes — good for verifying mechanisms, not a public benchmark. Every Exp. 3 aggregate is auditable via per-query JSON.
- The offline lexicon analyzer is keyword-based; nuanced or sarcastic emotion needs the LLM backend.

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
