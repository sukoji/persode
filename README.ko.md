[English](README.md) | **한국어**

<div align="center">

# Persode

**에피소드 기억 인식 저널링 에이전트 — 공식 구현**

Jin et al. (2025) [*Persode: Personalized Visual Journaling with Episodic Memory-Aware AI Agent*](https://arxiv.org/abs/2508.20585) 공식 구현.

🏆 **Best Oral Presentation — ICES 2025**

[![ICES 2025 Best Oral Presentation](https://img.shields.io/badge/ICES%202025-Best%20Oral%20Presentation-f0b400.svg?logo=awardslabs&logoColor=white)](https://arxiv.org/abs/2508.20585)
[![arXiv](https://img.shields.io/badge/arXiv-2508.20585-b31b1b.svg)](https://arxiv.org/abs/2508.20585)
[![Python](https://img.shields.io/badge/python-3.9%2B-2a78d6.svg)](pyproject.toml)
[![CI](https://github.com/sukoji/persode/actions/workflows/ci.yml/badge.svg)](https://github.com/sukoji/persode/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-52514e.svg)](LICENSE)

</div>

---

Persode는 인간과 유사한 기억 모델을 가진 저널링 챗봇입니다. 최근 사건은 **에빙하우스 곡선**을 따라 희미해지고, 감정적으로 강렬한 사건은 장기 저장으로 **통합(consolidation)** 되며, 검색은 **의미 유사도와 감정 현저성(salience)을 융합**해 적절한 에피소드를 회상한 뒤 일러스트 일기(성찰 텍스트 + 이미지 프롬프트)로 렌더링합니다.

이 저장소는 그 기억 핵심을 결정론적·오프라인으로 구현합니다. GPT-4o / DALL·E 3 호출은 투명 스텁으로 대체되어 API 키 없이 기억 모델을 단위 테스트할 수 있으며, 선택적 어댑터([`persode/llm.py`](persode/llm.py))로 전체 LLM 파이프라인을 사용할 수 있습니다. 아래 실험은 각 알고리즘 메커니즘을 설계에 비추어 검증하며, 유저 스터디는 향후 과제입니다.

## 아키텍처

<p align="center">
  <img src="docs/figure2_overview.png" width="92%" alt="Jin et al. (2025) Figure 2: Persode 시스템 아키텍처">
</p>
<p align="center"><sub><b>Figure 2</b> (<a href="https://arxiv.org/abs/2508.20585">논문</a>). 각 블록은 <code>persode/</code> 모듈에 대응하며, GPT-4o / DALL·E 3는 오프라인 결정론적 구현으로 대체됩니다.</sub></p>

| 모듈 | 논문 | 역할 |
|---|---|---|
| [`memory.py`](persode/memory.py) | §4.2, Eq. 1 | 에빙하우스 감쇠 `d(Δt)=e^(−λΔt)`와 기억 강도 점수 `S = d(Δt)·(wE·E+wR·R+wC·C)/(wE+wR+wC)`, 현저성 조절 통합 |
| [`analyzer.py`](persode/analyzer.py) | §4.2 | Event-Emotion Analyzer: 발화 → 사건, 감정, 강도 E, 해시태그 |
| [`store.py`](persode/store.py) | §3.2 | 벡터 저장소 + Memory Selection Block: 유사도와 현저성을 융합한 검색; 회상 시 기억 강화·감쇠 시계 리셋 |
| [`onboarding.py`](persode/onboarding.py) | §3.1, §4.1 | 온보딩 선호 → 챗봇 페르소나 + 시각 정체성 |
| [`templates.py`](persode/templates.py) | §3.3, §4.3 | Dual-Template 프레임워크: 성찰 일기 + few-shot 시각 프롬프트 템플릿 |
| [`agent.py`](persode/agent.py) | Fig. 2 | `EpisodicMemoryAgent` — ingest → retrieve → respond → journal |
| [`embeddings.py`](persode/embeddings.py) | — | 교체 가능한 임베더: 오프라인 해싱(기본) 또는 sentence-transformers |
| [`llm.py`](persode/llm.py) | §4.1, §4.3 | 선택적 GPT-4o / DALL·E 3 어댑터 + 오프라인 스텁 |

## 빠른 시작

```bash
pip install -e .          # numpy + matplotlib
python examples/demo.py   # 오프라인 엔드-투-엔드 세션
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

선택적 확장: `".[semantic]"` (sentence-transformers), `".[openai]"` (GPT-4o / DALL·E), `".[dev]"` (pytest).

## 실험

네 개의 결정론적 스크립트가 기억 모델의 각 메커니즘을 검증합니다. 고정 기준 시계와 수작업 라벨 시나리오([`experiments/_scenario.py`](experiments/_scenario.py))로 매 실행이 비트 단위로 동일하며, 그림과 기계 판독 JSON을 [`results/`](results)에 씁니다. 라벨은 객관적입니다(`E ≥ 0.6` = 중요, `나이 > 6일` = 장기).

```bash
python experiments/run_all.py
```

| # | 메커니즘 | 결과 |
|---|---|---|
| **1** | [망각곡선](experiments/exp1_forgetting_curve.py) | 논문의 6일 / ~75% 기준에서 λ = ln 4⁄6 ≈ 0.231/day(반감기 3일); 통합으로 강렬한 기억은 30일에 **S ≈ 0.044**, 중립 기억은 **≈ 0.0003**. |
| **2** | [기억 강도 점수(Eq. 1)](experiments/exp2_memory_scoring.py) | 감정 가중 점수가 한 달 된 강렬한 기억(`lost beloved dog`, E = 0.95)을 균형값의 **×2.6**, 7위 → 5위로 상승. |
| **3** | [현저성 인식 검색](experiments/exp3_retrieval.py) | 어휘적으로 먼 표현의 장기 감정 질의에서 융합(α = 0.5)이 **recall@4 0.80**, 순수 유사도는 **0.40**. |
| **4** | [Dual-Template 생성](experiments/exp4_visual_prompt.py) | 한 발화 → 일기 + 시각 프롬프트; 온보딩 속성 **24/24** 주입, 프로필별 프롬프트 상이, 감정 무드 공유. |

<p align="center">
  <img src="results/exp1_forgetting_curve.png" width="49%" alt="Exp 1 — 망각곡선">
  <img src="results/exp3_retrieval.png" width="49%" alt="Exp 3 — 검색 vs 베이스라인">
</p>

### Exp. 3 — 검색 상세

장기 감정 질의 5개를 어휘 겹침이 낮은 모호한 패러프레이즈로 평가합니다. α·top-k는 8,064개 config 그리드 서치([`tune_exp3_loop.py`](experiments/tune_exp3_loop.py)); recall ≥ 0.99인 config는 기각. 해싱 임베더.

| 전략 | recall@4 | MRR | topical-precision@4 |
|---|---:|---:|---:|
| recency-only | 0.00 | 0.00 | 0.65 |
| similarity-only (순수 RAG) | 0.40 | 0.40 | **1.00** |
| **fused (α = 0.5)** | **0.80** | **0.56** | 0.95 |

Robustness ([`results/exp3_retrieval.json`](results/exp3_retrieval.json)):

- **전체 10개 질의:** 융합과 순수 RAG가 recall 0.70으로 동률 — 이득은 장기 감정 회상에 한정되며 보편적이지 않음.
- **일반 표현:** 모호하지 않은 probe에서는 순수 RAG가 이미 recall 1.00 — 격차는 어휘 불일치에서만 발생.
- **α:** recall이 α ∈ [0.45, 0.95] 전 구간에서 0.80 유지; 순수 유사도(α = 1)·순수 현저성(α = 0)만 0.40으로 하락.
- **임베더:** 의미 임베더(`PERSODE_EMBEDDER=sentence-transformers`)에서는 순수 RAG가 recall 1.00 — 위 recall 이득은 어휘 임베더에 기인함. salience의 임베더-독립적 효과는 *우선순위화*로, 동등하게 관련된 두 기억이 있을 때 융합이 감정적으로 중요한 쪽을 먼저 랭크(JSON의 `salience_prioritization`).

<p align="center"><img src="results/exp3_alpha_ablation.png" width="72%" alt="Exp 3 — α 융합 ablation 스윕"></p>

## 테스트

```bash
python -m pytest    # 37개 테스트, 네트워크 불필요
```

감쇠 보정, Eq. 1 점수·통합, 검색 융합·강화, RAG 기반 응답, 저널 회상 중복 제거, analyzer 추출, 템플릿 결정성, 그리고 위 모든 수치를 고정하는 결과 회귀 검사를 커버합니다. 의미 임베더가 설치된 경우에만 실행되는 테스트가 하나 더 있습니다.

## 구현 노트

**논문에 명시.** Eq. 1 기억 강도 점수(§4.2); 에빙하우스 감쇠 `d(Δt)=e^(−λΔt)`(§4.2); 6일 / ~75% 단기 창(§3.2); Dual-Template 프레임워크(§3.3, §4.3); 온보딩 → 페르소나·시각 정체성(§3.1, §4.1); Event-Emotion Analyzer와 RAG Memory Selection Block(§3.2).

**이 코드에서 설정**(논문이 값을 열어둔 부분). λ = ln 4⁄6 (6일 / 25% 기준에서 유도); 통합 `λ_eff = λ·(1 − γ·k)` — 현저성 높은 기억이 단기 창 이후에도 유지되게 함; 검색 융합 `α·similarity + (1−α)·salience`, α = 0.5; 회상 시 강화(간격 반복); GPT-4o / DALL·E 3 대체용 오프라인 어휘·템플릿·해싱 스텁.

**미포함.** 유저 스터디(향후 과제)와 실제 이미지 생성; 평가 시나리오는 소규모 수작업 셋으로 공개 벤치마크가 아님; 오프라인 analyzer는 키워드 기반.

## 인용

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

## 라이선스

[MIT](LICENSE)
