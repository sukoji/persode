[English](README.md) | **한국어**

<div align="center">

# Persode

**에피소드 기억 인식 저널링 에이전트 — 오프라인 참조 구현**

Jin et al. (2025) [*Persode: Personalized Visual Journaling with Episodic Memory-Aware AI Agent*](https://arxiv.org/abs/2508.20585) 논문의 알고리즘 핵심을 충실히 재현합니다.

[![arXiv](https://img.shields.io/badge/arXiv-2508.20585-b31b1b.svg)](https://arxiv.org/abs/2508.20585)
[![Python](https://img.shields.io/badge/python-3.9%2B-2a78d6.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-25%20passing-1baf7a.svg)](tests)
[![License: MIT](https://img.shields.io/badge/license-MIT-52514e.svg)](LICENSE)
[![No API key required](https://img.shields.io/badge/API%20key-not%20required-52514e.svg)](#설계-선택)

</div>

---

사람은 모든 경험을 동일하게 기억하지 않습니다. 최근 사건은 빠르게 희미해지지만, 감정적으로 강렬하거나 자주 떠올리는 순간은 오래 남습니다. Persode는 이 관찰에 기반한 저널링 챗봇입니다. **에빙하우스 망각 곡선**이 단기 기억을 지배하고, **기억 강도 점수**가 장기 저장으로의 생존을 결정하며, 검색(retrieval)은 **의미적 유사도와 감정적 현저성(salience)** 을 융합해 적절한 기억이 적절한 순간에 떠오르게 합니다. 회상된 에피소드는 **일러스트 저널 항목**으로 렌더링됩니다: 성찰적 일기 텍스트와 개인화된 이미지 생성 프롬프트.

이 저장소는 전체 파이프라인을 **결정론적·오프라인** 으로 구현합니다. 모든 방정식, 임계값, 템플릿을 검사·단위 테스트·재현할 수 있으며 유료 API가 필요하지 않습니다. 선택적으로 GPT-4o / DALL·E 3 어댑터를 동일 인터페이스 뒤에 연결할 수 있습니다.

## 아키텍처

<p align="center">
  <img src="docs/figure2_overview.png" width="92%" alt="Jin et al. (2025) Figure 2: Persode 시스템 아키텍처 — 온보딩, 기억 인식 대화, 시각 저널 생성">
</p>
<p align="center"><sub><b>Figure 2</b> (<a href="https://arxiv.org/abs/2508.20585">논문</a>) — 시스템 개요: (1) 온보딩 선호 설정, (2) 기억 인식 대화, (3) 시각 저널 생성. 아래 각 블록은 <code>persode/</code> 모듈에 대응하며, GPT-4o / DALL·E 3 블록은 오프라인 결정론적 구현으로 대체됩니다.</sub></p>

| 모듈 | 논문 구성요소 | 역할 |
|---|---|---|
| [`persode/memory.py`](persode/memory.py) | §3.2, Eq. 1 | 에빙하우스 감쇠 `d(Δt)=e^(−λΔt)` — 6일 시점 25% 보존율로 보정, 기억 강도 점수 `S = d(Δt)·(wE·E + wR·R + wC·C)/(wE+wR+wC)` 및 현저성 조절 통합(consolidation) |
| [`persode/analyzer.py`](persode/analyzer.py) | Fig. 2 | Event-Emotion Analyzer: 발화 → 사건, 감정, 강도 E, 해시태그 (오프라인 어휘 사전 또는 GPT-4o) |
| [`persode/store.py`](persode/store.py) | Fig. 2 | 벡터 저장소 + **Memory Selection Block**: 코사인 유사도와 현저성을 융합한 검색; 회상 시 기억 강화 및 감쇠 시계 리셋 |
| [`persode/embeddings.py`](persode/embeddings.py) | — | 플러그형 임베더: 오프라인 해싱(기본) 또는 sentence-transformers |
| [`persode/onboarding.py`](persode/onboarding.py) | §3.1 | 온보딩 선호 → 챗봇 페르소나 프롬프트 + 시각적 정체성 |
| [`persode/templates.py`](persode/templates.py) | §3.3, §4.3–4.4 | **Dual-Template** 프레임워크: 성찰 일기 템플릿 + few-shot 시각 프롬프트 템플릿 |
| [`persode/agent.py`](persode/agent.py) | Fig. 2 | `EpisodicMemoryAgent` — 수집 → 검색 → 응답 → 저널 생성 |
| [`persode/llm.py`](persode/llm.py) | §4.1 | 선택적 GPT-4o / DALL·E 3 어댑터 및 오프라인 스텁 |

## 빠른 시작

```bash
pip install -e .          # numpy + matplotlib만 필요
python examples/demo.py   # 전체 세션, 완전 오프라인
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
print(entry.diary)                 # 성찰적 일기 항목
print(entry.visual_prompt.prompt)  # 개인화된 이미지 생성 프롬프트
```

선택적 의존성: `pip install -e ".[semantic]"` (sentence-transformers), `".[openai]"` (GPT-4o / DALL·E 어댑터), `".[dev]"` (pytest).

## 실험

[`experiments/`](experiments) 아래 네 개의 독립 스크립트가 논문 주장의 각 메커니즘을 검증합니다. 모두 결정론적(고정 기준 시각, [`experiments/_scenario.py`](experiments/_scenario.py)의 수동 라벨 시나리오)이며 수 초 내 오프라인 실행 후 [`results/`](results)에 그림과 JSON을 기록합니다.

```bash
python experiments/run_all.py          # 모든 그림 + JSON 재생성
python experiments/exp1_forgetting_curve.py
python experiments/exp2_memory_scoring.py
python experiments/exp3_retrieval.py   # results/exp3_tuned_config.json 있으면 사용
python experiments/exp4_visual_prompt.py
python experiments/tune_exp3_loop.py   # Exp. 3 하이퍼파라미터 탐색만
```

### Exp. 1 — 망각 곡선 보정

논문은 단기 기억을 6일 창에 고정하고 약 75% 보존율 하락을 기준으로 합니다. `e^(−6λ) = 0.25`를 풀면 **λ = ln 4⁄6 ≈ 0.231/일**(반감기 3일)이며, 스크립트의 assertion이 기계 정밀도로 이를 검증합니다. 오른쪽 패널은 통합 메커니즘을 보여 줍니다: 현저성이 감쇠를 늦추면(`λ_eff = λ·(1 − γ·k)`) 감정적으로 강렬한 기억은 한 달 후에도 검색 가능한 반면, 중립적 기억은 사실상 소멸합니다.

<p align="center"><img src="results/exp1_forgetting_curve.png" width="90%" alt="Exp 1 — 에빙하우스 감쇠 보정 및 현저성 조절 보존"></p>

**핵심 수치** ([`results/exp1_forgetting_curve.json`](results/exp1_forgetting_curve.json)): λ ≈ 0.231/일, 6일 보존율 0.25, 반감기 3일.

### Exp. 2 — 기억 강도 점수(Eq. 1) 가중치 ablation

시나리오 기억 10건을 네 가지 (wE, wR, wC) 가중치로 채점합니다. 그래프에서 읽을 수 있는 두 가지: (1) 최근성이 *절대* 점수를 지배(가장 젊은 두 감정 사건이 상위 — 단기 창 작동), (2) 가중치가 *노화 후 생존* 을 제어 — 감정 중심 가중치에서 한 달 된 "사랑하는 개를 잃음" 기억이 균형 가중치 대비 **×2.6** 점수를 기록하며, Eq. 1이 의도한 대로 장기 꼬리 순위가 재정렬됩니다.

<p align="center"><img src="results/exp2_memory_scoring.png" width="85%" alt="Exp 2 — Eq. 1 가중치 ablation 점 플롯"></p>

### Exp. 3 — 현저성 인식 검색(Memory Selection Block)

**6일 창을 넘는** 다섯 개의 장기 감정 쿼리에 대해 세 가지 검색 전략을 비교합니다. 쿼리는 저장된 기억 텍스트와 **어휘 겹침이 적은** 모호한 패러프레이즈이므로, 순수 RAG는 키워드 매칭에 의존할 수 없습니다. 하이퍼파라미터는 그리드 탐색으로 조정([`results/exp3_tuned_config.json`](results/exp3_tuned_config.json)); 헤드라인이 완벽한 점수의 설정은 과적합으로 거부됩니다.

| 전략 | target-recall@4 | target-MRR | topical-precision@4 | long-term recall@4 |
|---|---:|---:|---:|---:|
| recency-only (단기 버퍼) | 0.00 | 0.00 | 0.65 | 0.00 |
| similarity-only (순수 RAG) | 0.40 | 0.40 | **1.00** | 0.40 |
| **fused (Persode, α = 0.5)** | **0.80** | **0.56** | 0.95 | **0.80** |

**해석.** 최근성 버퍼는 구조적으로 장기 회상 점수가 0입니다. 프로브가 감정적으로 표현되지만 저장된 에피소드 텍스트와 어휘적으로 멀 때, 유사도만 RAG는 **2/5** 만 목표를 복구합니다. 융합 점수(α = 0.5, top-4)는 의미적 유사도와 감정적 현저성을 결합해 **4/5** 에 도달합니다 — 의심스러운 만점 없이 명확한 이득입니다. 쿼리별 상세: [`results/exp3_retrieval.json`](results/exp3_retrieval.json). 재조정: `python experiments/tune_exp3_loop.py`.

<p align="center"><img src="results/exp3_retrieval.png" width="85%" alt="Exp 3 — 베이스라인 대비 검색 품질"></p>

### Exp. 4 — Dual-Template 저널 생성

논문의 대표 비네트를 end-to-end로 실행합니다. 한 턴의 대화가 성찰적 일기 항목 **과** DALL·E 준비 시각 프롬프트가 됩니다. 대조적인 온보딩 프로필에서 동일 발화를 실행하면 시각 프롬프트가 결정론적으로 달라집니다 — 같은 사건, 다른 사람:

| | 프로필 **Mina** (15세, 트렌디, 도시) | 프로필 **Jun** (27세, 미니멀, 자연) |
|---|---|---|
| 감지된 에피소드 | `angry`, E = 0.96 | `angry`, E = 0.96 |
| 시각 프롬프트 | soft anime illustration, **15세 캐릭터, 염색 노란 머리, 안경 없음, 트렌디 패션, 생동감 있는 도시 배경**, depicting car splashed puddle water…, tense dramatic lighting, stormy frustrated mood | soft anime illustration, **27세 캐릭터, 짧은 검은 머리, 안경 착용, 미니멀 패션, 미니멀 자연 배경**, depicting car splashed puddle water…, tense dramatic lighting, stormy frustrated mood |

전체 기록(두 프로필 × 두 비네트, 일기 포함): [`results/exp4_journals.md`](results/exp4_journals.md)

## 테스트

```bash
python -m pytest    # 25개 테스트, 1초 미만, 네트워크 불필요
```

감쇠 보정·클램핑, Eq. 1 채점/가중치 정규화/통합, 검색 융합·강화, analyzer 추출, 프로필별 템플릿 결정론을 검증합니다.

## 설계 선택

- **오프라인 우선.** 논문 파이프라인은 GPT-4o와 DALL·E 3를 호출합니다. 이 구현은 투명한 결정론적 구성요소(어휘 analyzer, 템플릿 composer, 해싱 embedder)로 대체해 논문의 실제 기여인 *기억 수학* 을 격리해 테스트할 수 있습니다. [`persode/llm.py`](persode/llm.py)의 LLM 어댑터가 논문 원래 설정을 복원합니다.
- **단순 감쇠 대신 통합(consolidation).** 고정 지수 감쇠만 쓰면 한 달 안에 *모든* 기억이 지워집니다 — 논문의 장기 저장과 반대입니다. 처리 수준(Levels-of-Processing)에 따라 현저성이 감쇠를 늦추면(`λ_eff = λ·(1 − γ·k)`) Exp. 1의 교차 현상이 나타납니다.
- **강화(reinforcement).** 기억을 검색하면 회상 횟수가 증가하고 감쇠 시계가 리셋됩니다(간격 반복, MemoryBank/LUFY와 유사) — 자주 떠오르는 중요한 기억이 오래 유지됩니다.

## 범위 및 한계

- 논문의 UX 연구(N = 20)와 실제 이미지 생성은 범위 밖입니다. 오프라인 텍스트 출력은 의도적으로 단순한 템플릿입니다.
- 평가 시나리오는 논문 비네트에서 만든 소규모 수동 라벨 합성 세트입니다 — 메커니즘 검증에는 적합하나 공개 벤치마크는 아닙니다. Exp. 3은 쿼리별 JSON을 보고해 모든 집계 수치를 감사할 수 있습니다.
- 오프라인 어휘 analyzer는 키워드 기반입니다. 미묘하거나 풍자적인 감정은 LLM 백엔드가 필요합니다.
- **별도 실험 스크립트가 아닌 것:** α 융합 ablation 플롯, sentence-transformer vs 해싱 embedder 비교, 시간에 따른 강화 곡선, 논문 사용자 연구 — 위 네 핵심 스크립트만 오프라인으로 구현 가능한 메커니즘을 다룹니다.

## 인용

```bibtex
@article{jin2025persode,
  title   = {Persode: Personalized Visual Journaling with Episodic Memory-Aware AI Agent},
  author  = {Jin et al.},
  journal = {arXiv preprint arXiv:2508.20585},
  year    = {2025}
}
```

논문: [arXiv:2508.20585](https://arxiv.org/abs/2508.20585)

## 라이선스

[MIT](LICENSE)
