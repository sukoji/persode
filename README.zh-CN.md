[English](README.md) | [한국어](README.ko.md) | **中文**

<div align="center">

# Persode

**具备情景记忆感知能力的个性化视觉日记智能体**

Jin et al. (2025) [*Persode: Personalized Visual Journaling with Episodic Memory-Aware AI Agent*](https://arxiv.org/abs/2508.20585) 的官方实现。

[![ICES 2025 Best Oral Presentation](https://img.shields.io/badge/ICES%202025-Best%20Oral%20Presentation-f0b400.svg?logo=awardslabs&logoColor=white)](https://arxiv.org/abs/2508.20585)
[![arXiv](https://img.shields.io/badge/arXiv-2508.20585-b31b1b.svg)](https://arxiv.org/abs/2508.20585)
[![Python](https://img.shields.io/badge/python-3.9%2B-2a78d6.svg)](pyproject.toml)
[![CI](https://github.com/sukoji/persode/actions/workflows/ci.yml/badge.svg)](https://github.com/sukoji/persode/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-52514e.svg)](LICENSE)

</div>

Persode 是一个个性化日记聊天机器人：近期事件会按 **Ebbinghaus 遗忘曲线** 衰减，情绪强烈的事件会被巩固到长期记忆中；检索则融合语义相似度与情绪显著性，以找回相关的过往经历。系统随后生成反思性日记和个性化图片提示词。

本仓库以确定性、离线方式实现记忆核心。GPT-4o 和 DALL-E 3 调用由透明 stub 替代，因此无需 API key 即可测试；可选 adapter 支持完整 LLM pipeline。实验是对设计机制的检查，不应被理解为用户研究或对真实用户效果的证明。

## 快速开始

```bash
pip install -e .
python examples/demo.py
python -m pytest
```

可选依赖：`pip install -e ".[semantic]"` 使用 sentence-transformers；`".[openai]"` 使用 GPT-4o / DALL-E；`".[dev]"` 安装测试依赖。

## 模块

| 模块 | 作用 |
|---|---|
| [`memory.py`](persode/memory.py) | Ebbinghaus 衰减与记忆强度评分 |
| [`analyzer.py`](persode/analyzer.py) | 事件、情绪、强度与 hashtag 提取 |
| [`store.py`](persode/store.py) | 向量存储与显著性感知检索 |
| [`onboarding.py`](persode/onboarding.py) | persona、对话和视觉偏好 |
| [`templates.py`](persode/templates.py) | 反思日记与视觉提示词模板 |
| [`agent.py`](persode/agent.py) | 端到端 orchestration |

## 实验与限制

运行 `python experiments/run_all.py` 可重建 Exp. 1--4 的确定性结果；LoCoMo retrieval evaluation 需单独运行 `python experiments/exp5_locomo.py`。实验结果、数据来源、baseline 和限制请以 [English README](README.md) 为准。

Persode 的公开实现包含离线 lexicon analyzer、hashing embedder 和 template stub。论文未给定的数值与算法选择均在代码中明确记录；用户研究、真实图像生成以及基于 LLM analyzer 的完整评估仍然是后续工作。

## 引用

请使用 [English README](README.md) 中的 BibTeX 条目引用 Persode。

## License

[MIT](LICENSE)
