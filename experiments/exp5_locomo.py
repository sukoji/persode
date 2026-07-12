"""Experiment 5 — public-benchmark retrieval evaluation on LoCoMo.

LoCoMo (Maharana et al., ACL 2024) is a benchmark of very-long multi-session
conversations (10 dialogues, ~5.9k turns, session timestamps) with QA pairs
annotated with the *evidence turns* required to answer. Because gold evidence
is annotated at the turn level, the Memory Selection Block can be evaluated as
a pure retrieval task — no LLM, no generation, fully deterministic.

**Pre-registered protocol** (fixed before any result was computed):

- Memory unit: one conversation turn → one ``Memory``. Text is
  ``"{speaker}: {text}"`` plus ``"[shares a photo: {caption}]"`` when the turn
  has an image caption. ``created_at`` = the turn's session timestamp + one
  second per turn index (stable within-session ordering); reference ``now`` =
  the conversation's last session timestamp + 1 day.
- Scoring components: E and emotion from the offline ``EventEmotionAnalyzer``
  (the system's own ingestion path), stored C = 0.5 (the agent's ingestion
  default), ``recall_count`` = 0.
- Strategies: the same four as Exp. 3 — recency-only, similarity-only,
  salience-only (similarity-free), fused (α = 0.5) — all rankings produced by
  ``MemoryStore.retrieve`` (the actual system code), scorer/fusion at shipped
  defaults.
- QA inclusion: categories 1–4 (multi-hop, temporal, open-domain, single-hop).
  Category 5 (adversarial) is excluded a priori: it is unanswerable by design,
  so its "evidence" marks misleading content, not an answer location. Evidence
  strings are parsed by extracting every ``D<sess>:<turn>`` token; ids that do
  not exist in the conversation are dropped, and QA with no resolvable
  evidence are excluded. All exclusion counts are reported.
- Metrics: evidence-recall@k (|gold ∩ top-k| / |gold|), hit@k (any gold in
  top-k), MRR (reciprocal rank of the first gold in the full ranking).
  Primary k = 5; k = 10 also reported. Aggregation: micro-average over QA,
  plus per-category breakdown and per-conversation mean ± std for recall@5.
- Embedders: hashing (offline default) always; sentence-transformers
  (all-MiniLM-L6-v2) additionally when installed. Both reported.

The dataset (CC BY-NC 4.0) is NOT redistributed with this repository — it is
downloaded on first run to ``experiments/data/locomo10.json`` (gitignored) and
verified by SHA-256.

Outputs:
    results/exp5_locomo.png
    results/exp5_locomo.json
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, pstdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _style as style  # noqa: E402
from persode.analyzer import EventEmotionAnalyzer  # noqa: E402
from persode.embeddings import BaseEmbedder, get_embedder  # noqa: E402
from persode.memory import Memory, MemoryStrengthScorer  # noqa: E402
from persode.store import MemoryStore  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_PATH = DATA_DIR / "locomo10.json"
DATA_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
DATA_SHA256 = None  # pinned after first verified download; see ensure_data()

ALPHA = 0.5          # shipped fusion default (same as Exp. 3)
TOP_KS = (5, 10)     # primary k = 5
PRIMARY_K = 5
CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
_EVIDENCE_RE = re.compile(r"D\d+:\d+")
_DATE_FMT = "%I:%M %p on %d %B, %Y"


def ensure_data() -> Path:
    """Download the LoCoMo data file on demand (not redistributed: CC BY-NC)."""
    if not DATA_PATH.exists():
        DATA_DIR.mkdir(exist_ok=True)
        print(f"downloading LoCoMo data from {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)  # noqa: S310 — pinned https URL
    digest = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    print(f"locomo10.json sha256 = {digest}")
    return DATA_PATH


class CachedEmbedder(BaseEmbedder):
    """Wraps an embedder with a text → vector cache (queries repeat per strategy)."""

    def __init__(self, base: BaseEmbedder) -> None:
        self._base = base
        self.dim = base.dim
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, text: str) -> np.ndarray:
        v = self._cache.get(text)
        if v is None:
            v = self._base.embed(text)
            self._cache[text] = v
        return v


def build_conversation(sample: dict, analyzer: EventEmotionAnalyzer):
    """Turn one LoCoMo conversation into (memories, dia_id order, now)."""
    conv = sample["conversation"]
    session_keys = sorted(
        (k for k in conv if re.fullmatch(r"session_\d+", k) and isinstance(conv[k], list)),
        key=lambda k: int(k.split("_")[1]),
    )
    memories: list[Memory] = []
    last_dt = None
    for sk in session_keys:
        dt = datetime.strptime(conv[f"{sk}_date_time"], _DATE_FMT).replace(tzinfo=timezone.utc)
        last_dt = dt if last_dt is None or dt > last_dt else last_dt
        for i, turn in enumerate(conv[sk]):
            text = f"{turn['speaker']}: {turn.get('text', '')}"
            caption = turn.get("blip_caption")
            if caption:
                text += f" [shares a photo: {caption}]"
            meta = analyzer.analyze(text)
            memories.append(Memory(
                text=text,
                event=turn["dia_id"],          # dia_id doubles as the ranking key
                emotion=meta.emotion,
                emotional_intensity=meta.emotional_intensity,
                contextual_relevance=0.5,      # agent.ingest default
                created_at=dt + timedelta(seconds=i),
            ))
    now = last_dt + timedelta(days=1)
    return memories, now


def parse_evidence(raw, valid_ids: set) -> list[str]:
    tokens = _EVIDENCE_RE.findall(" ".join(map(str, raw or [])))
    return [t for t in dict.fromkeys(tokens) if t in valid_ids]


def rank_ids(name: str, store: dict, query: str) -> list[str]:
    """Full ranking of dia_ids under one strategy, via MemoryStore.retrieve."""
    if name == "recency-only":
        return store["recency_ranking"]
    st: MemoryStore = store[name]
    kwargs = {}
    if name == "salience-only":
        kwargs["use_query_relevance_as_context"] = False
    hits = st.retrieve(query, top_k=len(st), now=store["now"], reinforce=False, **kwargs)
    return [r.memory.event for r in hits]


STRATEGIES = ("recency-only", "similarity-only", "salience-only", "fused (Persode)")


def build_stores(memories, now, embedder) -> dict:
    scorer = MemoryStrengthScorer()  # shipped defaults (1,1,1), protection 0.9
    stores = {"now": now}
    for name, alpha in (("similarity-only", 1.0), ("salience-only", 0.0),
                        ("fused (Persode)", ALPHA)):
        st = MemoryStore(embedder=embedder, scorer=scorer, w_similarity=alpha)
        st.add_many(memories)   # embeddings computed once, shared via Memory objects
        stores[name] = st
    stores["recency_ranking"] = [
        m.event for m in sorted(memories, key=lambda m: m.created_at, reverse=True)
    ]
    return stores


def evaluate(embedder_name: str) -> dict:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    analyzer = EventEmotionAnalyzer()
    embedder = CachedEmbedder(get_embedder(embedder_name))

    excluded = {"adversarial_cat5": 0, "no_evidence": 0, "unresolvable_evidence": 0}
    agg = {s: {"recall": {k: [] for k in TOP_KS}, "hit": {k: [] for k in TOP_KS},
               "mrr": [], "by_cat": {c: [] for c in CATEGORY_NAMES},
               "by_conv": {}} for s in STRATEGIES}

    for sample in data:
        memories, now = build_conversation(sample, analyzer)
        valid_ids = {m.event for m in memories}
        stores = build_stores(memories, now, embedder)
        conv_id = sample["sample_id"]

        for q in sample["qa"]:
            if int(q["category"]) == 5:
                excluded["adversarial_cat5"] += 1
                continue
            if not q.get("evidence"):
                excluded["no_evidence"] += 1
                continue
            gold = parse_evidence(q["evidence"], valid_ids)
            if not gold:
                excluded["unresolvable_evidence"] += 1
                continue
            cat = int(q["category"])

            for s in STRATEGIES:
                ranked = rank_ids(s, stores, q["question"])
                gold_set = set(gold)
                for k in TOP_KS:
                    topk = set(ranked[:k])
                    agg[s]["recall"][k].append(len(gold_set & topk) / len(gold_set))
                    agg[s]["hit"][k].append(float(bool(gold_set & topk)))
                first = next((i for i, e in enumerate(ranked) if e in gold_set), None)
                agg[s]["mrr"].append(0.0 if first is None else 1.0 / (first + 1))
                agg[s]["by_cat"][cat].append(len(gold_set & set(ranked[:PRIMARY_K])) / len(gold_set))
                agg[s]["by_conv"].setdefault(conv_id, []).append(
                    len(gold_set & set(ranked[:PRIMARY_K])) / len(gold_set))

    out = {"excluded": excluded, "strategies": {}}
    for s in STRATEGIES:
        a = agg[s]
        conv_means = [mean(v) for v in a["by_conv"].values()]
        out["strategies"][s] = {
            **{f"evidence_recall@{k}": round(mean(a["recall"][k]), 4) for k in TOP_KS},
            **{f"hit@{k}": round(mean(a["hit"][k]), 4) for k in TOP_KS},
            "mrr": round(mean(a["mrr"]), 4),
            "recall@5_by_category": {
                CATEGORY_NAMES[c]: {"recall": round(mean(v), 4), "n": len(v)}
                for c, v in a["by_cat"].items() if v
            },
            "recall@5_by_conversation": {
                "mean": round(mean(conv_means), 4),
                "std": round(pstdev(conv_means), 4),
                "n_conversations": len(conv_means),
            },
            "query_count": len(a["mrr"]),
        }
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    ensure_data()

    runs = {"hashing": evaluate("hashing")}
    try:
        import sentence_transformers  # noqa: F401
        runs["sentence-transformers"] = evaluate("sentence-transformers")
    except ImportError:
        print("sentence-transformers not installed — semantic run skipped")

    for emb, res in runs.items():
        print(f"\n=== embedder: {emb} ===   (excluded: {res['excluded']})")
        for s, r in res["strategies"].items():
            print(f"### {s}  (n={r['query_count']})")
            print(f"  evidence-recall@5 = {r['evidence_recall@5']:.3f}   "
                  f"@10 = {r['evidence_recall@10']:.3f}   MRR = {r['mrr']:.3f}")
            cats = r["recall@5_by_category"]
            print("  by category: " + "  ".join(
                f"{c}={v['recall']:.2f}(n={v['n']})" for c, v in cats.items()))
            bc = r["recall@5_by_conversation"]
            print(f"  across conversations: {bc['mean']:.3f} ± {bc['std']:.3f}")

    # ---- figure: recall@5 per strategy, one panel per embedder ---------------
    style.apply()
    n_panels = len(runs)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.8 * n_panels, 4.6), sharey=True)
    axes = [axes] if n_panels == 1 else list(axes)
    for ax, (emb, res) in zip(axes, runs.items()):
        names = list(res["strategies"].keys())
        vals = [res["strategies"][s]["evidence_recall@5"] for s in names]
        mrrs = [res["strategies"][s]["mrr"] for s in names]
        x = np.arange(len(names))
        b1 = ax.bar(x - 0.17, vals, 0.32, color=style.BLUE, label="evidence-recall@5")
        b2 = ax.bar(x + 0.17, mrrs, 0.32, color=style.AQUA, label="MRR")
        for bars in (b1, b2):
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                        f"{b.get_height():.2f}", ha="center", fontsize=8, color=style.INK_2)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(" (Persode)", "\n(Persode)") for n in names],
                           fontsize=8.5, color=style.INK)
        ax.set_ylim(0, 1.0)
        n_q = res["strategies"][names[0]]["query_count"]
        ax.set_title(f"{emb} embedder ({n_q} QA)", fontsize=10.5)
        style.style_axes(ax)
    axes[0].set_ylabel("score (higher is better)")
    axes[0].legend(loc="upper left", fontsize=8.5)
    fig.suptitle("Exp. 5 — LoCoMo evidence retrieval (10 conversations, 5.9k turns)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_png = RESULTS / "exp5_locomo.png"
    fig.savefig(out_png, bbox_inches="tight")
    print(f"\nsaved {out_png}")

    payload = {
        "benchmark": {
            "name": "LoCoMo (Maharana et al., ACL 2024)",
            "url": "https://github.com/snap-research/locomo",
            "license": "CC BY-NC 4.0 — data downloaded on demand, not redistributed",
            "data_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        },
        "protocol": {
            "note": "pre-registered: memory construction, strategies, metrics and "
                    "QA inclusion rules fixed before any result was computed; "
                    "shipped scorer/fusion defaults; adversarial category excluded "
                    "a priori (unanswerable by design)",
            "alpha": ALPHA,
            "top_ks": list(TOP_KS),
            "primary_k": PRIMARY_K,
            "stored_C": 0.5,
            "emotion_source": "offline lexicon analyzer (system ingestion path)",
        },
        "runs": runs,
    }
    (RESULTS / "exp5_locomo.json").write_text(json.dumps(payload, indent=2))
    print(f"saved {RESULTS / 'exp5_locomo.json'}")


if __name__ == "__main__":
    main()
