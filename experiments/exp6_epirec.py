"""Experiment 6 — EpiRec: emotional resurfacing vs factual lookup, at scale.

EpiRec (github.com/sukoji/epirec) is a synthetic episodic-recall benchmark:
12 personas × 14 episodes with authored intensity-band/valence labels and
session timestamps, probed three ways per episode — factual, reflective with
explicit emotion words, and reflective with *no* emotion words and no content
words reused (the stratum that defeats lexical gates and plain similarity).
504 probes, single-target. Construction was pre-registered and mechanically
validated (see the EpiRec repo's GENERATION_SPEC.md); the corpus was frozen
before any Persode strategy ran on it, and the authored labels were never
inputs to the retrieval code — E is produced by the system's own analyzer.

Protocol (fixed a priori, same conventions as Exp. 3/5): per persona all
episodes form one store; strategies recency-only / similarity-only /
salience-only / fused (α = 0.5) / gated (the agent) rank via
``MemoryStore.retrieve`` at shipped defaults; stored C = 0.5; ``now`` = the
corpus reference date. Metrics: recall@3 (primary), recall@1, MRR — overall,
per probe type, and reflective probes stratified by intensity band. Hashing
embedder always; sentence-transformers when installed.

Outputs:
    results/exp6_epirec.png
    results/exp6_epirec.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXPERIMENTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENTS_DIR))
sys.path.insert(0, str(EXPERIMENTS_DIR.parent))
import _style as style  # noqa: E402
from persode.analyzer import SIGNIFICANCE_THRESHOLD, EventEmotionAnalyzer  # noqa: E402
from persode.memory import Memory, MemoryStrengthScorer  # noqa: E402
from persode.store import MemoryStore  # noqa: E402
from exp5_locomo import ALPHA, CachedEmbedder  # noqa: E402
from persode.embeddings import get_embedder  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# Corpus location: $EPIREC_PATH, else the sibling checkout.
DATA_PATH = Path(os.environ.get(
    "EPIREC_PATH", ROOT.parent / "epirec" / "data" / "epirec_v1.json"))

TYPES = ("factual", "reflective_explicit", "reflective_implicit")
STRATEGIES = ("recency-only", "similarity-only", "salience-only",
              "fused (Persode)", "gated (Persode)")

_GATE_ANALYZER = EventEmotionAnalyzer()
_GATE_CACHE: dict[str, bool] = {}


def query_is_significant(query: str) -> bool:
    v = _GATE_CACHE.get(query)
    if v is None:
        v = _GATE_ANALYZER.analyze(query).emotional_intensity >= SIGNIFICANCE_THRESHOLD
        _GATE_CACHE[query] = v
    return v


def build_stores(persona: dict, analyzer: EventEmotionAnalyzer, embedder, now):
    memories = []
    for e in persona["episodes"]:
        meta = analyzer.analyze(e["text"])
        memories.append(Memory(
            text=e["text"], event=e["id"], emotion=meta.emotion,
            emotional_intensity=meta.emotional_intensity,
            contextual_relevance=0.5,
            created_at=datetime.fromisoformat(e["date_time"]).replace(tzinfo=timezone.utc),
        ))
    scorer = MemoryStrengthScorer()
    stores = {"now": now, "recency_ranking": [
        m.event for m in sorted(memories, key=lambda m: m.created_at, reverse=True)]}
    for name, alpha in (("similarity-only", 1.0), ("salience-only", 0.0),
                        ("fused (Persode)", ALPHA)):
        st = MemoryStore(embedder=embedder, scorer=scorer, w_similarity=alpha)
        st.add_many(memories)
        stores[name] = st
    return stores


def rank_ids(name: str, stores: dict, query: str) -> list[str]:
    if name == "recency-only":
        return stores["recency_ranking"]
    kwargs = {}
    if name == "gated (Persode)":
        st = stores["fused (Persode)"]
        kwargs["w_similarity"] = ALPHA if query_is_significant(query) else 1.0
    else:
        st = stores[name]
        if name == "salience-only":
            kwargs["use_query_relevance_as_context"] = False
    hits = st.retrieve(query, top_k=len(st), now=stores["now"], reinforce=False, **kwargs)
    return [r.memory.event for r in hits]


def evaluate(embedder_name: str) -> tuple[dict, dict[str, list[dict]]]:
    corpus = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    now = datetime.fromisoformat(corpus["reference_now"])
    analyzer = EventEmotionAnalyzer()
    embedder = CachedEmbedder(get_embedder(embedder_name))

    agg = {s: {t: {"r3": [], "r1": [], "mrr": []} for t in TYPES} for s in STRATEGIES}
    strat_band = {s: {} for s in STRATEGIES}  # reflective probes by intensity band
    rankings = {s: [] for s in STRATEGIES}
    gate_fired = []

    for persona in corpus["personas"]:
        stores = build_stores(persona, analyzer, embedder, now)
        for e in persona["episodes"]:
            for pr in e["probes"]:
                if pr["type"] != "factual":
                    gate_fired.append(query_is_significant(pr["query"]))
                for s in STRATEGIES:
                    ranked = rank_ids(s, stores, pr["query"])
                    rankings[s].append({
                        "persona_id": persona["persona_id"],
                        "probe_id": pr["id"],
                        "ranked_episode_ids": ranked,
                    })
                    rank = ranked.index(e["id"]) + 1
                    m = agg[s][pr["type"]]
                    m["r3"].append(float(rank <= 3))
                    m["r1"].append(float(rank == 1))
                    m["mrr"].append(1.0 / rank)
                    if pr["type"] != "factual":
                        key = f"{pr['type']}·{e['intensity_band']}"
                        strat_band[s].setdefault(key, []).append(float(rank <= 3))

    out = {"strategies": {}, "gate": {
        "rule": f"fusion iff analyzer E >= {SIGNIFICANCE_THRESHOLD} on the query",
        "reflective_fused_fraction": round(float(np.mean(gate_fired)), 4),
    }}
    for s in STRATEGIES:
        res = {}
        for t in TYPES:
            res[t] = {k: round(float(np.mean(v)), 3) for k, v in agg[s][t].items()}
        res["overall"] = {k: round(float(np.mean(
            [x for t in TYPES for x in agg[s][t][k]])), 3) for k in ("r3", "r1", "mrr")}
        res["reflective_recall3_by_band"] = {
            k: {"recall": round(float(np.mean(v)), 3), "n": len(v)}
            for k, v in sorted(strat_band[s].items())}
        out["strategies"][s] = res
    return out, rankings


def main(embedder_choice: str = "all") -> None:
    RESULTS.mkdir(exist_ok=True)
    if not DATA_PATH.exists():
        raise SystemExit(f"EpiRec corpus not found at {DATA_PATH} — clone "
                         "github.com/sukoji/epirec next to this repo or set EPIREC_PATH")

    runs = {}
    exported_rankings = {}
    result, rankings = evaluate("hashing")
    runs["hashing"] = result
    exported_rankings["hashing"] = rankings
    try:
        if embedder_choice == "hashing":
            raise ImportError
        import sentence_transformers  # noqa: F401
        result, rankings = evaluate("sentence-transformers")
        runs["sentence-transformers"] = result
        exported_rankings["sentence-transformers"] = rankings
    except ImportError:
        print("semantic run skipped")

    for emb, res in runs.items():
        print(f"\n=== embedder: {emb} ===  (gate fires on "
              f"{res['gate']['reflective_fused_fraction']:.0%} of reflective probes)")
        for s, r in res["strategies"].items():
            print(f"### {s}")
            for t in (*TYPES, "overall"):
                print(f"  {t:21s} recall@3={r[t]['r3']:.3f}  recall@1={r[t]['r1']:.3f}  "
                      f"MRR={r[t]['mrr']:.3f}")

    # figure: recall@3 per probe type, strategies grouped; one panel per embedder
    style.apply()
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(7.4 * n, 4.8), sharey=True)
    axes = [axes] if n == 1 else list(axes)
    colors = [style.MUTED, style.BLUE, style.VIOLET, style.AQUA, style.YELLOW]
    for ax, (emb, res) in zip(axes, runs.items()):
        x = np.arange(len(TYPES))
        w = 0.16
        for i, s in enumerate(STRATEGIES):
            vals = [res["strategies"][s][t]["r3"] for t in TYPES]
            ax.bar(x + (i - 2) * w, vals, w, label=s, color=colors[i])
        ax.set_xticks(x)
        ax.set_xticklabels(["factual", "reflective\nexplicit", "reflective\nimplicit"],
                           fontsize=9, color=style.INK)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{emb} embedder", fontsize=10.5)
        style.style_axes(ax)
    axes[0].set_ylabel("target recall@3")
    axes[0].legend(loc="upper right", fontsize=7.5)
    fig.suptitle("Exp. 6 — EpiRec (504 probes): retrieval by probe type", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_png = RESULTS / "exp6_epirec.png"
    fig.savefig(out_png, bbox_inches="tight")
    print(f"\nsaved {out_png}")

    payload = {
        "benchmark": {"name": "EpiRec v1.0", "url": "https://github.com/sukoji/epirec",
                      "probes": 504, "personas": 12},
        "protocol": {
            "note": "pre-registered (EpiRec GENERATION_SPEC.md); corpus frozen before "
                    "any strategy ran; authored labels never fed to retrieval — E comes "
                    "from the system's own analyzer",
            "alpha": ALPHA, "top_k_primary": 3, "stored_C": 0.5,
        },
        "runs": runs,
    }
    (RESULTS / "exp6_epirec.json").write_text(json.dumps(payload, indent=2))
    print(f"saved {RESULTS / 'exp6_epirec.json'}")

    # The frozen EpiRec evaluator is the source of record for published CIs.
    evaluator = DATA_PATH.parent.parent / "scripts" / "evaluate_rankings.py"
    ranking_dir = RESULTS / "exp6_epirec_rankings"
    ranking_dir.mkdir(exist_ok=True)
    if not evaluator.exists():
        print("official EpiRec evaluator not found; raw rankings were still exported")
    for embedder_name, by_strategy in exported_rankings.items():
        for strategy, records in by_strategy.items():
            slug = strategy.lower().replace(" ", "_").replace("(", "").replace(")", "")
            rankings_path = ranking_dir / f"{embedder_name}_{slug}.jsonl"
            rankings_path.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
            if evaluator.exists():
                official_path = ranking_dir / f"{embedder_name}_{slug}.official.json"
                subprocess.run([sys.executable, str(evaluator), "--rankings", str(rankings_path),
                                "--output", str(official_path)], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedder", choices=("all", "hashing", "sentence-transformers"), default="all")
    main(parser.parse_args().embedder)
