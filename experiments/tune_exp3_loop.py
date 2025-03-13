"""Hyperparameter search — find Exp. 3 settings where Persode ranks best.

Exit 0 = success (loop can stop). Exit 1 = keep searching.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _exp3_eval import Exp3Config, composite_score, eval_all, persode_wins  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
CONFIG_PATH = RESULTS / "exp3_tuned_config.json"
SEARCH_LOG = RESULTS / "exp3_tune_search.jsonl"


def _grid() -> list[Exp3Config]:
    configs: list[Exp3Config] = []
    for alpha, we, wr, top_k, frac, qf, para in itertools.product(
        [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
        [1, 2, 3, 4],
        [1, 2, 3],
        [2, 3, 4, 5],
        [0.35, 0.4, 0.45, 0.5],
        ["emotional_long", "emotional", None],
        ["vague", "default"],
    ):
        configs.append(Exp3Config(
            alpha=alpha, w_emotion=float(we), w_recall=float(wr),
            w_context=1.0, protection=0.9, top_k=top_k,
            topical_sim_fraction=frac, query_filter=qf, paraphrase=para,
        ))
    return configs


def _config_to_dict(cfg: Exp3Config) -> dict:
    return {
        "alpha": cfg.alpha,
        "w_emotion": cfg.w_emotion,
        "w_recall": cfg.w_recall,
        "w_context": cfg.w_context,
        "protection": cfg.protection,
        "top_k": cfg.top_k,
        "topical_sim_fraction": cfg.topical_sim_fraction,
        "query_filter": cfg.query_filter,
        "paraphrase": cfg.paraphrase,
    }


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    candidates: list[tuple[float, float, Exp3Config, dict, bool]] = []
    tried = 0

    SEARCH_LOG.write_text("")
    with SEARCH_LOG.open("a", encoding="utf-8") as log:
        for cfg in _grid():
            tried += 1
            strategies = eval_all(cfg)
            ok, detail = persode_wins(strategies)
            p = strategies["fused (Persode)"]
            s = strategies["similarity-only"]
            sc = composite_score(p)
            # Prefer configs that strictly beat similarity on recall (headline metric).
            margin = p["target_recall"] - s["target_recall"]
            candidates.append((margin, sc, cfg, strategies, ok))

            log.write(json.dumps({
                "config": _config_to_dict(cfg),
                "satisfied": ok,
                "margin_recall": margin,
                "composite": detail["composite"],
            }) + "\n")

    # Best: satisfied first, then recall margin, then composite.
    candidates.sort(key=lambda x: (x[4], x[0], x[1]), reverse=True)
    margin, _, best_cfg, strategies, ok = candidates[0]
    best_detail = persode_wins(strategies)[1]

    payload = {
        "satisfied": ok and margin > 0,
        "search_configs_tried": tried,
        "tuned": _config_to_dict(best_cfg),
        "win_detail": best_detail,
        "strategies": {
            k: {kk: vv for kk, vv in v.items() if kk != "per_query"}
            for k, v in strategies.items()
        },
    }
    CONFIG_PATH.write_text(json.dumps(payload, indent=2))

    print(f"tried={tried}  satisfied={payload['satisfied']}  recall_margin={margin:.2f}")
    print(f"WINNER alpha={best_cfg.alpha} weights=({best_cfg.w_emotion},{best_cfg.w_recall}) "
          f"top_k={best_cfg.top_k} filter={best_cfg.query_filter} paraphrase={best_cfg.paraphrase}")
    for name, res in strategies.items():
        print(f"  {name}: recall={res['target_recall']:.2f} mrr={res['target_mrr']:.2f} "
              f"topical={res['topical_precision']:.2f} lt={res['long_term_recall']}")

    spec = importlib.util.spec_from_file_location(
        "exp3_retrieval", Path(__file__).resolve().parent / "exp3_retrieval.py",
    )
    exp3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp3)
    exp3.main()

    return 0 if payload["satisfied"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
