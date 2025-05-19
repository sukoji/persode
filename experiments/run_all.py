"""Run all four paper experiments and regenerate results/ artifacts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPS = [
    "exp1_forgetting_curve.py",
    "exp2_memory_scoring.py",
    "exp3_retrieval.py",
    "exp4_visual_prompt.py",
]


def _run(script: str) -> None:
    path = Path(__file__).resolve().parent / script
    print(f"\n{'=' * 60}\n>>> {script}\n{'=' * 60}")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    mod.main()


def main() -> None:
    for script in EXPS:
        _run(script)
    print("\nAll experiments finished. Outputs in results/")


if __name__ == "__main__":
    main()
