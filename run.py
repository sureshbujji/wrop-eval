#!/usr/bin/env python3
"""Run the WROP-style object-permanence exam.

Example:
    python run.py                        # full exam: 6 families x 20 questions
    python run.py --per-family 5         # quick smoke run (30 questions)
    python run.py --seed 11              # a different seeded exam
"""

from __future__ import annotations

import argparse
import os

from wrop_eval import build_exam, default_models, run_bench, save_results
from wrop_eval.data import FAMILIES

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main() -> None:
    parser = argparse.ArgumentParser(description="WROP-style exam harness.")
    parser.add_argument("--per-family", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "results.json"))
    args = parser.parse_args()

    questions = build_exam(per_family=args.per_family, seed=args.seed)
    models = default_models()
    results = run_bench(questions, models, seed=args.seed)

    print(f"exam: {len(questions)} questions "
          f"({args.per_family}/family x {len(FAMILIES)} families), seed={args.seed}")
    print()
    print(f"{'model':16s} {'interface':12s} {'acc':>6s} {'elo':>7s}")
    print("-" * 48)
    ranked = sorted(models, key=lambda m: results["models"][m.name]["elo"],
                    reverse=True)
    for m in ranked:
        r = results["models"][m.name]
        print(f"{m.name:16s} {m.interface:12s} {r['overall_accuracy']:6.3f} "
              f"{r['elo']:7.1f}")
    print()
    print("accuracy by family:")
    header = f"{'model':16s}" + "".join(f"{fam:>14s}" for fam, _, _ in FAMILIES)
    print(header)
    print("-" * len(header))
    for m in ranked:
        row = f"{m.name:16s}"
        for fam, _, _ in FAMILIES:
            row += f"{results['models'][m.name]['by_family'][fam]:14.3f}"
        print(row)
    print()
    print("dominant violation kinds per model:")
    for m in ranked:
        kinds = results["violation_counts"].get(m.name, {})
        top = sorted(kinds.items(), key=lambda kv: -kv[1])[:3]
        summary = ", ".join(f"{k.split(':')[1]} x{n}" for k, n in top)
        print(f"  {m.name:14s} {summary or '(none)'}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_results(results, args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
