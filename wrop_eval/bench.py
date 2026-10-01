"""Exam harness: run every model persona over the exam, score, and rank."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict

from .data import FAMILIES, Question, build_exam
from .elo import elo_ratings
from .models import Model, default_models
from .scorer import Violation, score_question


def run_bench(questions: list[Question], models: list[Model],
              seed: int = 7) -> dict:
    per_model: dict[str, dict] = {}
    wins: dict[tuple[str, str], float] = defaultdict(float)
    viol_kinds: dict[str, Counter] = defaultdict(Counter)

    passed: dict[str, dict[str, bool]] = {}
    for mi, model in enumerate(models):
        rng = random.Random(seed * 7919 + mi)
        per_model[model.name] = {
            "interface": model.interface,
            "blurb": model.blurb,
            "by_family": {},
            "overall_accuracy": 0.0,
            "n_passed": 0,
        }
        passed[model.name] = {}
        fam_hits: dict[str, list[int]] = defaultdict(list)
        for q in questions:
            pred = model.predict(q, rng)
            ok, viols = score_question(q, pred, model.name)
            passed[model.name][q.qid] = ok
            fam_hits[q.family].append(1 if ok else 0)
            for v in viols:
                viol_kinds[model.name][f"{q.family}:{v.kind}"] += 1
        for fam, _desc, _gen in FAMILIES:
            hits = fam_hits.get(fam, [])
            per_model[model.name]["by_family"][fam] = (
                sum(hits) / len(hits) if hits else 0.0)
        total = sum(sum(v) for v in fam_hits.values())
        n = sum(len(v) for v in fam_hits.values())
        per_model[model.name]["overall_accuracy"] = total / n if n else 0.0
        per_model[model.name]["n_passed"] = total

    # Pairwise Bradley-Terry: pass beats fail, ties split.
    names = [m.name for m in models]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            for q in questions:
                pa, pb = passed[a][q.qid], passed[b][q.qid]
                if pa and not pb:
                    wins[(a, b)] += 1.0
                elif pb and not pa:
                    wins[(b, a)] += 1.0
                else:
                    wins[(a, b)] += 0.5
                    wins[(b, a)] += 0.5
    elos = elo_ratings(names, wins)
    for n in names:
        per_model[n]["elo"] = round(elos[n], 1)

    return {
        "meta": {
            "n_questions": len(questions),
            "per_family": len(questions) // len(FAMILIES),
            "families": [f for f, _d, _g in FAMILIES],
            "seed": seed,
        },
        "models": per_model,
        "violation_counts": {m: dict(c) for m, c in viol_kinds.items()},
    }


def save_results(results: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
