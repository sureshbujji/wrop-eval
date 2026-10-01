"""Bradley-Terry pairwise ratings on the Elo scale.

Mirrors the WROP paper's evaluation: models are compared pairwise on each
exam question (pass beats fail, ties split), and strengths are estimated
with the standard MM algorithm, reported as Elo ratings anchored at 1500.
"""

from __future__ import annotations

import math


def elo_ratings(names: list[str], wins: dict[tuple[str, str], float],
                iters: int = 500, base: float = 1500.0) -> dict[str, float]:
    """wins[(a, b)] = total points a scored against b (ties count 0.5).

    Returns {name: elo}.
    """
    gamma = {n: 1.0 for n in names}  # strengths, updated by MM
    for _ in range(iters):
        new_gamma = {}
        for a in names:
            # points a earned against each opponent
            scored = sum(wins.get((a, b), 0.0) for b in names if b != a)
            denom = 0.0
            for b in names:
                if b == a:
                    continue
                games = wins.get((a, b), 0.0) + wins.get((b, a), 0.0)
                if games > 0:
                    denom += games / (gamma[a] + gamma[b])
            new_gamma[a] = scored / denom if denom > 0 else gamma[a]
        # normalize to keep numbers stable
        geo = math.exp(sum(math.log(max(g, 1e-300))
                           for g in new_gamma.values()) / len(new_gamma))
        gamma = {n: g / geo for n, g in new_gamma.items()}
    return {n: base + 400.0 * math.log10(max(g, 1e-300))
            for n, g in gamma.items()}
