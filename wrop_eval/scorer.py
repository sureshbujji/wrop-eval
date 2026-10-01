"""Automatic OP/OS violation detector.

The WROP paper judges generated continuations with human raters against
hand-authored ground truth. This harness replaces the rater panel with a
deterministic checker over the toy world, looking for the exact failure
modes the paper names:

  Object permanence (OP):
    - vanished      : an object present in the ground truth is missing
                      from the prediction ("objects vanish behind
                      occluders").
    - misplaced     : predicted position differs from ground truth beyond
                      tolerance (the object is somewhere it should not be).
    - teleport      : misplaced *and* the displacement exceeds what the
                      scene's max speed allows ("re-emerge at impossible
                      positions").

  Object solidity (OS):
    - interpenetration: two solid bodies overlap with positive area in a
                      predicted frame ("objects pass through solid
                      barriers undeflected"). Parent/child containment
                      pairs are exempt - a carried ball legitimately sits
                      inside its box.

A question passes only if every predicted target frame is violation-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .physics import EPS, Body

TOL = 2.0  # cells; predicted center must be within this of ground truth
TELEPORT_MARGIN = 1.5  # displacement beyond max_speed * frames * margin => teleport

STATIC_KINDS = {"wall", "platform", "occluder"}


@dataclass
class Violation:
    qid: str
    family: str
    model: str
    frame: int  # target-frame index
    kind: str   # vanished | misplaced | teleport | interpenetration
    detail: str


def _center(s: dict) -> tuple[float, float]:
    return (s["x"] + s["w"] / 2.0, s["y"] + s["h"] / 2.0)


def _as_body(oid: str, s: dict) -> Body:
    return Body(id=oid, kind=s["kind"], x=s["x"], y=s["y"],
                w=s["w"], h=s["h"], solid=s["solid"])


def score_question(question, predicted: list[dict], model_name: str,
                   tol: float = TOL) -> tuple[bool, list[Violation]]:
    """Score one predicted continuation. Returns (passed, violations)."""
    violations: list[Violation] = []
    max_speed = question.scene.max_speed
    parent_of = {b.id: b.parent for b in question.scene.bodies}

    def vkind(kind, frame, detail):
        return Violation(question.qid, question.family, model_name,
                         frame, kind, detail)

    for i, (gt, pr) in enumerate(zip(question.target, predicted)):
        # --- object permanence: every GT dynamic object must be present
        # and where physics says it should be ---
        for oid, g in gt.items():
            if g["kind"] in STATIC_KINDS:
                continue
            p = pr.get(oid)
            if p is None:
                violations.append(vkind(
                    "vanished", i,
                    f"{oid} present in ground truth, missing from prediction"))
                continue
            gx, gy = _center(g)
            px, py = _center(p)
            d = math.hypot(px - gx, py - gy)
            if d > tol:
                limit = max_speed * (i + 1) * TELEPORT_MARGIN
                kind = "teleport" if d > limit else "misplaced"
                violations.append(vkind(
                    kind, i,
                    f"{oid} displaced {d:.2f} cells from ground truth "
                    f"(tol {tol}, physical limit {limit:.1f})"))
        # --- object solidity: no two solids may interpenetrate ---
        solids = [(oid, _as_body(oid, s)) for oid, s in pr.items()
                  if s["solid"]]
        for a in range(len(solids)):
            for b in range(a + 1, len(solids)):
                oida, ba = solids[a]
                oidb, bb = solids[b]
                if parent_of.get(oida) == oidb or parent_of.get(oidb) == oida:
                    continue  # carried load inside its carrier is fine
                ax0, ay0, ax1, ay1 = ba.rect()
                bx0, by0, bx1, by1 = bb.rect()
                ox = min(ax1, bx1) - max(ax0, bx0)
                oy = min(ay1, by1) - max(ay0, by0)
                if ox > EPS and oy > EPS:
                    violations.append(vkind(
                        "interpenetration", i,
                        f"{oida} overlaps {oidb} by {ox:.2f}x{oy:.2f} cells"))
    return (not violations, violations)
