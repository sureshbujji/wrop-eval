"""Toy 2D physics for the WROP-style object-permanence exam.

A deliberately minimal side-view world: axis-aligned rectangles on a
continuous plane, constant velocities, optional gravity, rigid parenting
(objects carried inside containers), solid walls/platforms, and visual-only
occluders that hide objects without blocking them.

The simulator is exact and seeded-deterministic; the ground-truth target
frames it produces are the physically consistent "hand-authored" answers the
WROP paper judges model outputs against.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Callable

WORLD_W, WORLD_H = 32.0, 16.0
EPS = 1e-9


@dataclass
class Body:
    id: str
    kind: str  # 'ball' | 'box' | 'wall' | 'platform' | 'occluder'
    x: float
    y: float
    w: float
    h: float
    vx: float = 0.0
    vy: float = 0.0
    solid: bool = True    # participates in collision / solidity checks
    static: bool = False  # walls, platforms, occluders never move
    parent: str | None = None  # id of carrier; child follows it rigidly
    ox: float = 0.0  # offset from parent origin
    oy: float = 0.0

    def rect(self):
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def center(self):
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    def clone(self) -> "Body":
        return Body(**{f.name: getattr(self, f.name) for f in fields(self)})


def overlaps(a: Body, b: Body) -> bool:
    """Strict area overlap (edge contact is not a violation)."""
    ax0, ay0, ax1, ay1 = a.rect()
    bx0, by0, bx1, by1 = b.rect()
    return ((min(ax1, bx1) - max(ax0, bx0)) > EPS and
            (min(ay1, by1) - max(ay0, by0)) > EPS)


@dataclass
class Scene:
    bodies: list[Body]
    gravity: float = 0.0
    context_len: int = 6
    target_len: int = 4
    max_speed: float = 4.0  # fastest any body may legitimately move (cells/frame)
    # script(t, bodies_by_id): optional per-step velocity program, t is 1-based
    script: Callable[[int, dict[str, Body]], None] | None = None
    desc: str = ""


def snapshot(bodies: list[Body]) -> dict:
    """A frame: id -> plain JSON-serializable dict."""
    return {
        b.id: {"x": b.x, "y": b.y, "w": b.w, "h": b.h,
               "kind": b.kind, "solid": b.solid}
        for b in bodies
    }


def _first_static_overlap(b: Body, statics: list[Body]) -> Body | None:
    for s in statics:
        if s.id != b.id and overlaps(b, s):
            return s
    return None


def _step(by_id: dict[str, Body], gravity: float) -> None:
    statics = [b for b in by_id.values() if b.static and b.solid]
    movers = [b for b in by_id.values()
              if not b.static and b.parent is None and b.solid]
    children = [b for b in by_id.values()
                if not b.static and b.parent is not None]

    for b in movers:
        b.vy += gravity
        old_x, old_y = b.x, b.y
        # X axis: move, then clamp against solids.
        b.x = old_x + b.vx
        hit = _first_static_overlap(b, statics)
        if hit:
            if b.vx > 0:
                b.x = hit.x - b.w
            elif b.vx < 0:
                b.x = hit.x + hit.w
            else:
                b.x = old_x
            b.vx = 0.0
        # Y axis: move, then clamp (lands on top / rests underneath).
        b.y = old_y + b.vy
        hit = _first_static_overlap(b, statics)
        if hit:
            if b.vy > 0:
                b.y = hit.y - b.h
            elif b.vy < 0:
                b.y = hit.y + hit.h
            else:
                b.y = old_y
            b.vy = 0.0

    # Rigid parenting: children track their carrier exactly.
    for c in children:
        p = by_id[c.parent]
        c.x, c.y = p.x + c.ox, p.y + c.oy
        c.vx, c.vy = p.vx, p.vy

    # Elastic ball-ball collisions (parent/child pairs excluded).
    dyn = [b for b in by_id.values() if not b.static and b.solid]
    for i in range(len(dyn)):
        for j in range(i + 1, len(dyn)):
            a, d = dyn[i], dyn[j]
            if a.parent == d.id or d.parent == a.id:
                continue
            if overlaps(a, d):
                left, right = (a, d) if a.x <= d.x else (d, a)
                ov = (min(left.x + left.w, right.x + right.w)
                      - max(left.x, right.x))
                push = ov / 2.0 + EPS
                left.x -= push
                right.x += push
                left.vx, right.vx = right.vx, left.vx


def simulate(scene: Scene) -> list[dict]:
    """Run the scene; frames[0] is the initial state.

    Returns context_len + target_len frames.
    """
    bodies = [b.clone() for b in scene.bodies]
    by_id = {b.id: b for b in bodies}
    frames = [snapshot(bodies)]
    total = scene.context_len + scene.target_len
    for t in range(1, total):
        if scene.script is not None:
            scene.script(t, by_id)
        _step(by_id, scene.gravity)
        frames.append(snapshot(bodies))
    return frames
