"""Seeded WROP-style exam generators.

Six task families, mirroring the structure of the WROP benchmark
(three probing object permanence, three probing object solidity).
Each question is a small 2D scene split into context frames (the model
input) and target frames (what the model must continue). The key physical
event always lands inside the target window.

Families:
  op_occlusion    - ball passes behind a screen; must reappear at the
                    physically consistent position and time.
  op_containment  - ball rides inside a moving box; the box stops and the
                    ball must stop with it (rigid parenting).
  op_tunnel       - ball transits a long tunnel; must exit the far end,
                    not vanish inside.
  os_barrier      - ball rolls toward a solid wall; must stop at the wall,
                    never interpenetrate it.
  os_collision    - two balls on a head-on course; must collide and
                    rebound, not pass through each other.
  os_support      - ball falls under gravity onto a platform; must come to
                    rest on top of it, not fall through.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .physics import Body, Scene, simulate

CONTEXT_LEN = 6
TARGET_LEN = 4


@dataclass
class Question:
    qid: str
    family: str
    family_desc: str
    scene: Scene
    context: list[dict]  # input frames (what the model sees)
    target: list[dict]   # ground-truth continuation frames
    note: str = ""


def _ball(bid, x, y, vx=0.0, vy=0.0, size=2.0):
    return Body(id=bid, kind="ball", x=x, y=y, w=size, h=size, vx=vx, vy=vy)


def _wall(bid, x, w, y=0.0, h=16.0):
    return Body(id=bid, kind="wall", x=x, y=y, w=w, h=h, static=True)


def _platform(bid, x, y, w, h=2.0):
    return Body(id=bid, kind="platform", x=x, y=y, w=w, h=h, static=True)


def _occluder(bid, x, y, w, h):
    return Body(id=bid, kind="occluder", x=x, y=y, w=w, h=h,
                static=True, solid=False)


def _scene(bodies, **kw):
    kw.setdefault("context_len", CONTEXT_LEN)
    kw.setdefault("target_len", TARGET_LEN)
    return Scene(bodies=bodies, **kw)


def gen_occlusion(rng: random.Random) -> Scene:
    vx = rng.uniform(2.2, 2.8)
    x0 = rng.uniform(1.0, 4.0)
    return _scene(
        [_ball("ball", x0, 7.0, vx=vx),
         _occluder("screen", 13.0, 0.0, 5.0, 16.0)],
        max_speed=4.0,
        desc="ball passes behind a vertical screen",
    )


def gen_containment(rng: random.Random) -> Scene:
    vx = rng.uniform(1.8, 2.4)
    x0 = rng.uniform(4.0, 7.0)

    def script(t, by_id):
        # The carrier stops once the target window begins; the ball must
        # stop with it instead of sliding on.
        if t >= CONTEXT_LEN:
            by_id["box"].vx = 0.0

    box = Body(id="box", kind="box", x=x0, y=6.0, w=6.0, h=5.0, vx=vx)
    ball = Body(id="ball", kind="ball", x=x0 + 2.0, y=7.0, w=2.0, h=2.0,
                parent="box", ox=2.0, oy=1.0)
    return _scene([box, ball], max_speed=4.0, script=script,
                  desc="ball rides in a box that stops mid-scene")


def gen_tunnel(rng: random.Random) -> Scene:
    vx = rng.uniform(2.2, 2.8)
    x0 = rng.uniform(1.0, 3.0)
    return _scene(
        [_ball("ball", x0, 7.0, vx=vx),
         _occluder("tunnel", 10.0, 6.0, 12.0, 4.0)],
        max_speed=4.0,
        desc="ball transits a long horizontal tunnel",
    )


def gen_barrier(rng: random.Random) -> Scene:
    vx = rng.uniform(2.2, 2.8)
    x0 = rng.uniform(1.0, 3.0)
    return _scene(
        [_ball("ball", x0, 7.0, vx=vx),
         _wall("wall", 20.0, 4.0)],
        max_speed=4.0,
        desc="ball rolls into a solid wall and must stop",
    )


def gen_collision(rng: random.Random) -> Scene:
    v = rng.uniform(1.1, 1.4)
    ax0 = rng.uniform(7.0, 9.0)
    bx0 = rng.uniform(23.0, 25.0)
    return _scene(
        [_ball("a", ax0, 6.0, vx=v, size=3.0),
         _ball("b", bx0, 6.0, vx=-v, size=3.0)],
        max_speed=4.0,
        desc="two balls collide head-on and rebound",
    )


def gen_support(rng: random.Random) -> Scene:
    x0 = rng.uniform(12.0, 16.0)
    return _scene(
        [_ball("ball", x0, 1.0),
         _platform("floor", 8.0, 12.0, 16.0)],
        gravity=0.35,  # tuned so the ball lands in the target window (t=7)
        max_speed=6.0,
        desc="ball falls under gravity and rests on a platform",
    )


FAMILIES: list[tuple[str, str, object]] = [
    ("op_occlusion", "Object permanence - occlusion (behind screen)", gen_occlusion),
    ("op_containment", "Object permanence - containment (carried load)", gen_containment),
    ("op_tunnel", "Object permanence - tunnel transit", gen_tunnel),
    ("os_barrier", "Object solidity - solid barrier", gen_barrier),
    ("os_collision", "Object solidity - head-on collision", gen_collision),
    ("os_support", "Object solidity - support (rest on platform)", gen_support),
]


def build_exam(per_family: int = 20, seed: int = 7) -> list[Question]:
    """Build the full exam: per_family seeded questions per family."""
    questions = []
    for fi, (fam, desc, gen) in enumerate(FAMILIES):
        for i in range(per_family):
            rng = random.Random(seed * 1000 + fi * 100 + i)
            scene = gen(rng)
            frames = simulate(scene)
            questions.append(Question(
                qid=f"{fam}-{i:02d}",
                family=fam,
                family_desc=desc,
                scene=scene,
                context=frames[:scene.context_len],
                target=frames[scene.context_len:],
                note=scene.desc,
            ))
    return questions
