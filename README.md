# wrop-eval — A WROP-Style Object-Permanence Exam for Video Models

A fully self-contained re-implementation of the *exam mechanics* behind
**WROP (World Reasoning with Object Permanence)** — Zhang et al., "Training
Object Permanence in World Models" (arXiv:2609.28654, Sep 2026;
code at [hokindeng/object-permanence](https://github.com/hokindeng/object-permanence)).

The paper's headline problem: video generation models increasingly act as
world models, yet they fail at foundational physical reasoning — **objects
vanish behind occluders and re-emerge at impossible positions** (object
permanence, OP), and **objects pass through solid barriers undeflected**
(object solidity, OS). WROP tests this with 150 Blender generators across
six task families and a fixed 300-question exam, judged by human raters.

This repo rebuilds that exam as a seeded, stdlib-only toy world: six task
families (three OP, three OS), each question split into context frames (the
model input) and target frames (the continuation to predict), with the key
physical event always landing in the target window. An automatic
violation detector stands in for the human rater panel, and models are
ranked with Bradley–Terry pairwise ratings on the Elo scale, mirroring the
paper's evaluation.

## What it does

- `wrop_eval/physics.py` — minimal 2D toy physics: constant velocities,
  gravity, rigid parenting (carried loads), solid walls/platforms, visual-only
  occluders, elastic ball–ball collisions. Exact and seeded-deterministic.
- `wrop_eval/data.py` — six seeded exam generators, 20 questions per family
  by default (120 total), key event always in the target window.
- `wrop_eval/models.py` — six *behavioral* model personas across the paper's
  three interface classes (continuation / reference-to-video / edit); see
  table below.
- `wrop_eval/scorer.py` — automatic OP/OS violation detector: `vanished`,
  `misplaced`, `teleport` (permanence) and `interpenetration` (solidity).
- `wrop_eval/elo.py` — Bradley–Terry pairwise ratings on the Elo scale.
- `wrop_eval/bench.py` — harness: run every persona over the exam, score,
  aggregate per-family accuracy, compute Elo.
- `run.py` — CLI; prints the ranking tables, writes `results/results.json`.
- `tests/test_wrop.py` — 19 tests asserting real behavior (see below).

## The exam: six task families

| family | principle | scenario |
|---|---|---|
| `op_occlusion` | permanence | ball passes behind a screen; must reappear at the consistent position/time |
| `op_containment` | permanence | ball rides inside a moving box; the box stops and the ball must stop with it |
| `op_tunnel` | permanence | ball transits a long tunnel; must exit the far end, not vanish inside |
| `os_barrier` | solidity | ball rolls toward a solid wall; must stop at it, never interpenetrate |
| `os_collision` | solidity | two balls on a head-on course; must collide and rebound, not pass through |
| `os_support` | solidity | ball falls under gravity onto a platform; must rest on top, not fall through |

## Model personas (behavioral stand-ins, not API calls)

| persona | interface class | behavior |
|---|---|---|
| `oracle` | continuation | exact ground-truth physics — the ceiling |
| `linear-motion` | continuation | naive constant-velocity extrapolation; blind to walls, gravity, collisions, parenting |
| `mean-frame` | continuation | predicts the temporal mean of context — motion smears into a frozen ghost |
| `forget-occluded` | reference-to-video | perfect physics, but drops any object occluded in the last context frame |
| `copy-last` | edit | repeats the last context frame — a minimal edit that continues nothing |
| `noisy-diffuse` | edit | near-perfect physics with ±1-cell jitter and 12% per-object dropout |

## Results (seed 7, 120 questions)

| model | interface | accuracy | Elo |
|---|---|---|---|
| oracle | continuation | 1.000 | 1758.7 |
| forget-occluded | reference | 0.667 | 1594.6 |
| noisy-diffuse | edit | 0.350 | 1465.4 |
| linear-motion | continuation | 0.333 | 1458.7 |
| copy-last | edit | 0.192 | 1401.2 |
| mean-frame | continuation | 0.008 | 1321.5 |

Accuracy by family:

| model | op_occl. | op_contain. | op_tunnel | os_barrier | os_collis. | os_support |
|---|---|---|---|---|---|---|
| oracle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| forget-occluded | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| noisy-diffuse | 0.700 | 0.250 | 0.700 | 0.000 | 0.350 | 0.100 |
| linear-motion | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| copy-last | 0.000 | 1.000 | 0.000 | 0.100 | 0.050 | 0.000 |
| mean-frame | 0.000 | 0.000 | 0.000 | 0.050 | 0.000 | 0.000 |

The headline: **the exam separates the failure modes cleanly**.
`forget-occluded` is perfect except the two occlusion families, where it
scores exactly zero — the paper's "objects vanish behind occluders" failure,
isolated. `linear-motion` aces both occlusion families (constant velocity
handles reappearance) but fails every solidity family — the paper's
"objects pass through solid barriers undeflected" failure, isolated. The
dumbest baselines fail almost everywhere, *except* `copy-last` accidentally
passes containment: the carrier stops in the target window, so freezing is
momentarily correct — a nice reminder that a benchmark needs families a
lazy model can't luck into.

## Run it

```bash
python run.py                        # full exam: 6 families x 20 questions (~1s)
python run.py --per-family 5         # quick smoke run (30 questions)
python run.py --seed 11              # a different seeded exam
pytest tests/ -q                     # 19 tests, all passing
```

No API keys, no network, no video models — scenes, personas, and the
violation detector are a seeded stdlib-only simulation, so results are
bit-for-bit reproducible.

## Scope and honesty

This is a *mechanistic* re-implementation of the WROP exam structure, not a
re-run of the WROP paper. The "models" are behavioral simulators with
designed failure modes, not calls to Sora, Wan, MiniMax, or any real video
model — the harness measures whether the *exam* can separate these
behaviors, which is the prerequisite for pointing it at real models later.
The human rater panel is replaced by a deterministic violation detector over
a toy 2D world (rigid parenting, no rotation, no deformation), so the
absolute scores say nothing about any real system. Natural extensions: plug
in real video-model APIs behind the `Model.predict` interface, add the
paper's remaining generator diversity, and swap the detector for VLM or
human judgment on rendered clips.

## License

MIT — see [LICENSE](LICENSE).
