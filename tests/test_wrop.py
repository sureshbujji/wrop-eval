"""Tests for wrop-eval: physics, generators, scorer, models, Elo, bench."""
import json
import math
import random

import pytest

from wrop_eval import (
    FAMILIES, build_exam, default_models, elo_ratings, run_bench,
    save_results, score_question,
)
from wrop_eval.data import Question, gen_barrier, gen_collision, gen_support
from wrop_eval.models import ForgetOccluded, LinearMotion, Oracle
from wrop_eval.physics import Body, Scene, overlaps, simulate


# ---------- physics ----------

def test_ball_stops_at_wall_without_penetration():
    scene = Scene(
        bodies=[Body("b", "ball", 2.0, 7.0, 2.0, 2.0, vx=3.0),
                Body("w", "wall", 20.0, 0.0, 4.0, 16.0, static=True)],
        context_len=6, target_len=4)
    frames = simulate(scene)
    ball = [f["b"] for f in frames]
    wall = frames[0]["w"]
    for s in ball:
        assert s["x"] + s["w"] <= wall["x"] + 1e-9, "ball penetrated the wall"
    assert ball[-1]["x"] + ball[-1]["w"] == pytest.approx(wall["x"])
    assert all(s["x"] == ball[-1]["x"] for s in ball[-3:]), "ball should rest"


def test_head_on_collision_rebounds_without_overlap():
    scene = Scene(
        bodies=[Body("a", "ball", 8.0, 6.0, 3.0, 3.0, vx=1.25),
                Body("b", "ball", 24.0, 6.0, 3.0, 3.0, vx=-1.25)],
        context_len=6, target_len=4)
    frames = simulate(scene)
    for f in frames:
        a, b = f["a"], f["b"]
        ox = min(a["x"] + 3, b["x"] + 3) - max(a["x"], b["x"])
        assert ox <= 1e-9, "balls interpenetrated"
    # after the bounce the balls separate again
    gap0 = frames[5]["b"]["x"] - (frames[5]["a"]["x"] + 3)
    gap9 = frames[9]["b"]["x"] - (frames[9]["a"]["x"] + 3)
    assert gap9 > gap0, "balls should be moving apart after the collision"


def test_ball_rests_on_platform_under_gravity():
    scene = Scene(
        bodies=[Body("b", "ball", 14.0, 1.0, 2.0, 2.0),
                Body("p", "platform", 8.0, 12.0, 16.0, 2.0, static=True)],
        gravity=0.8, context_len=6, target_len=4)
    frames = simulate(scene)
    for f in frames:
        b, p = f["b"], f["p"]
        assert b["y"] + b["h"] <= p["y"] + 1e-9, "ball fell through platform"
    assert frames[-1]["b"]["y"] + 2.0 == pytest.approx(12.0), "ball rests on top"


def test_child_follows_parent_rigidly():
    box = Body("box", "box", 6.0, 6.0, 6.0, 5.0, vx=2.0)
    ball = Body("ball", "ball", 8.0, 7.0, 2.0, 2.0, parent="box", ox=2.0, oy=1.0)
    scene = Scene(bodies=[box, ball], context_len=3, target_len=1)
    frames = simulate(scene)
    for f in frames:
        assert f["ball"]["x"] == pytest.approx(f["box"]["x"] + 2.0)
        assert f["ball"]["y"] == pytest.approx(f["box"]["y"] + 1.0)


def test_occluder_does_not_block_motion():
    scene = Scene(
        bodies=[Body("b", "ball", 2.0, 7.0, 2.0, 2.0, vx=2.5),
                Body("s", "occluder", 13.0, 0.0, 5.0, 16.0,
                     static=True, solid=False)],
        context_len=6, target_len=4)
    frames = simulate(scene)
    xs = [f["b"]["x"] for f in frames]
    assert xs[-1] == pytest.approx(2.0 + 2.5 * 9)
    assert all(b2 - b1 == pytest.approx(2.5) for b1, b2 in zip(xs, xs[1:]))


# ---------- generators ----------

def _exam(per_family=6, seed=7):
    return build_exam(per_family=per_family, seed=seed)


def test_all_families_present_with_expected_counts():
    exam = _exam(per_family=4)
    assert len(exam) == 4 * len(FAMILIES) == 24
    assert {q.family for q in exam} == {f for f, _d, _g in FAMILIES}


def test_ground_truth_has_zero_violations_everywhere():
    exam = _exam(per_family=6)
    oracle = Oracle()
    for q in exam:
        ok, viols = score_question(q, oracle.predict(q, random.Random(0)),
                                   "oracle")
        assert ok, f"GT failed on {q.qid}: {viols}"


def test_exam_is_deterministic():
    a = _exam(per_family=5, seed=7)
    b = build_exam(per_family=5, seed=7)
    for qa, qb in zip(a, b):
        assert qa.target == qb.target
    c = build_exam(per_family=5, seed=8)
    assert any(qa.target != qc.target for qa, qc in zip(a, c))


# ---------- scorer ----------

def _first(exam, family):
    return next(q for q in exam if q.family == family)


def test_scorer_detects_vanished_object():
    exam = _exam(per_family=2)
    q = _first(exam, "op_occlusion")
    pred = [{oid: dict(s) for oid, s in f.items() if oid != "ball"}
            for f in q.target]
    ok, viols = score_question(q, pred, "m")
    assert not ok
    assert any(v.kind == "vanished" for v in viols)


def test_scorer_detects_interpenetration():
    exam = _exam(per_family=2)
    q = _first(exam, "os_barrier")
    pred = []
    for f in q.target:
        f2 = {oid: dict(s) for oid, s in f.items()}
        f2["ball"]["x"] = 21.0  # shove the ball into the wall (20..24)
        pred.append(f2)
    ok, viols = score_question(q, pred, "m")
    assert not ok
    assert any(v.kind == "interpenetration" for v in viols)


def test_scorer_detects_teleport():
    exam = _exam(per_family=2)
    q = _first(exam, "op_tunnel")
    pred = []
    for f in q.target:
        f2 = {oid: dict(s) for oid, s in f.items()}
        f2["ball"]["x"] += 25.0  # impossible jump
        pred.append(f2)
    ok, viols = score_question(q, pred, "m")
    assert not ok
    assert any(v.kind == "teleport" for v in viols)


def test_contained_ball_is_not_interpenetration():
    exam = _exam(per_family=2)
    q = _first(exam, "op_containment")
    ok, viols = score_question(q, [dict(f) for f in q.target], "m")
    assert ok, f"carried ball wrongly flagged: {viols}"


# ---------- model personas ----------

def test_oracle_passes_whole_exam():
    exam = _exam(per_family=4)
    oracle = Oracle()
    for q in exam:
        ok, _ = score_question(q, oracle.predict(q, random.Random(1)), "oracle")
        assert ok


def test_linear_motion_fails_barrier_and_collision():
    exam = _exam(per_family=8)
    model = LinearMotion()
    for fam, want in (("os_barrier", "interpenetration"),
                      ("os_collision", "interpenetration")):
        fails = 0
        for q in [q for q in exam if q.family == fam]:
            ok, viols = score_question(
                q, model.predict(q, random.Random(2)), "linear-motion")
            if not ok and any(v.kind == want for v in viols):
                fails += 1
        assert fails >= 1, f"linear-motion never {want} on {fam}"


def test_forget_occluded_drops_hidden_objects():
    exam = _exam(per_family=8)
    model = ForgetOccluded()
    for fam in ("op_occlusion", "op_tunnel"):
        qs = [q for q in exam if q.family == fam]
        fails = sum(
            not score_question(q, model.predict(q, random.Random(3)),
                               "forget-occluded")[0]
            for q in qs)
        assert fails == len(qs), f"forget-occluded should fail all of {fam}"


def test_copy_last_only_passes_the_stopped_carrier():
    exam = _exam(per_family=6)
    from wrop_eval.models import CopyLast
    model = CopyLast()
    acc = {}
    for fam, _d, _g in FAMILIES:
        qs = [q for q in exam if q.family == fam]
        acc[fam] = sum(
            score_question(q, model.predict(q, random.Random(4)),
                           "copy-last")[0] for q in qs) / len(qs)
    assert acc["op_containment"] == 1.0  # freezing is accidentally correct
    assert all(v == 0.0 for f, v in acc.items() if f != "op_containment")


def test_noisy_diffuse_is_imperfect_but_mostly_right():
    exam = _exam(per_family=10)
    from wrop_eval.models import NoisyDiffuse
    model = NoisyDiffuse()
    hits = [score_question(q, model.predict(q, random.Random(5)),
                           "noisy-diffuse")[0] for q in exam]
    acc = sum(hits) / len(hits)
    assert 0.0 < acc < 1.0, f"expected partial credit, got {acc}"


# ---------- elo & bench ----------

def test_elo_ranks_oracle_first_with_sane_values():
    names = ["oracle", "a", "b"]
    wins = {(("oracle", "a")): 10.0, (("a", "oracle")): 0.0,
            (("oracle", "b")): 10.0, (("b", "oracle")): 0.0,
            (("a", "b")): 6.0, (("b", "a")): 4.0}
    elos = elo_ratings(names, wins)
    assert elos["oracle"] > elos["a"] > elos["b"]
    assert all(math.isfinite(v) for v in elos.values())
    assert elos["oracle"] - elos["b"] > 100


def test_bench_results_are_json_serializable_and_reproducible():
    exam = _exam(per_family=4)
    models = default_models()
    r1 = run_bench(exam, models, seed=7)
    r2 = run_bench(exam, models, seed=7)
    assert r1 == r2
    json.dumps(r1)  # must not raise
    assert r1["meta"]["n_questions"] == 4 * len(FAMILIES)
    for m in models:
        assert 0.0 <= r1["models"][m.name]["overall_accuracy"] <= 1.0
