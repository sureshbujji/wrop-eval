"""Simulated video-model personas under test.

The WROP paper evaluates 14 real models across three interface classes:
continuation models (extend the clip), reference-to-video models (condition
on reference frames), and edit models (transform the input). Calling real
video-generation APIs is out of scope for a self-contained harness, so each
persona below is a *behavioral simulator*: a deterministic rule that maps
context frames to predicted target frames, exhibiting a characteristic
failure mode of its class:

  oracle            (continuation) - perfect physics; the ceiling.
  linear-motion     (continuation) - naive constant-velocity extrapolation;
                    ignores walls, gravity changes, collisions, parenting.
                    The classic "objects pass through barriers" failure.
  mean-frame        (continuation) - predicts the temporal mean of context;
                    everything smears into a frozen ghost.
  forget-occluded   (reference)    - perfect physics except any object that
                    was occluded in the last context frame is dropped from
                    the prediction. The classic "vanishes behind the
                    occluder" failure.
  copy-last         (edit)         - repeats the last context frame; the
                    minimal-edit model that fails to continue any motion.
  noisy-diffuse     (edit)         - near-perfect physics with seeded jitter
                    and occasional object dropout, like a diffusion model
                    that mostly tracks the scene but sometimes loses an
                    object.

Each persona is honest about what it is: a stand-in behavior, not an API
call to a real video model. The harness measures whether the *exam* can
separate these behaviors, which is the prerequisite for running it against
real models later.
"""

from __future__ import annotations

import random

from .physics import EPS

STATIC_KINDS = {"wall", "platform", "occluder"}


class Model:
    name: str = "model"
    interface: str = "continuation"  # continuation | reference | edit
    blurb: str = ""

    def predict(self, question, rng: random.Random) -> list[dict]:
        raise NotImplementedError

    def describe(self) -> str:
        return f"{self.name} [{self.interface}] - {self.blurb}"


def _copy_frame(frame: dict) -> dict:
    return {oid: dict(s) for oid, s in frame.items()}


class Oracle(Model):
    name = "oracle"
    interface = "continuation"
    blurb = "exact ground-truth physics; the ceiling every model is measured against"

    def predict(self, question, rng):
        return [_copy_frame(f) for f in question.target]


class LinearMotion(Model):
    name = "linear-motion"
    interface = "continuation"
    blurb = ("constant-velocity extrapolation from the last two context "
             "frames; blind to walls, gravity, collisions and parenting")

    def predict(self, question, rng):
        prev, last = question.context[-2], question.context[-1]
        out = []
        for k in range(1, question.scene.target_len + 1):
            frame = {}
            for oid, s in last.items():
                p = prev[oid]
                vx, vy = s["x"] - p["x"], s["y"] - p["y"]
                frame[oid] = dict(s, x=s["x"] + vx * k, y=s["y"] + vy * k)
            out.append(frame)
        return out


class MeanFrame(Model):
    name = "mean-frame"
    interface = "continuation"
    blurb = ("predicts the temporal mean of all context frames; motion "
             "smears into a frozen ghost")

    def predict(self, question, rng):
        ctx = question.context
        avg = {}
        for oid, s in ctx[0].items():
            n = len(ctx)
            avg[oid] = dict(
                s,
                x=sum(f[oid]["x"] for f in ctx) / n,
                y=sum(f[oid]["y"] for f in ctx) / n,
            )
        return [_copy_frame(avg) for _ in range(question.scene.target_len)]


class ForgetOccluded(Model):
    name = "forget-occluded"
    interface = "reference"
    blurb = ("perfect physics, but drops any object occluded in the last "
             "context frame - the 'vanishes behind the occluder' failure")

    def predict(self, question, rng):
        last = question.context[-1]
        occluders = [s for s in last.values() if s["kind"] == "occluder"]

        def hidden(s):
            for o in occluders:
                ox = (min(s["x"] + s["w"], o["x"] + o["w"])
                      - max(s["x"], o["x"]))
                oy = (min(s["y"] + s["h"], o["y"] + o["h"])
                      - max(s["y"], o["y"]))
                if ox > EPS and oy > EPS:
                    return True
            return False

        dropped = {oid for oid, s in last.items()
                   if s["kind"] not in STATIC_KINDS and hidden(s)}
        out = []
        for f in question.target:
            out.append({oid: dict(s) for oid, s in f.items()
                        if oid not in dropped})
        return out


class CopyLast(Model):
    name = "copy-last"
    interface = "edit"
    blurb = "repeats the last context frame; a minimal edit that continues nothing"

    def predict(self, question, rng):
        return [_copy_frame(question.context[-1])
                for _ in range(question.scene.target_len)]


class NoisyDiffuse(Model):
    name = "noisy-diffuse"
    interface = "edit"
    blurb = ("near-perfect physics with seeded +-1 cell jitter and 12% "
             "per-object-per-frame dropout")

    def predict(self, question, rng):
        out = []
        for f in question.target:
            frame = {}
            for oid, s in f.items():
                if s["kind"] not in STATIC_KINDS and rng.random() < 0.12:
                    continue  # object lost this frame
                frame[oid] = dict(
                    s,
                    x=s["x"] + rng.uniform(-1.0, 1.0),
                    y=s["y"] + rng.uniform(-1.0, 1.0),
                )
            out.append(frame)
        return out


def default_models() -> list[Model]:
    return [Oracle(), LinearMotion(), MeanFrame(),
            ForgetOccluded(), CopyLast(), NoisyDiffuse()]
