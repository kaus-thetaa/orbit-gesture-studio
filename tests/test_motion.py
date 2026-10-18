import math

import numpy as np
import pytest

from orbit.errors import ConfigError
from orbit.mapping import Mapping
from orbit.runtime.app import OrbitRuntime
from orbit.runtime.pipeline import Pipeline
from orbit.settings import MotionSettings, Settings
from orbit.vision.motion import (BUILTIN_MOTIONS, CustomMotion, MotionClassifier, MotionEngine, TrackPoint,
                                 augment, count_reversals, dtw_batch, N_POINTS)
from orbit.vision.motion_store import MotionRecorder, MotionStore
from fakes import FakeCamera, FakeKeyboard, FakeTracker
from synth import OPEN, make_hand

FPS = 30


def mj(u):  # minimum-jerk profile: natural bell-shaped hand speed
    return 10 * u**3 - 15 * u**4 + 6 * u**5


class Tl:
    """Builds a timed stream of palm positions (or None = hand lost)."""

    def __init__(self, size=100.0, pose="open_palm", start=(320.0, 240.0), jitter=0.4, seed=0):
        self.items, self.t, self.size, self.pose = [], 0.0, size, pose
        self.pos, self.jitter, self.rng = np.array(start, float), jitter, np.random.default_rng(seed)

    def _emit(self, pos, pose=None):
        p = pos + self.rng.normal(0, self.jitter, 2)
        self.items.append((self.t, TrackPoint(self.t, float(p[0]), float(p[1]), self.size, pose or self.pose)))
        self.t += 1.0 / FPS

    def still(self, d, pose=None):
        for _ in range(int(d * FPS)):
            self._emit(self.pos, pose)
        return self

    def move(self, fn, d, pose=None):
        anchor, n = self.pos.copy(), max(2, int(d * FPS))
        for k in range(1, n + 1):
            self._emit(anchor + self.size * np.array(fn(k / n)), pose)
        self.pos = anchor + self.size * np.array(fn(1.0))
        return self

    def lost(self, d):
        for _ in range(int(d * FPS)):
            self.items.append((self.t, None))
            self.t += 1.0 / FPS
        return self


def run(items, cfg=None, engine=None):
    eng = engine or MotionEngine(cfg or MotionSettings())
    events, results = [], []
    for t, p in items:
        ev, res = eng.update(p, t)
        events += ev
        if res:
            results.append(res)
    return eng, events, results


def names(events):
    return [(e.kind, e.name) for e in events]


# path functions (hand-lengths, image coords: +x right, +y DOWN)
swipe_right = lambda u: (3.0 * mj(u), 0.1 * mj(u))
swipe_left = lambda u: (-3.0 * mj(u), 0.1 * mj(u))
swipe_up = lambda u: (0.1 * mj(u), -3.0 * mj(u))
swipe_down = lambda u: (0.1 * mj(u), 3.0 * mj(u))
wave = lambda u: (1.2 * math.sin(2 * math.pi * 4 * u), 0.05 * math.sin(2 * math.pi * u))


def circle(cw, revs=1.05, r=1.0):
    def f(u):
        th = 2 * math.pi * revs * u
        return (r * math.sin(th) * (1 if cw else -1), r * (1 - math.cos(th)))
    return f


def one(items, **kw):
    eng, ev, res = run(items, **kw)
    assert len(res) == 1, [r.reason for r in res]
    return res[0], ev


# ============================================================ built-in motions
@pytest.mark.parametrize("fn,name", [(swipe_right, "swipe_right"), (swipe_left, "swipe_left"),
                                     (swipe_up, "swipe_up"), (swipe_down, "swipe_down")])
def test_swipes(fn, name):
    res, ev = one(Tl().still(0.5).move(fn, 0.4).still(0.5).items)
    assert res.prediction and res.prediction.name == name
    assert names(ev) == [("pressed", name), ("released", name)]


def test_wave_and_circles():
    r, _ = one(Tl().still(0.5).move(wave, 2.0).still(0.5).items)
    assert r.prediction.name == "wave"
    r, _ = one(Tl().still(0.5).move(circle(True), 1.0).still(0.5).items)
    assert r.prediction.name == "circle_cw"
    r, _ = one(Tl().still(0.5).move(circle(False), 1.0).still(0.5).items)
    assert r.prediction.name == "circle_ccw"


def test_circle_is_not_wave_and_wave_is_not_circle():
    r, _ = one(Tl().still(0.5).move(circle(True, revs=2.05), 1.6).still(0.5).items)
    assert r.prediction is None or r.prediction.name.startswith("circle")
    r, _ = one(Tl().still(0.5).move(wave, 2.0).still(0.5).items)
    assert not r.prediction.name.startswith("circle")


def test_stillness_jitter_and_slow_drift_never_fire():
    _, ev, res = run(Tl(jitter=1.5).still(10).items)
    assert ev == [] and res == []
    _, ev, res = run(Tl().still(0.5).move(lambda u: (3 * u, 0), 6.0).still(0.5).items)   # 0.5 hand-lengths/s
    assert ev == [] and res == []


def test_tiny_fast_movement_ignored():
    _, ev, res = run(Tl().still(0.5).move(lambda u: (0.5 * mj(u), 0), 0.2).still(0.5).items)
    assert ev == [] and res == []


def test_out_and_back_is_not_a_swipe():
    out_back = lambda u: (3.0 * mj(2 * u) if u < 0.5 else 3.0 * (1 - mj(2 * u - 1)), 0)
    r, ev = one(Tl().still(0.5).move(out_back, 0.9).still(0.5).items)
    assert r.prediction is None and ev == []


def test_hand_size_invariance():
    for size in (50.0, 100.0, 180.0):
        r, _ = one(Tl(size=size).still(0.5).move(swipe_right, 0.4).still(0.5).items)
        assert r.prediction.name == "swipe_right", size


def test_position_in_frame_invariance():
    for start in ((100, 100), (500, 400)):
        r, _ = one(Tl(start=start).still(0.5).move(swipe_left, 0.4).still(0.5).items)
        assert r.prediction.name == "swipe_left"


def test_short_dropout_tolerated_long_loss_ends_segment():
    tl = Tl().still(0.5)
    tl.move(lambda u: (3.0 * mj(u), 0), 0.2)
    tl.lost(2 / FPS)                                     # 2 lost frames < gap
    tl.move(lambda u: (3.0 * mj(u), 0), 0.2).still(0.5)
    r, _ = one(tl.items)
    assert r.prediction is not None
    tl = Tl().still(0.5).move(swipe_right, 0.4).lost(0.5)   # hand leaves right after the swipe
    r, _ = one(tl.items)
    assert r.prediction.name == "swipe_right"


def test_too_slow_movement_is_dropped_and_system_recovers():
    tl = Tl().still(0.5).move(wave, 4.0).still(0.6)       # > max_duration: discarded
    tl.move(swipe_right, 0.4).still(0.5)
    _, ev, res = run(tl.items)
    assert names(ev) == [("pressed", "swipe_right"), ("released", "swipe_right")]


def test_pose_gating():
    r, ev = one(Tl(pose="fist").still(0.5).move(swipe_right, 0.4).still(0.5).items)
    assert ev == [] and r.reason.startswith("pose is not open_palm")
    r, ev = one(Tl(pose="fist").still(0.5).move(swipe_right, 0.4).still(0.5).items, cfg=MotionSettings(require_pose=""))
    assert names(ev)[0] == ("pressed", "swipe_right")


def test_cooldown_suppresses_return_stroke():
    tl = Tl().still(0.5).move(swipe_right, 0.4).still(0.35).move(swipe_left, 0.4).still(0.5)   # return after a short pause
    _, ev, res = run(tl.items)
    assert names(ev) == [("pressed", "swipe_right"), ("released", "swipe_right")]
    assert len(res) == 2 and res[1].reason.startswith("cooldown")
    tl = Tl().still(0.5).move(swipe_right, 0.4).still(1.5).move(swipe_left, 0.4).still(0.5)   # unhurried return
    _, ev, _ = run(tl.items)
    assert [e.name for e in ev if e.kind == "pressed"] == ["swipe_right", "swipe_left"]


def test_strokes_without_a_pause_merge_into_one_movement():
    """Documented limitation: two strokes closer than ~0.2 s are one movement, hence not a swipe."""
    _, ev, res = run(Tl().still(0.5).move(swipe_right, 0.4).still(0.1).move(swipe_left, 0.4).still(0.5).items)
    assert ev == [] and len(res) == 1


def test_reversal_counter():
    x = np.sin(np.linspace(0, 2 * math.pi * 3, 200)) * 1.0
    assert count_reversals(x, 0.5) == 6
    assert count_reversals(x * 0.1, 0.5) == 0       # tiny swings ignored
    assert count_reversals(np.linspace(0, 3, 50), 0.5) == 0


# ============================================================ DTW
def dtw_naive(a, b, band):
    N = len(a)
    D = np.full((N + 1, N + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, N + 1):
        for j in range(max(1, i - band), min(N, i + band) + 1):
            D[i, j] = np.linalg.norm(a[i - 1] - b[j - 1]) + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[N, N] / N


def test_dtw_matches_reference_implementation():
    rng = np.random.default_rng(3)
    tpl = rng.normal(size=(5, 20, 2)).cumsum(axis=1)
    q = rng.normal(size=(20, 2)).cumsum(axis=0)
    got = dtw_batch(tpl, q, band=5)
    for k in range(5):
        assert got[k] == pytest.approx(dtw_naive(tpl[k], q, 5))


def test_dtw_identity_warp_and_difference():
    u = np.linspace(0, 1, N_POINTS)
    a = np.stack([3 * u, np.zeros_like(u)], 1)
    b = np.stack([3 * u**1.3, np.zeros_like(u)], 1)           # same shape, different speed profile
    c = np.stack([np.zeros_like(u), 3 * u], 1)                # different direction
    d = dtw_batch(np.stack([a]), a)[0]
    assert d == pytest.approx(0)
    assert dtw_batch(np.stack([a]), b)[0] < 0.25 * dtw_batch(np.stack([a]), c)[0]


def test_augmentation_is_plausible_and_deterministic():
    s = np.stack([np.linspace(0, 3, N_POINTS), np.zeros(N_POINTS)], 1)
    a1 = augment(s, np.random.default_rng(1), 5)
    a2 = augment(s, np.random.default_rng(1), 5)
    assert all(np.allclose(x, y) for x, y in zip(a1, a2))
    for v in a1:
        assert np.allclose(v[0], 0) and 2.0 < np.linalg.norm(v[-1]) < 3.8


# ============================================================ custom motions
def zigzag(u):
    if u < 1 / 3:
        return (1.5 * 3 * u, 0.0)
    if u < 2 / 3:
        v = 3 * u - 1
        return (1.5 - 1.5 * v, 1.2 * v)
    v = 3 * u - 2
    return (1.5 * v, 1.2)


def vee(u):   # down then up, to the right
    return (2.0 * u, 1.5 * (1 - abs(2 * u - 1)))


def performances(fn, n, seed=0, pose="open_palm"):
    """Run n varied performances of fn through the real segmenter; return their Segments."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        s, th, d = rng.uniform(0.85, 1.2), math.radians(rng.uniform(-8, 8)), rng.uniform(0.7, 1.3)
        rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        f = lambda u, s=s, rot=rot: tuple(rot @ (s * np.array(fn(mj(u)))))
        tl = Tl(size=float(rng.uniform(80, 140)), pose=pose, seed=i).still(0.4).move(f, d).still(0.5)
        _, _, res = run(tl.items)
        assert len(res) == 1
        out.append(res[0].segment)
    return out


def record(fn, n=14, name="zigzag", seed=0, **kw):
    rec = MotionRecorder(MotionSettings())
    for seg in performances(fn, n, seed):
        assert rec.add_segment(seg) == ""
    return rec.build(name, **kw)


def engine_with(*motions):
    eng = MotionEngine(MotionSettings())
    eng.custom.set_motions(list(motions))
    return eng


def test_custom_motion_recognised_and_confusers_rejected():
    z = record(zigzag)
    assert z.pose == "open_palm"
    eng = engine_with(z)
    tl = Tl(size=110).still(0.4)
    seg = performances(zigzag, 1, seed=99)[0]
    preds = eng.custom.classify(seg)
    assert preds and preds[0].name == "zigzag" and preds[0].confidence >= 0.5
    for fn in (swipe_right, swipe_down, lambda u: circle(True)(u), vee):
        s = performances(fn, 1, seed=5)[0]
        assert not [p for p in eng.custom.classify(s) if p.confidence >= 0.5]


def test_two_custom_motions_are_told_apart():
    eng = engine_with(record(zigzag, name="zigzag"), record(vee, name="vee", seed=1))
    for fn, want in ((zigzag, "zigzag"), (vee, "vee")):
        for k in range(3):
            seg = performances(fn, 1, seed=200 + k)[0]
            best = max(eng.custom.classify(seg), key=lambda p: p.confidence)
            assert best.name == want


def test_custom_fires_through_engine_and_takes_priority_over_builtin():
    # a custom "flick" that is geometrically a swipe_right must win over the built-in
    flick = record(swipe_right, name="flick")
    eng = engine_with(flick)
    _, ev, res = run(Tl().still(0.5).move(swipe_right, 0.4).still(0.5).items, engine=eng)
    assert names(ev)[0] == ("pressed", "flick")
    assert res[0].prediction.source == "custom"


def test_custom_pose_requirement():
    z = record(zigzag)
    eng = engine_with(z)
    seg = performances(zigzag, 1, seed=7, pose="fist")[0]
    assert eng.custom.classify(seg) == []
    anyp = record(zigzag, require_pose=False)
    assert anyp.pose == ""


def test_sensitivity_scales_threshold():
    z = record(zigzag)
    clf = MotionClassifier(MotionSettings())
    clf.set_motions([z])
    base = clf.threshold_for("zigzag")
    z.sensitivity = 2.0
    clf.set_motions([z])
    assert clf.threshold_for("zigzag") == pytest.approx(2 * base)


def test_more_samples_do_not_hurt_and_few_are_refused():
    rec = MotionRecorder(MotionSettings())
    for seg in performances(zigzag, 5):
        rec.add_segment(seg)
    with pytest.raises(ConfigError):
        rec.build("zigzag")


@pytest.mark.parametrize("name", ["", "Z", "Zig Zag", "1zig", "swipe_left", "circle_cw", "gaze_x", "x" * 40])
def test_bad_motion_names(name):
    rec = MotionRecorder(MotionSettings())
    for seg in performances(zigzag, 8):
        rec.add_segment(seg)
    with pytest.raises(ConfigError):
        rec.build(name)


def test_recorder_flags_outlier_and_builtin_overlap():
    rec = MotionRecorder(MotionSettings())
    for seg in performances(zigzag, 10):
        rec.add_segment(seg)
    rec.add_segment(performances(swipe_right, 1, seed=3)[0])          # a botched attempt
    assert 10 in rec.outlier_indices()
    assert rec.builtin_overlap() == {}
    sw = MotionRecorder(MotionSettings())
    for seg in performances(swipe_right, 10):
        sw.add_segment(seg)
    assert "swipe_right" in sw.builtin_overlap()


def test_store_roundtrip_and_corruption(tmp_path):
    st = MotionStore(tmp_path / "m.json")
    z = record(zigzag)
    st.add(z)
    with pytest.raises(ConfigError):
        st.add(record(zigzag))
    loaded = MotionStore(tmp_path / "m.json").load()
    assert [m.name for m in loaded] == ["zigzag"] and loaded[0].pose == "open_palm"
    assert np.allclose(loaded[0].samples, z.samples, atol=1e-3)
    assert st.remove("zigzag") and st.load() == []
    (tmp_path / "m.json").write_text("{bad")
    assert MotionStore(tmp_path / "m.json").load() == []
    assert (tmp_path / "m.json.corrupt").exists()


# ============================================================ mappings
def test_hold_mode_rejected_for_motions():
    s = Settings(); s.safety.enable_hotkeys = False
    rt = OrbitRuntime(settings=s, keyboard=FakeKeyboard())
    with pytest.raises(ConfigError):
        rt.store.add(Mapping("m1", "swipe_left", "focus", "hold", {}, True, 0))
    rt.store.add(Mapping("m2", "swipe_left", "focus", "toggle", {}, True, 0))
    assert "swipe_left" in rt.known_gestures() and "circle_cw" in rt.known_gestures()
    assert set(BUILTIN_MOTIONS) <= set(rt.motion_names())


# ============================================================ end to end
@pytest.fixture
def rt():
    s = Settings()
    s.safety.enable_hotkeys = False
    rt = OrbitRuntime(settings=s, keyboard=FakeKeyboard())
    rt.tracker, rt.camera = FakeTracker(), FakeCamera()
    rt.pipeline = Pipeline(s, rt.camera, rt.tracker, rt.gestures, rt.stabilizer, rt.mapping_engine, motion=rt.motion)
    rt.store.add(Mapping("snap-r", "swipe_right", "snap_window_right", "trigger", {}, True, 0))
    rt.dispatcher.start()
    yield rt
    rt.dispatcher.stop()


def hand_swipe(rt, hand_fingers, x0=140, x1=480):
    frame = np.zeros((480, 640, 3), np.uint8)
    xs = [x0] * 12 + [x0 + (x1 - x0) * mj(k / 12) for k in range(1, 13)] + [x1] * 14
    got = []
    for i, x in enumerate(xs):
        rt.tracker.next = make_hand(hand_fingers, center=(x, 330))
        rep = rt.pipeline.process_frame(frame, int(i * 1000 / FPS))
        if rep.motion:
            got.append(rep.motion)
    assert rt.dispatcher.join_idle(2.0)
    return got


def test_e2e_swipe_right_snaps_window_when_armed(rt):
    rt.safety.arm()
    got = hand_swipe(rt, OPEN)
    assert len(got) == 1 and got[0].prediction.name == "swipe_right"
    assert rt.ctx.keyboard.combos == [("win", "right")]


def test_e2e_dry_run_when_disarmed(rt):
    got = hand_swipe(rt, OPEN)
    assert got[0].prediction.name == "swipe_right"
    assert rt.ctx.keyboard.combos == []


def test_e2e_fist_swipe_is_ignored(rt):
    rt.safety.arm()
    got = hand_swipe(rt, dict())
    assert got and got[0].prediction is None and "pose" in got[0].reason
    assert rt.ctx.keyboard.combos == []


def test_e2e_camera_loss_resets_motion(rt):
    rt.pipeline.process_frame(np.zeros((480, 640, 3), np.uint8), 0)
    rt.pipeline.handle_camera_lost()
    assert not rt.motion.segmenter.active
