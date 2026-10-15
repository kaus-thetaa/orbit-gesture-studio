from orbit.settings import RecognitionSettings
from orbit.vision.stabilizer import DwellDetector, GestureStabilizer


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def kinds(events):
    return [(e.kind, e.name) for e in events]


def feed(stab, seq):
    out = []
    for n in seq:
        out.extend(kinds(stab.update(n)))
    return out


def make(confirm=3, release=3):
    return GestureStabilizer(RecognitionSettings(confirm_frames=confirm, release_frames=release))


def test_requires_consecutive_frames():
    s = make(3, 3)
    assert feed(s, ["a", "a"]) == []
    assert feed(s, ["a"]) == [("pressed", "a")]


def test_fires_once_not_every_frame():
    s = make(3, 3)
    assert feed(s, ["a"] * 20) == [("pressed", "a")]


def test_flicker_does_not_activate():
    s = make(3, 3)
    assert feed(s, ["a", "a", None, "a", "a", None, "a"]) == []


def test_dropout_shorter_than_release_keeps_hold():
    s = make(3, 3)
    ev = feed(s, ["a"] * 3 + [None, None] + ["a"] * 3)
    assert ev == [("pressed", "a")]


def test_release_after_enough_misses():
    s = make(3, 3)
    ev = feed(s, ["a"] * 3 + [None] * 3)
    assert ev == [("pressed", "a"), ("released", "a")]


def test_switching_gesture_releases_then_presses():
    s = make(3, 3)
    ev = feed(s, ["a"] * 3 + ["b"] * 3)
    assert ev == [("pressed", "a"), ("released", "a"), ("pressed", "b")]


def test_switching_when_confirm_less_than_release():
    s = make(2, 5)
    ev = feed(s, ["a"] * 2 + ["b"] * 5)
    assert ev == [("pressed", "a"), ("released", "a"), ("pressed", "b")]


def test_reset_releases_active():
    s = make(2, 3)
    feed(s, ["a", "a"])
    assert kinds(s.reset()) == [("released", "a")]
    assert s.reset() == []


def test_settings_changes_apply_live():
    cfg = RecognitionSettings(confirm_frames=5, release_frames=5)
    s = GestureStabilizer(cfg)
    assert feed(s, ["a"] * 3) == []
    cfg.confirm_frames = 4
    assert feed(s, ["a"]) == [("pressed", "a")]


def test_dwell_detector_dwell_and_cooldown():
    c = Clock()
    d = DwellDetector("gaze_left", "gaze", dwell_ms=500, release_ms=200, cooldown_ms=1000, clock=c)
    # not long enough
    for _ in range(4):
        c.t += 0.1
        assert d.update(True) == []
    c.t += 0.2
    assert kinds(d.update(True)) == [("pressed", "gaze_left")]
    # still looking: no repeat
    c.t += 0.2
    assert d.update(True) == []
    # look away -> released after release_ms
    c.t += 0.1
    assert d.update(False) == []
    c.t += 0.3
    assert kinds(d.update(False)) == [("released", "gaze_left")]
    # immediately look again: cooldown (1.0 s from release) blocks for 0.8 s
    for _ in range(8):
        c.t += 0.1
        assert d.update(True) == []
    # cooldown has now elapsed and the dwell is long satisfied -> fires
    c.t += 0.3
    assert kinds(d.update(True)) == [("pressed", "gaze_left")]


def test_dwell_detector_reading_motion_never_fires():
    c = Clock()
    d = DwellDetector("gaze_right", "gaze", dwell_ms=900, release_ms=200, cooldown_ms=0, clock=c)
    fired = []
    for i in range(100):  # glance right for 0.3s every second
        c.t += 0.1
        fired += d.update(i % 10 < 3)
    assert fired == []
