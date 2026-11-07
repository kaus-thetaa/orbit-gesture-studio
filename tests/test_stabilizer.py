from orbit.settings import RecognitionSettings
from orbit.vision.stabilizer import GestureStabilizer


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
