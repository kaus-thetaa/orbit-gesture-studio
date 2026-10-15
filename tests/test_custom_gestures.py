import numpy as np
import pytest

from orbit.errors import ConfigError
from orbit.settings import RecognitionSettings
from orbit.vision.custom_store import CustomGestureStore, GestureRecorder
from orbit.vision.gestures import CustomGestureClassifier, FingerPatternRecognizer, GestureRegistry
from synth import FIST, OPEN, TWO, make_hand

CFG = RecognitionSettings()
# a deliberately "unusual" pose: pinky + index only (rock sign)
ROCK = dict(index=True, pinky=True)


def record(fingers, n=12, noise=0.08, rot_jitter=6.0):
    rec = GestureRecorder(CFG)
    rng = np.random.default_rng(1)
    for i in range(n):
        h = make_hand(fingers, rotation_deg=float(rng.uniform(-rot_jitter, rot_jitter)),
                      scale=float(rng.uniform(10, 14)), noise=noise, seed=i)
        assert rec.try_capture(h) == ""
    return rec


def registry_with(gesture):
    clf = CustomGestureClassifier(CFG)
    clf.set_gestures([gesture])
    reg = GestureRegistry(CFG)
    reg.register(FingerPatternRecognizer(CFG))
    reg.register(clf)
    return reg


def test_custom_gesture_recognised_and_others_rejected():
    g = record(ROCK).build("rock")
    reg = registry_with(g)
    hits = [reg.classify(make_hand(ROCK, rotation_deg=r, noise=0.08, seed=100 + r)) for r in (-8, 0, 8)]
    assert all(h is not None and h.name == "rock" and h.source == "custom" for h in hits)
    for other in (OPEN, FIST, TWO):
        p = reg.classify(make_hand(other, noise=0.08, seed=7))
        assert p is None or p.name != "rock"


def test_sensitivity_changes_threshold():
    g = record(ROCK).build("rock")
    clf = CustomGestureClassifier(CFG)
    base = clf.threshold_for(g)
    g.sensitivity = 2.0
    assert clf.threshold_for(g) == pytest.approx(base * 2)


def test_too_few_samples_rejected():
    rec = GestureRecorder(CFG)
    for i in range(3):
        rec.try_capture(make_hand(ROCK, noise=0.05, seed=i))
    with pytest.raises(ConfigError):
        rec.build("rock")


@pytest.mark.parametrize("name", ["", "A", "Rock Sign", "1abc", "fist", "gaze_left", "x" * 40, "a-b"])
def test_invalid_names_rejected(name):
    rec = record(ROCK)
    with pytest.raises(ConfigError):
        rec.build(name)


def test_bad_captures_rejected():
    rec = GestureRecorder(CFG)
    assert rec.try_capture(None)
    assert rec.try_capture(make_hand(ROCK, center=(630, 300)))  # partly out of frame
    assert rec.try_capture(make_hand(ROCK, scale=1.5))          # too small
    assert rec.samples == []


def test_outlier_detection():
    rec = record(ROCK, n=12)
    rec.samples.append(CustomGestureClassifier.features(make_hand(OPEN)))  # wrong pose slipped in
    assert 12 in rec.outlier_indices()


def test_builtin_overlap_warning():
    rec = record(TWO, n=10)
    assert "two_fingers" in rec.builtin_overlap()
    assert record(ROCK, n=10).builtin_overlap() == {}


def test_store_roundtrip_and_duplicate(tmp_path):
    store = CustomGestureStore(tmp_path / "c.json")
    g = record(ROCK).build("rock")
    store.add(g)
    with pytest.raises(ConfigError):
        store.add(record(ROCK).build("rock"))
    store.add(record(ROCK).build("rock"), replace=True)
    s2 = CustomGestureStore(tmp_path / "c.json")
    loaded = s2.load()
    assert [x.name for x in loaded] == ["rock"]
    assert np.allclose(loaded[0].samples, store.gestures[0].samples, atol=1e-4)
    assert s2.remove("rock") and s2.load() == []


def test_corrupt_store_is_quarantined(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{not json")
    store = CustomGestureStore(p)
    assert store.load() == []
    assert (tmp_path / "c.json.corrupt").exists()
