import numpy as np
import pytest

from orbit.settings import RecognitionSettings
from orbit.vision.features import finger_states, normalize_landmarks
from orbit.vision.gestures import FingerPatternRecognizer, GestureRegistry
from orbit.vision.quality import check_hand_quality
from synth import FIST, OPEN, TWO, make_hand

CFG = RecognitionSettings()


def classify(hand):
    reg = GestureRegistry(CFG)
    reg.register(FingerPatternRecognizer(CFG))
    return reg.classify(hand)


@pytest.mark.parametrize("fingers,expected", [
    (TWO, "two_fingers"),
    (OPEN, "open_palm"),
    (FIST, "fist"),
    (dict(index=True), "one_finger"),
    (dict(index=True, middle=True, ring=True), "three_fingers"),
])
def test_builtin_poses(fingers, expected):
    pred = classify(make_hand(fingers))
    assert pred is not None and pred.name == expected
    assert pred.confidence > 0.9


def test_two_fingers_distinct_from_palm_and_fist():
    assert classify(make_hand(TWO)).name != "open_palm"
    assert classify(make_hand(TWO)).name != "fist"
    assert classify(make_hand(OPEN)).name != "two_fingers"
    assert classify(make_hand(FIST)).name != "two_fingers"


def test_thumb_is_ignored_for_two_fingers():
    for thumb in (False, True):
        assert classify(make_hand(TWO, thumb=thumb)).name == "two_fingers"


@pytest.mark.parametrize("rot", [0, 30, 90, 180, 270, -45])
@pytest.mark.parametrize("scale", [6.0, 12.0, 20.0])
@pytest.mark.parametrize("hand", ["Left", "Right"])
def test_invariance_rotation_scale_handedness(rot, scale, hand):
    h = make_hand(TWO, rotation_deg=rot, scale=scale, handedness=hand)
    assert classify(h).name == "two_fingers"


def test_ambiguous_finger_is_rejected_by_confidence():
    # middle finger halfway between curled and straight -> must not be a confident two_fingers
    h = make_hand(dict(index=True, middle=0.55))
    pred = classify(h)
    assert pred is None or pred.confidence < 0.9
    # and the finger margin itself should be near zero
    st = finger_states(h.landmarks, CFG)["middle"]
    assert abs(st.margin) < 0.6


def test_no_prediction_when_thresholds_make_it_ambiguous():
    strict = RecognitionSettings(min_gesture_confidence=0.99)
    reg = GestureRegistry(strict)
    reg.register(FingerPatternRecognizer(strict))
    assert reg.classify(make_hand(dict(index=True, middle=0.6))) is None


def test_normalize_is_invariant():
    a = normalize_landmarks(make_hand(TWO).landmarks, "Right")
    b = normalize_landmarks(make_hand(TWO, rotation_deg=77, scale=9, center=(200, 200)).landmarks, "Right")
    assert np.allclose(a, b, atol=1e-6)


def test_left_hand_matches_right_after_normalisation():
    a = normalize_landmarks(make_hand(TWO, handedness="Right").landmarks, "Right")
    b = normalize_landmarks(make_hand(TWO, handedness="Left").landmarks, "Left")
    assert np.allclose(a, b, atol=1e-6)


def test_quality_rejects_partial_and_tiny_hands():
    assert check_hand_quality(make_hand(TWO), CFG) is None
    assert check_hand_quality(make_hand(TWO, center=(620, 330)), CFG) == "partial_hand"
    assert check_hand_quality(make_hand(TWO, scale=1.5), CFG) == "hand_too_small"
