"""End-to-end: synthetic hand -> classifier -> stabiliser -> mapping -> dispatcher -> (fake) OS action."""
import numpy as np
import pytest

from orbit.runtime.app import OrbitRuntime
from orbit.runtime.pipeline import Pipeline
from orbit.settings import Settings
from fakes import FakeCamera, FakeKeyboard, FakeTracker
from synth import FIST, OPEN, TWO, make_hand

FRAME = np.zeros((480, 640, 3), np.uint8)


@pytest.fixture
def rt():
    s = Settings()
    s.recognition.confirm_frames = 4
    s.recognition.release_frames = 4
    s.safety.enable_hotkeys = False
    rt = OrbitRuntime(settings=s, keyboard=FakeKeyboard())
    rt.tracker = FakeTracker()
    rt.camera = FakeCamera()
    rt.pipeline = Pipeline(s, rt.camera, rt.tracker, rt.gestures, rt.stabilizer, rt.mapping_engine)
    rt.dispatcher.start()
    yield rt
    rt.dispatcher.stop()


def step(rt, hand, n=1):
    last = None
    for i in range(n):
        rt.tracker.next = hand
        last = rt.pipeline.process_frame(FRAME, 1000 + i)
    assert rt.dispatcher.join_idle(2.0)
    return last


def test_full_loop_hold_focus(rt):
    rt.safety.arm()
    step(rt, make_hand(TWO), 3)
    assert not rt.focus.active                      # needs 4 consecutive frames
    rep = step(rt, make_hand(TWO), 1)
    assert rt.focus.active and rep.stable_gesture == "two_fingers"
    step(rt, make_hand(TWO), 30)                    # holding does not re-fire
    assert rt.focus.active
    step(rt, None, 3)                               # tracking lost for fewer frames than release_frames
    assert rt.focus.active
    step(rt, None, 2)
    assert not rt.focus.active


def test_open_palm_and_fist_do_not_activate_focus(rt):
    rt.safety.arm()
    step(rt, make_hand(OPEN), 10)
    step(rt, make_hand(FIST), 10)
    assert not rt.focus.active


def test_dry_run_when_disarmed(rt):
    rep = None
    for _ in range(6):
        rep = step(rt, make_hand(TWO))
    assert not rt.focus.active
    assert rt.mapping_engine.active_ids() == []


def test_camera_loss_releases_everything(rt):
    rt.safety.arm()
    step(rt, make_hand(TWO), 5)
    assert rt.focus.active
    rt.pipeline.handle_camera_lost()
    assert rt.dispatcher.join_idle(2.0)
    assert not rt.focus.active


def test_partial_hand_is_rejected(rt):
    rt.safety.arm()
    rep = step(rt, make_hand(TWO, center=(630, 300)), 8)
    assert rep.reject_reason == "partial_hand" and not rt.focus.active


def test_ambiguous_pose_does_not_activate(rt):
    rt.safety.arm()
    step(rt, make_hand(dict(index=True, middle=0.55)), 10)
    assert not rt.focus.active


def test_switching_gesture_ends_hold(rt):
    rt.safety.arm()
    step(rt, make_hand(TWO), 5)
    assert rt.focus.active
    step(rt, make_hand(OPEN), 5)
    assert not rt.focus.active


def test_emergency_stop_mid_hold(rt):
    rt.safety.arm()
    step(rt, make_hand(TWO), 5)
    assert rt.focus.active
    rt.safety.emergency_stop()
    assert rt.dispatcher.join_idle(2.0)
    assert not rt.focus.active
    step(rt, make_hand(TWO), 10)                    # still holding the pose: stays inert
    assert not rt.focus.active
