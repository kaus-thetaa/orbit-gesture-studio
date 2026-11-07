"""Offscreen Qt smoke tests: windows build, react to reports, dialogs validate. Skipped if Qt is unavailable."""
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from orbit.gui.bridge import Bridge  # noqa: E402
from orbit.gui.main_window import MainWindow  # noqa: E402
from orbit.gui.mapping_dialog import MappingDialog  # noqa: E402
from orbit.gui.motion_recorder_dialog import MotionRecorderDialog  # noqa: E402
from orbit.gui.recorder_dialog import RecorderDialog  # noqa: E402
from orbit.vision.motion import MotionResult  # noqa: E402
from orbit.runtime.app import OrbitRuntime  # noqa: E402
from orbit.runtime.pipeline import FrameReport, Pipeline  # noqa: E402
from orbit.settings import Settings  # noqa: E402
from fakes import FakeCamera, FakeKeyboard, FakeTracker  # noqa: E402
from synth import TWO, make_hand  # noqa: E402

ROCK = dict(index=True, pinky=True)


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def env(qapp):
    s = Settings()
    s.safety.enable_hotkeys = False
    s.recognition.confirm_frames = 2
    rt = OrbitRuntime(settings=s, keyboard=FakeKeyboard())
    rt.tracker, rt.camera = FakeTracker(), FakeCamera()
    rt.pipeline = Pipeline(s, rt.camera, rt.tracker, rt.gestures, rt.stabilizer, rt.mapping_engine, motion=rt.motion)
    rt.dispatcher.start()
    bridge = Bridge()
    bridge.attach(rt)
    win = MainWindow(rt, bridge)
    yield rt, bridge, win
    rt.dispatcher.stop()
    win.close()


def feed(rt, bridge, hand, n=1):
    for i in range(n):
        rt.tracker.next = hand
        bridge.report.emit(rt.pipeline.process_frame(np.zeros((480, 640, 3), np.uint8), 100 + i))
        QtWidgets.QApplication.processEvents()


def test_window_shows_disarmed_and_table(env):
    rt, bridge, win = env
    assert "DISARMED" in win.pill.text()
    assert win.table.rowCount() == len(rt.store.all()) >= 1


def test_arm_disarm_and_live_labels(env):
    rt, bridge, win = env
    win.arm_btn.click()
    assert rt.safety.armed and "ARMED" in win.pill.text()
    feed(rt, bridge, make_hand(TWO), 4)
    assert win.stable_lbl.text() == "two_fingers"
    assert rt.dispatcher.join_idle(2)
    assert rt.focus.active and win.focus_lbl.text() == "Focus: ON"
    win.estop_btn.click()
    assert rt.dispatcher.join_idle(2)
    QtWidgets.QApplication.processEvents()
    assert not rt.focus.active and "DISARMED" in win.pill.text()


def test_toggle_enabled_from_table(env):
    rt, bridge, win = env
    item = win.table.item(0, 0)
    mid = item.data(256)
    item.setCheckState(__import__("PySide6.QtCore", fromlist=["Qt"]).Qt.CheckState.Unchecked)
    assert rt.store.get(mid).enabled is False


def test_mapping_dialog_validation_and_save(env, monkeypatch):
    rt, bridge, win = env
    warned = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: warned.append(a[2]))
    dlg = MappingDialog(win, rt, gesture="one_finger")
    dlg.action.setCurrentText("keyboard_shortcut")
    dlg.param_widgets["keys"].setText("alt+f4")           # denylisted
    dlg._save()
    assert warned and not dlg.result()
    dlg.param_widgets["keys"].setText("ctrl+shift+t")
    dlg._save()
    assert dlg.result() == 1
    assert any(m.gesture == "one_finger" and m.params["keys"] == "ctrl+shift+t" for m in rt.store.all())


def test_recorder_flow(env, monkeypatch):
    rt, bridge, win = env
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: None)
    dlg = RecorderDialog(win, rt, bridge)
    assert rt.mapping_engine.suspended
    rng = np.random.default_rng(0)
    for i in range(10):
        feed(rt, bridge, make_hand(ROCK, rotation_deg=float(rng.uniform(-5, 5)), noise=0.06, seed=i))
        assert dlg._capture()
    assert len(dlg.rec.samples) == 10 and dlg.samples.count() == 10
    feed(rt, bridge, make_hand(ROCK, noise=0.06, seed=99))
    assert "MATCH" in dlg.live.text()
    dlg.name.setText("Bad Name")
    dlg._save()
    assert not dlg.result()
    dlg.name.setText("rock_sign")
    dlg._save()
    assert dlg.result() == 1 and dlg.saved_name == "rock_sign"
    assert "rock_sign" in rt.custom_clf.names() and "rock_sign" in rt.known_gestures()
    dlg.done(1)
    assert not rt.mapping_engine.suspended


def test_no_camera_report_is_handled(env):
    rt, bridge, win = env
    bridge.report.emit(FrameReport(camera_status="reconnecting", reject_reason="camera_lost"))
    QtWidgets.QApplication.processEvents()
    assert "reconnecting" in win.video.text()


# ------------------------------------------------------------------ motion GUI
def seg_report(seg, pred=None, reason="unrecognized"):
    return FrameReport(motion=MotionResult(seg, pred, reason, {"path": 3.0, "straight": 0.9, "reversals": 0, "turn": 5}))


def test_mapping_dialog_hides_hold_for_motion_gestures(env):
    rt, bridge, win = env
    dlg = MappingDialog(win, rt, gesture="two_fingers")
    dlg.action.setCurrentText("focus")
    assert "hold" in [dlg.mode.itemText(i) for i in range(dlg.mode.count())]
    dlg.gesture.setCurrentText("swipe_left")
    assert "hold" not in [dlg.mode.itemText(i) for i in range(dlg.mode.count())]
    assert dlg.mode.currentText() == "toggle"


def test_live_tab_shows_motion_result(env):
    from orbit.vision.motion import MotionPrediction
    from test_motion import performances, swipe_right
    rt, bridge, win = env
    seg = performances(swipe_right, 1)[0]
    bridge.report.emit(seg_report(seg, MotionPrediction("swipe_right", 0.9, "builtin")))
    QtWidgets.QApplication.processEvents()
    assert "swipe_right" in win.motion_lbl.text()
    bridge.report.emit(seg_report(seg, None, "pose is not open_palm"))
    QtWidgets.QApplication.processEvents()
    assert "pose is not open_palm" in win.motion_lbl.text()


def test_motion_recorder_flow(env, monkeypatch):
    from test_motion import performances, zigzag, swipe_right
    rt, bridge, win = env
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: None)
    dlg = MotionRecorderDialog(win, rt, bridge)
    assert rt.mapping_engine.suspended
    for seg in performances(zigzag, 12):
        bridge.report.emit(seg_report(seg))
        QtWidgets.QApplication.processEvents()
    assert len(dlg.rec.samples) == 12 and dlg.samples.count() == 12
    assert "open_palm" in dlg.pose_lbl.text()
    # test mode: matching vs non-matching movement
    dlg.test_rb.setChecked(True)
    bridge.report.emit(seg_report(performances(zigzag, 1, seed=50)[0]))
    QtWidgets.QApplication.processEvents()
    assert "MATCH" in dlg.last.text()
    bridge.report.emit(seg_report(performances(swipe_right, 1, seed=50)[0]))
    QtWidgets.QApplication.processEvents()
    assert "no match" in dlg.last.text()
    assert len(dlg.rec.samples) == 12                       # test mode records nothing
    dlg.name.setText("swipe_left")                          # reserved
    dlg._save()
    assert not dlg.result()
    dlg.name.setText("zigzag")
    dlg._save()
    assert dlg.result() == 1 and dlg.saved_name == "zigzag"
    assert "zigzag" in rt.motion.custom.names() and "zigzag" in rt.known_gestures()
    assert rt.motion_store.get("zigzag").pose == "open_palm"
    win._reload_custom()
    assert win.motion_list.count() == 1
    assert not rt.mapping_engine.suspended


def test_motion_name_collision_with_static_custom(env, monkeypatch):
    from test_motion import performances, zigzag
    from test_custom_gestures import record, ROCK
    rt, bridge, win = env
    warned = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: warned.append(a[2]))
    rt.custom_store.add(record(ROCK).build("shared_name"))
    dlg = MotionRecorderDialog(win, rt, bridge)
    for seg in performances(zigzag, 9):
        bridge.report.emit(seg_report(seg))
    dlg.name.setText("shared_name")
    dlg._save()
    assert warned and "static gesture" in warned[-1] and not dlg.result()
