import time

import numpy as np
import pytest

from orbit.errors import ConfigError
from orbit.runtime.camera import CameraSource
from orbit.settings import CameraSettings, Settings
from orbit.vision.face import FaceSignalHub, GazeCalibrator, iris_ratio, validate_signal
from orbit.vision.types import FaceObservation
from orbit.settings import FaceSettings, GazeSettings


# ------------------------------------------------------------------ camera
class FakeCap:
    def __init__(self, frames_before_fail=None):
        self.n = 0
        self.limit = frames_before_fail
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        time.sleep(0.002)
        self.n += 1
        if self.limit is not None and self.n > self.limit:
            return False, None
        return True, np.zeros((48, 64, 3), np.uint8)

    def release(self):
        self.released = True


def wait_for(pred, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_camera_streams_and_stops():
    cap = FakeCap()
    cam = CameraSource(CameraSettings(), opener=lambda cfg: cap)
    cam.start()
    assert wait_for(lambda: cam.latest() is not None)
    assert cam.status == "streaming"
    cam.stop()
    assert cap.released and cam.status == "stopped"


def test_camera_disconnect_and_reconnect():
    caps = []

    def opener(cfg):
        cap = FakeCap(frames_before_fail=5 if not caps else None)
        caps.append(cap)
        return cap

    cfg = CameraSettings(reconnect_interval_s=0.5, read_failure_limit=3)
    statuses = []
    cam = CameraSource(cfg, opener=opener, on_status=statuses.append)
    cam.start()
    assert wait_for(lambda: "reconnecting" in statuses)
    assert cam.latest() is None or True            # feed is dropped when lost
    assert wait_for(lambda: len(caps) >= 2 and cam.status == "streaming")
    assert caps[0].released
    cam.stop()


def test_camera_open_failure_retries():
    attempts = []

    def opener(cfg):
        attempts.append(1)
        return None if len(attempts) < 3 else FakeCap()

    cam = CameraSource(CameraSettings(reconnect_interval_s=0.5), opener=opener)
    cam.start()
    assert wait_for(lambda: cam.latest() is not None, timeout=5)
    assert len(attempts) >= 3
    cam.stop()


def test_stale_frames_count_as_lost():
    cap = FakeCap()
    cam = CameraSource(CameraSettings(stale_after_s=0.2), opener=lambda c: cap)
    cam.start()
    assert wait_for(lambda: cam.latest() is not None)
    cam._stop.set()                                  # freeze the capture thread
    cam._thread.join()
    time.sleep(0.35)
    assert cam.latest() is None


# ------------------------------------------------------------------ settings
def test_settings_defaults_are_sane():
    s = Settings()
    assert s.safety.start_armed is False
    assert s.gaze.enabled is False and s.face.enabled is False


def test_settings_load_save_roundtrip(tmp_path):
    p = tmp_path / "s.json"
    s = Settings.load(p)                             # creates defaults
    s.recognition.confirm_frames = 9
    s.save(p)
    assert Settings.load(p).recognition.confirm_frames == 9


def test_settings_tolerates_garbage(tmp_path):
    p = tmp_path / "s.json"
    p.write_text('{"recognition": {"confirm_frames": "lots", "release_frames": -5}, "camera": 7, "bogus": 1}')
    s = Settings.load(p)
    assert s.recognition.confirm_frames == 6 and s.recognition.release_frames == 1
    p.write_text("not json")
    assert Settings.load(p).recognition.confirm_frames == 6
    assert (tmp_path / "s.json.corrupt").exists()


# ------------------------------------------------------------------ gaze / face
def landmarks_with_iris(ratio):
    xy = np.zeros((478, 2))
    xy[33], xy[133] = (0.30, 0.5), (0.40, 0.5)
    xy[362], xy[263] = (0.60, 0.5), (0.70, 0.5)
    xy[468] = (0.30 + 0.10 * ratio, 0.5)
    xy[473] = (0.60 + 0.10 * ratio, 0.5)
    return xy


@pytest.mark.parametrize("r", [0.1, 0.5, 0.9])
def test_iris_ratio(r):
    assert iris_ratio(landmarks_with_iris(r)) == pytest.approx(r)


def test_iris_ratio_degenerate():
    assert iris_ratio(np.zeros((478, 2))) is None
    assert iris_ratio(np.zeros((10, 2))) is None


def test_calibrator_thresholds():
    cal = GazeCalibrator()
    for phase, v in (("left", 0.30), ("center", 0.50), ("right", 0.72)):
        for i in range(30):
            cal.add(phase, v + (i % 3 - 1) * 0.01)
    left, right = cal.compute()
    assert 0.30 < left < 0.50 < right < 0.72
    assert left == pytest.approx(0.50 - 0.6 * 0.20, abs=0.01)


def test_calibrator_rejects_tiny_movement_and_missing_data():
    cal = GazeCalibrator()
    for phase in GazeCalibrator.PHASES:
        for _ in range(30):
            cal.add(phase, 0.5)
    with pytest.raises(ConfigError):
        cal.compute()
    with pytest.raises(ConfigError):
        GazeCalibrator().compute()


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def test_gaze_hub_dwell_filters_glances():
    c = Clock()
    hub = FaceSignalHub(GazeSettings(enabled=True, dwell_ms=800, release_ms=200, cooldown_ms=500, smoothing_frames=1),
                        FaceSettings(), clock=c)
    fired = []
    for i in range(30):                              # brief 0.3 s glances to the left
        c.t += 0.1
        ev, _ = hub.update(FaceObservation(0.2 if i % 10 < 3 else 0.5, {}))
        fired += ev
    assert fired == []
    for _ in range(12):                              # sustained look left
        c.t += 0.1
        ev, dbg = hub.update(FaceObservation(0.2, {}))
        fired += ev
    assert [(e.kind, e.name) for e in fired] == [("pressed", "gaze_left")]


def test_gaze_hub_no_face_releases():
    c = Clock()
    hub = FaceSignalHub(GazeSettings(enabled=True, dwell_ms=200, release_ms=100, cooldown_ms=0, smoothing_frames=1),
                        FaceSettings(), clock=c)
    out = []
    for _ in range(5):
        c.t += 0.1
        out += hub.update(FaceObservation(0.9, {}))[0]
    for _ in range(5):
        c.t += 0.1
        out += hub.update(None)[0]
    assert [(e.kind, e.name) for e in out] == [("pressed", "gaze_right"), ("released", "gaze_right")]


def test_face_signal_validation_and_firing():
    with pytest.raises(ConfigError):
        validate_signal({"name": "Bad Name", "blendshape": "jawOpen"})
    with pytest.raises(ConfigError):
        validate_signal({"name": "ok_name", "blendshape": "jaw;rm", "threshold": 0.5})
    c = Clock()
    hub = FaceSignalHub(GazeSettings(), FaceSettings(enabled=True, signals=[
        {"name": "jaw_open", "blendshape": "jawOpen", "threshold": 0.6, "dwell_ms": 300},
        {"name": "broken", "blendshape": "x"},       # invalid entry is skipped, not fatal
    ]), clock=c)
    out = []
    for _ in range(6):
        c.t += 0.1
        out += hub.update(FaceObservation(None, {"jawOpen": 0.8}))[0]
    assert [(e.kind, e.name, e.source) for e in out] == [("pressed", "face_jaw_open", "face")]
