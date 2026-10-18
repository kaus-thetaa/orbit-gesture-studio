import numpy as np
import pytest

from orbit.actions.base import ActionContext
from orbit.actions.builtin import build_default_registry
from orbit.actions.focus import FocusController
from orbit.actions.platform import detect_platform
from orbit.actions.audio import UnsupportedAudio, default_audio_backend, PulseAudio
from orbit.brain.core import Brain
from orbit.brain.context import ContextStore
from orbit.errors import ActionUnsupported
from orbit.mapping import Mapping
from orbit.runtime.app import OrbitRuntime
from orbit.runtime.pipeline import Pipeline
from orbit.settings import LipsSettings, Settings
from orbit.vision.lips import MouthTracker, lips_confidence, mouth_geometry
from fake_llm_server import FakeServer, STATE
from fakes import FakeAudio, FakeCamera, FakeKeyboard, FakeTracker
from synth import OPEN, make_hand

# ============================================================ mute action
def ctx(audio):
    return ActionContext(FakeKeyboard(), FocusController(), detect_platform(platform="linux"), (), audio)


def test_mute_hold_cycle_and_idempotence():
    a, reg = FakeAudio(), build_default_registry()
    act = reg.get("mute_audio")
    assert act.modes == ("hold", "toggle", "trigger")
    act.activate(ctx(a), {}); act.activate(ctx(a), {})
    assert a.muted and a.log == [True]
    act.deactivate(ctx(a), {}); act.deactivate(ctx(a), {})
    assert not a.muted and a.log == [True, False]


def test_release_never_unmutes_what_the_user_muted():
    a = FakeAudio(muted=True)
    act = build_default_registry().get("mute_audio")
    act.activate(ctx(a), {}); act.deactivate(ctx(a), {})
    assert a.muted and a.log == []


def test_mute_without_backend_fails_gracefully():
    act = build_default_registry().get("mute_audio")
    with pytest.raises(ActionUnsupported):
        act.activate(ctx(None), {})
    with pytest.raises(ActionUnsupported):
        act.activate(ctx(UnsupportedAudio("darwin")), {})
    assert isinstance(default_audio_backend("linux"), PulseAudio) and isinstance(default_audio_backend("darwin"), UnsupportedAudio)


# ============================================================ lips geometry
def face(cx, cy, width=60.0):
    xy = np.zeros((478, 2))
    xy[61], xy[291] = (cx - width / 2, cy), (cx + width / 2, cy)
    xy[13], xy[14] = (cx, cy - 4), (cx, cy + 4)
    return xy


def test_geometry_and_confidence():
    c, w = mouth_geometry(face(100, 200))
    assert np.allclose(c, (100, 200)) and w == pytest.approx(60)
    assert mouth_geometry(np.zeros((478, 2))) is None and mouth_geometry(np.zeros((10, 2))) is None
    assert lips_confidence(np.array([100.0, 200.0]), c, w, 1.0) == pytest.approx(1.0)
    assert lips_confidence(np.array([100.0, 260.0]), c, w, 1.0) == pytest.approx(0.0)       # one mouth-width away
    assert 0.4 < lips_confidence(np.array([100.0, 245.0]), c, w, 1.0) < 0.6          # 0.75 mouth-widths away


def finger_hand(cx=320, cy=330):
    return make_hand(dict(index=True), center=(cx, cy))


def test_tracker_gating_caching_and_failures():
    h = finger_hand()
    tip = h.landmarks[8, :2]
    calls = []

    def face_fn(frame, ts):
        calls.append(ts)
        return face(tip[0], tip[1])

    t = MouthTracker(LipsSettings(), face_fn)
    assert t.check(None, h, 1000) == pytest.approx(1.0, abs=0.05)
    t.check(None, h, 1033)
    t.check(None, h, 1066)
    assert len(calls) == 1                                 # cached: the face model isn't re-run every frame
    t.check(None, h, 1500)
    assert len(calls) == 2                                 # cache expired
    far = MouthTracker(LipsSettings(), lambda f, ts: face(tip[0], tip[1] - 300))
    assert far.check(None, h, 0) == 0.0                    # fingertip nowhere near the mouth
    tiny_face = MouthTracker(LipsSettings(), lambda f, ts: face(tip[0], tip[1], width=3))
    assert tiny_face.check(None, h, 0) is None
    wrong_depth = MouthTracker(LipsSettings(), lambda f, ts: face(tip[0], tip[1], width=400))   # hand tiny vs face -> not "on" lips
    assert wrong_depth.check(None, h, 0) is None
    broken = MouthTracker(LipsSettings(), lambda f, ts: 1 / 0)
    assert broken.check(None, h, 0) is None
    assert MouthTracker(LipsSettings(), lambda f, ts: None).check(None, h, 0) is None


# ============================================================ pipeline: finger on lips -> mute
@pytest.fixture
def rig():
    s = Settings()
    s.safety.enable_hotkeys = False
    s.recognition.confirm_frames, s.recognition.release_frames = 3, 3
    s.motion.enabled = False
    audio = FakeAudio()
    rt = OrbitRuntime(settings=s, keyboard=FakeKeyboard(), audio=audio)
    rt.tracker, rt.camera = FakeTracker(), FakeCamera()
    state = {"mouth": (0.0, 0.0), "calls": 0}

    def face_fn(frame, ts):
        state["calls"] += 1
        return face(*state["mouth"])

    rt.lips = MouthTracker(s.lips, face_fn)
    rt.pipeline = Pipeline(s, rt.camera, rt.tracker, rt.gestures, rt.stabilizer, rt.mapping_engine, lips=rt.lips)
    rt.store.add(Mapping("mute-shh", "finger_on_lips", "mute_audio", "hold", {}, True, 0))
    rt.dispatcher.start()
    yield rt, audio, state
    rt.dispatcher.stop()


def run(rt, hand, n, t0=0):
    frame = np.zeros((480, 640, 3), np.uint8)
    rep = None
    for i in range(n):
        rt.tracker.next = hand
        rep = rt.pipeline.process_frame(frame, t0 + i * 33)
    assert rt.dispatcher.join_idle(2.0)
    return rep


def test_finger_on_lips_mutes_while_held_and_restores(rig):
    rt, audio, state = rig
    rt.safety.arm()
    h = finger_hand()
    state["mouth"] = tuple(h.landmarks[8, :2])
    rep = run(rt, h, 6)
    assert rep.prediction.name == "finger_on_lips" and rep.stable_gesture == "finger_on_lips"
    assert audio.muted
    state["mouth"] = (tuple(h.landmarks[8, :2])[0] + 400, 50)           # finger leaves the mouth area
    run(rt, h, 6, t0=1000)
    assert not audio.muted and audio.log == [True, False]


def test_finger_near_but_not_on_lips_does_not_mute(rig):
    rt, audio, state = rig
    rt.safety.arm()
    h = finger_hand()
    tip = h.landmarks[8, :2]
    state["mouth"] = (tip[0] + 150, tip[1] + 100)
    rep = run(rt, h, 10)
    assert rep.prediction.name == "one_finger" and not audio.muted


def test_face_model_not_run_for_other_poses(rig):
    rt, audio, state = rig
    rt.safety.arm()
    run(rt, make_hand(OPEN), 10)
    assert state["calls"] == 0 and not audio.muted


def test_dry_run_when_disarmed_and_estop_unmutes(rig):
    rt, audio, state = rig
    h = finger_hand()
    state["mouth"] = tuple(h.landmarks[8, :2])
    run(rt, h, 6)
    assert not audio.muted                                              # disarmed: dry run
    run(rt, make_hand(OPEN), 6, t0=1000)                                # lower the finger (arming affects NEW presses only)
    rt.safety.arm()
    run(rt, h, 6, t0=2000)
    assert audio.muted
    rt.safety.emergency_stop()
    assert rt.dispatcher.join_idle(2.0)
    assert not audio.muted


def test_contextual_name_is_reserved_and_listed():
    from orbit.errors import ConfigError
    from orbit.vision.gestures import CustomGesture
    with pytest.raises(ConfigError):
        CustomGesture("finger_on_lips", np.zeros((10, 42))).validate()
    s = Settings(); s.safety.enable_hotkeys = False
    assert "finger_on_lips" in OrbitRuntime(settings=s, keyboard=FakeKeyboard(), audio=FakeAudio()).known_gestures()


# ============================================================ Brain over real HTTP (fake servers)
@pytest.fixture
def brain(tmp_path, monkeypatch):
    with FakeServer() as srv:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        s = Settings()
        s.brain.gemini_base_url, s.brain.ollama_host = srv.url, srv.url
        s.brain.hedge_after_ms = 0
        s.brain.cloud_timeout_s = 2.0
        b = Brain(s, store=ContextStore(tmp_path / "b.db"))
        b.net.check()
        yield b, srv
        b.store.close()


def test_brain_online_then_cloud_fails_then_offline(brain):
    b, srv = brain
    a = b.ask("capital of France?")
    assert a.source == "gemini" and a.text == "Samosa and tea" and "x-goog-api-key" in STATE.requests[-1][2]
    STATE.gemini_status = 503
    a = b.ask("another question")
    assert a.source == "ollama" and a.text == "Tea at 4:30"
    st = b.status()
    assert st["cloud"].startswith("cooldown") and st["local"] == "ok"
    assert b.router.store.usage(b.router.today(), "gemini")[0] == 1


def test_brain_without_key_goes_local(brain, monkeypatch):
    b, srv = brain
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setenv("ORBIT_HOME", str(srv.port))        # point secrets lookup at an empty dir
    import tempfile
    monkeypatch.setenv("ORBIT_HOME", tempfile.mkdtemp())
    assert b.ask("capital of France?").source == "ollama"
    assert b.status()["cloud"] == "no-key"


def test_brain_async_callback(brain):
    import threading
    b, _ = brain
    done, got = threading.Event(), []
    b.ask_async("hello there", lambda a: (got.append(a), done.set()))
    assert done.wait(3) and got[0].text


def test_brain_private_memory_stays_local(brain):
    b, _ = brain
    b.ask("remember privately: my locker code area is block C")
    n_before = len([r for r in STATE.requests if "generateContent" in r[1]])
    a = b.ask("where is my locker?")
    assert a.source == "ollama"
    assert len([r for r in STATE.requests if "generateContent" in r[1]]) == n_before     # nothing went to the cloud


# ============================================================ HUD + autostart + CLI
def test_autostart_entry_roundtrip(tmp_path):
    from orbit import autostart
    p = autostart.enable(tmp_path / "autostart" / "x.desktop", python="/home/me/my env/bin/python")
    text = p.read_text()
    assert 'Exec="/home/me/my env/bin/python" -m orbit hud' in text and "Terminal=false" in text
    assert autostart.disable(p) and not autostart.disable(p)


def test_hud_smoke(brain, monkeypatch):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from orbit.gui.hud import Hud
    b, _ = brain
    s = Settings()
    h = Hud(b, s.hud)
    h.show()
    h.show_text("Samosa, Tea", "menu", 5)
    assert h.width() > s.hud.orb_size + 40
    h.ask("capital of France?")
    for _ in range(50):
        app.processEvents()
        import time; time.sleep(0.05)
        if h.text and h.text != "...":
            break
    assert h.text == "Samosa and tea"                       # Gemini fake's reply
    h._clear_text()
    assert h.width() < 60 and h.text == ""
    h.refresh_status()
    h.close()


def test_cli_memory_and_routine_commands(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ORBIT_HOME", str(tmp_path))
    from orbit.cli import main
    assert main(["memory", "add", "my", "snack", "time", "is", "4:30"]) == 0
    assert main(["memory", "add", "--private", "locker", "in", "block", "C"]) == 0
    assert main(["memory", "add", "my", "password", "is", "x"]) == 1
    assert main(["memory", "list"]) == 0
    out = capsys.readouterr().out
    assert "[shared ] my snack time is 4:30" in out and "[private] locker in block C" in out
    assert main(["routine", "add-menu", "file:///etc/passwd"]) == 1
    assert main(["routine", "add-menu", "https://mess.example.edu/menu"]) == 0
    from orbit.settings import Settings
    assert Settings.load().brain.routines[0]["url"] == "https://mess.example.edu/menu"
