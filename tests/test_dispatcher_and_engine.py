import time

import pytest

from orbit.actions.base import ActionContext
from orbit.actions.builtin import build_default_registry
from orbit.actions.focus import FocusController
from orbit.actions.platform import detect_platform
from orbit.dispatcher import ActionDispatcher, ActionRequest
from orbit.mapping import Mapping, MappingStore
from orbit.runtime.mapping_engine import MappingEngine
from orbit.safety import SafetyController
from orbit.vision.types import GestureEvent
from fakes import FakeKeyboard


class Clock:
    t = 100.0

    def __call__(self):
        return self.t


class Rig:
    def __init__(self, tmp_path):
        self.kb = FakeKeyboard()
        self.focus = FocusController()
        self.focus_log = []
        self.focus.add_listener(lambda on, o: self.focus_log.append(on))
        self.reg = build_default_registry()
        ctx = ActionContext(self.kb, self.focus, detect_platform(platform="win32"), ("alt+f4",))
        self.safety = SafetyController(armed=False)
        self.results = []
        self.disp = ActionDispatcher(self.reg, ctx, self.safety, on_result=self.results.append)
        self.disp.start()
        self.store = MappingStore(self.reg, tmp_path / "m.json")
        self.store.load()
        self.clock = Clock()
        self.engine = MappingEngine(self.store, self.disp, self.safety, clock=self.clock)
        self.safety.add_listener(self._on_safety)

    def _on_safety(self, armed, reason):
        if not armed:
            self.disp.purge()
            self.engine.release_all(reason)

    def press(self, g):
        return self.engine.handle(GestureEvent("pressed", g, "hand", 0))

    def release(self, g):
        return self.engine.handle(GestureEvent("released", g, "hand", 0))

    def settle(self):
        assert self.disp.join_idle(2.0)

    def close(self):
        self.disp.stop()


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    yield r
    r.close()


def add(rig, **kw):
    base = dict(id="x", gesture="fist", action="log_message", mode="trigger", params={}, enabled=True, cooldown_ms=0)
    base.update(kw)
    rig.store.add(Mapping(**base))


# ---------------------------------------------------------------- hold (the required initial mapping)
def test_two_fingers_focus_hold(rig):
    rig.safety.arm()
    d = rig.press("two_fingers")
    rig.settle()
    assert d[0].executed and rig.focus.active
    assert rig.press("two_fingers") == []          # no repeat while held
    rig.release("two_fingers")
    rig.settle()
    assert not rig.focus.active
    assert rig.focus_log == [True, False]


def test_disarmed_is_dry_run(rig):
    d = rig.press("two_fingers")
    rig.settle()
    assert not d[0].executed and "dry run" in d[0].reason
    assert not rig.focus.active and rig.focus_log == []


def test_unmapped_gesture_ignored(rig):
    rig.safety.arm()
    assert rig.press("one_finger") == []


# ---------------------------------------------------------------- toggle / trigger / cooldown
def test_toggle(rig):
    rig.safety.arm()
    rig.store.remove("focus-two-fingers")
    add(rig, id="t", gesture="open_palm", action="focus", mode="toggle", cooldown_ms=0)
    rig.press("open_palm"); rig.settle()
    assert rig.focus.active
    rig.release("open_palm"); rig.settle()
    assert rig.focus.active                         # release does nothing in toggle mode
    rig.press("open_palm"); rig.settle()
    assert not rig.focus.active


def test_trigger_and_cooldown(rig):
    rig.safety.arm()
    add(rig, id="s", gesture="one_finger", action="snap_window_left", mode="trigger", cooldown_ms=1000)
    rig.press("one_finger"); rig.settle()
    d = rig.press("one_finger")
    assert d[0].reason == "cooldown"
    rig.clock.t += 1.5
    rig.press("one_finger"); rig.settle()
    assert rig.kb.combos == [("win", "left")] * 2


def test_hold_flicker_blocked_by_cooldown(rig):
    rig.safety.arm()
    rig.press("two_fingers"); rig.settle()
    rig.release("two_fingers"); rig.settle()
    d = rig.press("two_fingers")                    # immediately again: inside 500 ms cooldown
    assert d[0].reason == "cooldown"


# ---------------------------------------------------------------- safety
def test_disarm_releases_held_actions(rig):
    rig.safety.arm()
    rig.press("two_fingers"); rig.settle()
    assert rig.focus.active
    rig.safety.emergency_stop()
    rig.settle()
    assert not rig.focus.active and rig.engine.active_ids() == []


def test_suspended_blocks_everything(rig):
    rig.safety.arm()
    rig.engine.set_suspended(True)
    d = rig.press("two_fingers"); rig.settle()
    assert not d[0].executed and not rig.focus.active


def test_dispatcher_blocks_activate_when_disarmed_but_allows_deactivate(rig):
    req = ActionRequest("m", "focus", "activate", {"dim": 30})
    assert rig.disp.submit(req) is False
    assert rig.disp.submit(req, force=True) is True
    rig.settle()
    assert rig.focus.active
    assert rig.disp.submit(ActionRequest("m", "focus", "deactivate", {})) is True
    rig.settle()
    assert not rig.focus.active


def test_failing_action_is_contained(rig):
    rig.safety.arm()
    rig.store.remove("focus-two-fingers")
    add(rig, id="s", gesture="one_finger", action="snap_window_left", mode="trigger")
    rig.kb.fail = True
    rig.press("one_finger"); rig.settle()
    assert rig.results and not rig.results[-1].ok and "boom" in rig.results[-1].error
    rig.kb.fail = False
    rig.clock.t += 5
    rig.press("one_finger"); rig.settle()           # worker thread still alive
    assert rig.results[-1].ok


def test_unregistered_action_request_rejected_at_runtime(rig):
    rig.disp.submit(ActionRequest("m", "powershell", "activate", {}), force=True)
    rig.settle()
    assert not rig.results[-1].ok and "not registered" in rig.results[-1].error


def test_slow_action_does_not_block_caller(rig):
    class Slow(FakeKeyboard):
        def press_combo(self, combo):
            time.sleep(0.3)
    rig.disp.ctx.keyboard = Slow()
    rig.safety.arm()
    add(rig, id="s", gesture="one_finger", action="snap_window_left", mode="trigger")
    t0 = time.perf_counter()
    rig.press("one_finger")
    assert time.perf_counter() - t0 < 0.1           # handle() returns immediately
    rig.settle()


def test_safety_listener_failure_is_isolated():
    s = SafetyController()
    s.add_listener(lambda a, r: 1 / 0)
    got = []
    s.add_listener(lambda a, r: got.append(a))
    s.arm()
    assert got == [True]
