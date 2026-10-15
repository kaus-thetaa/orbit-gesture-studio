import json

import pytest

from orbit.actions.builtin import build_default_registry
from orbit.errors import ConfigError
from orbit.mapping import DEFAULT_MAPPINGS, Mapping, MappingStore, make_id

DENY = ["alt+f4"]


def store(tmp_path):
    return MappingStore(build_default_registry(), tmp_path / "m.json", denylist=lambda: DENY)


def m(**kw):
    base = dict(id="a", gesture="two_fingers", action="focus", mode="hold", params={}, enabled=True, cooldown_ms=500)
    base.update(kw)
    return Mapping(**base)


def test_first_run_writes_defaults(tmp_path):
    s = store(tmp_path)
    loaded = s.load()
    assert [x.id for x in loaded] == [d["id"] for d in DEFAULT_MAPPINGS]
    assert (tmp_path / "m.json").exists()
    # the initial required mapping: two_fingers -> focus -> hold
    d = s.for_gesture("two_fingers")
    assert (d.action, d.mode, d.enabled) == ("focus", "hold", True)
    assert s.for_gesture("open_palm") is None  # shipped disabled


def test_roundtrip(tmp_path):
    s = store(tmp_path)
    s.load()
    s.add(m(id="snap", gesture="one_finger", action="snap_window_left", mode="trigger"))
    s2 = store(tmp_path)
    ids = [x.id for x in s2.load()]
    assert "snap" in ids


@pytest.mark.parametrize("kw", [
    dict(action="rm_rf"),
    dict(mode="hold", action="snap_window_left"),
    dict(id="Bad Id!"),
    dict(gesture="Two Fingers"),
    dict(cooldown_ms=-1),
    dict(cooldown_ms=10**6),
    dict(cooldown_ms="500"),
    dict(enabled="yes"),
    dict(params={"dim": 999}),
    dict(action="keyboard_shortcut", mode="trigger", params={"keys": "alt+f4"}),
    dict(action="keyboard_shortcut", mode="trigger", params={"keys": "ctrl+t; calc"}),
])
def test_invalid_mappings_rejected(tmp_path, kw):
    s = store(tmp_path)
    s.load()
    with pytest.raises(ConfigError):
        s.add(m(id=kw.get("id", "new"), **{k: v for k, v in kw.items() if k != "id"}))


def test_duplicate_enabled_gesture_rejected(tmp_path):
    s = store(tmp_path)
    s.load()
    with pytest.raises(ConfigError):
        s.add(m(id="other"))            # two_fingers already used and enabled
    s.add(m(id="other", enabled=False))  # disabled duplicates are fine
    with pytest.raises(ConfigError):
        s.set_enabled("other", True)


def test_unknown_fields_rejected():
    with pytest.raises(ConfigError):
        Mapping.from_dict({"id": "a", "gesture": "fist", "action": "focus", "mode": "hold", "command": "calc.exe"})


def test_invalid_entries_are_skipped_not_fatal(tmp_path):
    (tmp_path / "m.json").write_text(json.dumps({"version": 1, "mappings": [
        {"id": "ok", "gesture": "fist", "action": "log_message", "mode": "hold"},
        {"id": "evil", "gesture": "open_palm", "action": "powershell", "mode": "trigger"},
        {"id": "evil2", "gesture": "one_finger", "action": "focus", "mode": "hold", "command": "calc"},
        "garbage",
    ]}))
    s = store(tmp_path)
    assert [x.id for x in s.load()] == ["ok"]
    assert len(s.load_errors) == 3


def test_corrupt_file_restores_defaults_and_keeps_backup(tmp_path):
    (tmp_path / "m.json").write_text("{ nope")
    s = store(tmp_path)
    assert s.load()
    assert (tmp_path / "m.json.corrupt").exists()
    assert s.load_errors


def test_update_remove_and_enable(tmp_path):
    s = store(tmp_path)
    s.load()
    s.set_enabled("focus-two-fingers", False)
    assert s.for_gesture("two_fingers") is None
    s.set_enabled("focus-two-fingers", True)
    s.update("focus-two-fingers", m(id="focus-two-fingers", mode="toggle", params={"dim": 50}))
    assert s.get("focus-two-fingers").mode == "toggle"
    assert s.remove("focus-two-fingers") and not s.remove("focus-two-fingers")
    with pytest.raises(ConfigError):
        s.update("nope", m())


def test_make_id_unique():
    assert make_id("fist", "focus", []) == "focus-fist"
    assert make_id("fist", "focus", ["focus-fist"]) == "focus-fist-2"
