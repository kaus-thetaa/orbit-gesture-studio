import json
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from orbit.brain.assistant import Assistant
from orbit.brain.context import ContextStore
from orbit.brain.providers import GeminiProvider, LLMRequest
from orbit.brain.router import Router
from orbit.brain.routines import (NoMenuForDate, Routine, RoutineRunner, RoutineScheduler, api_menu, parse_routines)
from orbit.brain.skills import fmt_range
from orbit.errors import ConfigError
from orbit.settings import BrainSettings
from fake_llm_server import FakeServer, STATE

SUN = datetime(2026, 10, 18, 16, 42)           # a Sunday
TODAY = "2026-10-18"


def meal(name, st, en, items):
    return {"name": name, "startTime": st, "endTime": en, "items": items}


def payload(dates=(TODAY,)):
    day = lambda d: {"day": "Sunday", "meals": {
        "breakfast": meal("Breakfast", "07:30", "09:30", ["Idli", "Sambar", "Chutney"]),
        "lunch": meal("Lunch", "12:30", "14:30", ["Rice", "Dal", "Paneer"]),
        "snacks": meal("Snacks", "16:30", "17:30", ["Samosa", "Tea"]),
        "dinner": meal("Dinner", "19:30", "21:30", ["Chapati", "Aloo"])}}
    return {"foodCourt": "Food Court 2", "week": "October 12 - 18, 2026", "menu": {d: day(d) for d in dates}}


# ------------------------------------------------------------ parsing
def test_api_menu_parses_spec_shape():
    meals, times, meta = api_menu(json.dumps(payload()), TODAY)
    assert meals["snacks"] == ["Samosa", "Tea"] and meals["breakfast"] == ["Idli", "Sambar", "Chutney"]
    assert times == {"breakfast": "07:30-09:30", "lunch": "12:30-14:30", "snacks": "16:30-17:30", "dinner": "19:30-21:30"}
    assert meta == {"food_court": "Food Court 2", "week": "October 12 - 18, 2026"}


def test_api_menu_errors_are_specific():
    with pytest.raises(NoMenuForDate):
        api_menu(json.dumps(payload(("2026-10-17",))), TODAY)
    with pytest.raises(ConfigError, match="not JSON"):
        api_menu("<html>maintenance</html>", TODAY)
    with pytest.raises(ConfigError, match="shape"):
        api_menu('{"unexpected": 1}', TODAY)
    with pytest.raises(NoMenuForDate):
        api_menu(json.dumps({"menu": {TODAY: {"day": "Sunday", "meals": "closed"}}}), TODAY)


def test_api_menu_sanitises_and_ignores_unknown():
    p = payload()
    p["menu"][TODAY]["meals"]["snacks"]["startTime"] = "25:99"                       # invalid time -> dropped, items kept
    p["menu"][TODAY]["meals"]["brunch"] = meal("Brunch", "10:00", "11:00", ["x"])    # unknown meal -> ignored
    p["menu"][TODAY]["meals"]["lunch"]["items"] = ["Rice", "rice", "<b>Dal</b>\x00", 5, "y" * 90]
    meals, times, _ = api_menu(json.dumps(p), TODAY)
    assert "snacks" not in times and meals["snacks"] == ["Samosa", "Tea"] and "brunch" not in meals
    assert meals["lunch"][:2] == ["Rice", "<b>Dal</b>"] and len(meals["lunch"][2]) == 60


def test_parse_routines_accepts_menu_api():
    ok, err = parse_routines([{"name": "mess_menu", "kind": "menu_api", "url": "https://tikm.coolstuff.work"}])
    assert ok and not err


# ------------------------------------------------------------ runner against a local server shaped like the API
class Api(BaseHTTPRequestHandler):
    days = {}
    status = 200
    seen = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urlparse(self.path)
        Api.seen.append((u.path, parse_qs(u.query)))
        if Api.status != 200:
            self.send_response(Api.status); self.send_header("Content-Length", "0"); self.end_headers(); return
        date = parse_qs(u.query).get("date", [None])[0]
        body = json.dumps({"foodCourt": "FC2", "week": "w", "menu": {d: v for d, v in Api.days.items() if d == date}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def api():
    Api.days, Api.status, Api.seen = payload()["menu"], 200, []
    s = ThreadingHTTPServer(("127.0.0.1", 0), Api)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{s.server_address[1]}"
    s.shutdown(); s.server_close()


@pytest.fixture
def store(tmp_path):
    s = ContextStore(tmp_path / "m.db")
    yield s
    s.close()


def runner(store):
    return RoutineRunner(store, Router(BrainSettings(), store, None, None, None), now=lambda: SUN)


def test_runner_fetches_todays_menu_from_api(store, api):
    res = runner(store).run(Routine("mess_menu", "menu_api", api))
    assert res.ok and res.parsed_by == "api"
    assert Api.seen == [("/api/menu", {"date": [TODAY]})]            # one small request, today only
    d = json.loads(store.get_doc("mess_menu")[1])
    assert d["meals"]["snacks"] == ["Samosa", "Tea"] and d["times"]["snacks"] == "16:30-17:30" and d["parsed_by"] == "api"


def test_runner_api_failures(store, api):
    Api.status = 500
    r = runner(store).run(Routine("mess_menu", "menu_api", api))
    assert not r.ok and "500" in r.message
    Api.status, Api.days = 200, {}
    r = runner(store).run(Routine("mess_menu", "menu_api", api))
    assert not r.ok and "no menu published" in r.message and r.retry_after_s == 3600
    assert store.get_doc("mess_menu") is None
    r = RoutineRunner(store, Router(BrainSettings(), store, None, None, None), now=lambda: SUN, fetch=lambda u: "<html>x</html>").run(Routine("mess_menu", "menu_api", api))
    assert not r.ok and "API changed" in r.message


def test_no_menu_today_waits_an_hour_not_thirty_seconds(store):
    from orbit.brain.routines import RoutineResult
    now = [1000.0]

    class R:
        calls = 0

        def run(self, r):
            R.calls += 1
            return RoutineResult(False, "no menu published for 2026-10-18", retry_after_s=3600.0)

    s = RoutineScheduler([Routine("mess_menu", "menu_api", "http://x")], store, R(), None, now=lambda: SUN, clock=lambda: now[0])
    s.tick(sync=True)
    now[0] += 600
    assert s.tick(sync=True) == []
    now[0] += 3100
    assert s.tick(sync=True) == ["mess_menu"] and R.calls == 2


# ------------------------------------------------------------ answers with times
@pytest.mark.parametrize("rng,expected", [("16:30-17:30", "4:30-5:30 PM"), ("07:30-09:30", "7:30-9:30 AM"),
                                          ("11:30-13:30", "11:30 AM-1:30 PM"), ("00:00-01:00", "12:00-1:00 AM"), ("bad", "bad")])
def test_fmt_range(rng, expected):
    assert fmt_range(rng) == expected


def assistant(store, now=SUN):
    cfg = BrainSettings()
    return Assistant(cfg, store, Router(cfg, store, None, None, None, today=lambda: TODAY), now=lambda: now)


def saved(store, api_url=None):
    meals, times, _ = api_menu(json.dumps(payload()), TODAY)
    store.put_doc("mess_menu", TODAY, json.dumps({"date": TODAY, "meals": meals, "times": times}))


def test_answers_include_times(store):
    saved(store)
    a = assistant(store)
    assert a.ask("what's for snacks").text == "Snacks 4:30-5:30 PM: Samosa, Tea"
    assert a.ask("what's for breakfast").text == "Breakfast 7:30-9:30 AM: Idli, Sambar, Chutney"
    assert a.ask("when is snacks").text == "Snacks 4:30-5:30 PM"
    assert a.ask("what time is dinner").text == "Dinner 7:30-9:30 PM"
    assert a.ask("what's the mess menu today").text.startswith("B: Idli, Sambar, Chutney | L: Rice, Dal, Paneer")


@pytest.mark.parametrize("hour,minute,expected", [
    (16, 42, "Now - Snacks 4:30-5:30 PM: Samosa, Tea"),
    (10, 0, "Next - Lunch 12:30-2:30 PM: Rice, Dal, Paneer"),
    (6, 0, "Next - Breakfast 7:30-9:30 AM: Idli, Sambar, Chutney"),
    (22, 0, "No more meals today"),
])
def test_whats_next_and_now(store, hour, minute, expected):
    saved(store)
    a = assistant(store, datetime(2026, 10, 18, hour, minute))
    assert a.ask("what's the next meal").text == expected
    assert a.ask("what food is being served now").text == expected


def test_whats_next_ignores_non_food_questions(store):
    saved(store)
    cfg = BrainSettings()
    from test_brain import Fake, Net
    g = Fake("gemini", "model answer")
    a = Assistant(cfg, store, Router(cfg, store, g, None, Net(True), today=lambda: TODAY), now=lambda: SUN)
    assert a.ask("what's next in my life").source == "gemini"


def test_context_line_has_times(store):
    saved(store)
    assert "snacks (16:30-17:30): Samosa, Tea" in assistant(store).menu.context_line()


# ------------------------------------------------------------ gemini: persisted thinking setting + bench + CLI
def test_thinking_variant_is_remembered_across_runs(tmp_path):
    with FakeServer() as srv:
        STATE.reject_thinking = {"minimal", "low"}
        state = tmp_path / "gemini_state.json"
        g1 = GeminiProvider("gemini-3.1-flash-lite", lambda: "test-key", srv.url, 3.0, state_path=state)
        g1.generate(LLMRequest("s", [("user", "x")]))
        assert g1.last_attempts == 3 and json.loads(state.read_text()) == {"gemini-3.1-flash-lite": 2}
        g2 = GeminiProvider("gemini-3.1-flash-lite", lambda: "test-key", srv.url, 3.0, state_path=state)   # a new CLI run
        g2.generate(LLMRequest("s", [("user", "x")]))
        assert g2.last_attempts == 1                                  # no wasted round trips
        bad = tmp_path / "bad.json"
        bad.write_text("{nope")
        GeminiProvider("m", lambda: "k", srv.url, state_path=bad)     # corrupt state is ignored, not fatal


def test_cli_bench_and_add_menu(tmp_path, monkeypatch, capsys):
    with FakeServer() as srv:
        monkeypatch.setenv("ORBIT_HOME", str(tmp_path))
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        (tmp_path / "settings.json").write_text(json.dumps({"brain": {"gemini_base_url": srv.url, "ollama_host": srv.url}}))
        from orbit.cli import main
        assert main(["brain", "bench", "-n", "3"]) == 0
        out = capsys.readouterr().out
        assert "gemini : gemini-3.1-flash-lite  first" in out and "ollama :" in out and "median" in out and "reuses the connection" in out
        assert main(["routine", "add-menu", "https://tikmit.com/"]) == 0
        from orbit.settings import Settings
        r = Settings.load().brain.routines[0]
        assert r["kind"] == "menu_api" and r["url"] == "https://tikm.coolstuff.work"
        assert main(["routine", "add-menu", "https://example.edu/mess"]) == 0
        assert Settings.load().brain.routines[0]["kind"] == "menu"
