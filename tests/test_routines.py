import json
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from orbit.brain.context import ContextStore
from orbit.brain.providers import LLMReply, ProviderUnavailable
from orbit.brain.router import Router
from orbit.brain.routines import (FetchError, Routine, RoutineRunner, RoutineScheduler, clean_meals, fetch_text,
                                  heuristic_menu, html_to_text, parse_routines)
from orbit.errors import ConfigError
from orbit.settings import BrainSettings

SAT = datetime(2026, 10, 17, 16, 42)          # a Saturday


# ------------------------------------------------------------ local web server
PAGES = {}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        page = PAGES.get(self.path)
        if page is None:
            self.send_response(404); self.send_header("Content-Length", "0"); self.end_headers(); return
        ctype, body = page
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def web():
    PAGES.clear()
    s = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{s.server_address[1]}"
    s.shutdown(); s.server_close()


WEEKLY_ROWS = """<html><head><title>x</title><script>var a=1;</script></head><body><h1>Mess Menu</h1>
<table><tr><th>Day</th><th>Breakfast</th><th>Lunch</th><th>Snacks</th><th>Dinner</th></tr>
<tr><td>Monday</td><td>Poha</td><td>Rice, Rajma</td><td>Bhel</td><td>Roti, Dal</td></tr>
<tr><td>Saturday</td><td>Idli, Sambar</td><td>Veg Biryani + Raita</td><td>Samosa, Tea</td><td>Chapati &amp; Aloo</td></tr>
<tr><td>Sunday</td><td>Puri</td><td>Chole</td><td>Cake</td><td>Pasta</td></tr></table></body></html>"""

WEEKLY_COLS = """<table><tr><th></th><th>Mon</th><th>Tue</th><th>Sat</th></tr>
<tr><td>Breakfast</td><td>Poha</td><td>Upma</td><td>Idli, Sambar</td></tr>
<tr><td>Lunch</td><td>Rice</td><td>Dal</td><td>Biryani</td></tr>
<tr><td>Snacks</td><td>Bhel</td><td>Vada</td><td>Samosa, Tea</td></tr>
<tr><td>Dinner</td><td>Roti</td><td>Rice</td><td>Chapati, Aloo</td></tr></table>"""

PLAIN = "Today's Menu\nBreakfast: Idli, Sambar\nLunch - Rice; Dal\nEvening Snacks: Samosa, Tea\nDinner | Chapati, Aloo\n"


# ------------------------------------------------------------ html / heuristics
def test_html_to_text_tables_and_noise():
    t = html_to_text(WEEKLY_ROWS)
    assert "var a" not in t and "Mess Menu" in t
    assert "Saturday | Idli, Sambar | Veg Biryani + Raita | Samosa, Tea | Chapati & Aloo" in t.splitlines()


@pytest.mark.parametrize("page", [WEEKLY_ROWS, WEEKLY_COLS])
def test_weekly_tables_pick_todays_column_or_row(page):
    m = heuristic_menu(html_to_text(page), "saturday")
    assert m["snacks"] == ["Samosa", "Tea"] and m["breakfast"] == ["Idli", "Sambar"]
    assert "Biryani" in " ".join(m["lunch"]) or m["lunch"] == ["Biryani"]
    assert heuristic_menu(html_to_text(page), "sunday") != m or page is WEEKLY_COLS


def test_weekly_cols_day_missing_returns_none_instead_of_guessing():
    assert heuristic_menu(html_to_text(WEEKLY_COLS), "wednesday") is None


def test_plain_lines():
    m = heuristic_menu(PLAIN, "saturday")
    assert m == {"breakfast": ["Idli", "Sambar"], "lunch": ["Rice", "Dal"], "snacks": ["Samosa", "Tea"], "dinner": ["Chapati", "Aloo"]}


def test_multi_day_prose_is_not_guessed():
    text = "Mon: Breakfast: a. Lunch: b\nTue Breakfast: c\nWed Breakfast: d\nSnacks: x\nDinner: y"
    assert heuristic_menu(text, "saturday") is None


def test_clean_meals_sanitises():
    m = clean_meals({"Breakfast": ["Idli", "idli", " Dosa\x00 ", "x" * 100, 5, ""], "Tea": ["Chai"], "weird": ["a"], "dinner": []})
    assert m["breakfast"] == ["Idli", "Dosa", "x" * 60] and m["snacks"] == ["Chai"] and "weird" not in m and "dinner" not in m
    assert len(clean_meals({"lunch": [str(i) for i in range(50)]})["lunch"]) == 20
    for bad in ({}, [], "x", {"lunch": "rice"}):
        with pytest.raises(ConfigError):
            clean_meals(bad)


def test_parse_routines():
    ok, err = parse_routines([{"name": "mess_menu", "kind": "menu", "url": "https://mess.example.edu/menu", "refresh_hours": 4}])
    assert [r.name for r in ok] == ["mess_menu"] and err == []
    bad = [{"name": "Bad Name", "url": "https://x.y"}, {"name": "ok_name", "url": "file:///etc/passwd"}, {"name": "ok_name", "url": "https://x.y", "command": "rm -rf /"},
           {"name": "ok_name", "kind": "shell", "url": "https://x.y"}, {"name": "ok_name", "url": "https://x.y", "refresh_hours": 0}, "junk"]
    ok, err = parse_routines(bad)
    assert ok == [] and len(err) == 6


# ------------------------------------------------------------ fetching
def test_fetch_html_and_text(web):
    PAGES["/a"] = ("text/html; charset=utf-8", WEEKLY_ROWS)
    PAGES["/b"] = ("text/plain", PLAIN)
    assert "Saturday | Idli, Sambar" in fetch_text(web + "/a")
    assert fetch_text(web + "/b") == PLAIN


def test_fetch_errors(web):
    PAGES["/pdf"] = ("application/pdf", b"%PDF-1.4")
    PAGES["/big"] = ("text/plain", "x" * 5000)
    with pytest.raises(FetchError, match="404"):
        fetch_text(web + "/missing")
    with pytest.raises(FetchError, match="PDF"):
        fetch_text(web + "/pdf")
    with pytest.raises(FetchError, match="too large"):
        fetch_text(web + "/big", max_bytes=1000)
    with pytest.raises(FetchError, match="http"):
        fetch_text("file:///etc/passwd")
    with pytest.raises(FetchError, match="could not reach"):
        fetch_text("http://127.0.0.1:1/x", timeout=1)


# ------------------------------------------------------------ runner
class Model:
    name = "gemini"

    def __init__(self, text=None, exc=None):
        self.text, self.exc, self.calls = text, exc, []

    def configured(self):
        return True

    def generate(self, req):
        self.calls.append(req)
        if self.exc:
            raise self.exc
        return LLMReply(self.text, "gemini", "m")


@pytest.fixture
def store(tmp_path):
    s = ContextStore(tmp_path / "r.db")
    yield s
    s.close()


def runner(store, model, pages):
    router = Router(BrainSettings(hedge_after_ms=0), store, model, None, None, today=lambda: "2026-10-17")
    return RoutineRunner(store, router, now=lambda: SAT, fetch=lambda url: pages[url]), router


R1 = Routine("mess_menu", "menu", "http://m/x")


def doc(store):
    row = store.get_doc("mess_menu")
    return json.loads(row[1]) if row else None


def test_heuristic_path_makes_no_model_call(store):
    m = Model("unused")
    r, _ = runner(store, m, {"http://m/x": html_to_text(WEEKLY_ROWS)})
    res = r.run(R1)
    assert res.ok and res.parsed_by == "heuristic" and m.calls == []
    d = doc(store)
    assert d["date"] == "2026-10-17" and d["meals"]["snacks"] == ["Samosa", "Tea"]


def test_model_fallback_for_unstructured_page(store):
    m = Model('```json\n{"breakfast": ["Idli"], "lunch": [], "snacks": ["Samosa", "Tea"], "dinner": ["Roti"]}\n```')
    r, _ = runner(store, m, {"http://m/x": "Welcome students!\nToday we serve idli for brekkie, samosa and tea in the evening, roti at night."})
    res = r.run(R1)
    assert res.ok and res.parsed_by == "model"
    assert doc(store)["meals"]["snacks"] == ["Samosa", "Tea"]
    sent = m.calls[0]
    assert sent.json_mode and "Saturday 2026-10-17" in sent.system and "idli for brekkie" in sent.messages[0][1]


def test_unparseable_page_keeps_raw_text_for_the_model_to_read(store):
    r, _ = runner(store, Model("not json at all"), {"http://m/x": "some page without structure"})
    res = r.run(R1)
    assert res.ok and res.parsed_by == "none"
    d = doc(store)
    assert d["meals"] == {} and "some page" in d["raw"]


def test_models_down_still_saves_raw(store):
    r, _ = runner(store, Model(exc=ProviderUnavailable("net")), {"http://m/x": "unstructured"})
    assert r.run(R1).parsed_by == "none"


def test_bad_parse_never_overwrites_todays_good_menu(store):
    pages = {"http://m/x": html_to_text(WEEKLY_ROWS)}
    r, _ = runner(store, Model("x"), pages)
    r.run(R1)
    good = doc(store)
    pages["http://m/x"] = "page layout changed completely"
    res = r.run(R1)
    assert not res.ok and doc(store) == good


def test_fetch_failure_and_empty_page(store):
    def boom(url):
        raise FetchError("could not reach m")
    router = Router(BrainSettings(), store, None, None, None)
    assert RoutineRunner(store, router, now=lambda: SAT, fetch=boom).run(R1).ok is False
    assert RoutineRunner(store, router, now=lambda: SAT, fetch=lambda u: "  \n").run(R1).message == "page was empty"
    assert store.get_doc("mess_menu") is None


def test_end_to_end_over_real_http(store, web):
    PAGES["/menu"] = ("text/html", WEEKLY_ROWS)
    router = Router(BrainSettings(), store, None, None, None)
    res = RoutineRunner(store, router, now=lambda: SAT).run(Routine("mess_menu", "menu", web + "/menu"))
    assert res.ok and doc(store)["meals"]["dinner"] == ["Chapati", "Aloo"]


# ------------------------------------------------------------ scheduler
class Net:
    def __init__(self, online=True):
        self.online, self.listeners = online, []

    def add_listener(self, cb):
        self.listeners.append(cb)


class FakeRunner:
    def __init__(self, ok=True):
        self.ok, self.calls = ok, []

    def run(self, r):
        from orbit.brain.routines import RoutineResult
        self.calls.append(r.name)
        return RoutineResult(self.ok, "Menu updated" if self.ok else "could not reach m")


def sched(store, runner, net=None, now=SAT, events=None, clock=None):
    return RoutineScheduler([R1], store, runner, net, now=lambda: now, clock=clock or (lambda: 1000.0),
                            on_event=(events.append if events is not None else None))


def test_runs_when_no_menu_then_not_again_same_day(store):
    ev, fr = [], FakeRunner()
    s = sched(store, fr, Net(True), events=ev)
    assert s.tick(sync=True) == ["mess_menu"]
    store.put_doc("mess_menu", "2026-10-17", "{}", SAT.timestamp())   # the runner would have saved this
    assert s.tick(sync=True) == [] and fr.calls == ["mess_menu"] and ev == ["Menu updated"]


def test_refreshes_when_stale_or_new_day(store):
    fr = FakeRunner()
    store.put_doc("mess_menu", "2026-10-17", "{}", SAT.timestamp() - 7 * 3600)
    assert sched(store, fr).tick(sync=True) == ["mess_menu"]           # older than refresh_hours (6)
    store.put_doc("mess_menu", "2026-10-16", "{}", SAT.timestamp() - 60)
    assert sched(store, fr).tick(sync=True) == ["mess_menu"]           # yesterday's menu
    store.put_doc("mess_menu", "2026-10-17", "{}", SAT.timestamp() - 60)
    assert sched(store, fr).tick(sync=True) == []


def test_offline_skips_and_online_event_wakes(store):
    fr, net = FakeRunner(), Net(False)
    s = sched(store, fr, net)
    assert s.tick(sync=True) == [] and fr.calls == []
    net.online = True
    net.listeners[0](True)
    assert s._wake.is_set() and s.tick(sync=True) == ["mess_menu"]


def test_failure_backs_off_then_retries(store):
    now = [1000.0]
    ev, fr = [], FakeRunner(ok=False)
    s = sched(store, fr, Net(True), events=ev, clock=lambda: now[0])
    s.tick(sync=True)
    assert ev == ["mess_menu: could not reach m"]
    assert s.tick(sync=True) == []               # inside the 30 s backoff
    now[0] += 31
    assert s.tick(sync=True) == ["mess_menu"]
    now[0] += 60
    assert s.tick(sync=True) == []               # second failure: 120 s backoff
    now[0] += 61
    assert s.tick(sync=True) == ["mess_menu"] and len(fr.calls) == 3


def test_crashing_runner_is_contained(store):
    class Boom:
        def run(self, r):
            raise RuntimeError("bug")
    ev = []
    s = sched(store, Boom(), Net(True), events=ev)
    s.tick(sync=True)
    assert ev and "error" in ev[0]


def test_disabled_routine_and_run_now(store):
    fr = FakeRunner()
    s = RoutineScheduler([Routine("mess_menu", "menu", "http://m/x", enabled=False)], store, fr, Net(True), now=lambda: SAT)
    assert s.tick(sync=True) == []
    assert s.run_now("mess_menu").ok
    with pytest.raises(ConfigError):
        s.run_now("nope")
