import json
import threading
import time
from datetime import datetime

import pytest

from orbit.brain.assistant import Assistant, shorten
from orbit.brain.context import ContextStore, looks_sensitive
from orbit.brain.netmon import NetMonitor
from orbit.brain.providers import (AuthError, LLMReply, LLMRequest, ProviderError, ProviderUnavailable, RateLimited)
from orbit.brain.router import AllProvidersFailed, Router
from orbit.brain.secrets import get_secret, set_secret
from orbit.errors import ConfigError
from orbit.settings import BrainSettings

NOW = datetime(2026, 10, 17, 16, 42)
TODAY = "2026-10-17"


class Fake:
    """A provider double: scripted reply / error / delay."""

    def __init__(self, name, reply="ok", exc=None, delay=0.0, configured=True):
        self.name, self.reply, self.exc, self.delay, self._cfg = name, reply, exc, delay, configured
        self.calls = []

    def configured(self):
        return self._cfg

    def generate(self, req):
        self.calls.append(req)
        time.sleep(self.delay)
        if self.exc:
            raise self.exc
        return LLMReply(self.reply, self.name, self.name + "-model", 30, 5, self.delay * 1000)


class Net:
    def __init__(self, online=True):
        self.online, self.soon = online, 0

    def check_soon(self):
        self.soon += 1


@pytest.fixture
def store(tmp_path):
    s = ContextStore(tmp_path / "b.sqlite3")
    yield s
    s.close()


def mk(store, gemini=None, ollama=None, net=None, **kw):
    cfg = BrainSettings(hedge_after_ms=kw.pop("hedge", 0), **kw)
    clock = kw.pop("clock", None)
    router = Router(cfg, store, gemini, ollama, net, today=lambda: TODAY)
    return cfg, router, Assistant(cfg, store, router, now=lambda: NOW)


REQ = LLMRequest("sys", [("user", "hi")])


# ================================================================ context store
def test_facts_dedupe_forget_and_relevance(store):
    assert store.remember("My snack time is 4:30pm")[1] is True
    assert store.remember("my snack time is 4:30PM")[1] is False
    store.remember("I study mechanical engineering", "private")
    rel = [f.text for f in store.relevant_facts("when is snack time")]
    assert rel == ["My snack time is 4:30pm"]
    assert store.forget("snack") == 1 and [f.text for f in store.facts()] == ["I study mechanical engineering"]


@pytest.mark.parametrize("text", ["my wifi password is hunter2", "my card 4111 1111 1111 1111", "AIzaSyA1234567890abcdef",
                                  "otp is 482913", "x" * 400, ""])
def test_secrets_and_junk_are_refused(store, text):
    with pytest.raises(ConfigError):
        store.remember(text)
    assert store.facts() == []


def test_sensitive_detector_allows_normal_facts():
    for ok in ("I like masala dosa", "snacks are at 4:30", "my room is 214"):
        assert looks_sensitive(ok) is None


def test_turns_expire_and_cap(store):
    store.add_turn("user", "old", now=1000.0)
    store.add_turn("user", "new", now=1990.0)
    assert store.recent_turns(5, max_age_s=100, now=2000.0) == [("user", "new")]
    for i in range(250):
        store.add_turn("user", str(i))
    assert len(store._db.execute("SELECT id FROM turns").fetchall()) == 200


def test_usage_counters(store):
    store.add_usage(TODAY, "gemini", 10, 5)
    store.add_usage(TODAY, "gemini", 20, 5)
    assert store.usage(TODAY, "gemini") == (2, 30, 10) and store.usage("2026-10-18", "gemini") == (0, 0, 0)


def test_store_persists(tmp_path):
    a = ContextStore(tmp_path / "x.db"); a.remember("persist me"); a.put_doc("d", "2026-10-17", "{}"); a.close()
    b = ContextStore(tmp_path / "x.db")
    assert [f.text for f in b.facts()] == ["persist me"] and b.get_doc("d")[0] == "2026-10-17"
    b.close()


# ================================================================ secrets
def test_secret_precedence_and_permissions(tmp_path):
    p = tmp_path / "s.json"
    assert get_secret("K", env={}, path=p) is None
    set_secret("K", " file-key ", p)
    assert get_secret("K", env={}, path=p) == "file-key"
    assert get_secret("K", env={"K": "env-key"}, path=p) == "env-key"
    assert oct(p.stat().st_mode & 0o777) == "0o600"
    p.chmod(0o644)
    assert get_secret("K", env={}, path=p) == "file-key" and oct(p.stat().st_mode & 0o777) == "0o600"


# ================================================================ router
def test_online_uses_gemini_and_counts_usage(store):
    g, o = Fake("gemini", "from cloud"), Fake("ollama", "from local")
    _, r, _ = mk(store, g, o, Net(True))
    assert r.ask(REQ).provider == "gemini" and o.calls == []
    assert store.usage(TODAY, "gemini")[0] == 1


def test_offline_uses_ollama_without_trying_cloud(store):
    g, o = Fake("gemini"), Fake("ollama", "local")
    _, r, _ = mk(store, g, o, Net(False))
    assert r.ask(REQ).provider == "ollama" and g.calls == []
    assert r.cloud_state() == "offline"


def test_no_key_uses_ollama(store):
    g, o = Fake("gemini", configured=False), Fake("ollama")
    _, r, _ = mk(store, g, o, Net(True))
    assert r.ask(REQ).provider == "ollama" and r.cloud_state() == "no-key"


def test_failover_on_cloud_error_and_breaker(store):
    g, o, n = Fake("gemini", exc=ProviderUnavailable("down")), Fake("ollama", "local"), Net(True)
    _, r, _ = mk(store, g, o, n)
    assert r.ask(REQ).provider == "ollama" and n.soon == 1
    assert r.cloud_state().startswith("cooldown")
    r.ask(REQ)
    assert len(g.calls) == 1                         # breaker keeps us off the dead provider


def test_rate_limit_switches_to_local_until_retry_after(store):
    g, o = Fake("gemini", exc=RateLimited("quota", 30)), Fake("ollama")
    clock = [100.0]
    cfg = BrainSettings(hedge_after_ms=0)
    r = Router(cfg, store, g, o, Net(True), clock=lambda: clock[0], today=lambda: TODAY)
    assert r.ask(REQ).provider == "ollama"
    assert r.cloud_state() == "cooldown:rate-limited"
    clock[0] += 31
    g.exc = None
    assert r.ask(REQ).provider == "gemini"


def test_auth_error_cools_down_long(store):
    g, o = Fake("gemini", exc=AuthError("bad key")), Fake("ollama")
    _, r, _ = mk(store, g, o, Net(True))
    r.ask(REQ)
    assert r.breakers["gemini"].open_until - r.clock() > 500


def test_budget_switches_to_local(store):
    g, o = Fake("gemini"), Fake("ollama")
    _, r, _ = mk(store, g, o, Net(True), gemini_daily_requests=10, switch_at_fraction=0.5)
    for _ in range(5):
        store.add_usage(TODAY, "gemini")
    assert r.cloud_state() == "budget" and r.ask(REQ).provider == "ollama"
    store.usage  # new day resets
    r.today = lambda: "2026-10-18"
    assert r.cloud_state() == "ok"


def test_local_only_never_touches_cloud(store):
    g, o = Fake("gemini"), Fake("ollama")
    _, r, _ = mk(store, g, o, Net(True))
    assert r.ask(REQ, local_only=True).provider == "ollama" and g.calls == []
    r2 = Router(BrainSettings(), store, g, None, Net(True), today=lambda: TODAY)
    with pytest.raises(AllProvidersFailed):
        r2.ask(REQ, local_only=True)
    assert g.calls == []


def test_both_fail_raises_with_reasons(store):
    g, o = Fake("gemini", exc=ProviderUnavailable("net")), Fake("ollama", exc=ProviderError("boom"))
    _, r, _ = mk(store, g, o, Net(True))
    with pytest.raises(AllProvidersFailed) as ei:
        r.ask(REQ)
    assert len(ei.value.reasons) == 2


def test_prefer_ollama(store):
    g, o = Fake("gemini"), Fake("ollama")
    _, r, _ = mk(store, g, o, Net(True), prefer="ollama")
    assert r.ask(REQ).provider == "ollama" and g.calls == []


def test_hedging_takes_the_faster_answer(store):
    g, o = Fake("gemini", "slow cloud", delay=1.2), Fake("ollama", "fast local", delay=0.05)
    _, r, _ = mk(store, g, o, Net(True), hedge=150)
    t0 = time.perf_counter()
    rep = r.ask(REQ)
    assert rep.provider == "ollama" and time.perf_counter() - t0 < 0.8
    assert len(g.calls) == 1                         # cloud was asked first and abandoned


def test_no_hedge_when_primary_is_fast(store):
    g, o = Fake("gemini", delay=0.02), Fake("ollama")
    _, r, _ = mk(store, g, o, Net(True), hedge=500)
    r.ask(REQ)
    assert o.calls == []


def test_hedge_still_falls_back_when_primary_errors_late(store):
    g, o = Fake("gemini", exc=ProviderUnavailable("x"), delay=0.3), Fake("ollama", delay=0.05)
    _, r, _ = mk(store, g, o, Net(True), hedge=100)
    assert r.ask(REQ).provider == "ollama"


# ================================================================ shorten
@pytest.mark.parametrize("raw,expected", [
    ("**Samosa** and *tea*", "Samosa and tea"),
    ("Sure! Snacks are samosa.\nAlso chai.", "Snacks are samosa."),
    ("- Item one\n- Item two", "Item one"),
    ("One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen", "One two three four five six seven eight nine ten eleven twelve thirteen fourteen"),
    ("", ""),
])
def test_shorten(raw, expected):
    assert shorten(raw, 14) == expected


def test_shorten_prefers_first_sentence_when_long():
    assert shorten("Snacks today are samosa and tea. They are served from four thirty until five thirty in the main hall ok.", 8) == "Snacks today are samosa and tea."


# ================================================================ assistant
MENU = {"date": TODAY, "meals": {"breakfast": ["Idli", "Sambar"], "lunch": ["Rice", "Dal", "Paneer"],
                                 "snacks": ["Samosa", "Tea"], "dinner": ["Chapati", "Aloo"]}}


def with_menu(store, doc=MENU):
    store.put_doc("mess_menu", doc["date"], json.dumps(doc))


def test_snacks_answered_instantly_without_any_model(store):
    with_menu(store)
    g, o = Fake("gemini"), Fake("ollama")
    _, _, a = mk(store, g, o, Net(True))
    for q in ("what's for snacks?", "whats for snacks", "Snacks today?", "what is there for tea"):
        ans = a.ask(q)
        assert ans.text == "Snacks: Samosa, Tea" and ans.source == "menu" and ans.latency_ms < 50, q
    assert g.calls == [] and o.calls == []


def test_menu_variants(store):
    with_menu(store)
    _, _, a = mk(store, Fake("gemini"), Fake("ollama"), Net(True))
    assert a.ask("what's for lunch").text == "Lunch: Rice, Dal, Paneer"
    assert a.ask("dinner?").text == "Dinner: Chapati, Aloo"
    assert a.ask("what's the mess menu today").text == "B: Idli, Sambar | L: Rice, Dal, Paneer | S: Samosa, Tea | D: Chapati, Aloo"
    assert a.ask("what's for snacks tomorrow").text == "Only today's menu is saved"


def test_menu_missing_stale_and_empty(store):
    _, _, a = mk(store, Fake("gemini"), Fake("ollama"), Net(True))
    assert a.ask("what's for snacks").text == "No menu saved yet"
    with_menu(store, {**MENU, "date": "2026-10-16"})
    assert a.ask("what's for snacks").text == "Snacks: Samosa, Tea (menu from 2026-10-16)"
    with_menu(store, {"date": TODAY, "meals": {"breakfast": ["Idli"]}})
    assert a.ask("snacks?").text == "No snacks listed"


@pytest.mark.parametrize("q", ["how many letters in tea", "I had tea", "do I prefer spicy food?", "anything good for vegetarian lunch tonight?",
                               "which snack has the fewest calories", "what should I eat for dinner to stay healthy and strong today"])
def test_non_menu_questions_go_to_the_model(store, q):
    with_menu(store)
    g = Fake("gemini", "model answer")
    _, _, a = mk(store, g, None, Net(True))
    ans = a.ask(q)
    assert ans.source == "gemini" and ans.text == "model answer"


def test_time_and_date_instant(store):
    _, _, a = mk(store, None, None, Net(True))
    assert a.ask("what time is it").text == "4:42 PM" and a.ask("what's the date").text == "Sat 17 Oct"


def test_model_path_shares_context_between_providers(store):
    store.remember("I am vegetarian")
    with_menu(store)
    g, o = Fake("gemini", "Paneer, rice"), Fake("ollama", "Paneer, rice")
    net = Net(True)
    _, r, a = mk(store, g, o, net)
    a.ask("anything good for vegetarian lunch tonight?")      # not a pure menu question -> model, sees facts + doc
    net.online = False
    r.breakers["gemini"].open_until = 0
    a.ask("any vegetarian dessert?")
    sys_cloud, sys_local = g.calls[0].system, o.calls[0].system
    for sysmsg in (sys_cloud, sys_local):
        assert "I am vegetarian" in sysmsg and "Mess menu (2026-10-17)" in sysmsg and "Saturday 17 October 2026" in sysmsg
    assert ("user", "anything good for vegetarian lunch tonight?") in o.calls[0].messages   # the local model saw the earlier turn too


def test_private_facts_are_never_sent_to_cloud(store):
    store.remember("my roommate is Arjun", "private")
    g, o = Fake("gemini", "cloud"), Fake("ollama", "Arjun")
    _, _, a = mk(store, g, o, Net(True))
    ans = a.ask("who is my roommate?")
    assert ans.source == "ollama" and g.calls == []
    assert "Arjun" in o.calls[0].system
    a2 = Assistant(BrainSettings(hedge_after_ms=0), store, Router(BrainSettings(), store, g, None, Net(True), today=lambda: TODAY), now=lambda: NOW)
    assert a2.ask("who is my roommate?").text == "Private: needs Ollama" and g.calls == []


def test_share_memory_switch_off_keeps_facts_local(store):
    store.remember("I prefer spicy food")
    g, o = Fake("gemini"), Fake("ollama", "ok")
    _, _, a = mk(store, g, o, Net(True), share_memory_with_cloud=False)
    assert a.ask("do I prefer spicy food?").source == "ollama" and g.calls == []
    assert a.ask("capital of France?").source in ("gemini", "ollama")      # no facts involved -> cloud is fine


def test_memory_commands(store):
    _, _, a = mk(store, None, None, Net(True))
    assert a.ask("remember that my snack time is 4:30").text == "Noted"
    assert a.ask("remember that my snack time is 4:30").text == "Already known"
    assert a.ask("remember privately: my locker is 214").text == "Noted (private)"
    assert "my snack time is 4:30" in a.ask("what do you know about me").text
    assert a.ask("remember my password is hunter2").text.startswith("Not stored")
    assert a.ask("forget snack").text == "Forgotten" and a.ask("forget zebra").text == "Nothing matched"
    assert [f.scope for f in store.facts()] == ["private"]


def test_models_down_messages(store):
    _, _, a = mk(store, Fake("gemini", exc=ProviderUnavailable("x")), Fake("ollama", exc=ProviderUnavailable("refused")), Net(True))
    assert a.ask("capital of France?").text == "Offline: start Ollama"


def test_model_text_is_shortened_and_learning_counts(store):
    store.remember("I study mechanical engineering")
    for _ in range(3):
        store.mark_used([store.facts()[0].id])
    long = "**Sure!** Here is a very long answer indeed, which keeps going well past the word limit. With many sentences."
    _, _, a = mk(store, Fake("gemini", long), None, Net(True))
    assert a.ask("random question").text == "Here is a very long answer indeed, which keeps going well past the word"
    assert store.facts()[0].uses >= 3         # frequently used facts rise to the top of future prompts


# ================================================================ network monitor
def test_netmon_notifies_on_change_only():
    states = iter([True, True, False, False, True])
    m = NetMonitor(probe=lambda: next(states))
    seen = []
    m.add_listener(seen.append)
    for _ in range(5):
        m.check()
    assert seen == [True, False, True]


def test_netmon_thread_and_check_soon():
    flag = {"v": False}
    m = NetMonitor(interval_s=30, offline_interval_s=30, probe=lambda: flag["v"])
    seen = []
    m.add_listener(seen.append)
    m.start()
    time.sleep(0.15)
    assert m.online is False
    flag["v"] = True
    m.check_soon()
    time.sleep(0.2)
    m.stop()
    assert seen == [False, True]
