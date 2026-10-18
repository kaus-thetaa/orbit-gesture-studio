"""Automatic routines: e.g. "when online, fetch the mess menu and keep today's plan".

A routine is just configuration - {"name", "kind", "url", "refresh_hours"} - never code or
commands. The only thing a routine can do is HTTP(S) GET the URL you gave it, turn the
page into text, extract structured data, and store it locally. One request per refresh.

Menu extraction cascade (cheapest first):
  1. weekly-table heuristics (rows = days, or columns = days) and "Snacks: a, b" lines
  2. the language model, given only the page text and today's weekday (JSON out)
  3. keep the raw text so the model can still answer from it later
Parsing happens ONCE at fetch time, so answering "what's for snacks?" later is a local lookup.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import urlencode, urlparse

from ..errors import ConfigError, OrbitError
from .context import ContextStore
from .providers import LLMRequest
from .router import AllProvidersFailed, Router

log = logging.getLogger(__name__)

MENU_DOC = "mess_menu"
MEAL_KEYS = ("breakfast", "lunch", "snacks", "dinner")
_MEAL_ALIASES = {"breakfast": "breakfast", "lunch": "lunch", "snack": "snacks", "snacks": "snacks", "tea": "snacks",
                 "evening snacks": "snacks", "evening snack": "snacks", "high tea": "snacks", "dinner": "dinner", "supper": "dinner"}
_DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_NAME = re.compile(r"^[a-z][a-z0-9_]{1,31}$")


class FetchError(OrbitError):
    pass


# ====================================================================== config
@dataclass
class Routine:
    name: str
    kind: str
    url: str
    refresh_hours: float = 6.0
    enabled: bool = True


def parse_routines(raw: list) -> Tuple[List[Routine], List[str]]:
    out, errors = [], []
    for item in raw or []:
        try:
            if not isinstance(item, dict):
                raise ConfigError("routine must be an object")
            extra = set(item) - {"name", "kind", "url", "refresh_hours", "enabled"}
            if extra:
                raise ConfigError(f"unknown field(s) {sorted(extra)}")
            name, kind, url = item.get("name"), item.get("kind", "menu"), item.get("url")
            if not isinstance(name, str) or not _NAME.match(name):
                raise ConfigError("bad routine name")
            if kind not in ("menu", "menu_api"):
                raise ConfigError(f"unsupported kind {kind!r} (supported: menu, menu_api)")
            u = urlparse(url) if isinstance(url, str) else None
            if not u or u.scheme not in ("http", "https") or not u.hostname:
                raise ConfigError("url must be http(s)")
            rh = item.get("refresh_hours", 6.0)
            if isinstance(rh, bool) or not isinstance(rh, (int, float)) or not 0.25 <= rh <= 48:
                raise ConfigError("refresh_hours must be between 0.25 and 48")
            out.append(Routine(name, kind, url, float(rh), bool(item.get("enabled", True))))
        except ConfigError as exc:
            errors.append(f"routine {item.get('name') if isinstance(item, dict) else item!r}: {exc}")
    return out, errors


# ====================================================================== fetching
class _Text(HTMLParser):
    SKIP = {"script", "style", "noscript", "head", "svg", "template"}
    BLOCK = {"p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "table", "section", "article", "main", "thead", "tbody"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in ("td", "th"):
            self.parts.append(" | ")
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    p = _Text()
    p.feed(html)
    lines, prev = [], None
    for ln in "".join(p.parts).splitlines():
        ln = re.sub(r"[ \t\xa0]+", " ", ln).strip()
        if "|" in ln:
            cells = [c.strip() for c in ln.split("|")]
            while cells and not cells[-1]:          # trailing empties are harmless; leading/middle ones keep columns aligned
                cells.pop()
            ln = ("| " if cells and not cells[0] else "") + " | ".join(cells).strip(" |") if cells else ""
        if ln and ln != prev:
            lines.append(ln)
        prev = ln
    return "\n".join(lines)


def fetch_text(url: str, timeout: float = 12.0, max_bytes: int = 2_000_000) -> str:
    u = urlparse(url)
    if u.scheme not in ("http", "https"):
        raise FetchError("only http(s) URLs are allowed")
    req = urllib.request.Request(url, headers={"User-Agent": "orbit-gesture-studio/0.3 (personal assistant; 1 request per refresh)",
                                               "Accept": "text/html,text/plain,text/csv;q=0.9,*/*;q=0.5"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(max_bytes + 1)
            ctype = resp.headers.get("Content-Type", "")
            charset = resp.headers.get_content_charset() or "utf-8"
    except urllib.error.HTTPError as exc:
        raise FetchError(f"HTTP {exc.code} from {u.hostname}") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise FetchError(f"could not reach {u.hostname}: {getattr(exc, 'reason', exc)}") from exc
    if len(raw) > max_bytes:
        raise FetchError("page too large")
    if any(t in ctype.lower() for t in ("pdf", "image/", "octet-stream", "msword", "officedocument")):
        raise FetchError(f"unsupported content type {ctype!r} (PDF/image menus are not supported yet)")
    text = raw.decode(charset, "replace")
    if "html" in ctype.lower() or re.search(r"<\s*(html|body|table|div)\b", text[:2000], re.I):
        text = html_to_text(text)
    return text


# ====================================================================== extraction
def _split_items(cell: str) -> List[str]:
    parts = re.split(r"\s*(?:,|;|•|·|\+|/|\n|\band\b|&)\s*", cell, flags=re.I)
    return [p.strip(" .-–") for p in parts if p.strip(" .-–")]


def clean_meals(obj) -> Dict[str, List[str]]:
    if not isinstance(obj, dict):
        raise ConfigError("not an object")
    meals: Dict[str, List[str]] = {}
    for k, v in obj.items():
        key = _MEAL_ALIASES.get(str(k).lower().strip())
        if not key or not isinstance(v, list):
            continue
        items = []
        for it in v:
            if isinstance(it, str):
                it = re.sub(r"[\x00-\x1f]", " ", it).strip()[:60]
                if it and it.lower() not in (x.lower() for x in items):
                    items.append(it)
        if items:
            meals.setdefault(key, [])
            meals[key] = (meals[key] + items)[:20]
    if not meals:
        raise ConfigError("no meals found")
    return meals


def _meal_of(label: str) -> Optional[str]:
    return _MEAL_ALIASES.get(re.sub(r"[^a-z ]", "", label.lower()).strip())


def _day_matches(label: str, weekday: str) -> bool:
    l = re.sub(r"[^a-z]", "", label.lower())
    return bool(l) and (l == weekday or (len(l) >= 3 and weekday.startswith(l[:3]) and l in (weekday[:3], weekday[:4], weekday)))


def heuristic_menu(text: str, weekday: str) -> Optional[Dict[str, List[str]]]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    rows = [[c.strip() for c in ln.split("|")] for ln in lines]

    # (a) days as rows, meals as column headers
    for i, cells in enumerate(rows):
        header = [_meal_of(c) for c in cells]
        if sum(1 for h in header if h) >= 2:
            for later in rows[i + 1:]:
                if later and _day_matches(later[0], weekday):
                    meals: Dict[str, List[str]] = {}
                    for col, meal in enumerate(header):
                        if meal and col < len(later) and later[col]:
                            meals.setdefault(meal, []).extend(_split_items(later[col]))
                    if meals:
                        return clean_meals(meals)
    # (b) meals as rows, days as column headers
    for i, cells in enumerate(rows):
        day_cols = [c for c, label in enumerate(cells) if any(_day_matches(label, d) for d in _DAYS)]
        if len(day_cols) >= 3:
            today_col = next((c for c in day_cols if _day_matches(cells[c], weekday)), None)
            if today_col is None:
                continue
            meals = {}
            for later in rows[i + 1:]:
                meal = _meal_of(later[0]) if later else None
                if meal and today_col < len(later) and later[today_col]:
                    meals.setdefault(meal, []).extend(_split_items(later[today_col]))
            if meals:
                return clean_meals(meals)
    # (c) plain "Snacks: Samosa, Tea" lines - only if the page is not a multi-day listing
    if len({d for d in _DAYS if re.search(rf"\b{d[:3]}", text, re.I)}) >= 3:
        return None
    meals = {}
    for ln in lines:
        m = re.match(r"^\s*([A-Za-z ]{3,20}?)\s*[:\-–|]\s*(.+)$", ln)
        if m and _meal_of(m.group(1)):
            meals.setdefault(_meal_of(m.group(1)), []).extend(_split_items(m.group(2)))
    return clean_meals(meals) if len(meals) >= 2 else None


def llm_menu(text: str, now: datetime, router: Router) -> Dict[str, List[str]]:
    system = ("Extract ONLY today's mess menu from the page text. Output JSON only, exactly: "
              '{"breakfast": [], "lunch": [], "snacks": [], "dinner": []} with short item names. '
              "If the page lists several days, use only the day given. Leave a meal empty if it is not listed. "
              f"Today is {now.strftime('%A')} {now.date().isoformat()}.")
    req = LLMRequest(system, [("user", text[:6000])], max_tokens=500, json_mode=True, timeout_s=25.0, temperature=0.0)
    reply = router.ask(req)
    raw = re.sub(r"^```(?:json)?|```$", "", reply.text.strip(), flags=re.M).strip()
    try:
        return clean_meals(json.loads(raw))
    except ValueError as exc:
        raise ConfigError(f"model returned invalid JSON: {raw[:80]!r}") from exc


_TIME = re.compile(r"^([01]?\d|2[0-3]):[0-5]\d$")


class NoMenuForDate(ConfigError):
    pass


def api_menu(text: str, today: str) -> Tuple[Dict[str, List[str]], Dict[str, str], dict]:
    """Parse the Food Court Menu API (GET /api/menu?date=YYYY-MM-DD):
    {"foodCourt": str, "week": str, "menu": {"YYYY-MM-DD": {"day": str, "meals": {"snacks": {"name","startTime","endTime","items":[...]}, ...}}}}
    """
    try:
        j = json.loads(text)
    except ValueError as exc:
        raise ConfigError(f"response is not JSON: {text[:160]!r}") from exc
    if not isinstance(j, dict) or not isinstance(j.get("menu"), dict):
        raise ConfigError(f"unexpected API response shape: {text[:160]!r}")
    day = j["menu"].get(today)
    if not isinstance(day, dict) or not isinstance(day.get("meals"), dict):
        raise NoMenuForDate(f"no menu published for {today}")
    raw: Dict[str, list] = {}
    times: Dict[str, str] = {}
    for key, meal in day["meals"].items():
        if not isinstance(meal, dict):
            continue
        mk = _MEAL_ALIASES.get(str(key).lower().strip()) or _MEAL_ALIASES.get(str(meal.get("name", "")).lower().strip())
        if not mk:
            continue
        raw[mk] = meal.get("items") or []
        st, en = meal.get("startTime"), meal.get("endTime")
        if isinstance(st, str) and isinstance(en, str) and _TIME.match(st) and _TIME.match(en):
            times[mk] = f"{st}-{en}"
    return clean_meals(raw), times, {"food_court": str(j.get("foodCourt", ""))[:80], "week": str(j.get("week", ""))[:80]}


# ====================================================================== running
@dataclass
class RoutineResult:
    ok: bool
    message: str
    parsed_by: str = ""
    retry_after_s: float = 0.0          # polite long wait (e.g. "no menu published today")


class RoutineRunner:
    def __init__(self, store: ContextStore, router: Router, now: Callable[[], datetime] = datetime.now,
                 fetch: Callable[[str], str] = fetch_text) -> None:
        self.store, self.router, self.now, self.fetch = store, router, now, fetch

    def _run_api(self, r: Routine) -> RoutineResult:
        now = self.now()
        today = now.date().isoformat()
        base = r.url.rstrip("/")
        url = base if "/api/menu" in base else base + "/api/menu"
        url += ("&" if "?" in url else "?") + urlencode({"date": today})
        try:
            text = self.fetch(url)
        except FetchError as exc:
            return RoutineResult(False, str(exc))
        try:
            meals, times, meta = api_menu(text, today)
        except NoMenuForDate as exc:
            return RoutineResult(False, str(exc), retry_after_s=3600.0)
        except ConfigError as exc:
            return RoutineResult(False, f"API changed? {exc}")
        doc = {"date": today, "url": url, "meals": meals, "times": times, "raw": "", "parsed_by": "api",
               "fetched_at": now.isoformat(timespec="seconds"), **meta}
        self.store.put_doc(MENU_DOC, today, json.dumps(doc, ensure_ascii=False), now.timestamp())
        return RoutineResult(True, "Menu updated", "api")

    def run(self, r: Routine) -> RoutineResult:
        if r.kind == "menu_api":
            return self._run_api(r)
        if r.kind != "menu":
            return RoutineResult(False, f"unsupported kind {r.kind}")
        try:
            text = self.fetch(r.url)
        except FetchError as exc:
            return RoutineResult(False, str(exc))
        if not text.strip():
            return RoutineResult(False, "page was empty")
        now = self.now()
        today = now.date().isoformat()
        meals, by = None, ""
        try:
            meals, by = heuristic_menu(text, now.strftime("%A").lower()), "heuristic"
        except ConfigError:
            meals = None
        if not meals:
            try:
                meals, by = llm_menu(text, now, self.router), "model"
            except (AllProvidersFailed, ConfigError) as exc:
                log.warning("menu extraction by model failed: %s", exc)
                meals = None
        prev = self.store.get_doc(MENU_DOC)
        if not meals:
            if prev and prev[0] == today and json.loads(prev[1]).get("meals"):
                return RoutineResult(False, "could not parse the page; kept today's earlier menu")
            by, meals = "none", {}
        doc = {"date": today, "url": r.url, "meals": meals, "raw": text[:3000], "parsed_by": by,
               "fetched_at": now.isoformat(timespec="seconds")}
        self.store.put_doc(MENU_DOC, today, json.dumps(doc, ensure_ascii=False), now.timestamp())
        if not meals:
            return RoutineResult(True, "saved raw page (could not extract a menu)", "none")
        return RoutineResult(True, "Menu updated", by)


class RoutineScheduler:
    BACKOFF = (30.0, 120.0, 600.0, 1800.0)

    def __init__(self, routines: List[Routine], store: ContextStore, runner: RoutineRunner, net=None,
                 now: Callable[[], datetime] = datetime.now, clock: Callable[[], float] = time.monotonic,
                 on_event: Optional[Callable[[str], None]] = None, tick_s: float = 30.0) -> None:
        self.routines, self.store, self.runner, self.net, self.now, self.clock = routines, store, runner, net, now, clock
        self.on_event, self.tick_s = on_event, tick_s
        self._fails: Dict[str, int] = {}
        self._next_try: Dict[str, float] = {}
        self._running: set = set()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        if net is not None:
            net.add_listener(lambda online: self._wake.set() if online else None)

    def due(self, r: Routine) -> bool:
        doc = self.store.get_doc(MENU_DOC)
        if not doc or doc[0] != self.now().date().isoformat():
            return True
        return self.now().timestamp() - doc[2] > r.refresh_hours * 3600

    def tick(self, sync: bool = False) -> List[str]:
        started = []
        if self.net is not None and self.net.online is False:
            return started
        for r in self.routines:
            with self._lock:
                if not r.enabled or r.name in self._running or self.clock() < self._next_try.get(r.name, 0.0) or not self.due(r):
                    continue
                self._running.add(r.name)
            started.append(r.name)
            if sync:
                self._run(r)
            else:
                threading.Thread(target=self._run, args=(r,), daemon=True, name=f"orbit-routine-{r.name}").start()
        return started

    def _run(self, r: Routine) -> None:
        try:
            res = self.runner.run(r)
        except Exception as exc:  # noqa: BLE001
            log.exception("routine %s crashed", r.name)
            res = RoutineResult(False, f"error: {exc}")
        with self._lock:
            self._running.discard(r.name)
            if res.ok:
                self._fails[r.name] = 0
                self._next_try[r.name] = 0.0
            else:
                n = self._fails.get(r.name, 0) + 1
                self._fails[r.name] = n
                wait = res.retry_after_s or self.BACKOFF[min(n - 1, len(self.BACKOFF) - 1)]
                self._next_try[r.name] = self.clock() + wait
        log.info("routine %s: %s", r.name, res.message)
        if self.on_event:
            try:
                self.on_event(res.message if res.ok else f"{r.name}: {res.message}")
            except Exception:
                log.exception("routine event callback failed")

    def run_now(self, name: str) -> RoutineResult:
        r = next((x for x in self.routines if x.name == name), None)
        if r is None:
            raise ConfigError(f"No routine named {name!r}")
        return self.runner.run(r)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="orbit-routines", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.tick()
            self._wake.wait(self.tick_s)
            self._wake.clear()
