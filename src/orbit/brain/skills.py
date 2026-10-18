"""Instant skills: answered from local data with NO model call (sub-millisecond)."""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Callable, Optional

from .context import ContextStore

MEALS = {
    "breakfast": ("breakfast", "brekkie", "morning"),
    "lunch": ("lunch", "noon"),
    "snacks": ("snack", "snacks", "tea", "tiffin", "evening"),
    "dinner": ("dinner", "supper", "night"),
}
_MENU_WORDS = re.compile(r"\b(menu|mess|canteen)\b")
# Anything asking for judgement/filtering goes to the model, which sees the menu in its context.
_COMPLEX = {"good", "best", "healthy", "vegetarian", "vegan", "veg", "nonveg", "calories", "calorie", "recommend", "should",
            "how", "why", "which", "who", "where", "spicy", "sweet", "without", "allergy", "allergic", "gluten", "protein",
            "compare", "better", "worse", "can", "could", "would", "anything", "something", "cheap", "cheapest"}
_CUES = {"what", "whats", "what's", "any", "today", "tonight", "serving", "menu", "mess", "there", "when", "time", "timing", "timings"}


def _has(text: str, words) -> bool:
    return any(re.search(rf"\b{re.escape(w)}\b", text) for w in words)


def meal_in(text: str) -> Optional[str]:
    t = text.lower()
    for meal, words in MEALS.items():
        if _has(t, words):
            return meal
    return None


def _hm(t: str):
    h, m = t.split(":")
    return int(h), int(m)


def fmt_range(rng: str) -> str:
    """'16:30-17:30' -> '4:30-5:30 PM'"""
    try:
        a, b = rng.split("-")
        (ah, am), (bh, bm) = _hm(a), _hm(b)
    except ValueError:
        return rng
    def one(h, m, suffix):
        return f"{(h % 12) or 12}:{m:02d}" + (f" {'PM' if h >= 12 else 'AM'}" if suffix else "")
    same = (ah >= 12) == (bh >= 12)
    return f"{one(ah, am, not same)}-{one(bh, bm, True)}"


def format_items(items, limit: int = 6) -> str:
    items = [i for i in items if i]
    shown = ", ".join(items[:limit])
    return shown + (f" +{len(items) - limit}" if len(items) > limit else "")


class MenuSkill:
    DOC = "mess_menu"

    def __init__(self, store: ContextStore, today: Callable[[], str], now: Optional[Callable[[], datetime]] = None) -> None:
        self.store, self.today = store, today
        self.now = now or datetime.now

    def load(self) -> Optional[dict]:
        row = self.store.get_doc(self.DOC)
        if not row:
            return None
        try:
            return json.loads(row[1])
        except ValueError:
            return None

    def try_answer(self, text: str) -> Optional[str]:
        t = text.lower()
        words = re.findall(r"[a-z']+", t)
        if len(words) > 7 or _COMPLEX & set(words):
            return None
        meal = meal_in(t)
        generic = bool(_MENU_WORDS.search(t))
        nxt = self._next_meal(t)
        if nxt is not None:
            return nxt
        if not meal and not generic:
            return None
        if meal and not generic and not (t.rstrip().endswith("?") or len(words) <= 2 or _CUES & set(words)):
            return None                       # e.g. "I had tea": a statement, not a menu question
        if re.search(r"\b(tomorrow|yesterday)\b", t):
            return "Only today's menu is saved"
        doc = self.load()
        if not doc:
            return "No menu saved yet"
        times = doc.get("times") or {}
        if meal and re.search(r"\b(when|time|timing|timings|till|until|open|start|starts)\b", t):
            return f"{meal.capitalize()} {fmt_range(times[meal])}" if meal in times else f"No {meal} time listed"
        meals = doc.get("meals") or {}
        stale = f" (menu from {doc.get('date')})" if doc.get("date") != self.today() else ""
        if not meals:
            return None                       # parse failed: let the model read the raw text in context
        if meal:
            items = meals.get(meal)
            label = f"{meal.capitalize()} {fmt_range(times[meal])}" if meal in times else meal.capitalize()
            return (f"{label}: {format_items(items)}" if items else f"No {meal} listed") + stale
        parts = [f"{m[0].upper()}: {format_items(meals[m], 3)}" for m in ("breakfast", "lunch", "snacks", "dinner") if meals.get(m)]
        return (" | ".join(parts) or "Menu is empty") + stale

    def _next_meal(self, t: str) -> Optional[str]:
        """'what's next' / 'what's on now' -> the current or next meal, using today's saved times."""
        if not re.search(r"\b(next|upcoming|now|current|currently)\b", t) or not re.search(r"\b(meal|food|eat|menu|mess|serving)\b", t):
            return None
        if meal_in(t):
            return None
        doc = self.load()
        if not doc or doc.get("date") != self.today() or not doc.get("times") or not doc.get("meals"):
            return None
        now = self.now()
        cur = now.hour * 60 + now.minute
        spans = []
        for meal, rng in doc["times"].items():
            try:
                a, b = rng.split("-")
                (ah, am), (bh, bm) = _hm(a), _hm(b)
            except ValueError:
                continue
            if doc["meals"].get(meal):
                spans.append((ah * 60 + am, bh * 60 + bm, meal))
        for start, end, meal in sorted(spans):
            if start <= cur < end:
                return f"Now - {meal.capitalize()} {fmt_range(doc['times'][meal])}: {format_items(doc['meals'][meal])}"
            if cur < start:
                return f"Next - {meal.capitalize()} {fmt_range(doc['times'][meal])}: {format_items(doc['meals'][meal])}"
        return "No more meals today"

    def context_line(self) -> str:
        doc = self.load()
        if not doc:
            return ""
        if doc.get("meals"):
            tm = doc.get("times") or {}
            body = "; ".join(f"{m}{' (' + tm[m] + ')' if m in tm else ''}: {format_items(v, 12)}" for m, v in doc["meals"].items() if v)
        else:
            body = (doc.get("raw") or "")[:1200]
        return f"Mess menu ({doc.get('date')}): {body}"


class TimeSkill:
    def __init__(self, now: Callable[[], datetime]) -> None:
        self.now = now

    def try_answer(self, text: str) -> Optional[str]:
        t = text.lower().strip(" ?!.")
        if re.fullmatch(r"(what('?s| is) )?(the )?(current )?time( is it)?( now)?|time( now)?|what time is it( now)?", t):
            return self.now().strftime("%I:%M %p").lstrip("0")
        if re.fullmatch(r"(what('?s| is) )?(the )?(today'?s )?(date|day)( is it)?( today)?|what day is it( today)?|date|day", t):
            return self.now().strftime("%a %d %b")
        return None
