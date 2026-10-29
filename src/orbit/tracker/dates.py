"""Forgiving date parsing for things people actually say: 'Oct 3', '3rd october 2026', 'next monday', 'in 2 weeks'."""
from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Optional

MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
DAYS = {d: i for i, d in enumerate(["mon", "tue", "wed", "thu", "fri", "sat", "sun"])}


def _mk(y: int, m: int, d: int) -> Optional[date]:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def parse_date(text: str, today: date) -> Optional[date]:
    t = re.sub(r"\s+", " ", text.strip().lower().replace(",", " ")).strip()
    t = re.sub(r"\b(on|the|of)\b", " ", t)
    t = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return None
    if t == "today":
        return today
    if t == "tomorrow":
        return today + timedelta(days=1)
    if t == "yesterday":
        return today - timedelta(days=1)
    m = re.fullmatch(r"in (\d{1,3}) (day|days|week|weeks)", t)
    if m:
        return today + timedelta(days=int(m.group(1)) * (7 if m.group(2).startswith("week") else 1))
    m = re.fullmatch(r"(next |this )?(mon|tue|wed|thu|fri|sat|sun)[a-z]*", t)
    if m:
        target = DAYS[m.group(2)]
        delta = (target - today.weekday()) % 7
        if m.group(1) == "next " and delta == 0:
            delta = 7
        elif m.group(1) == "next " and delta < 7 and today.weekday() < target:
            delta += 0
        return today + timedelta(days=delta)
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", t)
    if m:
        return _mk(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.fullmatch(r"(\d{1,2})[/.-](\d{1,2})(?:[/.-](\d{2,4}))?", t)       # day-first (India)
    if m:
        y = int(m.group(3)) if m.group(3) else today.year
        return _mk(y + 2000 if y < 100 else y, int(m.group(2)), int(m.group(1)))
    m = re.fullmatch(r"(\d{1,2}) ([a-z]{3})[a-z]*(?: (\d{4}))?", t)
    if m and m.group(2) in MONTHS:
        return _mk(int(m.group(3) or today.year), MONTHS[m.group(2)], int(m.group(1)))
    m = re.fullmatch(r"([a-z]{3})[a-z]* (\d{1,2})(?: (\d{4}))?", t)
    if m and m.group(1) in MONTHS:
        return _mk(int(m.group(3) or today.year), MONTHS[m.group(1)], int(m.group(2)))
    return None
