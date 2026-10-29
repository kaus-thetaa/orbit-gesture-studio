"""Talk to the spreadsheet: "I applied to Google for SWE intern on Oct 3", "Google rejected me",
"when is my next interview?", "status of Google", "undo".

Order: deterministic rules (instant, private) -> local model extraction for free-form statements.
Every write is reversible with "undo" (rolling backups), so there is no confirmation step.
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import date, datetime
from typing import Any, Callable, Dict, Optional, Tuple

from ..errors import OrbitError
from .dates import parse_date
from .store import STATUSES, Tracker, TrackerError

log = logging.getLogger(__name__)

_DATE_TAIL = re.compile(
    r"\s*(?:on|for|at|is on|is|scheduled (?:for|on))?\s*"
    r"(?P<d>today|tomorrow|yesterday|(?:next |this )?(?:mon|tue|wed|thu|fri|sat|sun)[a-z]*|in \d+ (?:days?|weeks?)|\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/.-]\d{1,2}(?:[/.-]\d{2,4})?"
    r"|\d{1,2}(?:st|nd|rd|th)? (?:of )?[a-z]{3,9}(?:,? \d{4})?|[a-z]{3,9} \d{1,2}(?:st|nd|rd|th)?(?:,? \d{4})?)\s*[.!]?$", re.I)
_NAME = r"[A-Za-z0-9][\w&.'+\- ]{0,60}?"


def split_date(text: str, today: date) -> Tuple[str, Optional[date]]:
    m = _DATE_TAIL.search(text)
    if m:
        d = parse_date(m.group("d"), today)
        if d:
            return text[: m.start()].strip(" ,.-"), d
    return text.strip(" ,.-!"), None


def _nice(name: str) -> str:
    name = name.strip(" ,.!?")
    return name.title() if name.islower() else name


def _when(d: date, today: date) -> str:
    n = (d - today).days
    return "today" if n == 0 else "tomorrow" if n == 1 else f"in {n} days" if n > 1 else f"{-n} days ago"


class TrackerSkill:
    def __init__(self, tracker: Tracker, router=None, local_only: bool = True,
                 now: Callable[[], datetime] = datetime.now, clock: Callable[[], float] = time.monotonic) -> None:
        self.t, self.router, self.local_only, self.now, self.clock = tracker, router, local_only, now, clock
        self._last_write = -1e9

    # ------------------------------------------------------------------
    def handle(self, text: str) -> Optional[str]:
        raw = " ".join(text.split())
        low = raw.lower().rstrip("?!. ")
        self.t.today = lambda: self.now().date()
        try:
            r = self._undo(low) or self._query(low, raw) or self._write_rules(raw, low)
            if r is None:
                r = self._write_llm(raw, low)
        except TrackerError as exc:
            return str(exc)[:140]
        return r

    # ------------------------------------------------------------------ undo
    def _undo(self, low: str) -> Optional[str]:
        if re.fullmatch(r"undo( that| last| the last)?( (tracker|sheet|spreadsheet|change|update|entry))?", low) and self.clock() - self._last_write < 900:
            self._last_write = -1e9
            return self.t.undo()
        return None

    # ------------------------------------------------------------------ questions
    def _query(self, low: str, raw: str) -> Optional[str]:
        today = self.now().date()
        if re.search(r"\b(next|upcoming)\b.*\binterview\b|\bwhen\b.*\b(is|are)\b.*\b(my )?(next )?interview", low):
            r = self.t.next_interview()
            if not r:
                return "No upcoming interviews"
            d = r["Interview Date"]
            return f"{r['Company']}{' - ' + r['Role'] if r.get('Role') else ''}: {d.day} {d:%b} ({_when(d, today)})"
        m = re.search(r"\b(?:status|news|update)\b.*?\b(?:of|on|from|about|with|for)\s+(?P<c>.+)$|^(?:did|has|have)\s+(?P<c2>.+?)\s+(?:reply|replied|respond|responded|call|called|get back)", low)
        if m:
            comp = (m.group("c") or m.group("c2")).strip(" ?.!")
            hits = self.t.find("interviews", comp)
            if not hits:
                return f"No row for {_nice(comp)}"
            if len(hits) > 1:
                return f"{len(hits)} rows for {_nice(comp)}: " + ", ".join(f"{h.get('Role') or '?'}" for h in hits[:3])
            h = hits[0]
            bits = [h.get("Status") or "Applied"]
            if isinstance(h.get("Interview Date"), date):
                bits.append(f"interview {h['Interview Date'].day} {h['Interview Date']:%b}")
            if h.get("Result"):
                bits.append(str(h["Result"]))
            return f"{h['Company']}: " + ", ".join(bits)
        if re.search(r"\bhow many\b.*\b(applications?|applied|interviews?|offers?|rejections?|rejected)\b", low):
            c = self.t.counts()
            if re.search(r"offer", low):
                return f"{c.get('Offer', 0)} offer(s)"
            if re.search(r"reject", low):
                return f"{c.get('Rejected', 0)} rejection(s)"
            if re.search(r"interview", low):
                n = c.get("Interview Scheduled", 0) + c.get("Interviewed", 0)
                return f"{n} interview(s)"
            return f"{c['total']} application(s)"
        if re.fullmatch(r"(list|show)( me)?( all)?( my)? (interviews|applications)( this week)?", low):
            rows = self.t.rows("interviews")
            if not rows:
                return "Nothing tracked yet"
            return "; ".join(f"{r['Company']} ({r.get('Status') or 'Applied'})" for r in rows[:5]) + (f" +{len(rows) - 5}" if len(rows) > 5 else "")
        return None

    # ------------------------------------------------------------------ writes: rules
    def _mark(self) -> None:
        self._last_write = self.clock()

    def _upsert(self, company: str, role: Optional[str], values: Dict[str, Any]) -> str:
        company = _nice(company)
        values = self.t.normalize("interviews", values)              # parse dates/status once, up front
        hits = self.t.find("interviews", company, role)
        target = None
        if len(hits) == 1:
            target = hits[0]["_row"]
        elif len(hits) > 1:
            raise TrackerError(f"{len(hits)} rows match {company}; mention the role")
        elif role:                                                   # a new role: fill a blank-role row, otherwise it's a new application
            blanks = [r for r in self.t.find("interviews", company) if not r.get("Role")]
            if len(blanks) == 1:
                target, values = blanks[0]["_row"], {**values, "Role": role}
        if target is not None:
            row = self.t.update("interviews", company, values, row=target)
            self._mark()
            bits = [f"{k}: {v.day} {v:%b}" if isinstance(v, date) else (str(v) if k == "Status" else f"{k}: {v}") for k, v in values.items()]
            return f"{row['Company']}: " + ", ".join(bits)
        vals = {"Company": company, **({"Role": role} if role else {}), **values}
        self.t.add("interviews", vals)
        self._mark()
        ao = values.get("Applied On")
        extra = f", applied {ao.day} {ao:%b}" if isinstance(ao, date) else ""
        return f"Added {company}" + (f" - {role}" if role else "") + extra

    def _write_rules(self, raw: str, low: str) -> Optional[str]:
        if raw.rstrip().endswith("?"):
            return None
        today = self.now().date()
        m = re.match(rf"^(?:i )?(?:have |'ve |just )*applied (?:to|at|for) (?P<c>{_NAME})(?: (?:for|as|role of|position of)(?: an?)? (?P<r>.+))?$", raw, re.I)
        if m:
            rest = m.group("r")
            company = m.group("c")
            when = None
            if rest:
                rest, when = split_date(rest, today)
            else:
                company, when = split_date(company, today)
            return self._upsert(company, rest or None, {"Applied On": when or today, "Status": "Applied"})
        m = re.match(rf"^(?:(?P<c>{_NAME}) (?:has |have )?(?:rejected|turned down|declined) me|(?:i )?(?:got |was |have been )?rejected (?:by|from|at) (?P<c2>{_NAME}))$", raw, re.I)
        if m:
            return self._upsert(m.group("c") or m.group("c2"), None, {"Status": "Rejected"})
        m = re.match(rf"^(?:i )?(?:got|received|have|just got)(?: an?)? offer (?:from|at) (?P<c>{_NAME})$|^(?P<c2>{_NAME}) (?:gave|sent|offered) me (?:an )?offer$", raw, re.I)
        if m:
            return self._upsert(m.group("c") or m.group("c2"), None, {"Status": "Offer"})
        m = re.match(rf"^(?:i )?(?:have |got |just got )?(?:an? )?(?:(?P<rd>[\w ]{{2,20}}) )?interview (?:with|at) (?P<c>.+)$", raw, re.I)
        m2 = re.match(rf"^(?P<c>{_NAME}) (?:(?P<rd>[\w]{{2,20}}) )?interview (?P<tail>.+)$", raw, re.I)
        if m and not re.match(r"^(?:how|what|when|where)\b", low):
            comp, when = split_date(m.group("c"), today)
            if when:
                return self._upsert(comp, None, {"Interview Date": when, "Status": "Interview Scheduled"})
        elif m2:
            _, when = split_date(m2.group("tail"), today)
            if when and not re.match(r"^(?:my|the|an?|next|when|how|what)\b", m2.group("c"), re.I):
                return self._upsert(m2.group("c"), None, {"Interview Date": when, "Status": "Interview Scheduled"})
        m = re.match(r"^(?:add|log|track|record)(?: an?)? (?:interview|application|job)[:,]? (?P<rest>.+)$", raw, re.I)
        if m:
            parts = [p.strip() for p in re.split(r"[,|;]", m.group("rest")) if p.strip()]
            vals: Dict[str, Any] = {}
            company, role = parts[0], None
            for p in parts[1:]:
                kv = re.match(r"^(applied|interview|status)(?: on)? (.+)$", p, re.I)
                if kv:
                    k = {"applied": "Applied On", "interview": "Interview Date", "status": "Status"}[kv.group(1).lower()]
                    vals[k] = kv.group(2)
                elif role is None:
                    role = p
            return self._upsert(company, role, vals)
        return None

    # ------------------------------------------------------------------ writes: model fallback
    def _write_llm(self, raw: str, low: str) -> Optional[str]:
        if self.router is None or raw.rstrip().endswith("?"):
            return None
        if not re.search(r"\b(i|i've|ive|my)\b", low) or not re.search(r"\b(applied|application|interview|interviewed|offer|rejected|rejection|shortlisted|selected)\b", low):
            return None
        from ..brain.providers import LLMRequest
        from ..brain.providers import ProviderError
        system = ("Extract a job-tracking update. Output JSON only: "
                  '{"action": "add"|"update"|"none", "company": str, "role": str|null, "values": '
                  '{"Applied On": str|null, "Status": one of ' + "|".join(STATUSES) + '|null, "Interview Date": str|null, "Round": str|null, "Result": str|null, "Notes": str|null}}. '
                  'Use "none" if the message is not about a specific company. Dates as written by the user.')
        try:
            rep = self.router.ask(LLMRequest(system, [("user", raw)], max_tokens=200, json_mode=True, temperature=0.0, timeout_s=20.0),
                                  local_only=self.local_only)
            data = json.loads(re.sub(r"^```(?:json)?|```$", "", rep.text.strip(), flags=re.M).strip())
        except (ProviderError, ValueError):
            return None
        if not isinstance(data, dict) or data.get("action") not in ("add", "update") or not isinstance(data.get("company"), str) or not data["company"].strip():
            return None
        values = {k: v for k, v in (data.get("values") or {}).items() if isinstance(v, str) and v.strip()}
        if not values and data["action"] == "add":
            values = {"Applied On": "today"}
        return self._upsert(data["company"], data.get("role") if isinstance(data.get("role"), str) else None, values)
