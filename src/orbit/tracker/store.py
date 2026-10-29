"""The workbook. ONE file; each table is a sheet; nothing else is ever created next to it.

Safety: atomic save (temp file + rename), a rolling backup kept in ORBIT's own folder (not beside your
file) so `undo` works, refuses to write while LibreOffice has the file open, never overwrites a file it
cannot read, and neutralises formula injection (cells starting with = + - @).
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import threading
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..errors import OrbitError
from .dates import parse_date

log = logging.getLogger(__name__)


class TrackerError(OrbitError):
    pass


@dataclass(frozen=True)
class Column:
    name: str
    kind: str = "text"                      # text | date | enum
    choices: Tuple[str, ...] = ()


STATUSES = ("Applied", "Screening", "Interview Scheduled", "Interviewed", "Offer", "Rejected", "Withdrawn", "No Response")
_STATUS_SYNONYMS = {"reject": "Rejected", "rejected": "Rejected", "rejection": "Rejected", "offer": "Offer", "offered": "Offer",
                    "selected": "Offer", "scheduled": "Interview Scheduled", "interview": "Interview Scheduled",
                    "interviewed": "Interviewed", "done": "Interviewed", "screening": "Screening", "applied": "Applied",
                    "withdrew": "Withdrawn", "withdrawn": "Withdrawn", "ghosted": "No Response", "no response": "No Response",
                    "waiting": "Applied", "pending": "Applied"}

DEFAULT_TABLES: Dict[str, List[Column]] = {
    "interviews": [
        Column("Company"), Column("Role"), Column("Applied On", "date"), Column("Status", "enum", STATUSES),
        Column("Interview Date", "date"), Column("Round"), Column("Result"), Column("Notes"), Column("Updated", "date"),
    ]
}
KEY_COLUMN = "Company"
MAX_TEXT = 200


def _sheet_title(table: str) -> str:
    return table.replace("_", " ").title()


def clean_text(v: Any) -> str:
    s = re.sub(r"[\x00-\x08\x0b-\x1f]", " ", str(v)).strip()[:MAX_TEXT]
    return "'" + s if s[:1] in ("=", "+", "-", "@") else s          # no formula injection


@dataclass
class Tracker:
    path: Path
    tables: Dict[str, List[Column]] = field(default_factory=lambda: dict(DEFAULT_TABLES))
    backup_dir: Optional[Path] = None
    today: Callable[[], date] = date.today
    keep_backups: int = 10

    def __post_init__(self) -> None:
        self.path = Path(self.path).expanduser()
        self.backup_dir = Path(self.backup_dir) if self.backup_dir else None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ file handling
    def _openpyxl(self):
        try:
            import openpyxl
        except ImportError as exc:
            raise TrackerError("the tracker needs openpyxl (pip install openpyxl)") from exc
        return openpyxl

    def _check_unlocked(self) -> None:
        lock = self.path.parent / f".~lock.{self.path.name}#"
        if lock.exists():
            raise TrackerError("The spreadsheet is open in LibreOffice; close it first")

    def ensure(self) -> Path:
        with self._lock:
            if self.path.exists():
                return self.path
            ox = self._openpyxl()
            wb = ox.Workbook()
            wb.remove(wb.active)
            for table, cols in self.tables.items():
                self._build_sheet(wb.create_sheet(_sheet_title(table)), cols)
            self._save(wb, backup=False)
            return self.path

    def _build_sheet(self, ws, cols: Sequence[Column]) -> None:
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
        from openpyxl.worksheet.datavalidation import DataValidation
        for i, c in enumerate(cols, 1):
            cell = ws.cell(row=1, column=i, value=c.name)
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="2F4F6F")
            cell.alignment = Alignment(vertical="center")
            ws.column_dimensions[get_column_letter(i)].width = 14 if c.kind == "date" else 26 if c.kind == "enum" else 28
            if c.kind == "enum" and c.choices:
                dv = DataValidation(type="list", formula1='"' + ",".join(c.choices) + '"', allow_blank=True)
                ws.add_data_validation(dv)
                dv.add(f"{get_column_letter(i)}2:{get_column_letter(i)}1000")
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = f"A1:{get_column_letter(len(cols))}1"

    def _load(self):
        ox = self._openpyxl()
        self.ensure()
        try:
            return ox.load_workbook(self.path)
        except Exception as exc:
            raise TrackerError(f"could not read {self.path.name} ({exc}); it was NOT modified") from exc

    def _backup(self) -> None:
        if not self.backup_dir or not self.path.exists():
            return
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        shutil.copy2(self.path, self.backup_dir / f"{self.path.stem}.{stamp}.xlsx")
        old = sorted(self.backup_dir.glob(f"{self.path.stem}.*.xlsx"))
        for f in old[: max(0, len(old) - self.keep_backups)]:
            f.unlink()

    def _save(self, wb, backup: bool = True) -> None:
        self._check_unlocked()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if backup:
            self._backup()
        tmp = self.path.parent / f".{self.path.name}.tmp"
        try:
            wb.save(tmp)
            os.replace(tmp, self.path)
        except PermissionError as exc:
            raise TrackerError(f"cannot write {self.path.name}: permission denied") from exc
        finally:
            if tmp.exists():
                tmp.unlink()

    def undo(self) -> str:
        with self._lock:
            if not self.backup_dir:
                raise TrackerError("no backups kept")
            files = sorted(self.backup_dir.glob(f"{self.path.stem}.*.xlsx"))
            if not files:
                raise TrackerError("nothing to undo")
            self._check_unlocked()
            last = files[-1]
            shutil.copy2(last, self.path)
            last.unlink()
            return "Undone"

    # ------------------------------------------------------------------ values
    def _cols(self, table: str) -> List[Column]:
        if table not in self.tables:
            raise TrackerError(f"Unknown table {table!r} (have: {', '.join(self.tables)})")
        return self.tables[table]

    def normalize(self, table: str, values: Dict[str, Any]) -> Dict[str, Any]:
        by_name = {c.name.lower().replace(" ", "_"): c for c in self._cols(table)}
        by_name.update({c.name.lower(): c for c in self._cols(table)})
        out: Dict[str, Any] = {}
        for k, v in values.items():
            col = by_name.get(str(k).lower().strip().replace(" ", "_")) or by_name.get(str(k).lower().strip())
            if col is None or v is None or str(v).strip() == "":
                continue
            if col.name == "Updated":
                continue
            if col.kind == "date":
                d = v if isinstance(v, date) else parse_date(str(v), self.today())
                if d is None:
                    raise TrackerError(f"I couldn't read the date {v!r} for {col.name}")
                out[col.name] = d
            elif col.kind == "enum":
                key = str(v).strip().lower()
                match = next((c for c in col.choices if c.lower() == key), None) or _STATUS_SYNONYMS.get(key)
                if match is None or (col.choices and match not in col.choices):
                    raise TrackerError(f"Status must be one of: {', '.join(col.choices)}")
                out[col.name] = match
            else:
                out[col.name] = clean_text(v)
        return out

    # ------------------------------------------------------------------ rows
    def rows(self, table: str) -> List[Dict[str, Any]]:
        with self._lock:
            cols = self._cols(table)
            ws = self._sheet(self._load(), table, cols)
            out = []
            for r in range(2, ws.max_row + 1):
                row = {c.name: ws.cell(row=r, column=i).value for i, c in enumerate(cols, 1)}
                if any(v not in (None, "") for v in row.values()):
                    row = {k: (v.date() if isinstance(v, datetime) else v) for k, v in row.items()}
                    row["_row"] = r
                    out.append(row)
            return out

    def _sheet(self, wb, table: str, cols: Sequence[Column]):
        title = _sheet_title(table)
        if title not in wb.sheetnames:
            self._build_sheet(wb.create_sheet(title), cols)
        return wb[title]

    def _write_row(self, ws, cols: Sequence[Column], r: int, values: Dict[str, Any]) -> None:
        for i, c in enumerate(cols, 1):
            if c.name in values:
                cell = ws.cell(row=r, column=i, value=values[c.name])
                if c.kind == "date":
                    cell.number_format = "DD MMM YYYY"
        for i, c in enumerate(cols, 1):
            if c.name == "Updated":
                cell = ws.cell(row=r, column=i, value=self.today())
                cell.number_format = "DD MMM YYYY"

    def add(self, table: str, values: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            cols = self._cols(table)
            vals = self.normalize(table, values)
            if KEY_COLUMN not in vals:
                raise TrackerError(f"{KEY_COLUMN} is required")
            if table == "interviews" and "Status" not in vals:
                vals["Status"] = "Applied"
            wb = self._load()
            ws = self._sheet(wb, table, cols)
            r = 2
            while any(ws.cell(row=r, column=i).value not in (None, "") for i in range(1, len(cols) + 1)):
                r += 1
            self._write_row(ws, cols, r, vals)
            self._save(wb)
            return {**vals, "_row": r}

    def find(self, table: str, company: str, role: Optional[str] = None) -> List[Dict[str, Any]]:
        c = company.strip().lower()
        hits = [r for r in self.rows(table) if c and c in str(r.get(KEY_COLUMN, "")).lower()]
        exact = [r for r in hits if str(r[KEY_COLUMN]).lower() == c]
        hits = exact or hits
        if role:                                    # strict: a different role is a different application
            hits = [r for r in hits if role.lower() in str(r.get("Role") or "").lower()]
        return hits

    def update(self, table: str, company: str, changes: Dict[str, Any], role: Optional[str] = None,
               row: Optional[int] = None) -> Dict[str, Any]:
        with self._lock:
            cols = self._cols(table)
            vals = self.normalize(table, changes)
            if not vals:
                raise TrackerError("Nothing to change")
            if row is not None:
                hits = [r for r in self.rows(table) if r["_row"] == row]
            else:
                hits = self.find(table, company, role)
            if not hits:
                raise TrackerError(f"No row for {company!r}")
            if len(hits) > 1:
                raise TrackerError(f"{len(hits)} rows match {company!r}; mention the role")
            wb = self._load()
            ws = self._sheet(wb, table, cols)
            self._write_row(ws, cols, hits[0]["_row"], vals)
            self._save(wb)
            return {**hits[0], **vals}

    # ------------------------------------------------------------------ questions
    def next_interview(self) -> Optional[Dict[str, Any]]:
        t = self.today()
        up = [r for r in self.rows("interviews") if isinstance(r.get("Interview Date"), date) and r["Interview Date"] >= t
              and r.get("Status") not in ("Rejected", "Withdrawn")]
        return min(up, key=lambda r: r["Interview Date"]) if up else None

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {"total": 0}
        for r in self.rows("interviews"):
            out["total"] += 1
            s = r.get("Status") or "Applied"
            out[s] = out.get(s, 0) + 1
        return out
