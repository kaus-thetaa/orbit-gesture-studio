import json
from datetime import date, datetime

import pytest

from orbit.brain.providers import LLMReply, ProviderUnavailable
from orbit.tracker.dates import parse_date
from orbit.tracker.skill import TrackerSkill, split_date
from orbit.tracker.store import STATUSES, Tracker, TrackerError, clean_text

TODAY = date(2026, 10, 18)                 # a Sunday
NOW = datetime(2026, 10, 18, 11, 0)


@pytest.fixture
def tr(tmp_path):
    return Tracker(tmp_path / "docs" / "orbit_tracker.xlsx", backup_dir=tmp_path / "bk", today=lambda: TODAY)


# ============================================================ dates
@pytest.mark.parametrize("text,expected", [
    ("2026-11-05", date(2026, 11, 5)), ("5 Nov", date(2026, 11, 5)), ("Nov 5", date(2026, 11, 5)), ("5th november 2027", date(2027, 11, 5)),
    ("on the 3rd of Oct", date(2026, 10, 3)), ("05/11/2026", date(2026, 11, 5)), ("5/11", date(2026, 11, 5)),
    ("today", TODAY), ("tomorrow", date(2026, 10, 19)), ("yesterday", date(2026, 10, 17)), ("in 3 days", date(2026, 10, 21)),
    ("in 2 weeks", date(2026, 11, 1)), ("friday", date(2026, 10, 23)), ("next monday", date(2026, 10, 19)), ("next sunday", date(2026, 10, 25)),
])
def test_parse_date(text, expected):
    assert parse_date(text, TODAY) == expected


@pytest.mark.parametrize("bad", ["", "soon", "31 feb", "13/13/2026", "banana 5"])
def test_parse_date_rejects_nonsense(bad):
    assert parse_date(bad, TODAY) is None


def test_split_date():
    assert split_date("SWE intern on Oct 3", TODAY) == ("SWE intern", date(2026, 10, 3))
    assert split_date("Google tomorrow", TODAY) == ("Google", date(2026, 10, 19))
    assert split_date("Data Engineer", TODAY) == ("Data Engineer", None)


# ============================================================ store
def test_one_file_one_sheet_per_table_and_nothing_else(tr):
    p = tr.ensure()
    assert p.exists() and sorted(x.name for x in p.parent.iterdir()) == ["orbit_tracker.xlsx"]
    import openpyxl
    wb = openpyxl.load_workbook(p)
    ws = wb["Interviews"]
    assert [c.value for c in ws[1]] == ["Company", "Role", "Applied On", "Status", "Interview Date", "Round", "Result", "Notes", "Updated"]
    assert ws.freeze_panes == "A2" and ws.data_validations.dataValidation


def test_add_find_update_and_dates_are_real_dates(tr):
    tr.add("interviews", {"company": "Google", "role": "SWE Intern", "applied_on": "Oct 3"})
    tr.add("interviews", {"Company": "Microsoft", "Role": "SDE", "Status": "screening"})
    rows = tr.rows("interviews")
    assert [r["Company"] for r in rows] == ["Google", "Microsoft"]
    assert rows[0]["Applied On"] == date(2026, 10, 3) and rows[0]["Status"] == "Applied" and rows[0]["Updated"] == TODAY
    assert rows[1]["Status"] == "Screening"
    tr.update("interviews", "google", {"Interview Date": "5 Nov", "Status": "interview"})
    g = tr.find("interviews", "Google")[0]
    assert g["Interview Date"] == date(2026, 11, 5) and g["Status"] == "Interview Scheduled"
    import openpyxl
    cell = openpyxl.load_workbook(tr.path)["Interviews"]["E2"]
    assert cell.number_format == "DD MMM YYYY"                                       # sorts/filters as a date in Excel


def test_validation_errors(tr):
    with pytest.raises(TrackerError, match="Company is required"):
        tr.add("interviews", {"Role": "x"})
    with pytest.raises(TrackerError, match="couldn't read the date"):
        tr.add("interviews", {"Company": "A", "Applied On": "someday"})
    with pytest.raises(TrackerError, match="Status must be one of"):
        tr.add("interviews", {"Company": "A", "Status": "maybe"})
    with pytest.raises(TrackerError, match="Unknown table"):
        tr.add("nope", {"Company": "A"})
    with pytest.raises(TrackerError, match="No row for"):
        tr.update("interviews", "Nobody", {"Status": "Offer"})
    assert tr.rows("interviews") == []                                               # failed writes left nothing behind


def test_ambiguous_company_needs_role(tr):
    tr.add("interviews", {"Company": "Google", "Role": "SWE Intern"})
    tr.add("interviews", {"Company": "Google", "Role": "Data Analyst"})
    with pytest.raises(TrackerError, match="mention the role"):
        tr.update("interviews", "Google", {"Status": "Offer"})
    tr.update("interviews", "Google", {"Status": "Offer"}, role="analyst")
    assert [r["Status"] for r in tr.rows("interviews")] == ["Applied", "Offer"]


@pytest.mark.parametrize("evil", ["=HYPERLINK(\"http://x\",\"y\")", "+1+1", "-2+3", "@SUM(A1)"])
def test_formula_injection_is_neutralised(tr, evil):
    tr.add("interviews", {"Company": "Evil Corp", "Notes": evil})
    import openpyxl
    cell = openpyxl.load_workbook(tr.path)["Interviews"]["H2"]
    assert cell.data_type == "s" and cell.value.startswith("'")
    assert clean_text("normal") == "normal" and len(clean_text("x" * 500)) == 200


def test_atomic_backup_undo_and_rolling(tr, tmp_path):
    tr.add("interviews", {"Company": "A"})
    tr.add("interviews", {"Company": "B"})
    assert len(list((tmp_path / "bk").glob("*.xlsx"))) == 2 and [p.name for p in tr.path.parent.iterdir()] == ["orbit_tracker.xlsx"]
    assert tr.undo() == "Undone" and [r["Company"] for r in tr.rows("interviews")] == ["A"]
    assert tr.undo() == "Undone" and tr.rows("interviews") == []
    with pytest.raises(TrackerError, match="nothing to undo"):
        tr.undo()
    tr.keep_backups = 3
    for i in range(8):
        tr.add("interviews", {"Company": f"C{i}"})
    assert len(list((tmp_path / "bk").glob("*.xlsx"))) == 3


def test_refuses_to_write_while_libreoffice_has_it_open(tr):
    tr.add("interviews", {"Company": "A"})
    (tr.path.parent / f".~lock.{tr.path.name}#").write_text("x")
    with pytest.raises(TrackerError, match="open in LibreOffice"):
        tr.add("interviews", {"Company": "B"})
    assert [r["Company"] for r in tr.rows("interviews")] == ["A"]


def test_corrupt_file_is_never_overwritten(tr):
    tr.ensure()
    tr.path.write_bytes(b"this is not an xlsx")
    with pytest.raises(TrackerError, match="NOT modified"):
        tr.add("interviews", {"Company": "A"})
    assert tr.path.read_bytes() == b"this is not an xlsx"


def test_user_edits_in_excel_are_preserved(tr):
    """ORBIT reloads the file each time, so manual edits (and extra sheets) survive its updates."""
    import openpyxl
    tr.add("interviews", {"Company": "Google"})
    wb = openpyxl.load_workbook(tr.path)
    wb["Interviews"]["G2"] = "Great call"
    wb.create_sheet("My Own Sheet")["A1"] = "keep me"
    wb.save(tr.path)
    tr.update("interviews", "Google", {"Status": "Offer"})
    wb = openpyxl.load_workbook(tr.path)
    assert wb["Interviews"]["G2"].value == "Great call" and wb["My Own Sheet"]["A1"].value == "keep me"


def test_next_interview_and_counts(tr):
    tr.add("interviews", {"Company": "Past", "Interview Date": "1 Oct", "Status": "Interviewed"})
    tr.add("interviews", {"Company": "Later", "Interview Date": "20 Nov", "Status": "Interview Scheduled"})
    tr.add("interviews", {"Company": "Soon", "Interview Date": "25 Oct", "Status": "Interview Scheduled"})
    tr.add("interviews", {"Company": "Dead", "Interview Date": "20 Oct", "Status": "Rejected"})
    assert tr.next_interview()["Company"] == "Soon"
    assert tr.counts() == {"total": 4, "Interviewed": 1, "Interview Scheduled": 2, "Rejected": 1}
    assert set(STATUSES) >= {"Offer", "Rejected"}


# ============================================================ natural language
class Clock:
    t = 0.0

    def __call__(self):
        return self.t


class Router:
    def __init__(self, text=None, exc=None):
        self.text, self.exc, self.calls = text, exc, []

    def ask(self, req, local_only=False):
        self.calls.append((req, local_only))
        if self.exc:
            raise self.exc
        return LLMReply(self.text, "ollama", "m")


@pytest.fixture
def skill(tr):
    clock = Clock()
    s = TrackerSkill(tr, None, now=lambda: NOW, clock=clock)
    s.clock_obj = clock
    return s


def test_rule_based_writes(skill, tr):
    assert skill.handle("I applied to Google for SWE intern on Oct 3") == "Added Google - SWE intern, applied 3 Oct"
    assert skill.handle("applied to microsoft") == "Added Microsoft, applied 18 Oct"
    assert skill.handle("Google rejected me") == "Google: Rejected"
    assert skill.handle("I got an offer from Microsoft") == "Microsoft: Offer"
    assert skill.handle("I have an interview with Amazon on 12 Nov") == "Added Amazon"
    assert skill.handle("Google interview on nov 5") == "Google: Interview Date: 5 Nov, Interview Scheduled"
    rows = {r["Company"]: r for r in tr.rows("interviews")}
    assert rows["Google"]["Status"] == "Interview Scheduled" and rows["Google"]["Interview Date"] == date(2026, 11, 5)
    assert rows["Microsoft"]["Status"] == "Offer" and rows["Amazon"]["Interview Date"] == date(2026, 11, 12)
    assert rows["Google"]["Role"] == "SWE intern"


def test_structured_add_command(skill, tr):
    assert skill.handle("add interview: Flipkart, Data Analyst, applied Oct 3, interview 5 Nov") == "Added Flipkart - Data Analyst, applied 3 Oct"
    r = tr.find("interviews", "flipkart")[0]
    assert r["Interview Date"] == date(2026, 11, 5) and r["Role"] == "Data Analyst"


def test_queries(skill, tr):
    assert skill.handle("when is my next interview?") == "No upcoming interviews"
    skill.handle("I have an interview with Amazon on 12 Nov")
    skill.handle("I applied to Google for SWE intern on Oct 3")
    skill.handle("Google interview on 25 Oct")
    assert skill.handle("when is my next interview?") == "Google - SWE intern: 25 Oct (in 7 days)"
    assert skill.handle("what's the status of google") == "Google: Interview Scheduled, interview 25 Oct"
    assert skill.handle("did amazon reply?") == "Amazon: Interview Scheduled, interview 12 Nov"
    assert skill.handle("status of zebra corp") == "No row for Zebra Corp"
    assert skill.handle("how many applications do I have") == "2 application(s)"
    assert skill.handle("how many interviews") == "2 interview(s)"
    assert skill.handle("list my interviews") == "Amazon (Interview Scheduled); Google (Interview Scheduled)"


def test_undo_window(skill, tr):
    skill.handle("applied to Google")
    assert skill.handle("undo") == "Undone" and tr.rows("interviews") == []
    assert skill.handle("undo") is None                                    # nothing recent: not ours
    skill.handle("applied to Google")
    skill.clock_obj.t += 2000
    assert skill.handle("undo") is None                                    # too old: don't undo silently


def test_non_tracker_text_passes_through(skill):
    for t in ("what's for snacks", "how do I prepare for an interview?", "I applied sunscreen", "capital of France?", "remember that my interview is hard",
              "tell me a joke", "an interview"):
        assert skill.handle(t) is None, t


def test_ambiguity_and_errors_are_one_line(skill, tr):
    skill.handle("applied to Google for SWE intern")
    skill.handle("applied to Google for Data Analyst")
    assert skill.handle("Google rejected me").endswith("mention the role")
    assert skill.handle("add interview: X, Y, applied banana") == "I couldn't read the date 'banana' for Applied On"
    assert len(tr.rows("interviews")) == 2


def test_llm_fallback_is_local_only_and_validated(tr):
    r = Router(json.dumps({"action": "add", "company": "Zoho", "role": "Backend Dev", "values": {"Applied On": "Oct 10", "Status": "applied"}}))
    s = TrackerSkill(tr, r, local_only=True, now=lambda: NOW)
    assert s.handle("so yesterday I sent my resume over to Zoho for the backend dev job, that counts as an application I guess") == "Added Zoho - Backend Dev, applied 10 Oct"
    req, local_only = r.calls[0]
    assert local_only is True and req.json_mode
    assert tr.find("interviews", "zoho")[0]["Applied On"] == date(2026, 10, 10)


@pytest.mark.parametrize("reply", ['{"action": "none"}', "not json", '{"action": "add", "company": ""}', '{"action": "drop table", "company": "x"}', "[]"])
def test_llm_garbage_changes_nothing(tr, reply):
    s = TrackerSkill(tr, Router(reply), now=lambda: NOW)
    assert s.handle("my application to Acme is something I should track") is None
    assert tr.rows("interviews") == []


def test_llm_offline_falls_through_and_questions_skip_llm(tr):
    s = TrackerSkill(tr, Router(exc=ProviderUnavailable("x")), now=lambda: NOW)
    assert s.handle("I think I applied somewhere but forget where, my application is odd") is None
    r = Router("{}")
    s = TrackerSkill(tr, r, now=lambda: NOW)
    assert s.handle("should I track my application to Acme?") is None and r.calls == []


def test_llm_dates_and_bad_status_are_rejected_not_written(tr):
    r = Router(json.dumps({"action": "add", "company": "Acme", "values": {"Status": "super hired", "Applied On": "someday"}}))
    s = TrackerSkill(tr, r, now=lambda: NOW)
    out = s.handle("I want to note my application at Acme, it was a while back")
    assert out and ("Status must be" in out or "couldn't read" in out)
    assert tr.rows("interviews") == []


def test_new_role_at_same_company_is_a_new_row_but_blank_role_gets_filled(skill, tr):
    skill.handle("I have an interview with Google on 5 Nov")              # row without a role
    assert skill.handle("applied to Google for SWE intern on Oct 3").startswith("Google: ")   # fills the blank row
    assert len(tr.rows("interviews")) == 1 and tr.rows("interviews")[0]["Role"] == "SWE intern"
    assert skill.handle("applied to Google for Data Analyst").startswith("Added Google - Data Analyst")
    assert len(tr.rows("interviews")) == 2
    assert skill.handle("Google rejected me").endswith("mention the role")
