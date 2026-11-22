import unittest
from datetime import date
from pathlib import Path

from fedp0.calendar import Calendar

ROOT = Path(__file__).resolve().parents[1]
CALENDAR = Calendar.load(ROOT / "data/fomc_calendar.json")


class CalendarTest(unittest.TestCase):
    def test_next_meeting(self):
        meeting = CALENDAR.next_policy(date(2026, 11, 22))
        self.assertEqual(meeting.decision_date, date(2026, 12, 9))
        self.assertEqual(meeting.effective_date, date(2026, 12, 10))

    def test_decision_day_is_still_upcoming(self):
        meeting = CALENDAR.next_policy(date(2026, 12, 9))
        self.assertEqual(meeting.decision_date, date(2026, 12, 9))

    def test_upcoming_four(self):
        rows = CALENDAR.upcoming(date(2026, 11, 22), 4)
        self.assertEqual(
            [row.decision_date.isoformat() for row in rows],
            ["2026-12-09", "2027-01-27", "2027-03-17", "2027-04-28"],
        )

    def test_strategy_meeting_excluded(self):
        decisions = {row.decision_date for row in CALENDAR.policy_meetings()}
        self.assertNotIn(date(2027, 10, 27), decisions)
        self.assertTrue(
            any(row.type == "strategy" for row in CALENDAR.meetings)
        )

    def test_clean_month_detection(self):
        self.assertTrue(CALENDAR.has_policy_in_month(2027, 1))
        self.assertTrue(CALENDAR.has_policy_in_month(2027, 3))
        self.assertFalse(CALENDAR.has_policy_in_month(2027, 2))
        self.assertFalse(CALENDAR.has_policy_in_month(2027, 5))


if __name__ == "__main__":
    unittest.main()
