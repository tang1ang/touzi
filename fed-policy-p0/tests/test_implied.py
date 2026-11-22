import unittest
from datetime import date
from pathlib import Path

from fedp0.calendar import Calendar
from fedp0.implied import (
    decompose_post_rate,
    estimate_step,
    implied_rate,
    split_month_days,
)
from fedp0.market import Quote

ROOT = Path(__file__).resolve().parents[1]
CALENDAR = Calendar.load(ROOT / "data/fomc_calendar.json")


class ImpliedRateTest(unittest.TestCase):
    def test_price_to_rate(self):
        self.assertAlmostEqual(implied_rate(95.96), 4.04)
        self.assertAlmostEqual(implied_rate(96.12), 3.88)

    def test_december_day_split(self):
        self.assertEqual(split_month_days(2026, 12, date(2026, 12, 10)), (9, 22))

    def test_late_month_meeting_split(self):
        self.assertEqual(split_month_days(2027, 4, date(2027, 4, 29)), (28, 2))

    def test_effective_next_month_means_no_post_days(self):
        self.assertEqual(split_month_days(2026, 4, date(2026, 5, 1)), (30, 0))

    def test_decompose_december(self):
        value = decompose_post_rate(4.04, 3.88, 2026, 12, date(2026, 12, 10))
        self.assertAlmostEqual(value, 4.105454545, places=6)

    def test_decompose_requires_post_days(self):
        with self.assertRaises(ValueError):
            decompose_post_rate(4.04, 3.88, 2026, 4, date(2026, 5, 1))


class EstimateStepTest(unittest.TestCase):
    def _quote(self, symbol, close, volume=1000):
        return Quote(symbol=symbol, as_of=date(2026, 11, 20), close=close, volume=volume)

    def test_december_uses_meeting_month(self):
        quotes = {"ZQZ26": self._quote("ZQZ26", 95.96)}
        meeting = CALENDAR.next_policy(date(2026, 11, 22))
        est = estimate_step(meeting, quotes["ZQZ26"], 3.88, CALENDAR, quotes)
        self.assertEqual(est.method, "meeting_month")
        self.assertAlmostEqual(est.post_rate, 4.105454545, places=6)
        self.assertAlmostEqual(est.change_bp, 22.545, places=3)
        self.assertEqual(est.reliability, "high")

    def test_january_uses_next_clean_month(self):
        meeting = CALENDAR.upcoming(date(2026, 11, 22), 2)[1]
        quotes = {
            "ZQF27": self._quote("ZQF27", 95.88),
            "ZQG27": self._quote("ZQG27", 95.755),
        }
        est = estimate_step(meeting, quotes["ZQF27"], 4.105454545, CALENDAR, quotes)
        self.assertEqual(est.method, "next_clean_month")
        self.assertAlmostEqual(est.post_rate, 4.245, places=6)
        self.assertEqual(est.reliability, "high")


if __name__ == "__main__":
    unittest.main()
