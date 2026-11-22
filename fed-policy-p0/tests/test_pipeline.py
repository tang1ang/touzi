import unittest
from datetime import date
from pathlib import Path

from fedp0.pipeline import build_baseline

ROOT = Path(__file__).resolve().parents[1]


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = build_baseline(
            calendar_path=ROOT / "data/fomc_calendar.json",
            zq_path=ROOT / "data/market/zq_curve_2026-11-20.csv",
            rates_path=ROOT / "data/market/rate_snapshot_2026-11-22.csv",
            as_of=date(2026, 11, 22),
        )

    def test_next_meeting_probabilities(self):
        summary = self.result["summary"]
        self.assertEqual(summary["next_meeting_date"], "2026-12-09")
        self.assertAlmostEqual(summary["hike"], 0.901818, places=6)
        self.assertAlmostEqual(summary["hold"], 0.098182, places=6)
        self.assertAlmostEqual(summary["cut"], 0.0)
        self.assertEqual(summary["most_likely"], "hike")

    def test_path(self):
        path = self.result["path"]
        self.assertEqual(len(path), 4)
        self.assertEqual(path[0]["method"], "meeting_month")
        self.assertEqual(path[1]["method"], "next_clean_month")
        self.assertAlmostEqual(path[-1]["post_rate"], 4.405, places=6)
        self.assertAlmostEqual(path[-1]["cumulative_change_bp"], 52.5, places=6)

    def test_probabilities_sum_to_one(self):
        for row in self.result["path"]:
            prob = row["probability"]
            self.assertAlmostEqual(prob["cut"] + prob["hold"] + prob["hike"], 1.0)

    def test_curve_cross_checks_pass(self):
        checks = self.result["curve_cross_check"]
        self.assertEqual(len(checks), 6)
        self.assertTrue(all(row["pass"] for row in checks))


if __name__ == "__main__":
    unittest.main()
