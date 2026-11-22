import unittest
from datetime import date
from pathlib import Path

from fedp1.pipeline import build_p1

ROOT = Path(__file__).resolve().parents[1]


class PipelineP1Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = build_p1(
            calendar_path=ROOT / "data/fomc_calendar.json",
            zq_path=ROOT / "data/market/zq_curve_2026-11-20.csv",
            rates_path=ROOT / "data/market/rate_snapshot_2026-11-22.csv",
            macro_dir=ROOT / "data/macro",
            text_dir=ROOT / "data/text",
            as_of=date(2026, 11, 22),
        )

    def test_schema_and_next_meeting(self):
        result = self.result
        self.assertEqual(result["schema_version"], "p1.0.0")
        self.assertEqual(result["as_of"], "2026-11-22")
        summary = result["combination"]["summary"]
        self.assertEqual(summary["next_meeting_date"], "2026-12-09")

    def test_probabilities_sum_to_one(self):
        summary = self.result["combination"]["summary"]
        for key in ("market", "macro", "combined"):
            prob = summary[key]["probability"]
            self.assertAlmostEqual(prob["cut"] + prob["hold"] + prob["hike"], 1.0, places=5)

    def test_paths_aligned(self):
        result = self.result
        self.assertEqual(len(result["market_prior"]["path"]), 4)
        self.assertEqual(len(result["macro"]["path"]), 4)
        self.assertEqual(len(result["combination"]["path"]), 4)
        for market_row, macro_row, combo_row in zip(
            result["market_prior"]["path"],
            result["macro"]["path"],
            result["combination"]["path"],
        ):
            self.assertEqual(market_row["meeting"]["decision_date"], combo_row["meeting"]["decision_date"])
            self.assertEqual(macro_row["meeting"]["decision_date"], combo_row["meeting"]["decision_date"])

    def test_text_signal(self):
        text = self.result["text"]
        self.assertEqual(text["statement"]["date"], "2026-10-28")
        self.assertEqual(text["statement"]["action"], "maintain")
        self.assertEqual(text["statement"]["hike_dissents"], 2)
        self.assertEqual(text["speeches"]["count"], 5)
        self.assertGreaterEqual(text["text_score"], -1.0)
        self.assertLessEqual(text["text_score"], 1.0)

    def test_macro_signal(self):
        macro = self.result["macro"]
        self.assertAlmostEqual(macro["taylor"]["policy_gap_bp"], -59.02, places=2)
        self.assertGreater(macro["taylor"]["macro_target_rate"], macro["taylor"]["taylor_rate"])
        self.assertEqual(len(macro["sources"]), 11)

    def test_provenance_and_caveats(self):
        result = self.result
        items = " ".join(
            (row.get("item", "") or "") + " " + (row.get("source", "") or "")
            for row in result["provenance"]
        )
        self.assertIn("FRED", items)
        self.assertIn("FOMC", items)
        caveats = " ".join(result["caveats"])
        self.assertNotIn("尚未叠加宏观反应函数", caveats)
        self.assertIn("有序 Probit", caveats)


if __name__ == "__main__":
    unittest.main()
