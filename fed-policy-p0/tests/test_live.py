"""实时 P2 管道测试：训练截止、无前视、概率归一、新鲜度与渲染。"""

from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

from fedp1.live import (
    LiveConfig,
    _freshness,
    _round_prob,
    predict_live,
    render_live_markdown,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = {
    "calendar_path": ROOT / "data/fomc_calendar.json",
    "zq_path": ROOT / "data/backtest/zq_history.csv",
    "rates_path": ROOT / "data/backtest/rates_long.csv",
    "macro_dir": ROOT / "data/macro",
    "statements_path": ROOT / "data/backtest/statements.json",
}
AS_OF = date(2026, 11, 22)
QUOTE_SNAPSHOT = ROOT / "data/market/zq_curve_2026-11-20.csv"


class RoundProbabilityTests(unittest.TestCase):
    def test_round_prob_normalizes_and_removes_dust(self):
        rounded = _round_prob({"cut": 1.8e-9, "hold": 0.014129283, "hike": 0.98587})
        self.assertAlmostEqual(sum(rounded.values()), 1.0, places=5)
        self.assertGreaterEqual(rounded["cut"], 0.0)


class LivePredictTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.result = predict_live(
            **DATA,
            as_of=AS_OF,
            config=LiveConfig(n_meetings=3),
            snapshot_dir=cls.tmp.name,
            quote_snapshot_path=QUOTE_SNAPSHOT,
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_next_meeting_and_count(self):
        self.assertEqual(self.result["schema_version"], "live.p2.0.0")
        predictions = self.result["predictions"]
        self.assertEqual(len(predictions), 3)
        self.assertEqual(predictions[0]["decision_date"], "2026-12-09")

    def test_probabilities_normalized(self):
        for row in self.result["predictions"]:
            probabilities = [
                row["market"]["probability"],
                row["raw_probability"],
                row["calibrated_probability"],
            ]
            for probability in probabilities:
                self.assertAlmostEqual(sum(probability.values()), 1.0, places=5)
            self.assertAlmostEqual(
                sum(row["calibrators"][self.result["model"]["calibrator"]].values()),
                1.0,
                places=5,
            )

    def test_training_has_no_lookahead(self):
        training = self.result["training"]
        self.assertEqual(training["end"], "2026-10-28")
        self.assertGreaterEqual(training["n"], 40)
        self.assertLess(date.fromisoformat(training["end"]), AS_OF)
        for row in self.result["predictions"]:
            self.assertGreater(date.fromisoformat(row["decision_date"]), AS_OF)

    def test_decomposition_and_interval(self):
        for row in self.result["predictions"]:
            self.assertIn("edge_pp", row)
            self.assertIn("anchor_edge_pp", row)
            self.assertIn("calibration_shift_pp", row)
            self.assertIn("direction_flip_by_calibration", row)
            low, high = row["magnitude_interval_bp"]
            self.assertLessEqual(low, high)
            self.assertGreaterEqual(row["magnitude_bp"], low)
            self.assertLessEqual(row["magnitude_bp"], high)

    def test_freshness_ok_for_current_snapshot(self):
        quality = self.result["data_quality"]
        self.assertFalse(quality["critical_stale"])
        self.assertIn(quality["market"]["latest"], {"2026-11-20", "2026-11-21"})
        self.assertTrue(quality["statements"]["ok"])
        self.assertTrue(all(item["exists"] for item in quality["inputs"]))

    def test_render_contains_key_sections(self):
        markdown = render_live_markdown(self.result)
        for text in ("逐会议预测", "数据新鲜度", "校准抬升", "局限", "2026-12-09"):
            self.assertIn(text, markdown)


class StaleDataTests(unittest.TestCase):
    def test_freshness_flags_far_future_as_of(self):
        from fedp1.backtest import load_rates_long, load_statements, load_zq_history

        freshness = _freshness(
            load_zq_history(DATA["zq_path"]),
            load_rates_long(DATA["rates_path"]),
            DATA["macro_dir"],
            load_statements(DATA["statements_path"]),
            date(2030, 1, 1),
            LiveConfig(),
        )
        self.assertTrue(freshness["critical_stale"])
        self.assertTrue(freshness["stale_sources"])


if __name__ == "__main__":
    unittest.main()
