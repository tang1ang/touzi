import unittest
from datetime import date
from pathlib import Path

from fedp1.macro import (
    MacroConfig,
    annualized_change,
    build_macro_path,
    build_macro_signal,
    latest_on_or_before,
    yoy,
)

ROOT = Path(__file__).resolve().parents[1]


def _monthly(values: list[float], start_year: int = 2024) -> list[tuple[date, float]]:
    return [
        (date(start_year + index // 12, index % 12 + 1, 1), value)
        for index, value in enumerate(values)
    ]


class MacroPrimitiveTest(unittest.TestCase):
    def test_latest_on_or_before(self):
        rows = _monthly([1.0, 2.0, 3.0])
        self.assertEqual(latest_on_or_before(rows, date(2024, 2, 15)), (date(2024, 2, 1), 2.0))

    def test_yoy(self):
        rows = _monthly([100.0] + [0.0] * 11 + [103.0])
        result = yoy(rows, date(2025, 1, 15))
        self.assertAlmostEqual(result["pct"], 3.0)

    def test_annualized_change(self):
        rows = [(date(2023, 1, 1), 100.0), (date(2024, 1, 1), 110.0)]
        result = annualized_change(rows, date(2024, 1, 2), 365)
        self.assertAlmostEqual(result["pct"], 10.0072, places=3)

    def test_unknown_date_raises(self):
        with self.assertRaises(ValueError):
            latest_on_or_before(_monthly([1.0]), date(2023, 12, 1))


class MacroSignalTest(unittest.TestCase):
    def setUp(self):
        self.signal = build_macro_signal(
            macro_dir=ROOT / "data/macro",
            as_of=date(2026, 11, 22),
            current_rate=3.88,
        )

    def test_features(self):
        features = self.signal.features
        self.assertAlmostEqual(features["core_pce"]["yoy_pct"], 3.1212, places=3)
        self.assertAlmostEqual(features["core_cpi"]["ann3m_pct"], 3.7179, places=3)
        self.assertAlmostEqual(features["labor"]["unemployment_gap_pp"], -0.2884, places=3)
        self.assertAlmostEqual(features["labor"]["payrolls_3m_avg_k"], 137.0, places=1)
        self.assertAlmostEqual(features["growth"]["gdp_qoq_ann_pct"], 3.3696, places=3)

    def test_taylor_rule(self):
        taylor = self.signal.taylor
        self.assertAlmostEqual(taylor["output_gap_pct"], 0.5769, places=3)
        self.assertAlmostEqual(taylor["taylor_rate"], 4.4702, places=3)
        self.assertAlmostEqual(taylor["policy_gap_bp"], -59.02, places=2)
        self.assertGreater(taylor["macro_target_rate"], taylor["taylor_rate"])

    def test_scores_bounded(self):
        for value in self.signal.scores.values():
            self.assertGreaterEqual(value, -1.0)
            self.assertLessEqual(value, 1.0)
        self.assertGreater(self.signal.scores["inflation"], 0.5)
        self.assertGreater(self.signal.scores["taylor_signal"], 0.5)

    def test_macro_path(self):
        market_path = [
            {"index": i, "meeting": {"decision_date": f"2027-0{i}-15"}}
            for i in range(1, 5)
        ]
        path = build_macro_path(self.signal, market_path, current_rate=3.88)
        self.assertEqual(len(path), 4)
        for row in path:
            prob = row["probability"]
            self.assertAlmostEqual(prob["cut"] + prob["hold"] + prob["hike"], 1.0, places=6)
            self.assertLessEqual(abs(row["change_bp"]), 25.0)
        self.assertGreater(path[-1]["cumulative_bp"], 50.0)
        self.assertLess(path[-1]["cumulative_bp"], 60.0)

    def test_config_override(self):
        signal = build_macro_signal(
            macro_dir=ROOT / "data/macro",
            as_of=date(2026, 11, 22),
            current_rate=3.88,
            config=MacroConfig(convergence_lambda=0.5),
        )
        self.assertEqual(signal.taylor["convergence_lambda"], 0.5)


if __name__ == "__main__":
    unittest.main()
