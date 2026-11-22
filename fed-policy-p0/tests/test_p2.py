"""P2 市场锚定、校准、幅度与 walk-forward 测试。"""

from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path

from fedp1.backtest import BacktestConfig
from fedp1.calibration import (
    apply_shrinkage,
    base_rates,
    binary_ece,
    fit_magnitude,
    fit_platt,
    fit_shrinkage,
    magnitude_metrics,
)
from fedp1.macro import build_macro_signal
from fedp1.p2 import (
    P2Config,
    anchored_combination,
    render_p2_markdown,
    run_cutoff_sweep,
    run_p2_backtest,
    select_calibrator,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = {
    "calendar_path": ROOT / "data/fomc_calendar.json",
    "zq_path": ROOT / "data/backtest/zq_history.csv",
    "rates_path": ROOT / "data/backtest/rates_long.csv",
    "macro_dir": ROOT / "data/macro",
    "statements_path": ROOT / "data/backtest/statements.json",
}


class CalibrationTests(unittest.TestCase):
    def test_shrinkage_normalizes_and_reaches_base_at_alpha_one(self):
        base = base_rates(["hold", "hold", "cut"])
        probability = {"cut": 0.05, "hold": 0.85, "hike": 0.10}
        shrunk = apply_shrinkage(probability, 1.0, base)
        self.assertAlmostEqual(sum(shrunk.values()), 1.0, places=9)
        for name in ("cut", "hold", "hike"):
            self.assertAlmostEqual(shrunk[name], base[name], places=9)

    def test_shrinkage_reduces_loss_for_overconfident_wrong_forecasts(self):
        labels = ["hold", "hold", "cut", "cut", "hold", "cut"]
        # 故意把类别预测反转为高损失样本。
        probabilities = [
            {"cut": 1.0, "hold": 0.0, "hike": 0.0}
            if label == "hold"
            else {"cut": 0.0, "hold": 0.0, "hike": 1.0}
            for label in labels
        ]
        fitted = fit_shrinkage(probabilities, labels, min_n=1)
        self.assertGreater(fitted.alpha, 0.0)
        self.assertLess(fitted.calibrated_log_loss, fitted.raw_log_loss)

    def test_platt_unfitted_keeps_identity(self):
        model = fit_platt([{"cut": 0.1, "hold": 0.8, "hike": 0.1}] * 3, ["hold"] * 3)
        out = model.apply({"cut": 0.1, "hold": 0.8, "hike": 0.1})
        self.assertAlmostEqual(out["hold"], 0.8, places=6)
        self.assertFalse(model.hike.fitted)

    def test_ece_is_zero_for_perfect_binary_forecast(self):
        probabilities = [
            {"cut": 0.0, "hold": 0.0, "hike": 1.0},
            {"cut": 1.0, "hold": 0.0, "hike": 0.0},
        ]
        labels = ["hike", "cut"]
        self.assertAlmostEqual(binary_ece(probabilities, labels, "hike"), 0.0, places=9)
        self.assertAlmostEqual(binary_ece(probabilities, labels, "cut"), 0.0, places=9)

    def test_calibrator_selection_uses_out_of_sample_losses(self):
        train_rows = []
        for index in range(10):
            label = "hike" if index % 2 else "cut"
            raw = {"cut": 0.49, "hold": 0.02, "hike": 0.49}
            perfect = (
                {"cut": 0.0, "hold": 0.0, "hike": 1.0}
                if label == "hike"
                else {"cut": 1.0, "hold": 0.0, "hike": 0.0}
            )
            train_rows.append(
                {
                    "actual": {"label": label},
                    "p2": {
                        "raw_probability": raw,
                        "shrinkage_probability": raw,
                        "platt_probability": perfect,
                    },
                }
            )
        choice, losses = select_calibrator(train_rows, P2Config())
        self.assertEqual(choice, "platt")
        self.assertLess(losses["platt"], losses["raw"])

    def test_calibrator_selection_falls_back_to_raw_with_small_sample(self):
        choice, losses = select_calibrator([], P2Config())
        self.assertEqual(choice, "raw")
        self.assertEqual(losses, {})


class AnchorTests(unittest.TestCase):
    def setUp(self):
        self.config = P2Config()

    def _call(self, market, macro, text, reliability="high", blend=1.0):
        return anchored_combination(
            market,
            macro,
            text,
            (0.85, 0.10, 0.05),
            blend,
            reliability,
            self.config,
        )

    def test_near_certain_market_is_kept(self):
        result = self._call(
            {"cut": 0.0, "hold": 0.98, "hike": 0.02},
            {"cut": 0.0, "hold": 0.0, "hike": 1.0},
            0.9,
        )
        self.assertFalse(result["applied"])
        self.assertAlmostEqual(result["probability"]["hold"], 0.98, places=9)

    def test_same_direction_does_not_deviate(self):
        # 2026-07-29 型：市场略偏鹰，宏观与文本更鹰；锚定保留市场判断。
        result = self._call(
            {"cut": 0.0, "hold": 0.66, "hike": 0.34},
            {"cut": 0.0, "hold": 0.0, "hike": 1.0},
            0.8,
        )
        self.assertFalse(result["applied"])
        self.assertEqual(result["probability"]["hold"], 0.66)

    def test_disagreement_does_not_deviate(self):
        result = self._call(
            {"cut": 0.50, "hold": 0.45, "hike": 0.05},
            {"cut": 0.0, "hold": 0.0, "hike": 1.0},
            -0.8,
        )
        self.assertFalse(result["applied"])

    def test_contrarian_agreement_deviates_with_cap(self):
        result = self._call(
            {"cut": 0.50, "hold": 0.45, "hike": 0.05},
            {"cut": 0.0, "hold": 0.0, "hike": 1.0},
            0.8,
        )
        self.assertTrue(result["applied"])
        self.assertGreater(result["probability"]["hike"], 0.05)
        for name in ("cut", "hold", "hike"):
            self.assertLessEqual(
                abs(result["probability"][name] - {"cut": 0.50, "hold": 0.45, "hike": 0.05}[name])
                * 100.0,
                self.config.max_deviation_pp + 1e-9,
            )

    def test_low_reliability_is_kept(self):
        result = self._call(
            {"cut": 0.50, "hold": 0.45, "hike": 0.05},
            {"cut": 0.0, "hold": 0.0, "hike": 1.0},
            0.8,
            reliability="low",
        )
        self.assertFalse(result["applied"])

    def test_zero_blend_is_market(self):
        result = self._call(
            {"cut": 0.50, "hold": 0.45, "hike": 0.05},
            {"cut": 0.0, "hold": 0.0, "hike": 1.0},
            0.8,
            blend=0.0,
        )
        self.assertFalse(result["applied"])
        self.assertEqual(result["probability"]["hike"], 0.05)


class MagnitudeTests(unittest.TestCase):
    def test_small_sample_falls_back_to_identity(self):
        model = fit_magnitude([10.0, 20.0], [-10.0, -20.0], min_n=6)
        self.assertFalse(model.fitted)
        self.assertAlmostEqual(model.predict(30.0), 30.0, places=9)

    def test_linear_fit_and_metrics(self):
        xs = [-50.0, -25.0, 0.0, 25.0, 50.0, 75.0]
        ys = [-50.0, -25.0, 0.0, 25.0, 50.0, 75.0]
        model = fit_magnitude(xs, ys, shrink=0.5, min_n=6)
        self.assertTrue(model.fitted)
        self.assertAlmostEqual(model.predict(50.0), 50.0, places=6)
        metrics = magnitude_metrics(xs, ys)
        self.assertEqual(metrics["mae_bp"], 0.0)
        self.assertEqual(metrics["exact_grid_rate"], 1.0)

    def test_large_move_metrics_detect_50bp(self):
        metrics = magnitude_metrics([-28.0, 25.0], [-50.0, 25.0])
        self.assertLess(metrics["large_move_recall"], 1.0)


class MacroFallbackTests(unittest.TestCase):
    def test_credit_spread_falls_back_to_baa10y_before_hy_history(self):
        config = BacktestConfig()
        signal = build_macro_signal(
            macro_dir=ROOT / "data/macro",
            as_of=date(2021, 6, 9),
            current_rate=0.25,
            publication_lags=config.publication_lags,
        )
        self.assertEqual(
            signal.features["financial_conditions"]["credit_spread_series"],
            "BAA10Y",
        )
        self.assertIsNone(signal.features["financial_conditions"]["hy_oas"])

    def test_credit_spread_uses_hy_oas_when_available(self):
        config = BacktestConfig()
        signal = build_macro_signal(
            macro_dir=ROOT / "data/macro",
            as_of=date(2024, 6, 5),
            current_rate=5.25,
            publication_lags=config.publication_lags,
        )
        self.assertEqual(
            signal.features["financial_conditions"]["credit_spread_series"],
            "BAMLH0A0HYM2",
        )
        self.assertIsNotNone(signal.features["financial_conditions"]["hy_oas"])


class P2IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_p2_backtest(
            **DATA,
            start=date(2023, 1, 1),
            end=date(2026, 11, 22),
            config=P2Config(),
        )

    def test_every_meeting_has_signal_and_actual(self):
        for row in self.result["meetings"]:
            self.assertIn("market", row)
            self.assertIn("macro", row)
            self.assertIn("text", row)
            self.assertIn("actual", row)
            self.assertAlmostEqual(
                sum(row["p2"]["calibrated_probability"].values()), 1.0, places=6
            )

    def test_walk_forward_train_count_never_uses_future(self):
        for index, row in enumerate(self.result["meetings"]):
            selected = row["p2"]["selected"]
            self.assertEqual(selected["train_n"], index)
            self.assertLess(row["as_of"], row["decision_date"])

    def test_anchor_prevents_known_p1_false_hikes(self):
        cases = {
            case["decision_date"]: case
            for case in self.result["signal_contribution"]["p1_wrong_cases"]
        }
        self.assertIn("2023-06-14", cases)
        self.assertIn("2026-07-29", cases)
        for case in cases.values():
            self.assertEqual(case["market"], "hold")
            self.assertEqual(case["p1"], "hike")
            self.assertEqual(case["p2"], "hold")
            self.assertEqual(case["actual"], "hold")

    def test_metrics_and_data_quality_are_present(self):
        metrics = self.result["metrics"]
        for key in ("p2_calibrated", "p2_raw", "p1_fixed", "market"):
            self.assertIn(key, metrics)
            self.assertGreaterEqual(metrics[key]["accuracy"], 0.0)
            self.assertLessEqual(metrics[key]["accuracy"], 1.0)
        quality = self.result["data_quality"]
        self.assertEqual(quality["quote_age_max_days"], 0)
        self.assertGreater(quality["min_pricing_volume"], 0)
        self.assertEqual(quality["low_reliability"], [])

    def test_report_renders_key_sections(self):
        text = render_p2_markdown(self.result, None)
        for heading in (
            "## 一、核心结论",
            "## 二、指标对比",
            "## 三、逐会议明细",
            "## 四、概率校准",
            "## 五、幅度预测",
            "## 六、混淆矩阵",
            "## 七、信号贡献与动态权重",
            "## 十、数据与局限",
        ):
            self.assertIn(heading, text)
        self.assertIn("锚定避免的 P1 误报", text)
        self.assertIn("50bp 及以上事件", text)

    def test_cutoff_sweep_returns_three_runs(self):
        sweep = run_cutoff_sweep(
            **DATA,
            start=date(2025, 1, 1),
            end=date(2026, 11, 22),
            config=P2Config(),
            cutoffs=(7, 14, 30),
        )
        self.assertEqual(sweep["cutoffs"], [7, 14, 30])
        for cutoff in ("7", "14", "30"):
            self.assertIn(cutoff, sweep["runs"])
            self.assertEqual(sweep["runs"][cutoff]["config"]["cutoff_days"], int(cutoff))


if __name__ == "__main__":
    unittest.main()
