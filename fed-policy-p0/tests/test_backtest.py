import math
import tempfile
import unittest
from datetime import date
from pathlib import Path

from fedp0.calendar import Calendar, Meeting
from fedp1.backtest import (
    BacktestConfig,
    _class_report,
    actual_decision,
    latest_on_or_before,
    load_zq_history,
    market_signal,
    quote_on,
    text_probabilities,
    text_signal_for,
)


def make_meeting(year: int, month: int, day: int, effective: date) -> Meeting:
    decision = date(year, month, day)
    return Meeting(
        year=year,
        month=month,
        start=decision,
        end=decision,
        decision_date=decision,
        effective_date=effective,
    )


def make_statement(day: str, paragraphs: list[str], action: str = "maintain") -> dict:
    return {
        "date": day,
        "_day": date.fromisoformat(day),
        "url": "",
        "vote": None,
        "action": action,
        "hike_dissents": 0,
        "cut_dissents": 0,
        "paragraphs": paragraphs,
    }


class ActualDecisionTest(unittest.TestCase):
    def test_cut_50bp(self):
        meeting = make_meeting(2024, 9, 18, date(2024, 9, 19))
        rates = {
            "DFEDTARL": [(date(2024, 9, 18), 5.25), (date(2024, 9, 19), 4.75)],
            "DFEDTARU": [(date(2024, 9, 18), 5.50), (date(2024, 9, 19), 5.00)],
        }
        actual = actual_decision(meeting, rates)
        self.assertEqual(actual["label"], "cut")
        self.assertEqual(actual["change_bp"], -50.0)

    def test_hike_25bp(self):
        meeting = make_meeting(2026, 9, 16, date(2026, 9, 17))
        rates = {
            "DFEDTARL": [(date(2026, 9, 16), 3.50), (date(2026, 9, 17), 3.75)],
            "DFEDTARU": [(date(2026, 9, 16), 3.75), (date(2026, 9, 17), 4.00)],
        }
        actual = actual_decision(meeting, rates)
        self.assertEqual(actual["label"], "hike")
        self.assertEqual(actual["change_bp"], 25.0)

    def test_hold(self):
        meeting = make_meeting(2025, 1, 29, date(2025, 1, 30))
        rates = {
            "DFEDTARL": [(date(2025, 1, 29), 4.25), (date(2025, 1, 30), 4.25)],
            "DFEDTARU": [(date(2025, 1, 29), 4.50), (date(2025, 1, 30), 4.50)],
        }
        actual = actual_decision(meeting, rates)
        self.assertEqual(actual["label"], "hold")
        self.assertEqual(actual["change_bp"], 0.0)


class NoLookaheadTest(unittest.TestCase):
    def test_quote_on_uses_last_available(self):
        history = {
            "ZQZ26": [
                (date(2026, 11, 5), 95.0, 10.0),
                (date(2026, 11, 20), 96.0, 20.0),
            ]
        }
        quote = quote_on(history, "ZQZ26", date(2026, 11, 10))
        self.assertEqual(quote.as_of, date(2026, 11, 5))
        self.assertEqual(quote.close, 95.0)
        self.assertIsNone(quote_on(history, "ZQZ26", date(2026, 11, 1)))

    def test_latest_on_or_before(self):
        rows = [(date(2024, 1, 1), 1.0), (date(2024, 2, 1), 2.0)]
        self.assertEqual(latest_on_or_before(rows, date(2024, 1, 15)), (date(2024, 1, 1), 1.0))
        self.assertIsNone(latest_on_or_before(rows, date(2023, 12, 31)))

    def test_market_signal_ignores_post_cutoff_prices(self):
        meeting = make_meeting(2024, 3, 20, date(2024, 3, 21))
        calendar = Calendar([meeting])
        rates = {
            "EFFR": [(date(2024, 3, 13), 5.33)],
            "DFEDTARL": [(date(2024, 3, 20), 5.25)],
            "DFEDTARU": [(date(2024, 3, 20), 5.50)],
        }
        base = {
            "ZQH24": [(date(2024, 3, 13), 94.67, 1000.0)],
            "ZQJ24": [(date(2024, 3, 13), 94.67, 1000.0)],
        }
        spiked = {
            "ZQH24": base["ZQH24"] + [(date(2024, 3, 20), 99.0, 1000.0)],
            "ZQJ24": base["ZQJ24"] + [(date(2024, 3, 20), 99.0, 1000.0)],
        }
        config = BacktestConfig()
        first = market_signal(meeting, date(2024, 3, 13), calendar, base, rates, config)
        second = market_signal(meeting, date(2024, 3, 13), calendar, spiked, rates, config)
        self.assertEqual(first["post_rate"], second["post_rate"])
        self.assertEqual(first["probability"], second["probability"])

    def test_load_zq_history_sorts_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "zq.csv"
            path.write_text(
                "symbol,as_of,open,high,low,close,volume,open_interest\n"
                "ZQZ26,2026-11-20,96,96,96,96,2,0\n"
                "ZQZ26,2026-11-05,95,95,95,95,1,0\n",
                encoding="utf-8",
            )
            history = load_zq_history(path)
            self.assertEqual(history["ZQZ26"][0][0], date(2026, 11, 5))
            self.assertEqual(history["ZQZ26"][1][0], date(2026, 11, 20))


class TextSignalTest(unittest.TestCase):
    def setUp(self):
        self.statements = [
            make_statement("2024-01-31", ["The Committee decided to maintain the target range."]),
            make_statement(
                "2024-03-20",
                ["Inflation remains elevated and the labor market remains tight."],
            ),
            make_statement(
                "2024-05-01",
                ["The Committee decided to lower the target range for the federal funds rate."],
                action="lower",
            ),
        ]
        self.config = BacktestConfig()

    def test_uses_statement_before_cutoff_only(self):
        signal = text_signal_for(self.statements, date(2024, 3, 13), self.config)
        self.assertEqual(signal["as_of_statement"], "2024-01-31")
        self.assertIsNone(signal["previous_statement"])

    def test_uses_previous_for_diff(self):
        signal = text_signal_for(self.statements, date(2024, 3, 27), self.config)
        self.assertEqual(signal["as_of_statement"], "2024-03-20")
        self.assertEqual(signal["previous_statement"], "2024-01-31")
        self.assertGreater(signal["text_score"], 0.0)

    def test_later_statement_does_not_leak(self):
        early = text_signal_for(self.statements, date(2024, 3, 13), self.config)
        only_early = text_signal_for(self.statements[:1], date(2024, 3, 13), self.config)
        self.assertEqual(early["text_score"], only_early["text_score"])


class TextProbabilityTest(unittest.TestCase):
    def test_symmetric_and_normalized(self):
        probabilities = text_probabilities(0.0, 2.0)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=9)
        self.assertAlmostEqual(probabilities["cut"], probabilities["hike"], places=9)

    def test_sign(self):
        hawkish = text_probabilities(0.8, 2.0)
        dovish = text_probabilities(-0.8, 2.0)
        self.assertGreater(hawkish["hike"], hawkish["cut"])
        self.assertGreater(dovish["cut"], dovish["hike"])


class ClassReportTest(unittest.TestCase):
    def test_metrics(self):
        probabilities = [
            {"cut": 0.1, "hold": 0.8, "hike": 0.1},
            {"cut": 0.1, "hold": 0.2, "hike": 0.7},
        ]
        labels = ["hold", "hike"]
        report = _class_report(probabilities, labels)
        self.assertEqual(report["accuracy"], 1.0)
        self.assertAlmostEqual(report["brier"], 0.1, places=9)
        expected_logloss = -(math.log(0.8) + math.log(0.7)) / 2
        self.assertAlmostEqual(report["log_loss"], expected_logloss, places=4)
        self.assertEqual(report["confusion"]["hold"]["hold"], 1)
        self.assertEqual(report["confusion"]["hike"]["hike"], 1)

    def test_wrong_prediction_confusion(self):
        probabilities = [
            {"cut": 0.0, "hold": 1.0, "hike": 0.0},
            {"cut": 0.0, "hold": 1.0, "hike": 0.0},
        ]
        report = _class_report(probabilities, ["hold", "hike"])
        self.assertEqual(report["accuracy"], 0.5)
        self.assertEqual(report["confusion"]["hike"]["hold"], 1)


if __name__ == "__main__":
    unittest.main()
