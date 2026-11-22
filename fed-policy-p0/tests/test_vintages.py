from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import date
from pathlib import Path

from fedp1.macro import build_macro_signal, load_vintages, select_vintage
from fedp1.p2 import _vintage_line, _vintage_quality

ROOT = Path(__file__).resolve().parents[1]


class VintageLoadingTests(unittest.TestCase):
    def test_selects_latest_vintage_on_or_before_as_of(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v.csv"
            with path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["series", "vintage_date", "observation_date", "value"])
                writer.writerow(["CPILFESL", "2021-01-20", "2020-12-01", "100"])
                writer.writerow(["CPILFESL", "2021-01-20", "2021-01-01", "101"])
                writer.writerow(["CPILFESL", "2021-06-09", "2021-04-01", "102"])
            vintages = load_vintages(path)

        chosen = select_vintage(vintages, "CPILFESL", date(2021, 3, 1))
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen[0], date(2021, 1, 20))
        self.assertEqual(chosen[1][-1], (date(2021, 1, 1), 101.0))

        chosen = select_vintage(vintages, "CPILFESL", date(2021, 6, 9))
        self.assertEqual(chosen[0], date(2021, 6, 9))
        self.assertEqual(chosen[1][-1], (date(2021, 4, 1), 102.0))

    def test_returns_none_when_no_vintage_is_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v.csv"
            path.write_text(
                "series,vintage_date,observation_date,value\n"
                "CPILFESL,2021-06-09,2021-04-01,102\n",
                encoding="utf-8",
            )
            vintages = load_vintages(path)
        self.assertIsNone(select_vintage(vintages, "CPILFESL", date(2021, 1, 1)))
        self.assertIsNone(select_vintage(vintages, "UNRATE", date(2021, 6, 9)))

    def test_observed_values_do_not_postdate_the_vintage(self):
        vintages = load_vintages(ROOT / "data/macro/vintages/CPILFESL.csv")
        vintage_day, rows = select_vintage(vintages, "CPILFESL", date(2021, 6, 9))
        self.assertLessEqual(max(day for day, _ in rows), vintage_day)
        # 2021-05 CPI 于 2021-06-10 发布，截止日 2021-06-09 不应看到。
        self.assertLessEqual(max(day for day, _ in rows), date(2021, 4, 1))


class VintageSignalTests(unittest.TestCase):
    def test_build_macro_signal_marks_vintage_and_fallback_sources(self):
        vintages = load_vintages(ROOT / "data/macro/vintages/CPILFESL.csv")
        signal = build_macro_signal(
            ROOT / "data/macro",
            date(2021, 6, 9),
            0.25,
            vintages=vintages,
        )
        self.assertEqual(
            signal.sources["CPILFESL"], "ALFRED vintage 2021-06-09"
        )
        self.assertNotIn("ALFRED", signal.sources["UNRATE"])


class VintageQualityTests(unittest.TestCase):
    def rows(self):
        return [
            {
                "macro": {
                    "sources": {
                        "CPILFESL": "ALFRED vintage 2021-06-09",
                        "BAMLH0A0HYM2": "current + 1d lag",
                    }
                }
            },
            {
                "macro": {
                    "sources": {
                        "CPILFESL": "ALFRED vintage 2021-09-15",
                        "BAMLH0A0HYM2": "ALFRED vintage 2024-01-24",
                    }
                }
            },
        ]

    def test_counts_vintage_and_fallback_pairs(self):
        quality = _vintage_quality(self.rows(), True)
        self.assertTrue(quality["enabled"])
        self.assertEqual(quality["vintage_pairs"], 3)
        self.assertEqual(quality["fallback_pairs"], 1)
        self.assertAlmostEqual(quality["vintage_share"], 0.75, places=6)
        self.assertEqual(quality["series"]["BAMLH0A0HYM2"]["fallback"], 1)

    def test_renders_coverage_and_fallback_note(self):
        line = _vintage_line(_vintage_quality(self.rows(), True))
        self.assertIn("3/4", line)
        self.assertIn("75.0%", line)
        self.assertIn("BAMLH0A0HYM2", line)

    def test_renders_disabled_note(self):
        line = _vintage_line(_vintage_quality(self.rows(), False))
        self.assertIn("未启用", line)


if __name__ == "__main__":
    unittest.main()
