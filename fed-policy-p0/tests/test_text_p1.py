import unittest
from datetime import date
from pathlib import Path

from fedp1.text import (
    score_speeches,
    score_statement,
    score_statement_diff,
    score_text,
)

ROOT = Path(__file__).resolve().parents[1]


def _statement(text: str, action: str = "raise", hike: int = 0, cut: int = 0) -> dict:
    return {
        "date": "2026-10-28",
        "url": "https://example.com",
        "action": action,
        "vote": "10-2",
        "hike_dissents": hike,
        "cut_dissents": cut,
        "paragraphs": [text],
    }


class TextScoreTest(unittest.TestCase):
    def test_hawkish_text(self):
        result = score_text(
            "Inflation remains elevated and the labor market remains tight. "
            "The Committee will deliver price stability."
        )
        self.assertGreater(result.score, 0.5)
        self.assertGreater(result.hawk_hits, result.dove_hits)

    def test_dovish_text(self):
        result = score_text(
            "The labor market has cooled and inflation is slowing. "
            "The Committee should cut the target range to support employment."
        )
        self.assertLess(result.score, -0.3)

    def test_negation_flips(self):
        positive = score_text("Further tightening is needed.")
        negated = score_text("It is not clear that further tightening is needed.")
        self.assertGreater(positive.score, 0)
        self.assertLess(negated.score, positive.score)
        self.assertTrue(any(term["negated"] for term in negated.terms))

    def test_neutral_text(self):
        result = score_text("The Committee met on Tuesday and released a statement.")
        self.assertEqual(result.score, 0.0)


class StatementScoreTest(unittest.TestCase):
    def test_hawkish_statement(self):
        result = score_statement(
            _statement(
                "Inflation remains elevated. Economic activity is expanding at a strong pace.",
                action="maintain",
                hike=2,
            )
        )
        self.assertGreater(result["score"], 0.2)
        self.assertEqual(result["hike_dissents"], 2)
        self.assertAlmostEqual(result["dissent_score"], 0.5)

    def test_dovish_statement(self):
        result = score_statement(
            _statement(
                "The labor market has cooled and inflation is slowing.",
                action="cut",
                cut=2,
            )
        )
        self.assertLess(result["score"], 0)

    def test_diff_direction(self):
        previous = {
            "date": "2026-09-16",
            "paragraphs": ["The labor market has cooled and inflation is slowing."],
        }
        current = {
            "date": "2026-10-28",
            "paragraphs": ["Inflation remains elevated and the labor market remains tight."],
        }
        forward = score_statement_diff(current, previous)
        backward = score_statement_diff(previous, current)
        self.assertGreater(forward["diff_score"], 0)
        self.assertLess(backward["diff_score"], 0)


class SpeechScoreTest(unittest.TestCase):
    def test_role_weighting(self):
        speeches = [
            {
                "date": "2026-11-20",
                "speaker": "Chair",
                "role": "chair",
                "title": "Outlook",
                "url": "u1",
                "paragraphs": ["Inflation remains elevated."],
            },
            {
                "date": "2026-11-20",
                "speaker": "President",
                "role": "president",
                "title": "Outlook",
                "url": "u2",
                "paragraphs": ["The labor market has cooled."],
            },
        ]
        result = score_speeches(speeches, date(2026, 11, 22))
        self.assertEqual(result["count"], 2)
        chair = next(item for item in result["items"] if item["role"] == "chair")
        president = next(item for item in result["items"] if item["role"] == "president")
        self.assertGreater(chair["weight"], president["weight"])
        self.assertGreater(result["weighted_score"], 0)


if __name__ == "__main__":
    unittest.main()
