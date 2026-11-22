"""FOMC 会议日历：区分政策会议与战略会议。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

POLICY = "policy"
STRATEGY = "strategy"


@dataclass(frozen=True)
class Meeting:
    year: int
    month: int
    start: date
    end: date
    decision_date: date
    effective_date: date
    type: str = POLICY

    @property
    def is_policy(self) -> bool:
        return self.type == POLICY

    @classmethod
    def from_dict(cls, row: dict) -> "Meeting":
        meeting = cls(
            year=int(row["year"]),
            month=int(row["month"]),
            start=date.fromisoformat(row["start"]),
            end=date.fromisoformat(row["end"]),
            decision_date=date.fromisoformat(row["decision_date"]),
            effective_date=date.fromisoformat(row["effective_date"]),
            type=row.get("type", POLICY),
        )
        if meeting.decision_date != meeting.end:
            raise ValueError(f"decision_date 必须等于会议最后一天: {row}")
        if meeting.effective_date <= meeting.decision_date:
            raise ValueError(f"effective_date 必须晚于决议日: {row}")
        return meeting

    def to_dict(self) -> dict:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "decision_date": self.decision_date.isoformat(),
            "effective_date": self.effective_date.isoformat(),
            "type": self.type,
        }


class Calendar:
    def __init__(self, meetings: list[Meeting], source: str = "", retrieved_at: str = ""):
        self.meetings = sorted(meetings, key=lambda m: m.start)
        self.source = source
        self.retrieved_at = retrieved_at

    @classmethod
    def load(cls, path: str | Path) -> "Calendar":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        meetings = [Meeting.from_dict(row) for row in payload["meetings"]]
        return cls(meetings, payload.get("source", ""), payload.get("retrieved_at", ""))

    def policy_meetings(self) -> list[Meeting]:
        return [m for m in self.meetings if m.is_policy]

    def upcoming(self, as_of: date, n: int | None = None) -> list[Meeting]:
        """返回 as_of 当日及之后仍待召开的政策会议。"""
        rows = [m for m in self.policy_meetings() if m.decision_date >= as_of]
        return rows[:n] if n is not None else rows

    def next_policy(self, as_of: date) -> Meeting:
        rows = self.upcoming(as_of, 1)
        if not rows:
            raise ValueError(f"{as_of} 之后没有政策会议")
        return rows[0]

    def has_policy_in_month(self, year: int, month: int) -> bool:
        return any(m.is_policy and m.year == year and m.month == month for m in self.meetings)
