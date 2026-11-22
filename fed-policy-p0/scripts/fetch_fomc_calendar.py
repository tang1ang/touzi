#!/usr/bin/env python3
"""抓取并解析美联储 FOMC 日历。

用法:
    python3 scripts/fetch_fomc_calendar.py --start-year 2026 --end-year 2027
"""

from __future__ import annotations

import argparse
import calendar as calmod
import json
import re
import urllib.request
from datetime import date, timedelta
from pathlib import Path

URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
ROOT = Path(__file__).resolve().parents[1]
MONTHS = {name.lower(): idx for idx, name in enumerate(calmod.month_name) if name}

YEAR_HEADER = re.compile(r'<a id="\d+">(\d{4}) FOMC Meetings</a>')
MONTH_ROW = re.compile(
    r'class="[^"]*fomc-meeting__month[^"]*"[^>]*><strong>([A-Za-z]+)</strong></div>\s*'
    r'<div class="[^"]*fomc-meeting__date[^"]*"[^>]*>([^<]+)</div>'
)


def fetch(url: str) -> str:
    request = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (compatible; FedPolicyRadar/0.1)"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_sections(html: str) -> list[tuple[int, int, int]]:
    headers = list(YEAR_HEADER.finditer(html))
    return [
        (int(match.group(1)), match.start(), headers[i + 1].start() if i + 1 < len(headers) else len(html))
        for i, match in enumerate(headers)
    ]


def parse_date_range(text: str) -> tuple[int, int] | None:
    match = re.search(r"(\d{1,2})\s*(?:[-–]\s*(\d{1,2}))?", text)
    if not match:
        return None
    start = int(match.group(1))
    end = int(match.group(2) or match.group(1))
    return start, end


def parse_meetings(html: str, start_year: int, end_year: int) -> list[dict]:
    meetings: list[dict] = []
    for year, begin, end in parse_sections(html):
        if not start_year <= year <= end_year:
            continue
        section = html[begin:end]
        rows = list(MONTH_ROW.finditer(section))
        for index, row in enumerate(rows):
            month_name = row.group(1).lower()
            month = MONTHS.get(month_name)
            parsed = parse_date_range(row.group(2))
            if month is None or parsed is None:
                continue
            day_start, day_end = parsed
            note_end = rows[index + 1].start() if index + 1 < len(rows) else len(section)
            note = section[row.end() : note_end]
            meeting_type = "strategy" if "Monetary Policy Strategy Meeting" in note else "policy"

            start = date(year, month, day_start)
            decision = date(year, month, day_end)
            meetings.append(
                {
                    "year": year,
                    "month": month,
                    "start": start.isoformat(),
                    "end": decision.isoformat(),
                    "decision_date": decision.isoformat(),
                    "effective_date": (decision + timedelta(days=1)).isoformat(),
                    "type": meeting_type,
                }
            )
    deduped = {row["start"]: row for row in meetings}
    return sorted(deduped.values(), key=lambda row: row["start"])


def main() -> None:
    parser = argparse.ArgumentParser(description="抓取 FOMC 日历")
    parser.add_argument("--start-year", type=int, default=2026)
    parser.add_argument("--end-year", type=int, default=2027)
    parser.add_argument("--out", default=str(ROOT / "data/fomc_calendar.json"))
    parser.add_argument("--url", default=URL)
    args = parser.parse_args()

    html = fetch(args.url)
    meetings = parse_meetings(html, args.start_year, args.end_year)
    if not meetings:
        raise SystemExit("未解析到任何会议，请检查页面结构是否变化")

    payload = {
        "source": args.url,
        "retrieved_at": date.today().isoformat(),
        "note": "decision_date 为会议最后一天；effective_date 为决议生效日（次一自然日）。"
        "type=strategy 表示不含利率决议的货币政策战略会议。",
        "meetings": meetings,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"parsed={len(meetings)} out={out}")
    for row in meetings:
        print(row["start"], row["decision_date"], row["type"])


if __name__ == "__main__":
    main()
