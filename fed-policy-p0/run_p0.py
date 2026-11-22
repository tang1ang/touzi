#!/usr/bin/env python3
"""P0 入口：生成市场基准 JSON 与 Markdown 报告。"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from fedp0.pipeline import build_baseline
from fedp0.report import render_markdown

ROOT = Path(__file__).resolve().parent


def _as_of(value: str | None) -> date:
    if value:
        return date.fromisoformat(value)
    return datetime.now(ZoneInfo("Asia/Shanghai")).date()


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 P0 美联储市场基准")
    parser.add_argument("--as-of", help="基准日 YYYY-MM-DD，默认取上海时区当天")
    parser.add_argument("--calendar", default=str(ROOT / "data/fomc_calendar.json"))
    parser.add_argument(
        "--zq", default=str(ROOT / "data/market/zq_curve_2026-11-20.csv")
    )
    parser.add_argument(
        "--rates", default=str(ROOT / "data/market/rate_snapshot_2026-11-22.csv")
    )
    parser.add_argument("--meetings", type=int, default=4)
    parser.add_argument("--step", type=float, default=0.25)
    parser.add_argument("--min-post-days", type=int, default=7)
    parser.add_argument(
        "--out-json", default=str(ROOT / "output/p0_market_baseline.json")
    )
    parser.add_argument("--out-md", default=str(ROOT / "output/p0_market_baseline.md"))
    args = parser.parse_args()

    as_of = _as_of(args.as_of)
    baseline = build_baseline(
        calendar_path=args.calendar,
        zq_path=args.zq,
        rates_path=args.rates,
        as_of=as_of,
        n_meetings=args.meetings,
        step=args.step,
        min_post_days=args.min_post_days,
    )
    baseline["generated_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(
        timespec="seconds"
    )

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    out_md.write_text(render_markdown(baseline), encoding="utf-8")

    summary = baseline["summary"]
    print(f"as_of={baseline['as_of']}")
    print(
        f"next={summary['next_meeting_date']} "
        f"cut={summary['cut']:.4f} hold={summary['hold']:.4f} hike={summary['hike']:.4f}"
    )
    print(
        f"post_rate={summary['expected_post_rate']:.4f} "
        f"change_bp={summary['expected_change_bp']:+.2f}"
    )
    print(f"json={out_json}")
    print(f"md={out_md}")


if __name__ == "__main__":
    main()
