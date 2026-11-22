#!/usr/bin/env python3
"""P1 入口：市场先验 + 宏观反应函数 + 沟通文本 -> 联合信号 JSON / Markdown。"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from fedp1.pipeline import build_p1
from fedp1.report import render_markdown

ROOT = Path(__file__).resolve().parent


def _as_of(value: str | None) -> date:
    if value:
        return date.fromisoformat(value)
    return datetime.now(ZoneInfo("Asia/Shanghai")).date()


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 P1 美联储联合信号")
    parser.add_argument("--as-of", help="基准日 YYYY-MM-DD，默认取上海时区当天")
    parser.add_argument("--calendar", default=str(ROOT / "data/fomc_calendar.json"))
    parser.add_argument(
        "--zq", default=str(ROOT / "data/market/zq_curve_2026-11-20.csv")
    )
    parser.add_argument(
        "--rates", default=str(ROOT / "data/market/rate_snapshot_2026-11-22.csv")
    )
    parser.add_argument("--macro-dir", default=str(ROOT / "data/macro"))
    parser.add_argument("--text-dir", default=str(ROOT / "data/text"))
    parser.add_argument("--meetings", type=int, default=4)
    parser.add_argument("--step", type=float, default=0.25)
    parser.add_argument("--min-post-days", type=int, default=7)
    parser.add_argument("--market-weight", type=float, default=0.60)
    parser.add_argument("--macro-weight", type=float, default=0.25)
    parser.add_argument("--text-weight", type=float, default=0.15)
    parser.add_argument("--text-scale", type=float, default=2.0)
    parser.add_argument("--smoothing-eps", type=float, default=0.005)
    parser.add_argument(
        "--out-json", default=str(ROOT / "output/p1_combined_signal.json")
    )
    parser.add_argument("--out-md", default=str(ROOT / "output/p1_combined_signal.md"))
    args = parser.parse_args()

    as_of = _as_of(args.as_of)
    result = build_p1(
        calendar_path=args.calendar,
        zq_path=args.zq,
        rates_path=args.rates,
        macro_dir=args.macro_dir,
        text_dir=args.text_dir,
        as_of=as_of,
        n_meetings=args.meetings,
        step=args.step,
        min_post_days=args.min_post_days,
        weights=(args.market_weight, args.macro_weight, args.text_weight),
        text_scale=args.text_scale,
        smoothing_eps=args.smoothing_eps,
    )
    result["generated_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(
        timespec="seconds"
    )

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    out_md.write_text(render_markdown(result), encoding="utf-8")

    summary = result["combination"]["summary"]
    print(f"as_of={result['as_of']}")
    print(
        f"next={summary['next_meeting_date']} "
        f"market_hike={summary['market']['probability']['hike']:.4f} "
        f"macro_hike={summary['macro']['probability']['hike']:.4f} "
        f"combined_hike={summary['combined']['probability']['hike']:.4f}"
    )
    print(
        f"combined_expected_bp={summary['combined']['expected_bp']:+.2f} "
        f"macro_contrib={summary['contributions_pp']['macro']['hike']:+.2f}pp "
        f"text_contrib={summary['contributions_pp']['text']['hike']:+.2f}pp"
    )
    last = result["combination"]["path"][-1]
    print(
        f"path_end={last['meeting']['decision_date']} "
        f"rate={last['combined']['expected_post_rate']:.3f} "
        f"cumulative_bp={last['combined']['cumulative_bp']:+.2f}"
    )
    print(f"json={out_json}")
    print(f"md={out_md}")


if __name__ == "__main__":
    main()
