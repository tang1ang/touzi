#!/usr/bin/env python3
"""回测入口：逐次 FOMC 会议预测 vs 实际决议，输出 JSON / Markdown。"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from fedp1.backtest import BacktestConfig, render_markdown, run_backtest

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="美联储利率决议回测")
    parser.add_argument("--start", default="2023-11-22", help="窗口起始日")
    parser.add_argument("--end", default="2026-11-22", help="窗口结束日")
    parser.add_argument("--calendar", default=str(ROOT / "data/fomc_calendar.json"))
    parser.add_argument("--zq", default=str(ROOT / "data/backtest/zq_history.csv"))
    parser.add_argument("--rates", default=str(ROOT / "data/backtest/rates_long.csv"))
    parser.add_argument("--macro-dir", default=str(ROOT / "data/macro"))
    parser.add_argument(
        "--statements", default=str(ROOT / "data/backtest/statements.json")
    )
    parser.add_argument("--cutoff-days", type=int, default=7)
    parser.add_argument("--step", type=float, default=0.25)
    parser.add_argument("--min-post-days", type=int, default=7)
    parser.add_argument("--market-weight", type=float, default=0.60)
    parser.add_argument("--macro-weight", type=float, default=0.25)
    parser.add_argument("--text-weight", type=float, default=0.15)
    parser.add_argument("--text-scale", type=float, default=2.0)
    parser.add_argument("--smoothing-eps", type=float, default=0.005)
    parser.add_argument(
        "--out-json", default=str(ROOT / "output/backtest_2023_2026.json")
    )
    parser.add_argument(
        "--out-md", default=str(ROOT / "output/backtest_2023_2026.md")
    )
    args = parser.parse_args()

    config = BacktestConfig(
        cutoff_days=args.cutoff_days,
        step=args.step,
        min_post_days=args.min_post_days,
        weights=(args.market_weight, args.macro_weight, args.text_weight),
        text_scale=args.text_scale,
        smoothing_eps=args.smoothing_eps,
    )
    result = run_backtest(
        calendar_path=args.calendar,
        zq_path=args.zq,
        rates_path=args.rates,
        macro_dir=args.macro_dir,
        statements_path=args.statements,
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
        config=config,
    )

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    out_md.write_text(render_markdown(result), encoding="utf-8")

    combined = result["metrics"]["combined"]
    market = result["metrics"]["market"]
    print(f"meetings={result['sample_size']} window={result['window']}")
    print(
        f"combined accuracy={combined['accuracy']:.4f} brier={combined['brier']:.4f} "
        f"logloss={combined['log_loss']:.4f}"
    )
    print(
        f"market   accuracy={market['accuracy']:.4f} brier={market['brier']:.4f} "
        f"logloss={market['log_loss']:.4f}"
    )
    for row in result["meetings"]:
        print(
            f"{row['decision_date']} actual={row['actual']['label']:4s} "
            f"combined={row['combined']['most_likely']:4s} "
            f"hike={row['combined']['probability']['hike']:.3f} "
            f"market_hike={row['market']['probability']['hike']:.3f}"
        )
    print(f"json={out_json}")
    print(f"md={out_md}")


if __name__ == "__main__":
    main()
