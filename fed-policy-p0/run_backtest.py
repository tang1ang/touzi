#!/usr/bin/env python3
"""回测入口：P1 固定权重 / P2 市场锚定 + 动态权重 + 校准，输出 JSON / Markdown。"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from fedp1.backtest import BacktestConfig, render_markdown, run_backtest
from fedp1.p2 import P2Config, render_p2_markdown, run_cutoff_sweep, run_p2_backtest

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="美联储利率决议回测")
    parser.add_argument("--mode", choices=("p1", "p2"), default="p2")
    parser.add_argument("--start", default="2021-01-01", help="窗口起始日")
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
    parser.add_argument("--certainty-threshold", type=float, default=0.80)
    parser.add_argument("--max-deviation-pp", type=float, default=12.0)
    parser.add_argument("--min-calibration-n", type=int, default=8)
    parser.add_argument("--min-magnitude-n", type=int, default=6)
    parser.add_argument(
        "--no-sweep", action="store_true", help="P2 不做 7/14/30 天截止期敏感性"
    )
    parser.add_argument(
        "--cutoffs", default="7,14,30", help="敏感性分析的截止期（逗号分隔）"
    )
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--sweep-json", default=None)
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    out_json = Path(
        args.out_json
        or ROOT
        / (
            "output/backtest_p2_2021_2026.json"
            if args.mode == "p2"
            else "output/backtest_2021_2026.json"
        )
    )
    out_md = Path(
        args.out_md
        or ROOT
        / (
            "output/backtest_p2_2021_2026.md"
            if args.mode == "p2"
            else "output/backtest_2021_2026.md"
        )
    )

    if args.mode == "p1":
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
            start=start,
            end=end,
            config=config,
        )
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        out_md.write_text(render_markdown(result), encoding="utf-8")
        combined = result["metrics"]["combined"]
        print(f"meetings={result['sample_size']} window={result['window']}")
        print(
            f"combined accuracy={combined['accuracy']:.4f} "
            f"brier={combined['brier']:.4f} logloss={combined['log_loss']:.4f}"
        )
        print(f"json={out_json}")
        print(f"md={out_md}")
        return

    config = P2Config(
        cutoff_days=args.cutoff_days,
        step=args.step,
        min_post_days=args.min_post_days,
        text_scale=args.text_scale,
        smoothing_eps=args.smoothing_eps,
        certainty_threshold=args.certainty_threshold,
        max_deviation_pp=args.max_deviation_pp,
        min_calibration_n=args.min_calibration_n,
        min_magnitude_n=args.min_magnitude_n,
    )
    result = run_p2_backtest(
        calendar_path=args.calendar,
        zq_path=args.zq,
        rates_path=args.rates,
        macro_dir=args.macro_dir,
        statements_path=args.statements,
        start=start,
        end=end,
        config=config,
    )
    sweep = None
    if not args.no_sweep:
        cutoffs = tuple(int(item) for item in args.cutoffs.split(",") if item.strip())
        sweep = run_cutoff_sweep(
            calendar_path=args.calendar,
            zq_path=args.zq,
            rates_path=args.rates,
            macro_dir=args.macro_dir,
            statements_path=args.statements,
            start=start,
            end=end,
            config=config,
            cutoffs=cutoffs,
        )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    out_md.write_text(render_p2_markdown(result, sweep), encoding="utf-8")
    if sweep is not None:
        sweep_json = Path(args.sweep_json or ROOT / "output/cutoff_sweep_p2_2021_2026.json")
        sweep_json.parent.mkdir(parents=True, exist_ok=True)
        sweep_json.write_text(
            json.dumps(sweep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    calibrated = result["metrics"]["p2_calibrated"]
    raw = result["metrics"]["p2_raw"]
    market = result["metrics"]["market"]
    p1 = result["metrics"]["p1_fixed"]
    print(f"meetings={result['sample_size']} window={result['window']}")
    print(
        f"p2_calibrated accuracy={calibrated['accuracy']:.4f} "
        f"brier={calibrated['brier']:.4f} logloss={calibrated['log_loss']:.4f}"
    )
    print(
        f"p2_raw        accuracy={raw['accuracy']:.4f} "
        f"brier={raw['brier']:.4f} logloss={raw['log_loss']:.4f}"
    )
    print(
        f"p1_fixed      accuracy={p1['accuracy']:.4f} "
        f"brier={p1['brier']:.4f} logloss={p1['log_loss']:.4f}"
    )
    print(
        f"market        accuracy={market['accuracy']:.4f} "
        f"brier={market['brier']:.4f} logloss={market['log_loss']:.4f}"
    )
    print(
        f"magnitude     mae={result['magnitude']['p2_magnitude']['mae_bp']:.2f}bp "
        f"rmse={result['magnitude']['p2_magnitude']['rmse_bp']:.2f}bp"
    )
    for row in result["meetings"]:
        selected = row["p2"]["selected"]
        weights = selected["weights"]
        print(
            f"{row['decision_date']} actual={row['actual']['label']:4s} "
            f"market={row['market']['most_likely']:4s} "
            f"p2raw={row['p2']['raw_most_likely']:4s} "
            f"p2cal={row['p2']['calibrated_most_likely']:4s} "
            f"hike={row['p2']['calibrated_probability']['hike']:.3f} "
            f"bp={row['p2']['magnitude_bp']:+.1f} "
            f"w={weights['market']:.2f}/{weights['macro']:.2f}/{weights['text']:.2f} "
            f"b={selected['blend']:.2f}"
        )
    print(f"json={out_json}")
    print(f"md={out_md}")
    if sweep is not None:
        print(f"sweep={Path(args.sweep_json or ROOT / 'output/cutoff_sweep_p2_2021_2026.json')}")


if __name__ == "__main__":
    main()
