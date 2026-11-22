#!/usr/bin/env python3
"""实时 P2 入口：当前数据快照 -> 利率调整概率 / 幅度 / 市场 edge。

默认读取本地最新数据；加 --refresh 先抓取网络数据再预测。
每次运行写出 prediction.json、prediction.md、manifest.json，并追加 history.jsonl。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from fedp1.live import LiveConfig, predict_live, render_live_markdown

ROOT = Path(__file__).resolve().parent
DEFAULT_VINTAGES = ROOT / "data/macro/vintages/vintages_long.csv"


def _as_of(value: str | None) -> date:
    if value:
        return date.fromisoformat(value)
    return datetime.now(ZoneInfo("America/New_York")).date()


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _latest_quote_snapshot() -> Path | None:
    candidates = sorted((ROOT / "data/market").glob("zq_curve_*.csv"))
    return candidates[-1] if candidates else None


def _refresh(as_of: date) -> int:
    command = [
        sys.executable,
        str(ROOT / "scripts/refresh_live_data.py"),
        "--as-of",
        as_of.isoformat(),
    ]
    print(f"== 刷新数据: {' '.join(command)}")
    result = subprocess.run(command, cwd=ROOT, check=False)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="生成美联储利率决议实时 P2 预测")
    parser.add_argument("--as-of", help="数据截止日 YYYY-MM-DD，默认纽约时区当天")
    parser.add_argument("--meetings", type=int, default=3, help="预测未来会议数")
    parser.add_argument("--refresh", action="store_true", help="运行前先刷新数据")
    parser.add_argument(
        "--fail-on-stale", action="store_true", help="关键数据过期时拒绝输出"
    )
    parser.add_argument("--no-vintages", action="store_true", help="训练不使用 ALFRED vintage")
    parser.add_argument("--calendar", default=str(ROOT / "data/fomc_calendar.json"))
    parser.add_argument(
        "--zq", default=str(ROOT / "data/backtest/zq_history.csv")
    )
    parser.add_argument("--rates", default=str(ROOT / "data/backtest/rates_long.csv"))
    parser.add_argument("--macro-dir", default=str(ROOT / "data/macro"))
    parser.add_argument(
        "--statements", default=str(ROOT / "data/backtest/statements.json")
    )
    parser.add_argument("--snapshot-dir", help="市场快照输出目录，默认临时目录")
    parser.add_argument(
        "--quote-snapshot",
        help="补充行情快照 CSV（补齐历史长表缺失的新合约）；默认取 data/market 最新一份",
    )
    parser.add_argument("--out-dir", default=str(ROOT / "output/live"))
    parser.add_argument("--edge-threshold-pp", type=float, default=5.0)
    args = parser.parse_args()

    as_of = _as_of(args.as_of)
    if args.refresh:
        code = _refresh(as_of)
        if code != 0:
            print("数据刷新失败，继续使用本地最新数据", file=sys.stderr)

    vintages_path = None
    if not args.no_vintages and DEFAULT_VINTAGES.exists():
        vintages_path = DEFAULT_VINTAGES

    quote_snapshot = (
        Path(args.quote_snapshot) if args.quote_snapshot else _latest_quote_snapshot()
    )

    config = LiveConfig(
        n_meetings=args.meetings,
        edge_threshold_pp=args.edge_threshold_pp,
        fail_on_stale=args.fail_on_stale,
    )
    result = predict_live(
        calendar_path=args.calendar,
        zq_path=args.zq,
        rates_path=args.rates,
        macro_dir=args.macro_dir,
        statements_path=args.statements,
        as_of=as_of,
        config=config,
        vintages_path=vintages_path,
        snapshot_dir=args.snapshot_dir,
        quote_snapshot_path=quote_snapshot,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "prediction.json"
    md_path = out_dir / "prediction.md"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md_path.write_text(render_live_markdown(result), encoding="utf-8")

    manifest = {
        "schema_version": result["schema_version"],
        "generated_at": result["generated_at"],
        "as_of": result["as_of"],
        "vintages": str(vintages_path) if vintages_path else None,
        "quote_snapshot": str(quote_snapshot) if quote_snapshot else None,
        "outputs": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in (json_path, md_path)
        },
        "inputs": result["data_quality"]["inputs"],
        "stale_sources": result["data_quality"]["stale_sources"],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with (out_dir / "history.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "generated_at": result["generated_at"],
                    "as_of": result["as_of"],
                    "training_end": result["training"]["end"],
                    "predictions": [
                        {
                            "decision_date": row["decision_date"],
                            "market": row["market"]["probability"],
                            "calibrated": row["calibrated_probability"],
                            "most_likely": row["most_likely"],
                            "magnitude_bp": row["magnitude_bp"],
                            "edge_pp": row["edge_pp"],
                            "action_hint": row["action_hint"],
                        }
                        for row in result["predictions"]
                    ],
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    first = result["predictions"][0]
    print(f"as_of={result['as_of']} train_end={result['training']['end']} n={result['training']['n']}")
    print(
        f"next={first['decision_date']} "
        f"market_hike={first['market']['probability']['hike']:.4f} "
        f"p2_hike={first['calibrated_probability']['hike']:.4f} "
        f"most_likely={first['most_likely']} "
        f"magnitude={first['magnitude_bp']:+.1f}bp "
        f"hint={first['action_hint']}"
    )
    stale = result["data_quality"]["stale_sources"]
    print(f"stale={','.join(stale) if stale else 'none'}")
    print(f"json={json_path}")
    print(f"md={md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
