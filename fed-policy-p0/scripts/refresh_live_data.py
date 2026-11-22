#!/usr/bin/env python3
"""刷新实时预测所需的数据快照。

依次刷新：FOMC 日历、FRED 宏观、利率/声明、文本，以及 ZQ 行情（尽力而为）。
单步失败默认保留旧数据并继续；--strict 时任一失败即退出非零。
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fedp0.contracts import contract_symbol  # noqa: E402


def run_step(name: str, command: list[str], timeout: int = 600) -> bool:
    print(f"\n== {name}: {' '.join(command)}")
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"{name}: 超时")
        return False
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        print(f"{name}: 失败 ({result.returncode}) {detail[-1] if detail else ''}")
        return False
    tail = (result.stdout or "").strip().splitlines()
    for line in tail[-4:]:
        print(line)
    return True


def merge_zq(new_path: Path, target: Path) -> int:
    """按 (symbol, as_of) 合并新行情到历史长表，新数据优先。"""
    rows: dict[tuple[str, str], dict] = {}
    fieldnames = None
    if target.exists():
        with target.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames
            for row in reader:
                rows[(row["symbol"], row["as_of"])] = row
    added = 0
    with new_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = fieldnames or reader.fieldnames
        for row in reader:
            key = (row["symbol"], row["as_of"])
            if key not in rows:
                added += 1
            rows[key] = row
    ordered = sorted(rows.values(), key=lambda row: (row["symbol"], row["as_of"]))
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ordered)
    return added


def _meeting_symbols(as_of: date, count: int = 6) -> list[str]:
    """按未来会议月份生成 ZQ 合约，末尾多取一个月覆盖次月干净合约。"""
    path = ROOT / "data/fomc_calendar.json"
    if not path.exists():
        return []
    calendar = json.loads(path.read_text(encoding="utf-8"))
    meetings = [
        row
        for row in calendar.get("meetings", [])
        if row.get("type", "policy") == "policy"
        and date.fromisoformat(row["decision_date"]) >= as_of
    ]
    meetings.sort(key=lambda row: row["decision_date"])
    symbols = []
    for row in meetings[:count]:
        day = date.fromisoformat(row["decision_date"])
        symbol = contract_symbol(day.year, day.month)
        if symbol not in symbols:
            symbols.append(symbol)
    if symbols:
        last = date.fromisoformat(meetings[min(count, len(meetings)) - 1]["decision_date"])
        year, month = (last.year + 1, 1) if last.month == 12 else (last.year, last.month + 1)
        symbol = contract_symbol(year, month)
        if symbol not in symbols:
            symbols.append(symbol)
    return symbols


def refresh_market(as_of: date, strict: bool) -> bool:
    symbols = _meeting_symbols(as_of)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "zq_live.csv"
        if symbols:
            command = [
                "node",
                "scripts/fetch_barchart_history.js",
                "--symbols",
                ",".join(symbols),
                "--from",
                (as_of - timedelta(days=60)).isoformat(),
                "--out",
                str(out),
            ]
        else:
            command = [
                "node",
                "scripts/fetch_barchart_history.js",
                "--start",
                as_of.strftime("%Y-%m"),
                "--months",
                "5",
                "--out",
                str(out),
            ]
        ok = run_step("ZQ 行情（Barchart，第三方延迟）", command, timeout=600)
        if not ok or not out.exists():
            return False
        before = (ROOT / "data/backtest/zq_history.csv").stat().st_size
        added = merge_zq(out, ROOT / "data/backtest/zq_history.csv")
        after = (ROOT / "data/backtest/zq_history.csv").stat().st_size
        print(f"ZQ 合并：新增 {added} 行，文件 {before} -> {after} bytes")
        return True


def main() -> int:
    parser = argparse.ArgumentParser(description="刷新实时预测数据")
    parser.add_argument("--as-of", help="基准日，默认纽约时区当天")
    parser.add_argument("--strict", action="store_true", help="任一步骤失败即退出非零")
    parser.add_argument("--skip-market", action="store_true", help="跳过 ZQ 行情刷新")
    args = parser.parse_args()

    as_of = (
        date.fromisoformat(args.as_of)
        if args.as_of
        else datetime.now(ZoneInfo("America/New_York")).date()
    )
    print(f"refresh as_of={as_of}")

    steps = [
        (
            "FOMC 日历",
            [
                sys.executable,
                "scripts/fetch_fomc_calendar.py",
                "--start-year",
                "2021",
                "--end-year",
                str(as_of.year + 1),
            ],
        ),
        ("FRED 宏观", [sys.executable, "scripts/fetch_macro.py", "--as-of", as_of.isoformat()]),
        (
            "利率与声明",
            [sys.executable, "scripts/fetch_backtest_data.py", "--force"],
        ),
        ("文本", [sys.executable, "scripts/fetch_fomc_text.py", "--as-of", as_of.isoformat()]),
    ]
    results = {}
    for name, command in steps:
        results[name] = run_step(name, command)
    if not args.skip_market:
        results["ZQ 行情"] = refresh_market(as_of, args.strict)
    else:
        results["ZQ 行情"] = None

    print("\n== 汇总")
    failed = []
    for name, ok in results.items():
        status = "跳过" if ok is None else ("成功" if ok else "失败")
        print(f"- {name}: {status}")
        if ok is False:
            failed.append(name)
    if args.strict and failed:
        print("严格模式：存在失败步骤")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
