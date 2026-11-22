#!/usr/bin/env python3
"""准备回测数据：历史 FOMC 声明与政策利率长表。

用法:
    python3 scripts/fetch_backtest_data.py

数据来源:
    - FOMC 声明: https://www.federalreserve.gov/newsevents/pressreleases/monetary{YYYYMMDD}a.htm
    - 目标区间/EFFR: FRED (DFEDTARU / DFEDTARL / EFFR)

说明: 本沙箱出网需经 HTTP 代理，Python urllib 直连会超时，因此统一通过 curl 抓取。
"""

from __future__ import annotations

import csv
import io
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fetch_fomc_text import STATEMENT_URL, parse_statement  # noqa: E402

BACKTEST_DIR = ROOT / "data/backtest"
RAW_DIR = BACKTEST_DIR / "raw"
WINDOW_START = date(2021, 1, 1)
WINDOW_END = date(2026, 11, 22)
# 文本信号需要会议之前的声明，因此多抓取半年以上，保证首次回测会议也有两份历史声明。
STATEMENT_HISTORY_START = date(2020, 6, 1)
# 2020 年会议不在当前日历页（页面只保留 2021 年起），但首批 2021 会议需要两份会前声明。
PRE_WINDOW_STATEMENTS = (date(2020, 11, 5), date(2020, 12, 16))
SERIES = ("DFEDTARU", "DFEDTARL", "EFFR")
# ICE BofA 高收益 OAS 在 FRED 只有约三年历史；2021-2023 回测用 BAA10Y 作为金融条件替代。
MACRO_EXTRA = ("BAA10Y",)
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


def curl(url: str, timeout: int = 90) -> bytes:
    result = subprocess.run(
        ["curl", "-sS", "--max-time", str(timeout), url],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl 失败({result.returncode}): {url} {result.stderr[:200]}")
    return result.stdout


def fetch_rates() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, date, float]] = []
    for series in SERIES:
        path = RAW_DIR / f"{series}.csv"
        if not path.exists() or path.stat().st_size == 0:
            path.write_bytes(curl(FRED_CSV.format(series=series)))
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                raw = row.get(series, "")
                if raw in ("", "."):
                    continue
                day = date.fromisoformat(row["observation_date"])
                if day <= WINDOW_END:
                    rows.append((series, day, float(raw)))
    rows.sort(key=lambda item: (item[0], item[1]))
    out = BACKTEST_DIR / "rates_long.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["series", "as_of", "value", "source"])
        for series, day, value in rows:
            writer.writerow([series, day.isoformat(), f"{value:.4f}", f"FRED:{series}"])
    print(f"rates -> {out} ({len(rows)} rows)")
    return out


def fetch_macro_extra() -> None:
    """补充长历史金融条件序列，供扩展回测窗口使用。"""
    raw_macro = ROOT / "data/macro/raw"
    raw_macro.mkdir(parents=True, exist_ok=True)
    for series in MACRO_EXTRA:
        path = raw_macro / f"{series}.csv"
        if path.exists() and path.stat().st_size > 0:
            print(f"macro {series} exists -> {path}")
            continue
        raw = curl(FRED_CSV.format(series=series)).decode("utf-8")
        reader = csv.DictReader(io.StringIO(raw))
        rows = [
            (row["observation_date"], row[series])
            for row in reader
            if row.get(series) not in (None, "", ".")
        ]
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["observation_date", "value"])
            writer.writerows(rows)
        print(f"macro {series} -> {path} ({len(rows)} rows)")


def fetch_statements() -> Path:
    calendar = json.loads((ROOT / "data/fomc_calendar.json").read_text(encoding="utf-8"))
    meetings = [
        row
        for row in calendar["meetings"]
        if row.get("type", "policy") == "policy"
        and STATEMENT_HISTORY_START
        <= date.fromisoformat(row["decision_date"])
        <= WINDOW_END
    ]
    meetings.sort(key=lambda row: row["decision_date"])
    days = [date.fromisoformat(row["decision_date"]) for row in meetings]
    days = sorted(set(days) | set(PRE_WINDOW_STATEMENTS))
    statements = []
    for day in days:
        url = STATEMENT_URL.format(date=day.strftime("%Y%m%d"))
        page = curl(url).decode("utf-8", errors="replace")
        parsed = parse_statement(page, day, url)
        if parsed["vote"] is None:
            joined = " ".join(parsed["paragraphs"]).lower()
            parsed["hike_dissents"] = len(
                re.findall(r"preferred to (?:raise|increase)\b", joined)
            )
            parsed["cut_dissents"] = len(
                re.findall(r"preferred to (?:lower|cut)\b", joined)
            )
        statements.append(parsed)
        print(
            f"statement {day} action={parsed['action']} vote={parsed['vote']} "
            f"hike_d={parsed['hike_dissents']} cut_d={parsed['cut_dissents']} "
            f"paragraphs={len(parsed['paragraphs'])}"
        )
    payload = {
        "source": "Federal Reserve FOMC statements",
        "retrieved_at": date.today().isoformat(),
        "backtest_window": {
            "start": WINDOW_START.isoformat(),
            "end": WINDOW_END.isoformat(),
        },
        "statement_history_window": {
            "start": STATEMENT_HISTORY_START.isoformat(),
            "end": WINDOW_END.isoformat(),
        },
        "statements": statements,
    }
    out = BACKTEST_DIR / "statements.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"statements -> {out} ({len(statements)})")
    return out


if __name__ == "__main__":
    fetch_rates()
    fetch_macro_extra()
    fetch_statements()
