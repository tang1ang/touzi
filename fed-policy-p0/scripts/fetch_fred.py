#!/usr/bin/env python3
"""从 FRED 抓取目标区间与 EFFR，生成利率快照 CSV。

用法:
    python3 scripts/fetch_fred.py --as-of 2026-11-22
"""

from __future__ import annotations

import argparse
import csv
import io
import urllib.request
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SERIES = ["DFEDTARU", "DFEDTARL", "EFFR"]
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


def fetch_series(series: str) -> list[tuple[date, float]]:
    request = urllib.request.Request(
        FRED_CSV.format(series=series),
        headers={"User-Agent": "Mozilla/5.0 (compatible; FedPolicyRadar/0.1)"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        text = response.read().decode("utf-8", errors="replace")
    rows: list[tuple[date, float]] = []
    for row in csv.DictReader(io.StringIO(text)):
        raw = row.get(series, "")
        if raw in ("", "."):
            continue
        rows.append((date.fromisoformat(row["observation_date"]), float(raw)))
    return rows


def latest_on_or_before(rows: list[tuple[date, float]], as_of: date) -> tuple[date, float]:
    eligible = [(day, value) for day, value in rows if day <= as_of]
    if not eligible:
        raise ValueError(f"截至 {as_of} 没有可用观测")
    return max(eligible)


def main() -> None:
    parser = argparse.ArgumentParser(description="抓取 FRED 利率快照")
    parser.add_argument("--as-of", help="基准日 YYYY-MM-DD，默认上海时区当天")
    parser.add_argument(
        "--out", help="输出 CSV；默认 data/market/rate_snapshot_<as_of>.csv"
    )
    args = parser.parse_args()

    as_of = (
        date.fromisoformat(args.as_of)
        if args.as_of
        else datetime.now(ZoneInfo("Asia/Shanghai")).date()
    )
    out = Path(args.out or ROOT / f"data/market/rate_snapshot_{as_of.isoformat()}.csv")

    lines = ["series,as_of,value,source"]
    for series in SERIES:
        rows = fetch_series(series)
        day, value = latest_on_or_before(rows, as_of)
        lines.append(f"{series},{day.isoformat()},{value:.2f},FRED:{series}")
        print(f"{series}: {day} {value:.2f}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"out={out}")


if __name__ == "__main__":
    main()
