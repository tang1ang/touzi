#!/usr/bin/env python3
"""从 FRED 抓取 P1 宏观数据，落盘为可复现的原始序列与元数据。

用法:
    python3 scripts/fetch_macro.py --as-of 2026-11-22
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import urllib.request
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

SERIES = [
    {"id": "PCEPILFE", "name": "核心 PCE 价格指数", "unit": "index", "frequency": "monthly"},
    {"id": "CPILFESL", "name": "核心 CPI", "unit": "index", "frequency": "monthly"},
    {"id": "UNRATE", "name": "失业率", "unit": "percent", "frequency": "monthly"},
    {"id": "NROU", "name": "CBO 自然失业率", "unit": "percent", "frequency": "quarterly"},
    {"id": "PAYEMS", "name": "非农就业人数", "unit": "thousands", "frequency": "monthly"},
    {"id": "GDPC1", "name": "实际 GDP（链式 2017 美元）", "unit": "billions", "frequency": "quarterly"},
    {"id": "NFCI", "name": "芝加哥联储全国金融条件指数", "unit": "index", "frequency": "weekly"},
    {"id": "DGS2", "name": "2 年期美债收益率", "unit": "percent", "frequency": "daily"},
    {"id": "DGS10", "name": "10 年期美债收益率", "unit": "percent", "frequency": "daily"},
    {"id": "T10YIE", "name": "10 年期盈亏平衡通胀率", "unit": "percent", "frequency": "daily"},
    {"id": "BAMLH0A0HYM2", "name": "美国高收益债期权调整利差", "unit": "percent", "frequency": "daily"},
]


def fetch_series(series: str) -> list[tuple[date, float]]:
    request = urllib.request.Request(
        FRED_CSV.format(series=series),
        headers={"User-Agent": "Mozilla/5.0 (compatible; FedPolicyRadar/0.1)"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        text = response.read().decode("utf-8-sig", errors="replace")
    rows: list[tuple[date, float]] = []
    for row in csv.DictReader(io.StringIO(text)):
        raw = (row.get(series) or "").strip()
        if raw in ("", "."):
            continue
        rows.append((date.fromisoformat(row["observation_date"]), float(raw)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="抓取 P1 宏观数据")
    parser.add_argument("--as-of", help="基准日 YYYY-MM-DD，默认上海时区当天")
    parser.add_argument("--out-dir", default=str(ROOT / "data/macro"))
    args = parser.parse_args()

    as_of = (
        date.fromisoformat(args.as_of)
        if args.as_of
        else datetime.now(ZoneInfo("Asia/Shanghai")).date()
    )
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    retrieved_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")

    metadata: dict = {
        "source": "Federal Reserve Bank of St. Louis (FRED)",
        "base_url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}",
        "as_of": as_of.isoformat(),
        "retrieved_at": retrieved_at,
        "note": "保存截至基准日的最新可用观测；FRED 为当前版本（可能含修订），非实时 vintage。",
        "series": {},
    }

    for spec in SERIES:
        series = spec["id"]
        rows = [(day, value) for day, value in fetch_series(series) if day <= as_of]
        if not rows:
            raise RuntimeError(f"{series} 截至 {as_of} 没有可用观测")
        path = raw_dir / f"{series}.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["observation_date", "value"])
            for day, value in rows:
                writer.writerow([day.isoformat(), f"{value:.6f}".rstrip("0").rstrip(".")])
        latest_day, latest_value = rows[-1]
        metadata["series"][series] = {
            **spec,
            "latest_date": latest_day.isoformat(),
            "latest_value": latest_value,
            "observations": len(rows),
            "url": f"https://fred.stlouisfed.org/series/{series}",
        }
        print(f"{series}: {len(rows)} 行，最新 {latest_day} = {latest_value:g}")

    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"out={out_dir}")


if __name__ == "__main__":
    main()
