#!/usr/bin/env python3
"""抓取 ALFRED 宏观实时 vintage 数据，供回测消除修订前视。

背景：FRED 的 ``vintage_date`` 参数实测被忽略，直接下载当前版本会包含
决议日之后才发布/修订的观测。ALFRED 提供按日期归档的真实 vintage。

用法:
    python3 scripts/fetch_macro_vintages.py [--force]

输出:
    data/macro/vintages/vintages_long.csv
    data/macro/vintages/metadata.json

长表字段: series, vintage_date, observation_date, value

每个序列的 vintage 日期取回测会议的预测截止日（决议日前 7 天），
因此回测可以直接按截止日取当时已知的观测，无需再叠加发布滞后。
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import subprocess
import sys
import zipfile
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data/macro/raw"
OUT_DIR = ROOT / "data/macro/vintages"
CALENDAR = ROOT / "data/fomc_calendar.json"
WINDOW_START = date(2021, 1, 1)
WINDOW_END = date(2026, 11, 22)
CUTOFF_DAYS = (7, 14, 30)
# ALFRED 单次请求的日期输入长度有限，按 30 个日期分批。
BATCH_SIZE = 30
# 回测宏观信号最多回看 12 个月（同比）与 30 天（利率变化），2019 年起足够。
OBS_START_FLOOR = date(2019, 1, 1)
SERIES = (
    "PCEPILFE",
    "CPILFESL",
    "UNRATE",
    "PAYEMS",
    "GDPC1",
    "NROU",
    "NFCI",
    "BAMLH0A0HYM2",
    "BAA10Y",
    "DGS2",
    "DGS10",
    "T10YIE",
)
URL = "https://alfred.stlouisfed.org/series/downloaddata?seid={series}"


def cutoff_dates() -> list[date]:
    payload = json.loads(CALENDAR.read_text(encoding="utf-8"))
    meetings = [
        row
        for row in payload["meetings"]
        if row.get("type", "policy") == "policy"
        and WINDOW_START <= date.fromisoformat(row["decision_date"]) <= WINDOW_END
    ]
    return sorted(
        {
            date.fromisoformat(row["decision_date"]) - timedelta(days=days)
            for row in meetings
            for days in CUTOFF_DAYS
        }
    )


def series_bounds(series: str) -> tuple[date, date]:
    path = RAW_DIR / f"{series}.csv"
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    days = [date.fromisoformat(row["observation_date"]) for row in rows]
    return min(days), max(days)


def curl_post(series: str, dates: list[date], start: date, end: date) -> tuple[bytes, str]:
    fields = [
        ("form[file_type]", "2"),
        ("form[file_format]", "csv"),
        ("form[units]", "lin"),
        ("form[entered_vintage_dates]", " ".join(day.isoformat() for day in dates)),
        ("form[obs_start_date]", start.isoformat()),
        ("form[obs_end_date]", end.isoformat()),
        ("form[download_data]", "Download data"),
    ]
    command = ["curl", "-sS", "-L", "--max-time", "180"]
    for key, value in fields:
        command += ["--data-urlencode", f"{key}={value}"]
    command.append(URL.format(series=series))
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"curl 失败({result.returncode}): {series} {result.stderr[:200]}")
    return result.stdout, result.stdout[:2000].decode("utf-8", errors="ignore")


def parse_error(html: str) -> str | None:
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    match = re.search(r"(Data is not available after \d{4}-\d{2}-\d{2})", text)
    if match:
        return match.group(1)
    match = re.search(
        r"(Observation start date can not be before \d{4}-\d{2}-\d{2})", text
    )
    if match:
        return match.group(1)
    return None


def parse_zip(payload: bytes, series: str) -> list[tuple[str, date, date, float]]:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not names:
            raise RuntimeError(f"{series} vintage zip 内没有 CSV")
        raw = archive.read(names[0]).decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(raw))
    fieldnames = reader.fieldnames or []
    vintages: dict[str, date] = {}
    for name in fieldnames:
        if name == "observation_date":
            continue
        match = re.fullmatch(rf"{re.escape(series)}_(\d{{8}})", name)
        if match:
            vintages[name] = date(
                int(match.group(1)[:4]),
                int(match.group(1)[4:6]),
                int(match.group(1)[6:]),
            )
    if not vintages:
        raise RuntimeError(f"{series} vintage CSV 缺少日期列: {fieldnames[:3]}")
    rows: list[tuple[str, date, date, float]] = []
    for row in reader:
        day_raw = row.get("observation_date") or ""
        try:
            observation = date.fromisoformat(day_raw)
        except ValueError:
            continue
        for column, vintage in vintages.items():
            value_raw = row.get(column, "")
            if value_raw in ("", ".", None):
                continue
            rows.append((series, vintage, observation, float(value_raw)))
    return rows


def valid_dates(dates: list[date], start: date) -> list[date]:
    # BAMLH0A0HYM2 等序列历史较短；早于首个观测的日期无法取 vintage。
    return [day for day in dates if day >= start]


def fetch_series(series: str, dates: list[date], force: bool) -> list[tuple]:
    cache = OUT_DIR / f"{series}.csv"
    cached: list[tuple] = []
    if cache.exists() and not force:
        cached = [
            (
                row["series"],
                date.fromisoformat(row["vintage_date"]),
                date.fromisoformat(row["observation_date"]),
                float(row["value"]),
            )
            for row in csv.DictReader(cache.open(newline="", encoding="utf-8"))
        ]

    first_obs, last_obs = series_bounds(series)
    start = max(OBS_START_FLOOR, first_obs)
    end = min(last_obs, date(2026, 11, 19))
    requested = valid_dates(dates, start)
    cached_dates = {row[1] for row in cached}
    missing = [day for day in requested if day not in cached_dates]
    if not missing:
        print(f"{series}: 复用缓存 {cache} ({len(cached)} rows)")
        return cached
    if cached:
        print(f"{series}: 缓存 {len(cached_dates)} vintages，补抓 {len(missing)} 个日期")

    rows: list[tuple] = list(cached)
    for offset in range(0, len(missing), BATCH_SIZE):
        batch = missing[offset : offset + BATCH_SIZE]
        payload, preview = curl_post(series, batch, start, end)
        if not payload.startswith(b"PK"):
            message = parse_error(preview)
            if message and "after" in message:
                new_end = date.fromisoformat(message.rsplit(" ", 1)[-1])
                print(f"{series}: 结束日回退 {end} -> {new_end}")
                end = new_end
                payload, preview = curl_post(series, batch, start, end)
            elif message and "before" in message:
                new_start = date.fromisoformat(message.rsplit(" ", 1)[-1])
                print(f"{series}: 起始日回退 {start} -> {new_start}")
                start = new_start
                batch = [day for day in batch if day >= start]
                payload, preview = curl_post(series, batch, start, end)
        if not payload.startswith(b"PK"):
            raise RuntimeError(f"{series} {batch[0]}..{batch[-1]} 下载失败: {preview[:200]}")
        parsed = parse_zip(payload, series)
        rows.extend(parsed)
        print(
            f"{series}: {batch[0]}..{batch[-1]} -> "
            f"{len({row[1] for row in parsed})} vintages / {len(parsed)} rows"
        )
    rows.sort(key=lambda item: (item[1], item[2]))
    with cache.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["series", "vintage_date", "observation_date", "value"])
        for item in rows:
            writer.writerow([item[0], item[1].isoformat(), item[2].isoformat(), item[3]])
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="忽略缓存重新抓取")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dates = cutoff_dates()
    print(f"vintage cutoffs: {len(dates)} ({dates[0]} .. {dates[-1]})")

    all_rows: list[tuple] = []
    for series in SERIES:
        all_rows.extend(fetch_series(series, dates, args.force))

    out = OUT_DIR / "vintages_long.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["series", "vintage_date", "observation_date", "value"])
        for item in sorted(all_rows, key=lambda row: (row[0], row[1], row[2])):
            writer.writerow([item[0], item[1].isoformat(), item[2].isoformat(), item[3]])
    metadata = {
        "source": "ALFRED (Archival Federal Reserve Economic Data)",
        "url": "https://alfred.stlouisfed.org/",
        "retrieved_at": date.today().isoformat(),
        "cutoff_days": list(CUTOFF_DAYS),
        "cutoff_dates": [day.isoformat() for day in dates],
        "series": list(SERIES),
        "note": "vintage_date 为会议预测截止日；observation_date <= vintage_date。",
    }
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"vintages -> {out} ({len(all_rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
