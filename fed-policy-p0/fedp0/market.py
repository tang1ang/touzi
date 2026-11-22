"""行情与利率快照读取。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class Quote:
    symbol: str
    as_of: date
    close: float
    volume: float = 0.0
    open_interest: float = 0.0


@dataclass(frozen=True)
class RateSnapshot:
    effr: float
    effr_as_of: date | None
    target_lower: float
    target_upper: float
    target_as_of: date
    rate_basis: str

    @property
    def midpoint(self) -> float:
        return (self.target_lower + self.target_upper) / 2

    @property
    def decision_rate(self) -> float:
        """推算使用的当前利率：优先 EFFR（ZQ 结算标的），否则用目标区间中值。"""
        return self.effr if self.rate_basis == "EFFR" else self.midpoint


def load_quotes(path: str | Path) -> dict[str, Quote]:
    """读取 ZQ 曲线 CSV；同一合约多行时取最新日期。"""
    quotes: dict[str, Quote] = {}
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            symbol = row["symbol"].strip().upper()
            quote = Quote(
                symbol=symbol,
                as_of=date.fromisoformat(row["as_of"]),
                close=float(row["close"]),
                volume=float(row.get("volume") or 0),
                open_interest=float(row.get("open_interest") or 0),
            )
            if symbol not in quotes or quote.as_of >= quotes[symbol].as_of:
                quotes[symbol] = quote
    if not quotes:
        raise ValueError(f"行情文件为空: {path}")
    return quotes


def load_rate_snapshot(path: str | Path, as_of: date) -> RateSnapshot:
    """读取 series/as_of/value/source 格式的快照，取每个序列不晚于 as_of 的最新值。"""
    latest: dict[str, tuple[date, float]] = {}
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            series = row["series"].strip()
            row_date = date.fromisoformat(row["as_of"])
            if row_date > as_of:
                continue
            value = float(row["value"])
            if series not in latest or row_date >= latest[series][0]:
                latest[series] = (row_date, value)

    for required in ("DFEDTARL", "DFEDTARU"):
        if required not in latest:
            raise ValueError(f"利率快照缺少 {required}")

    lower_date, lower = latest["DFEDTARL"]
    upper_date, upper = latest["DFEDTARU"]
    if "EFFR" in latest:
        effr_date, effr = latest["EFFR"]
        if lower <= effr <= upper + 0.001:
            return RateSnapshot(
                effr=effr,
                effr_as_of=effr_date,
                target_lower=lower,
                target_upper=upper,
                target_as_of=max(lower_date, upper_date),
                rate_basis="EFFR",
            )
    return RateSnapshot(
        effr=0.0,
        effr_as_of=None,
        target_lower=lower,
        target_upper=upper,
        target_as_of=max(lower_date, upper_date),
        rate_basis="target_midpoint",
    )
