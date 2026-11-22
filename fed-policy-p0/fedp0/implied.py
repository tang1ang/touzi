"""隐含利率与会议步长分解。"""

from __future__ import annotations

import calendar as calmod
from dataclasses import dataclass
from datetime import date

from .calendar import Calendar, Meeting
from .contracts import contract_symbol
from .market import Quote

PRICE_TO_RATE = 100.0
DEFAULT_MIN_POST_DAYS = 7
PRICE_NOISE_BP = 1.0


def implied_rate(price: float) -> float:
    """ZQ 报价转隐含利率（百分数）：r = 100 - price。"""
    return PRICE_TO_RATE - price


def month_days(year: int, month: int) -> int:
    return calmod.monthrange(year, month)[1]


def next_month(year: int, month: int) -> tuple[int, int]:
    return (year + 1, 1) if month == 12 else (year, month + 1)


def split_month_days(year: int, month: int, effective_date: date) -> tuple[int, int]:
    """返回会议当月「会前天数 / 会后天数」。

    决议在 decision_date 公布，effective_date 起生效，决议日当天仍按旧利率计息。
    若 effective_date 落在下月（月末会议），当月全部为会前天数。
    """
    total = month_days(year, month)
    if (effective_date.year, effective_date.month) != (year, month):
        return total, 0
    pre = effective_date.day - 1
    return pre, total - pre


def decompose_post_rate(
    month_rate: float,
    pre_rate: float,
    year: int,
    month: int,
    effective_date: date,
) -> float:
    """按日权重从月度均值反推会后利率：r_post = (r_month*D - r_pre*D_pre)/D_post。"""
    total = month_days(year, month)
    pre_days, post_days = split_month_days(year, month, effective_date)
    if post_days <= 0:
        raise ValueError(f"{year}-{month:02d} 无会后天数，无法用当月合约反推")
    return (month_rate * total - pre_rate * pre_days) / post_days


@dataclass
class StepEstimate:
    meeting: Meeting
    symbol: str
    method: str
    contract_close: float
    contract_rate: float
    contract_volume: float
    pre_rate: float
    post_rate: float
    pre_days: int
    post_days: int
    total_days: int
    amplification: float
    estimation_noise_bp: float
    reliability: str
    note: str

    @property
    def change_bp(self) -> float:
        # 利率以百分数表示：0.01 = 1bp，故乘 100。
        return (self.post_rate - self.pre_rate) * 100


def _grade(noise_bp: float) -> str:
    if noise_bp <= 2.0:
        return "high"
    if noise_bp <= 5.0:
        return "medium"
    return "low"


def estimate_step(
    meeting: Meeting,
    quote: Quote,
    pre_rate: float,
    calendar: Calendar,
    quotes: dict[str, Quote],
    min_post_days: int = DEFAULT_MIN_POST_DAYS,
) -> StepEstimate:
    """估计一次会议后的利率。

    优先使用「次月无会议」的干净合约；否则用当月合约按日权重分解。
    """
    year, month = meeting.year, meeting.month
    total = month_days(year, month)
    pre_days, post_days = split_month_days(year, month, meeting.effective_date)
    month_rate = implied_rate(quote.close)

    ny, nm = next_month(year, month)
    next_symbol = contract_symbol(ny, nm)
    next_quote = quotes.get(next_symbol)
    clean_next = next_quote is not None and not calendar.has_policy_in_month(ny, nm)

    if clean_next:
        post_rate = implied_rate(next_quote.close)
        method = "next_clean_month"
        amplification = 1.0
        noise = PRICE_NOISE_BP
        note = f"次月（{ny}-{nm:02d}）无政策会议，{next_symbol} 隐含利率即为会后利率"
    else:
        post_rate = decompose_post_rate(month_rate, pre_rate, year, month, meeting.effective_date)
        method = "meeting_month"
        amplification = total / post_days if post_days else float("inf")
        noise = amplification * PRICE_NOISE_BP
        note = f"按决议生效日拆分 {total} 天：会前 {pre_days} 天、会后 {post_days} 天"
        if post_days < min_post_days:
            note += "；会后天数偏少，反推放大明显，可靠性下调"

    return StepEstimate(
        meeting=meeting,
        symbol=quote.symbol,
        method=method,
        contract_close=quote.close,
        contract_rate=month_rate,
        contract_volume=quote.volume,
        pre_rate=pre_rate,
        post_rate=post_rate,
        pre_days=pre_days,
        post_days=post_days,
        total_days=total,
        amplification=amplification,
        estimation_noise_bp=noise,
        reliability=_grade(noise),
        note=note,
    )
