"""P0 管道：读取快照 -> 隐含路径 -> 三分类概率 -> JSON / Markdown。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from .calendar import Calendar, Meeting
from .contracts import contract_symbol, parse_contract
from .implied import StepEstimate, estimate_step, implied_rate, month_days
from .market import Quote, RateSnapshot, load_quotes, load_rate_snapshot
from .probability import most_likely, three_class_probabilities

METHOD_LABELS = {
    "next_clean_month": "次月干净合约",
    "meeting_month": "当月合约日权重",
}

TOLERANCE_BP = 2.0
LIQUIDITY_MIN_VOLUME = 1000


def _r(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _meeting_dict(meeting: Meeting) -> dict:
    return {
        "year": meeting.year,
        "month": meeting.month,
        "start": meeting.start.isoformat(),
        "end": meeting.end.isoformat(),
        "decision_date": meeting.decision_date.isoformat(),
        "effective_date": meeting.effective_date.isoformat(),
    }


def _rate_on(day: date, current_rate: float, anchors: list[tuple[date, float]]) -> float:
    rate = current_rate
    for effective, value in anchors:
        if effective <= day:
            rate = value
    return rate


def _model_month_rate(
    year: int, month: int, current_rate: float, anchors: list[tuple[date, float]]
) -> float:
    total = month_days(year, month)
    total_rate = sum(
        _rate_on(date(year, month, day), current_rate, anchors)
        for day in range(1, total + 1)
    )
    return total_rate / total


def _thresholds(prob: dict) -> dict:
    return {
        "probability": {
            "cut": _r(prob["cut"]),
            "hold": _r(prob["hold"]),
            "hike": _r(prob["hike"]),
        },
        "expected_bp": _r(prob["expected_bp"], 2),
        "expected_steps": _r(prob["expected_steps"], 4),
        "beyond_one_step": bool(prob["beyond_one_step"]),
        "most_likely": most_likely(prob),
    }


def build_baseline(
    calendar_path: str | Path,
    zq_path: str | Path,
    rates_path: str | Path,
    as_of: date,
    n_meetings: int = 4,
    step: float = 0.25,
    min_post_days: int = 7,
) -> dict:
    calendar = Calendar.load(calendar_path)
    quotes = load_quotes(zq_path)
    rates = load_rate_snapshot(rates_path, as_of)

    meetings = calendar.upcoming(as_of, n_meetings)
    if not meetings:
        raise ValueError(f"{as_of} 之后没有可用的政策会议")

    current_rate = rates.decision_rate
    steps: list[StepEstimate] = []
    pre_rate = current_rate
    for meeting in meetings:
        symbol = contract_symbol(meeting.year, meeting.month)
        quote = quotes.get(symbol)
        if quote is None:
            raise ValueError(f"缺少会议当月合约行情: {symbol} ({meeting.decision_date})")
        estimate = estimate_step(
            meeting, quote, pre_rate, calendar, quotes, min_post_days=min_post_days
        )
        steps.append(estimate)
        pre_rate = estimate.post_rate

    used_quotes = [quotes[est.symbol] for est in steps]
    for est in steps:
        if est.method == "next_clean_month":
            nxt = contract_symbol(*_next_month(est.meeting.year, est.meeting.month))
            if nxt in quotes:
                used_quotes.append(quotes[nxt])
    quote_date = max(q.as_of for q in used_quotes)

    anchors = [(est.meeting.effective_date, est.post_rate) for est in steps]
    path: list[dict] = []
    for index, est in enumerate(steps, start=1):
        prob = three_class_probabilities(est.post_rate, est.pre_rate, step)
        cumulative_bp = (est.post_rate - current_rate) * 100
        entry = {
            "index": index,
            "meeting": _meeting_dict(est.meeting),
            "contract": est.symbol,
            "contract_month": f"{est.meeting.year}-{est.meeting.month:02d}",
            "contract_close": _r(est.contract_close, 4),
            "contract_implied_rate": _r(est.contract_rate),
            "contract_volume": int(est.contract_volume),
            "method": est.method,
            "method_label": METHOD_LABELS.get(est.method, est.method),
            "pre_rate": _r(est.pre_rate),
            "post_rate": _r(est.post_rate),
            "change_bp": _r(est.change_bp, 2),
            "cumulative_change_bp": _r(cumulative_bp, 2),
            "days": {
                "month": est.total_days,
                "pre": est.pre_days,
                "post": est.post_days,
            },
            "amplification": _r(est.amplification, 3),
            "estimation_noise_bp": _r(est.estimation_noise_bp, 2),
            "reliability": est.reliability,
            "note": est.note,
        }
        entry.update(_thresholds(prob))
        path.append(entry)

    cross_checks = _cross_checks(steps, quotes, current_rate, anchors, as_of)
    warnings = _warnings(steps, quotes, rates, quote_date, as_of, cross_checks)

    path_symbols = {est.symbol for est in steps}
    market_curve: list[dict] = []
    for symbol, quote in quotes.items():
        try:
            year, month = parse_contract(symbol)
        except ValueError:
            continue
        market_curve.append(
            {
                "symbol": symbol,
                "month": f"{year}-{month:02d}",
                "as_of": quote.as_of.isoformat(),
                "close": _r(quote.close, 4),
                "implied_rate": _r(implied_rate(quote.close)),
                "volume": int(quote.volume),
                "used_in_path": symbol in path_symbols,
            }
        )
    market_curve.sort(key=lambda row: row["month"])

    first, last = steps[0], steps[-1]
    next_prob = path[0]
    summary = {
        "next_meeting_date": first.meeting.decision_date.isoformat(),
        "cut": next_prob["probability"]["cut"],
        "hold": next_prob["probability"]["hold"],
        "hike": next_prob["probability"]["hike"],
        "most_likely": next_prob["most_likely"],
        "expected_post_rate": path[0]["post_rate"],
        "expected_change_bp": path[0]["change_bp"],
        "path_end_meeting": last.meeting.decision_date.isoformat(),
        "path_end_rate": path[-1]["post_rate"],
        "path_cumulative_bp": path[-1]["cumulative_change_bp"],
        "path_hikes_priced": _r(path[-1]["cumulative_change_bp"] / (step * 100), 2),
    }

    return {
        "schema_version": "p0.1.0",
        "as_of": as_of.isoformat(),
        "policy_basis": {
            "current_rate": _r(current_rate),
            "rate_basis": rates.rate_basis,
            "current_rate_as_of": rates.effr_as_of.isoformat() if rates.effr_as_of else None,
            "target_range": {
                "lower": _r(rates.target_lower, 4),
                "upper": _r(rates.target_upper, 4),
                "midpoint": _r(rates.midpoint),
                "as_of": rates.target_as_of.isoformat(),
            },
            "step_bp": int(round(step * 100)),
            "decision_effective_lag_days": 1,
            "contract": "CME 30-Day Fed Funds Futures (ZQ)",
            "settlement": "月度平均 EFFR",
            "method_priority": "次月无政策会议时用次月合约；否则用当月合约按生效日拆分日权重",
            "min_post_days": min_post_days,
        },
        "summary": summary,
        "path": path,
        "market_curve": market_curve,
        "curve_cross_check": cross_checks,
        "data_quality": {
            "quote_date": quote_date.isoformat(),
            "quote_age_days": (as_of - quote_date).days,
            "target_date": rates.target_as_of.isoformat(),
            "target_age_days": (as_of - rates.target_as_of).days,
            "effr_date": rates.effr_as_of.isoformat() if rates.effr_as_of else None,
            "effr_age_days": (as_of - rates.effr_as_of).days if rates.effr_as_of else None,
            "liquidity_min_volume": LIQUIDITY_MIN_VOLUME,
            "warnings": warnings,
        },
        "provenance": [
            {
                "item": "FOMC 会议日历",
                "source": "Federal Reserve",
                "url": calendar.source,
                "as_of": calendar.retrieved_at,
            },
            {
                "item": "目标区间上/下限",
                "source": "FRED (DFEDTARU / DFEDTARL)",
                "url": "https://fred.stlouisfed.org/series/DFEDTARU",
                "as_of": rates.target_as_of.isoformat(),
            },
            {
                "item": "有效联邦基金利率",
                "source": "FRED (EFFR)",
                "url": "https://fred.stlouisfed.org/series/EFFR",
                "as_of": rates.effr_as_of.isoformat() if rates.effr_as_of else None,
            },
            {
                "item": "ZQ 期货收盘价",
                "source": "Barchart（第三方延迟行情）",
                "url": "https://www.barchart.com/futures/quotes/ZQZ26",
                "as_of": quote_date.isoformat(),
            },
        ],
        "caveats": [
            "ZQ 以月度平均 EFFR 结算；会前天数沿用当前 EFFR，未对月内资金面波动建模。",
            "行情来自 Barchart 第三方延迟数据，非 CME 直连，仅用于 P0 基准。",
            "ZQ 最小价格变动 0.005，对应 0.5bp 利率精度；日权重分解会放大该误差。",
            "月末会议会后天数很少时，优先使用次月干净合约，避免反推放大。",
            "2027-10-26/27 为货币政策战略会议，不含利率决议，已排除出政策路径。",
            "P0 只给出市场隐含先验，尚未叠加宏观反应函数与美联储沟通文本信号。",
        ],
    }


def _next_month(year: int, month: int) -> tuple[int, int]:
    return (year + 1, 1) if month == 12 else (year, month + 1)


def _cross_checks(
    steps: list[StepEstimate],
    quotes: dict[str, Quote],
    current_rate: float,
    anchors: list[tuple[date, float]],
    as_of: date,
) -> list[dict]:
    """用拟合出的阶梯利率重算月均，与市场合约对比。"""
    first = steps[0].meeting
    last = steps[-1].meeting
    end_year, end_month = _next_month(last.year, last.month)
    checks: list[dict] = []

    for symbol, quote in sorted(quotes.items(), key=lambda kv: kv[1].as_of):
        try:
            year, month = _parse(symbol)
        except ValueError:
            continue
        if (year, month) < (first.year, first.month) or (year, month) > (end_year, end_month):
            continue
        model_rate = _model_month_rate(year, month, current_rate, anchors)
        observed = implied_rate(quote.close)
        diff_bp = (model_rate - observed) * 100
        checks.append(
            {
                "month": f"{year}-{month:02d}",
                "symbol": symbol,
                "as_of": quote.as_of.isoformat(),
                "observed_rate": _r(observed),
                "model_rate": _r(model_rate),
                "diff_bp": _r(diff_bp, 2),
                "pass": abs(diff_bp) <= TOLERANCE_BP,
            }
        )
    return checks


def _parse(symbol: str) -> tuple[int, int]:
    return parse_contract(symbol)


def _warnings(
    steps: list[StepEstimate],
    quotes: dict[str, Quote],
    rates: RateSnapshot,
    quote_date: date,
    as_of: date,
    cross_checks: list[dict],
) -> list[str]:
    warnings: list[str] = []
    age = (as_of - quote_date).days
    if age > 0:
        warnings.append(f"行情为 {quote_date} 收盘价，距基准日 {age} 天（非实时）")
    if rates.effr_as_of and (as_of - rates.effr_as_of).days > 0:
        warnings.append(
            f"EFFR 最新观测为 {rates.effr_as_of}，距基准日 {(as_of - rates.effr_as_of).days} 天"
        )
    thin = [
        step.symbol
        for step in steps
        if quotes[step.symbol].volume < LIQUIDITY_MIN_VOLUME
    ]
    if thin:
        warnings.append(f"成交量偏低（<{LIQUIDITY_MIN_VOLUME} 手）：{', '.join(thin)}")
    weak = [step.symbol for step in steps if step.reliability != "high"]
    if weak:
        warnings.append(f"估算可靠性非 high：{', '.join(weak)}")
    failed = [row["month"] for row in cross_checks if not row["pass"]]
    if failed:
        warnings.append(f"曲线一致性校验偏差 > {TOLERANCE_BP}bp：{', '.join(failed)}")
    return warnings
