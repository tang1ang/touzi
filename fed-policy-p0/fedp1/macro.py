"""P1 宏观反应函数：通胀、就业、增长、金融条件与 Taylor 规则缺口。"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from fedp0.probability import most_likely, three_class_probabilities


@dataclass(frozen=True)
class MacroConfig:
    target_inflation: float = 2.0
    r_star: float = 0.5
    okun_coefficient: float = 2.0
    taylor_inflation_coefficient: float = 0.5
    taylor_output_coefficient: float = 0.5
    inflation_half_width: float = 1.0
    unemployment_half_width: float = 0.5
    payrolls_breakeven_k: float = 100.0
    payrolls_half_width_k: float = 100.0
    potential_growth: float = 1.8
    growth_half_width: float = 2.0
    nfci_half_width: float = 0.5
    hy_oas_center: float = 4.0
    hy_oas_half_width: float = 2.0
    baa10y_center: float = 2.0
    baa10y_half_width: float = 1.0
    labor_momentum_adjust_weight: float = 0.10
    financial_adjust_weight: float = 0.25
    convergence_lambda: float = 0.25
    max_step: float = 0.25


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _scale(value: float, center: float, half_width: float) -> float:
    if half_width <= 0:
        raise ValueError("half_width 必须为正")
    return _clip((value - center) / half_width)


def load_series(path: str | Path) -> list[tuple[date, float]]:
    rows: list[tuple[date, float]] = []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append((date.fromisoformat(row["observation_date"]), float(row["value"])))
    rows.sort(key=lambda item: item[0])
    if not rows:
        raise ValueError(f"序列为空: {path}")
    return rows


def latest_on_or_before(rows: list[tuple[date, float]], day: date) -> tuple[date, float]:
    eligible = [item for item in rows if item[0] <= day]
    if not eligible:
        raise ValueError(f"截至 {day} 没有可用观测")
    return eligible[-1]


def observation_before(rows: list[tuple[date, float]], day: date) -> tuple[date, float]:
    return latest_on_or_before(rows, day)


def yoy(rows: list[tuple[date, float]], as_of: date) -> dict:
    latest_day, latest_value = latest_on_or_before(rows, as_of)
    base_day, base_value = observation_before(rows, latest_day - timedelta(days=365))
    return {
        "latest_date": latest_day.isoformat(),
        "latest": latest_value,
        "base_date": base_day.isoformat(),
        "base": base_value,
        "pct": (latest_value / base_value - 1.0) * 100.0,
    }


def annualized_change(rows: list[tuple[date, float]], as_of: date, days: int) -> dict:
    latest_day, latest_value = latest_on_or_before(rows, as_of)
    base_day, base_value = observation_before(rows, latest_day - timedelta(days=days))
    elapsed = (latest_day - base_day).days
    if elapsed <= 0:
        raise ValueError("通胀动量基期与最新观测相同")
    value = ((latest_value / base_value) ** (365.25 / elapsed) - 1.0) * 100.0
    return {
        "latest_date": latest_day.isoformat(),
        "latest": latest_value,
        "base_date": base_day.isoformat(),
        "base": base_value,
        "elapsed_days": elapsed,
        "pct": value,
    }


def quarterly_annualized(rows: list[tuple[date, float]], as_of: date) -> dict:
    latest_day, latest_value = latest_on_or_before(rows, as_of)
    eligible = [item for item in rows if item[0] < latest_day]
    if not eligible:
        raise ValueError("GDP 序列不足两期")
    previous_day, previous_value = eligible[-1]
    value = ((latest_value / previous_value) ** 4 - 1.0) * 100.0
    return {
        "latest_date": latest_day.isoformat(),
        "latest": latest_value,
        "previous_date": previous_day.isoformat(),
        "previous": previous_value,
        "pct": value,
    }


def payrolls_momentum(rows: list[tuple[date, float]], as_of: date, months: int = 3) -> dict:
    latest_day, latest_value = latest_on_or_before(rows, as_of)
    eligible = [item for item in rows if item[0] <= latest_day]
    if len(eligible) <= months:
        raise ValueError("非农序列不足")
    base_day, base_value = eligible[-(months + 1)]
    average_k = (latest_value - base_value) / months
    return {
        "latest_date": latest_day.isoformat(),
        "base_date": base_day.isoformat(),
        "average_k": average_k,
    }


def change_over_days(rows: list[tuple[date, float]], as_of: date, days: int) -> dict:
    latest_day, latest_value = latest_on_or_before(rows, as_of)
    base_day, base_value = observation_before(rows, latest_day - timedelta(days=days))
    return {
        "latest_date": latest_day.isoformat(),
        "latest": latest_value,
        "base_date": base_day.isoformat(),
        "base": base_value,
        "change": latest_value - base_value,
        "change_bp": (latest_value - base_value) * 100.0,
    }


@dataclass
class MacroSignal:
    features: dict
    scores: dict
    taylor: dict
    config: MacroConfig = field(default_factory=MacroConfig)

    def to_dict(self) -> dict:
        return {
            "config": {
                "target_inflation": self.config.target_inflation,
                "r_star": self.config.r_star,
                "okun_coefficient": self.config.okun_coefficient,
                "taylor_inflation_coefficient": self.config.taylor_inflation_coefficient,
                "taylor_output_coefficient": self.config.taylor_output_coefficient,
                "convergence_lambda": self.config.convergence_lambda,
                "max_step": self.config.max_step,
            },
            "features": self.features,
            "scores": {key: round(value, 4) for key, value in self.scores.items()},
            "taylor": {
                key: (round(value, 4) if isinstance(value, float) else value)
                for key, value in self.taylor.items()
            },
        }


def build_macro_signal(
    macro_dir: str | Path,
    as_of: date,
    current_rate: float,
    config: MacroConfig | None = None,
    publication_lags: dict[str, int] | None = None,
) -> MacroSignal:
    """构建宏观信号。

    publication_lags 为「发布滞后天数」：某序列只使用 as_of - lag 之前已发布的观测，
    用于回测中避免前视偏差；默认不滞后（与实时管道口径一致）。
    """
    cfg = config or MacroConfig()
    raw_dir = Path(macro_dir) / "raw"
    lags = publication_lags or {}

    def series(series_id: str) -> list[tuple[date, float]]:
        rows = load_series(raw_dir / f"{series_id}.csv")
        lag = int(lags.get(series_id, 0))
        if lag <= 0:
            return rows
        cutoff = as_of - timedelta(days=lag)
        return [row for row in rows if row[0] <= cutoff]

    pce_yoy = yoy(series("PCEPILFE"), as_of)
    pce_3m = annualized_change(series("PCEPILFE"), as_of, 92)
    cpi_yoy = yoy(series("CPILFESL"), as_of)
    cpi_3m = annualized_change(series("CPILFESL"), as_of, 92)

    unemployment_day, unemployment = latest_on_or_before(series("UNRATE"), as_of)
    nrou_day, nrou = latest_on_or_before(series("NROU"), as_of)
    unemployment_gap = unemployment - nrou

    payrolls = payrolls_momentum(series("PAYEMS"), as_of, 3)
    gdp_yoy = yoy(series("GDPC1"), as_of)
    gdp_qoq = quarterly_annualized(series("GDPC1"), as_of)

    nfci = change_over_days(series("NFCI"), as_of, 28)
    # ICE BofA 高收益 OAS 在 FRED 仅约三年历史。更早的会议改用 BAA10Y
    # （穆迪 Baa 企业债与 10 年国债利差）作为信用利差替代，并在特征中标注。
    try:
        credit_day, credit_spread = latest_on_or_before(
            series("BAMLH0A0HYM2"), as_of
        )
        credit_series = "BAMLH0A0HYM2"
        credit_center = cfg.hy_oas_center
        credit_half_width = cfg.hy_oas_half_width
    except ValueError:
        credit_day, credit_spread = latest_on_or_before(series("BAA10Y"), as_of)
        credit_series = "BAA10Y"
        credit_center = cfg.baa10y_center
        credit_half_width = cfg.baa10y_half_width
    dgs2 = change_over_days(series("DGS2"), as_of, 30)
    dgs10_day, dgs10 = latest_on_or_before(series("DGS10"), as_of)
    t10yie_day, t10yie = latest_on_or_before(series("T10YIE"), as_of)

    inflation_score = _clip(
        0.5 * _scale(pce_yoy["pct"], cfg.target_inflation, cfg.inflation_half_width)
        + 0.3 * _scale(pce_3m["pct"], cfg.target_inflation, cfg.inflation_half_width)
        + 0.2 * _scale(cpi_3m["pct"], cfg.target_inflation, cfg.inflation_half_width)
    )
    labor_momentum_score = _scale(
        payrolls["average_k"], cfg.payrolls_breakeven_k, cfg.payrolls_half_width_k
    )
    labor_score = _clip(
        0.7 * _scale(-unemployment_gap, 0.0, cfg.unemployment_half_width)
        + 0.3 * labor_momentum_score
    )
    growth_score = _scale(gdp_qoq["pct"], cfg.potential_growth, cfg.growth_half_width)
    financial_score = _clip(
        0.6 * (-_scale(nfci["latest"], 0.0, cfg.nfci_half_width))
        + 0.4 * (-_scale(credit_spread, credit_center, credit_half_width))
    )

    output_gap = -cfg.okun_coefficient * unemployment_gap
    taylor_rate = (
        cfg.r_star
        + pce_yoy["pct"]
        + cfg.taylor_inflation_coefficient * (pce_yoy["pct"] - cfg.target_inflation)
        + cfg.taylor_output_coefficient * output_gap
    )
    labor_adjust = cfg.labor_momentum_adjust_weight * labor_momentum_score
    financial_adjust = cfg.financial_adjust_weight * financial_score
    macro_target_rate = taylor_rate + labor_adjust + financial_adjust
    policy_gap_bp = (current_rate - taylor_rate) * 100.0
    taylor_signal = _clip(-policy_gap_bp / 50.0)
    composite_score = _clip(
        0.5 * taylor_signal
        + 0.2 * inflation_score
        + 0.1 * labor_score
        + 0.1 * growth_score
        + 0.1 * financial_score
    )

    features = {
        "core_pce": {
            "latest_date": pce_yoy["latest_date"],
            "latest": round(pce_yoy["latest"], 4),
            "yoy_pct": round(pce_yoy["pct"], 4),
            "yoy_base_date": pce_yoy["base_date"],
            "ann3m_pct": round(pce_3m["pct"], 4),
            "ann3m_base_date": pce_3m["base_date"],
        },
        "core_cpi": {
            "latest_date": cpi_yoy["latest_date"],
            "yoy_pct": round(cpi_yoy["pct"], 4),
            "yoy_base_date": cpi_yoy["base_date"],
            "ann3m_pct": round(cpi_3m["pct"], 4),
            "ann3m_base_date": cpi_3m["base_date"],
        },
        "labor": {
            "unemployment_date": unemployment_day.isoformat(),
            "unemployment": unemployment,
            "nrou_date": nrou_day.isoformat(),
            "nrou": round(nrou, 4),
            "unemployment_gap_pp": round(unemployment_gap, 4),
            "payrolls_3m_avg_k": round(payrolls["average_k"], 2),
            "payrolls_latest_date": payrolls["latest_date"],
        },
        "growth": {
            "gdp_latest_date": gdp_qoq["latest_date"],
            "gdp_yoy_pct": round(gdp_yoy["pct"], 4),
            "gdp_qoq_ann_pct": round(gdp_qoq["pct"], 4),
        },
        "financial_conditions": {
            "nfci": nfci["latest"],
            "nfci_date": nfci["latest_date"],
            "nfci_change_4w": round(nfci["change"], 4),
            "credit_spread_series": credit_series,
            "credit_spread": credit_spread,
            "credit_spread_date": credit_day.isoformat(),
            "hy_oas": credit_spread if credit_series == "BAMLH0A0HYM2" else None,
            "hy_oas_date": (
                credit_day.isoformat() if credit_series == "BAMLH0A0HYM2" else None
            ),
        },
        "market_rates": {
            "dgs2": dgs2["latest"],
            "dgs2_date": dgs2["latest_date"],
            "dgs2_change_1m_bp": round(dgs2["change_bp"], 2),
            "dgs10": dgs10,
            "dgs10_date": dgs10_day.isoformat(),
            "t10yie": t10yie,
            "t10yie_date": t10yie_day.isoformat(),
        },
    }

    scores = {
        "inflation": inflation_score,
        "labor": labor_score,
        "labor_momentum": labor_momentum_score,
        "growth": growth_score,
        "financial_conditions": financial_score,
        "taylor_signal": taylor_signal,
        "composite": composite_score,
    }

    taylor = {
        "r_star": cfg.r_star,
        "target_inflation": cfg.target_inflation,
        "output_gap_pct": round(output_gap, 4),
        "taylor_rate": round(taylor_rate, 4),
        "labor_momentum_adjustment_pp": round(labor_adjust, 4),
        "financial_conditions_adjustment_pp": round(financial_adjust, 4),
        "macro_target_rate": round(macro_target_rate, 4),
        "policy_gap_bp": round(policy_gap_bp, 2),
        "current_rate": round(current_rate, 4),
        "convergence_lambda": cfg.convergence_lambda,
        "note": "Taylor 规则含通胀缺口与 Okun 产出缺口；就业动量与金融条件作为目标利率修正项。",
    }

    return MacroSignal(features=features, scores=scores, taylor=taylor, config=cfg)


def build_macro_path(
    signal: MacroSignal,
    market_path: list[dict],
    current_rate: float,
    step: float = 0.25,
) -> list[dict]:
    """独立宏观路径：每次会议向 macro_target_rate 收敛 lambda，单次最多一步。"""
    cfg = signal.config
    target = signal.taylor["macro_target_rate"]
    pre_rate = current_rate
    rows: list[dict] = []
    for market_row in market_path:
        gap = target - pre_rate
        move = max(-cfg.max_step, min(cfg.max_step, gap * cfg.convergence_lambda))
        post_rate = pre_rate + move
        prob = three_class_probabilities(post_rate, pre_rate, step)
        rows.append(
            {
                "index": market_row["index"],
                "meeting": market_row["meeting"],
                "pre_rate": round(pre_rate, 6),
                "change_bp": round(move * 100.0, 2),
                "post_rate": round(post_rate, 6),
                "cumulative_bp": round((post_rate - current_rate) * 100.0, 2),
                "probability": {
                    "cut": round(prob["cut"], 6),
                    "hold": round(prob["hold"], 6),
                    "hike": round(prob["hike"], 6),
                },
                "expected_bp": round(prob["expected_bp"], 2),
                "expected_steps": round(prob["expected_steps"], 4),
                "most_likely": most_likely(prob),
            }
        )
        pre_rate = post_rate
    return rows


def macro_sources(macro_dir: str | Path) -> list[dict]:
    path = Path(macro_dir) / "metadata.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for series_id, row in payload.get("series", {}).items():
        out.append(
            {
                "item": row.get("name", series_id),
                "series": series_id,
                "source": "FRED",
                "url": row.get("url"),
                "as_of": row.get("latest_date"),
                "retrieved_at": payload.get("retrieved_at"),
            }
        )
    return out
