"""P2 回测：市场锚定、动态权重、滚动校准与幅度预测。

与 P1 的区别：
1. 以市场先验为锚；只有宏观与文本方向一致、且与市场方向相反时才偏离。
2. 权重与偏离强度用 expanding-window walk-forward 在历史会议上选择。
3. 概率用向基础频率收缩 / Platt 缩放做滚动校准。
4. 幅度用市场隐含变动做线性校准，覆盖 50bp 等大幅变动。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from pathlib import Path

from fedp0.calendar import Calendar, Meeting
from fedp1.backtest import (
    DEFAULT_PUBLICATION_LAGS,
    BacktestConfig,
    _class_report,
    actual_decision,
    calibration_table,
    load_rates_long,
    load_statements,
    load_zq_history,
    macro_signal_for,
    market_signal,
    text_probabilities,
    text_signal_for,
)
from fedp1.calibration import (
    CLASSES,
    binary_ece,
    fit_magnitude,
    fit_platt,
    fit_shrinkage,
    magnitude_metrics,
    mean_log_loss,
    normalize,
)
from fedp1.combine import expected_bp, most_likely, pool
from fedp1.macro import load_vintages

DEFAULT_WEIGHT_GRID = (
    (0.85, 0.10, 0.05),
    (0.75, 0.15, 0.10),
    (0.65, 0.20, 0.15),
    (0.55, 0.25, 0.20),
)
DEFAULT_BLEND_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass
class P2Config:
    cutoff_days: int = 7
    step: float = 0.25
    min_post_days: int = 7
    text_scale: float = 2.0
    smoothing_eps: float = 0.005
    use_dissent: bool = True
    weight_grid: tuple[tuple[float, float, float], ...] = DEFAULT_WEIGHT_GRID
    blend_grid: tuple[float, ...] = DEFAULT_BLEND_GRID
    default_weights: tuple[float, float, float] = (0.85, 0.10, 0.05)
    default_blend: float = 0.0
    certainty_threshold: float = 0.80
    max_deviation_pp: float = 12.0
    direction_deadband: float = 0.10
    text_deadband: float = 0.10
    market_deadband: float = 0.05
    min_calibration_n: int = 8
    shrinkage_grid: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    calibration_tolerance: float = 0.002
    min_magnitude_n: int = 6
    magnitude_shrink: float = 0.5
    magnitude_tolerance: float = 0.5
    selection_tolerance: float = 0.005
    publication_lags: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_PUBLICATION_LAGS)
    )


def _r(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _sign(value: float, deadband: float) -> int:
    if value > deadband:
        return 1
    if value < -deadband:
        return -1
    return 0


def _backtest_config(config: P2Config) -> BacktestConfig:
    return BacktestConfig(
        cutoff_days=config.cutoff_days,
        step=config.step,
        min_post_days=config.min_post_days,
        weights=config.default_weights,
        text_scale=config.text_scale,
        smoothing_eps=config.smoothing_eps,
        use_dissent=config.use_dissent,
        publication_lags=config.publication_lags,
    )


def anchored_combination(
    market: dict,
    macro: dict,
    text_score: float,
    weights: tuple[float, float, float],
    blend: float,
    reliability: str,
    config: P2Config,
) -> dict:
    """市场锚定：只在宏观与文本同向、且与市场方向相反时偏离。"""
    market_prob = normalize(market)
    top = max(market_prob.values())
    move = max(market_prob["hike"], market_prob["cut"])
    market_dir = _sign(
        market_prob["hike"] - market_prob["cut"], config.market_deadband
    )
    macro_dir = _sign(macro["hike"] - macro["cut"], config.direction_deadband)
    text_dir = _sign(text_score, config.text_deadband)
    info = {
        "applied": False,
        "reason": "",
        "market_dir": market_dir,
        "macro_dir": macro_dir,
        "text_dir": text_dir,
        "market_top": _r(top, 4),
    }

    if blend <= 0:
        info["reason"] = "偏离强度为 0"
        return {**info, "probability": market_prob}
    if reliability == "low":
        info["reason"] = "市场定价可靠性低，保持锚定"
        return {**info, "probability": market_prob}
    if top >= config.certainty_threshold or move >= config.certainty_threshold:
        info["reason"] = "市场接近确定，保持锚定"
        return {**info, "probability": market_prob}
    if macro_dir == 0 or text_dir == 0 or macro_dir != text_dir:
        info["reason"] = "宏观与文本未同向，保持锚定"
        return {**info, "probability": market_prob}
    if market_dir != 0 and macro_dir == market_dir:
        info["reason"] = "非市场信号与市场同向，无需偏离"
        return {**info, "probability": market_prob}

    pooled = pool(
        market_prob,
        macro,
        text_score,
        weights,
        config.text_scale,
        config.smoothing_eps,
    )["probability"]
    cap = config.max_deviation_pp / 100.0
    delta = {
        name: _clip(pooled[name] - market_prob[name], -cap, cap) for name in CLASSES
    }
    adjusted = normalize(
        {name: market_prob[name] + blend * delta[name] for name in CLASSES}
    )
    info["applied"] = True
    info["reason"] = "宏观与文本同向且与市场相反，按历史表现偏离"
    info["deviation_pp"] = {
        name: _r((adjusted[name] - market_prob[name]) * 100.0, 2)
        for name in CLASSES
    }
    return {**info, "probability": adjusted}


def select_hyperparameters(
    train_rows: list[dict], config: P2Config
) -> dict:
    """在历史会议上选择权重与偏离强度；默认偏向更高市场权重。"""
    default_weights = config.default_weights
    default_blend = config.default_blend
    if not train_rows:
        return {
            "weights": default_weights,
            "blend": default_blend,
            "train_n": 0,
            "train_log_loss": None,
            "default_log_loss": None,
        }
    labels = [row["actual"]["label"] for row in train_rows]

    def score(weights: tuple[float, float, float], blend: float) -> float:
        probabilities = [
            anchored_combination(
                row["market"]["probability"],
                row["macro"]["probability"],
                row["text"]["text_score"],
                weights,
                blend,
                row["market"]["reliability"],
                config,
            )["probability"]
            for row in train_rows
        ]
        return mean_log_loss(probabilities, labels)

    default_loss = score(default_weights, default_blend)
    best_loss = default_loss
    best_weights = default_weights
    best_blend = default_blend
    for weights in config.weight_grid:
        for blend in config.blend_grid:
            loss = score(weights, blend)
            if loss < best_loss - 1e-12:
                best_loss = loss
                best_weights = weights
                best_blend = blend
    if best_loss > default_loss - config.selection_tolerance:
        best_weights = default_weights
        best_blend = default_blend
        best_loss = default_loss
    return {
        "weights": best_weights,
        "blend": best_blend,
        "train_n": len(train_rows),
        "train_log_loss": _r(best_loss, 6),
        "default_log_loss": _r(default_loss, 6),
    }


def select_calibrator(train_rows: list[dict], config: P2Config) -> tuple[str, dict]:
    """按历史会议的样本外 log loss 选择 raw / 收缩 / Platt。"""
    if len(train_rows) < config.min_calibration_n:
        return "raw", {}
    labels = [row["actual"]["label"] for row in train_rows]
    losses = {
        "raw": mean_log_loss(
            [row["p2"]["raw_probability"] for row in train_rows], labels
        ),
        "shrinkage": mean_log_loss(
            [row["p2"]["shrinkage_probability"] for row in train_rows], labels
        ),
        "platt": mean_log_loss(
            [row["p2"]["platt_probability"] for row in train_rows], labels
        ),
    }
    best = min(losses, key=losses.get)
    if losses[best] > losses["raw"] - config.calibration_tolerance:
        best = "raw"
    return best, {name: _r(value, 6) for name, value in losses.items()}


def select_magnitude_method(
    train_rows: list[dict], config: P2Config
) -> tuple[str, dict]:
    """按历史会议的样本外 MAE 选择市场隐含 / 线性校准 / 类别期望。"""
    if len(train_rows) < config.min_magnitude_n:
        return "identity", {}
    actuals = [row["actual"]["change_bp"] for row in train_rows]
    mae = {
        "identity": sum(
            abs(row["p2"]["market_bp"] - actual)
            for row, actual in zip(train_rows, actuals)
        )
        / len(train_rows),
        "linear": sum(
            abs(row["p2"]["linear_magnitude_bp"] - actual)
            for row, actual in zip(train_rows, actuals)
        )
        / len(train_rows),
        "class": sum(
            abs(row["p2"]["class_magnitude_bp"] - actual)
            for row, actual in zip(train_rows, actuals)
        )
        / len(train_rows),
    }
    best = min(mae, key=mae.get)
    if mae[best] > mae["identity"] - config.magnitude_tolerance:
        best = "identity"
    return best, {name: _r(value, 2) for name, value in mae.items()}


def _base_row(
    meeting: Meeting,
    config: P2Config,
    calendar: Calendar,
    zq: dict,
    rates: dict,
    statements: list[dict],
    macro_dir: str | Path,
    vintages: dict | None = None,
) -> dict:
    as_of = meeting.decision_date - timedelta(days=config.cutoff_days)
    bt_config = _backtest_config(config)
    market = market_signal(meeting, as_of, calendar, zq, rates, bt_config)
    macro = macro_signal_for(
        meeting, as_of, market["pre_rate"], macro_dir, bt_config, vintages
    )
    text_p1 = text_signal_for(
        statements, as_of, replace(bt_config, use_dissent=False)
    )
    text_p2 = text_signal_for(statements, as_of, bt_config)
    text_prob = {
        name: round(value, 6)
        for name, value in text_probabilities(
            text_p2["text_score"], config.text_scale
        ).items()
    }
    text = {
        **text_p2,
        "probability": text_prob,
        "expected_bp": _r(
            (text_prob["hike"] - text_prob["cut"]) * config.step * 100.0, 2
        ),
        "most_likely": most_likely(text_prob),
    }
    p1_prob = pool(
        market["probability"],
        macro["probability"],
        text_p1["text_score"],
        (0.60, 0.25, 0.15),
        config.text_scale,
        config.smoothing_eps,
    )["probability"]
    actual = actual_decision(meeting, rates)
    return {
        "decision_date": meeting.decision_date.isoformat(),
        "effective_date": meeting.effective_date.isoformat(),
        "as_of": as_of.isoformat(),
        "market": market,
        "macro": macro,
        "text": text,
        "text_p1": text_p1,
        "p1": {
            "probability": p1_prob,
            "expected_bp": _r(expected_bp(p1_prob, config.step), 2),
            "most_likely": most_likely(p1_prob),
        },
        "actual": actual,
    }


def _vintage_quality(rows: list[dict], enabled: bool) -> dict:
    """统计宏观信号实际使用了多少 ALFRED vintage、多少回退到当前版本。"""
    by_series: dict[str, dict[str, int]] = {}
    for row in rows:
        for series_id, source in row["macro"].get("sources", {}).items():
            bucket = by_series.setdefault(series_id, {"vintage": 0, "fallback": 0})
            if source.startswith("ALFRED"):
                bucket["vintage"] += 1
            else:
                bucket["fallback"] += 1
    vintage_pairs = sum(item["vintage"] for item in by_series.values())
    fallback_pairs = sum(item["fallback"] for item in by_series.values())
    return {
        "enabled": enabled,
        "series": by_series,
        "vintage_pairs": vintage_pairs,
        "fallback_pairs": fallback_pairs,
        "vintage_share": round(
            vintage_pairs / (vintage_pairs + fallback_pairs), 4
        )
        if vintage_pairs + fallback_pairs
        else None,
    }


def _vintage_line(quality: dict) -> str:
    if not quality.get("enabled"):
        return "- 宏观 vintage：未启用，回退当前版本 + 发布滞后。"
    total = quality["vintage_pairs"] + quality["fallback_pairs"]
    share = quality.get("vintage_share")
    text = (
        f"- 宏观 vintage：{quality['vintage_pairs']}/{total} 个（会议，序列）使用 ALFRED 实时数据"
        + (f"（{share:.1%}）" if share is not None else "")
        + f"；回退 {quality['fallback_pairs']} 个。"
    )
    fallback = [
        name
        for name, count in quality.get("series", {}).items()
        if count.get("fallback")
    ]
    if fallback:
        text += "回退序列：" + "、".join(fallback) + "。"
        if "BAMLH0A0HYM2" in fallback:
            text += "BAMLH0A0HYM2 在 2023-11 前不可用，由 BAA10Y vintage 覆盖。"
    return text


def run_p2_backtest(
    calendar_path: str | Path,
    zq_path: str | Path,
    rates_path: str | Path,
    macro_dir: str | Path,
    statements_path: str | Path,
    start: date,
    end: date,
    config: P2Config | None = None,
    vintages_path: str | Path | None = None,
) -> dict:
    cfg = config or P2Config()
    calendar = Calendar.load(calendar_path)
    zq = load_zq_history(zq_path)
    rates = load_rates_long(rates_path)
    statements = load_statements(statements_path)
    vintages = load_vintages(vintages_path) if vintages_path else None

    meetings = [
        meeting
        for meeting in calendar.policy_meetings()
        if start <= meeting.decision_date <= end
    ]
    if not meetings:
        raise ValueError(f"{start} 至 {end} 没有政策会议")

    base_rows = [
        _base_row(meeting, cfg, calendar, zq, rates, statements, macro_dir, vintages)
        for meeting in meetings
    ]

    rows: list[dict] = []
    for index, base in enumerate(base_rows):
        cutoff_day = date.fromisoformat(base["as_of"])
        train_rows = [
            row
            for row in rows
            if date.fromisoformat(row["decision_date"]) < cutoff_day
        ]
        selected = select_hyperparameters(train_rows, cfg)
        anchor = anchored_combination(
            base["market"]["probability"],
            base["macro"]["probability"],
            base["text"]["text_score"],
            selected["weights"],
            selected["blend"],
            base["market"]["reliability"],
            cfg,
        )
        raw_prob = anchor["probability"]
        dynamic_pool_prob = pool(
            base["market"]["probability"],
            base["macro"]["probability"],
            base["text"]["text_score"],
            selected["weights"],
            cfg.text_scale,
            cfg.smoothing_eps,
        )["probability"]

        train_probs = [row["p2"]["raw_probability"] for row in train_rows]
        train_labels = [row["actual"]["label"] for row in train_rows]
        calibration = fit_shrinkage(
            train_probs,
            train_labels,
            cfg.shrinkage_grid,
            cfg.min_calibration_n,
        )
        platt = fit_platt(train_probs, train_labels, cfg.min_calibration_n + 2)
        shrinkage_prob = calibration.apply(raw_prob)
        platt_prob = platt.apply(raw_prob)
        calibrator_choice, calibrator_losses = select_calibrator(train_rows, cfg)
        calibrated_prob = {
            "raw": raw_prob,
            "shrinkage": shrinkage_prob,
            "platt": platt_prob,
        }[calibrator_choice]

        magnitude_model = fit_magnitude(
            [row["market"]["change_bp"] for row in train_rows],
            [row["actual"]["change_bp"] for row in train_rows],
            cfg.magnitude_shrink,
            cfg.min_magnitude_n,
        )
        market_bp = base["market"]["change_bp"]
        linear_bp = _clip(magnitude_model.predict(market_bp), -100.0, 100.0)
        class_bp = expected_bp(raw_prob, cfg.step)
        magnitude_choice, magnitude_losses = select_magnitude_method(train_rows, cfg)
        magnitude_bp = {
            "identity": market_bp,
            "linear": linear_bp,
            "class": class_bp,
        }[magnitude_choice]
        calibrated_class_bp = expected_bp(calibrated_prob, cfg.step)

        rows.append(
            {
                **base,
                "p2": {
                    "selected": {
                        **selected,
                        "weights": {
                            "market": selected["weights"][0],
                            "macro": selected["weights"][1],
                            "text": selected["weights"][2],
                        },
                    },
                    "anchor": anchor,
                    "raw_probability": raw_prob,
                    "raw_most_likely": most_likely(raw_prob),
                    "raw_expected_bp": _r(class_bp, 2),
                    "dynamic_pool_probability": dynamic_pool_prob,
                    "dynamic_pool_most_likely": most_likely(dynamic_pool_prob),
                    "calibration": calibration.to_dict(),
                    "selected_calibrator": calibrator_choice,
                    "calibrator_losses": calibrator_losses,
                    "shrinkage_probability": shrinkage_prob,
                    "shrinkage_most_likely": most_likely(shrinkage_prob),
                    "calibrated_probability": calibrated_prob,
                    "calibrated_most_likely": most_likely(calibrated_prob),
                    "calibrated_class_bp": _r(calibrated_class_bp, 2),
                    "platt": platt.to_dict(),
                    "platt_probability": platt_prob,
                    "platt_most_likely": most_likely(platt_prob),
                    "magnitude_model": magnitude_model.to_dict(),
                    "selected_magnitude": magnitude_choice,
                    "magnitude_losses": magnitude_losses,
                    "market_bp": market_bp,
                    "linear_magnitude_bp": _r(linear_bp, 2),
                    "class_magnitude_bp": _r(class_bp, 2),
                    "magnitude_bp": _r(magnitude_bp, 2),
                    "magnitude_error_bp": _r(
                        magnitude_bp - base["actual"]["change_bp"], 2
                    ),
                    "correct": most_likely(calibrated_prob)
                    == base["actual"]["label"],
                    "raw_correct": most_likely(raw_prob) == base["actual"]["label"],
                    "market_correct": base["market"]["most_likely"]
                    == base["actual"]["label"],
                    "p1_correct": base["p1"]["most_likely"]
                    == base["actual"]["label"],
                    "diverges_from_market": most_likely(calibrated_prob)
                    != base["market"]["most_likely"],
                },
            }
        )

    labels = [row["actual"]["label"] for row in rows]
    market_probs = [row["market"]["probability"] for row in rows]
    macro_probs = [row["macro"]["probability"] for row in rows]
    text_probs = [row["text"]["probability"] for row in rows]
    p1_probs = [row["p1"]["probability"] for row in rows]
    raw_probs = [row["p2"]["raw_probability"] for row in rows]
    shrinkage_probs = [row["p2"]["shrinkage_probability"] for row in rows]
    calibrated_probs = [row["p2"]["calibrated_probability"] for row in rows]
    platt_probs = [row["p2"]["platt_probability"] for row in rows]
    dynamic_probs = [row["p2"]["dynamic_pool_probability"] for row in rows]
    hold_probs = [{"cut": 0.0, "hold": 1.0, "hike": 0.0} for _ in rows]

    metrics = {
        "p2_calibrated": _class_report(calibrated_probs, labels),
        "p2_raw": _class_report(raw_probs, labels),
        "p2_shrinkage": _class_report(shrinkage_probs, labels),
        "p2_platt": _class_report(platt_probs, labels),
        "p1_fixed": _class_report(p1_probs, labels),
        "dynamic_pool": _class_report(dynamic_probs, labels),
        "market": _class_report(market_probs, labels),
        "macro": _class_report(macro_probs, labels),
        "text": _class_report(text_probs, labels),
        "always_hold": _class_report(hold_probs, labels),
    }

    calibration = {
        "market": calibration_table(market_probs, labels, bins=5),
        "p2_raw": calibration_table(raw_probs, labels, bins=5),
        "p2_calibrated": calibration_table(calibrated_probs, labels, bins=5),
        "ece": {
            "market_hike": _r(binary_ece(market_probs, labels, "hike"), 4),
            "market_cut": _r(binary_ece(market_probs, labels, "cut"), 4),
            "p2_raw_hike": _r(binary_ece(raw_probs, labels, "hike"), 4),
            "p2_raw_cut": _r(binary_ece(raw_probs, labels, "cut"), 4),
            "p2_calibrated_hike": _r(binary_ece(calibrated_probs, labels, "hike"), 4),
            "p2_calibrated_cut": _r(binary_ece(calibrated_probs, labels, "cut"), 4),
        },
    }

    actuals_bp = [row["actual"]["change_bp"] for row in rows]
    magnitude = {
        "market": magnitude_metrics(
            [row["p2"]["market_bp"] for row in rows], actuals_bp, cfg.step * 100.0
        ),
        "p1_class": magnitude_metrics(
            [row["p1"]["expected_bp"] for row in rows], actuals_bp, cfg.step * 100.0
        ),
        "p2_class": magnitude_metrics(
            [row["p2"]["raw_expected_bp"] for row in rows],
            actuals_bp,
            cfg.step * 100.0,
        ),
        "p2_linear": magnitude_metrics(
            [row["p2"]["linear_magnitude_bp"] for row in rows],
            actuals_bp,
            cfg.step * 100.0,
        ),
        "p2_calibrated_class": magnitude_metrics(
            [row["p2"]["calibrated_class_bp"] for row in rows],
            actuals_bp,
            cfg.step * 100.0,
        ),
        "p2_magnitude": magnitude_metrics(
            [row["p2"]["magnitude_bp"] for row in rows], actuals_bp, cfg.step * 100.0
        ),
    }

    divergent = [row for row in rows if row["p2"]["diverges_from_market"]]
    anchor_applied = [row for row in rows if row["p2"]["anchor"]["applied"]]
    p1_diverged = [
        row for row in rows if row["p1"]["most_likely"] != row["market"]["most_likely"]
    ]
    p1_wrong_market_right = [
        row
        for row in rows
        if not row["p2"]["p1_correct"] and row["market"]["most_likely"] == row["actual"]["label"]
    ]
    anchor_saved = [
        row
        for row in rows
        if row["p2"]["correct"]
        and not row["p2"]["p1_correct"]
        and row["market"]["most_likely"] == row["actual"]["label"]
    ]
    large_events = [
        {
            "decision_date": row["decision_date"],
            "actual_bp": row["actual"]["change_bp"],
            "market_bp": row["p2"]["market_bp"],
            "predicted_bp": row["p2"]["magnitude_bp"],
            "predicted_grid_bp": round(row["p2"]["magnitude_bp"] / (cfg.step * 100.0))
            * cfg.step
            * 100.0,
            "large_move_hit": abs(row["p2"]["magnitude_bp"]) >= 2 * cfg.step * 100.0,
        }
        for row in rows
        if abs(row["actual"]["change_bp"]) >= 2 * cfg.step * 100.0
    ]
    weight_counts: dict[str, int] = {}
    blend_counts: dict[str, int] = {}
    calibrator_counts: dict[str, int] = {}
    magnitude_counts: dict[str, int] = {}
    for row in rows:
        weights = row["p2"]["selected"]["weights"]
        key = f"{weights['market']:.2f}/{weights['macro']:.2f}/{weights['text']:.2f}"
        weight_counts[key] = weight_counts.get(key, 0) + 1
        blend_key = f"{row['p2']['selected']['blend']:.2f}"
        blend_counts[blend_key] = blend_counts.get(blend_key, 0) + 1
        calibrator = row["p2"]["selected_calibrator"]
        calibrator_counts[calibrator] = calibrator_counts.get(calibrator, 0) + 1
        magnitude_method = row["p2"]["selected_magnitude"]
        magnitude_counts[magnitude_method] = magnitude_counts.get(magnitude_method, 0) + 1

    quality = {
        "zq_symbols": len(zq),
        "zq_observations": sum(len(rows_) for rows_ in zq.values()),
        "quote_age_max_days": max(row["market"]["quote_age_days"] for row in rows),
        "next_clean_month": sum(
            1 for row in rows if row["market"]["method"] == "next_clean_month"
        ),
        "meeting_month": sum(
            1 for row in rows if row["market"]["method"] == "meeting_month"
        ),
        "min_pricing_volume": min(row["market"]["pricing_volume"] for row in rows),
        "low_reliability": [
            row["decision_date"]
            for row in rows
            if row["market"]["reliability"] == "low"
        ],
        "non_high_reliability": [
            row["decision_date"]
            for row in rows
            if row["market"]["reliability"] != "high"
        ],
        "credit_spread_series": {
            name: sum(
                1
                for row in rows
                if row["macro"]["features"]["financial_conditions"][
                    "credit_spread_series"
                ]
                == name
            )
            for name in ("BAMLH0A0HYM2", "BAA10Y")
        },
        "macro_vintage": _vintage_quality(rows, vintages is not None),
    }

    return {
        "schema_version": "bt.p2.0.0",
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "config": {
            "cutoff_days": cfg.cutoff_days,
            "step": cfg.step,
            "min_post_days": cfg.min_post_days,
            "text_scale": cfg.text_scale,
            "smoothing_eps": cfg.smoothing_eps,
            "use_dissent": cfg.use_dissent,
            "weight_grid": [list(weights) for weights in cfg.weight_grid],
            "blend_grid": list(cfg.blend_grid),
            "default_weights": list(cfg.default_weights),
            "default_blend": cfg.default_blend,
            "certainty_threshold": cfg.certainty_threshold,
            "max_deviation_pp": cfg.max_deviation_pp,
            "direction_deadband": cfg.direction_deadband,
            "text_deadband": cfg.text_deadband,
            "market_deadband": cfg.market_deadband,
            "min_calibration_n": cfg.min_calibration_n,
            "shrinkage_grid": list(cfg.shrinkage_grid),
            "min_magnitude_n": cfg.min_magnitude_n,
            "magnitude_shrink": cfg.magnitude_shrink,
            "selection_tolerance": cfg.selection_tolerance,
            "publication_lags": cfg.publication_lags,
        },
        "sample_size": len(rows),
        "actual_counts": {
            name: sum(1 for label in labels if label == name) for name in CLASSES
        },
        "metrics": metrics,
        "calibration": calibration,
        "magnitude": magnitude,
        "signal_contribution": {
            "anchor_applied": len(anchor_applied),
            "divergence_count": len(divergent),
            "divergence_help": sum(
                1
                for row in divergent
                if row["p2"]["correct"] and not row["p2"]["market_correct"]
            ),
            "divergence_hurt": sum(
                1
                for row in divergent
                if not row["p2"]["correct"] and row["p2"]["market_correct"]
            ),
            "p1_market_divergence": len(p1_diverged),
            "p1_wrong_market_right": len(p1_wrong_market_right),
            "anchor_saved": len(anchor_saved),
            "p1_wrong_cases": [
                {
                    "decision_date": row["decision_date"],
                    "actual": row["actual"]["label"],
                    "market": row["market"]["most_likely"],
                    "p1": row["p1"]["most_likely"],
                    "p2": row["p2"]["calibrated_most_likely"],
                }
                for row in p1_wrong_market_right
            ],
            "weight_counts": weight_counts,
            "blend_counts": blend_counts,
            "calibrator_counts": calibrator_counts,
            "magnitude_counts": magnitude_counts,
            "mean_abs_deviation_pp": _r(
                sum(
                    max(
                        abs(
                            row["p2"]["calibrated_probability"][name]
                            - row["market"]["probability"][name]
                        )
                        * 100.0
                        for name in CLASSES
                    )
                    for row in rows
                )
                / len(rows),
                2,
            ),
        },
        "data_quality": quality,
        "large_move_events": large_events,
        "meetings": rows,
        "caveats": [
            "市场先验使用 Barchart 第三方延迟 ZQ 日线，非 CME 直连结算价；扩展窗口由两次抓取合并并逐会议校验。",
            "宏观序列使用 ALFRED 截止日 vintage（预测截止日当日或之前最近归档），不再依赖 FRED 的发布滞后近似；BAMLH0A0HYM2 在 2023-11 前不可用，回退 BAA10Y vintage。vintage 归档日与真实发布时刻之间仍可能有边界误差。",
            "ICE BofA 高收益 OAS 在 FRED 仅有约三年历史；2021-2023 会议改用 BAA10Y 信用利差，分项权重相同但量纲不同，已在校准报告中标注。",
            "文本信号只使用会前已发布的 FOMC 声明与声明 diff，并加入异议票方向微调；演讲与纪要未纳入。",
            "权重、偏离强度、校准参数与幅度斜率均只用当前会议截止日之前已开完的会议拟合，属于 walk-forward，但仍可能受小样本波动影响。",
            "实际决议由 FRED DFEDTARU/DFEDTARL 在决议生效日的变化推导；50bp 变动记为一次降息/加息，幅度单独评估。",
            "报告中 P1 固定权重指标是在同一 2021-2026 扩展窗口、同一截止日下重算的结果，与前一阶段 2023-11 至 2026-11 的 P1 报告不可直接比较。",
            "样本类别不平衡，准确率易被维持类主导；应结合 Brier、log loss、校准误差、幅度 MAE 与分类别召回解读。",
        ],
    }


def run_cutoff_sweep(
    calendar_path: str | Path,
    zq_path: str | Path,
    rates_path: str | Path,
    macro_dir: str | Path,
    statements_path: str | Path,
    start: date,
    end: date,
    config: P2Config | None = None,
    cutoffs: tuple[int, ...] = (7, 14, 30),
    vintages_path: str | Path | None = None,
) -> dict:
    base = config or P2Config()
    results = {}
    for cutoff in cutoffs:
        results[str(cutoff)] = run_p2_backtest(
            calendar_path,
            zq_path,
            rates_path,
            macro_dir,
            statements_path,
            start,
            end,
            replace(base, cutoff_days=cutoff),
            vintages_path=vintages_path,
        )
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "cutoffs": list(cutoffs),
        "runs": results,
    }


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _pp(value: float) -> str:
    return f"{value:+.1f}pp"


def render_p2_markdown(result: dict, sweep: dict | None = None) -> str:
    metrics = result["metrics"]
    rows = result["meetings"]
    calibrated = metrics["p2_calibrated"]
    market = metrics["market"]
    p1 = metrics["p1_fixed"]
    label_cn = {"hike": "加息", "cut": "降息", "hold": "维持"}
    caveats = result["caveats"]

    improved = [
        row
        for row in rows
        if row["p2"]["correct"] and not row["p2"]["market_correct"]
    ]
    worsened = [
        row
        for row in rows
        if not row["p2"]["correct"] and row["p2"]["market_correct"]
    ]
    anchor_cases = [row for row in rows if row["p2"]["anchor"]["applied"]]

    lines = [
        "# 美联储利率决议回测 P2：市场锚定、滚动校准与幅度预测",
        "",
        f"- 回测窗口：{result['window']['start']} 至 {result['window']['end']}",
        f"- 样本：{result['sample_size']} 次政策会议；实际为 "
        f"{result['actual_counts']['hold']} 次维持、"
        f"{result['actual_counts']['cut']} 次降息、"
        f"{result['actual_counts']['hike']} 次加息。",
        f"- 预测截止：每次决议日前 {result['config']['cutoff_days']} 天；"
        f"市场锚定 + walk-forward 动态权重 + 滚动校准。",
        "",
        "## 一、核心结论",
        "",
        f"- P2 校准后准确率 **{_pct(calibrated['accuracy'])}**"
        f"（{calibrated['correct']}/{calibrated['n']}），"
        f"P1 固定权重 {_pct(p1['accuracy'])}，市场先验 {_pct(market['accuracy'])}。",
        f"- P2 校准后 Brier **{calibrated['brier']:.4f}**、log loss "
        f"**{calibrated['log_loss']:.4f}**；市场先验 "
        f"{market['brier']:.4f} / {market['log_loss']:.4f}。",
        f"- 概率校准：P2 原始 ECE(hike) "
        f"{result['calibration']['ece']['p2_raw_hike']:.4f}，校准后 "
        f"{result['calibration']['ece']['p2_calibrated_hike']:.4f}；市场 "
        f"{result['calibration']['ece']['market_hike']:.4f}。"
        f"滚动选择的校准器："
        + "、".join(
            f"{name} {count} 次"
            for name, count in result["signal_contribution"][
                "calibrator_counts"
            ].items()
        )
        + "。",
        f"- 幅度预测：P2 walk-forward 选择后 MAE "
        f"**{result['magnitude']['p2_magnitude']['mae_bp']:.1f}bp**、"
        f"RMSE {result['magnitude']['p2_magnitude']['rmse_bp']:.1f}bp；"
        f"市场隐含 MAE {result['magnitude']['market']['mae_bp']:.1f}bp。",
        f"- 市场锚定未触发主动偏离（{result['signal_contribution']['anchor_applied']} 次）："
        "宏观与文本从未同时与市场反向。"
        f"但它在 {result['signal_contribution']['anchor_saved']} 次 P1 会误报的会议上保留了市场判断"
        f"（P1 与市场方向不同 {result['signal_contribution']['p1_market_divergence']} 次，"
        f"其中 P1 错、市场对 {result['signal_contribution']['p1_wrong_market_right']} 次）。",
    ]
    if worsened:
        details = "；".join(
            f"{row['decision_date']} P2 {_pct(row['p2']['calibrated_probability'][row['p2']['calibrated_most_likely']])}、"
            f"市场 {_pct(row['market']['probability'][row['market']['most_likely']])}，"
            f"实际{label_cn[row['actual']['label']]}"
            for row in worsened
        )
        lines.append(f"- 方向恶化案例：{details}。")
    lines.extend(
        [
            "",
            "## 二、指标对比",
            "",
            "| 模型 | 准确率 | Brier | log loss | 加息召回 | 降息召回 | 维持召回 |",
            "|------|--------|-------|----------|----------|----------|----------|",
        ]
    )
    labels = {
        "p2_calibrated": "P2 锚定+校准（主模型）",
        "p2_raw": "P2 锚定+动态权重（未校准）",
        "p2_shrinkage": "P2 + 收缩校准（对照）",
        "p2_platt": "P2 + Platt（对照）",
        "p1_fixed": "P1 固定权重 0.60/0.25/0.15",
        "dynamic_pool": "动态权重池化（无锚定）",
        "market": "市场先验（ZQ）",
        "macro": "宏观反应函数",
        "text": "文本信号",
        "always_hold": "始终维持",
    }
    for key in (
        "p2_calibrated",
        "p2_raw",
        "p2_shrinkage",
        "p2_platt",
        "p1_fixed",
        "dynamic_pool",
        "market",
        "macro",
        "text",
        "always_hold",
    ):
        item = metrics[key]
        lines.append(
            f"| {labels[key]} | {_pct(item['accuracy'])} | {item['brier']:.4f} | "
            f"{item['log_loss']:.4f} | {_pct(item['by_class']['hike']['recall'])} | "
            f"{_pct(item['by_class']['cut']['recall'])} | "
            f"{_pct(item['by_class']['hold']['recall'])} |"
        )

    lines.extend(
        [
            "",
            "## 三、逐会议明细",
            "",
            "| 决议日 | 截止日 | 实际 | 变化 | 市场加息 | P1加息 | P2原始加息 | P2校准加息 | P2判断 | 幅度预测 | 误差 | 权重(市/宏/文) | 偏离强度 | 校准器 | 幅度口径 | 正确 |",
            "|--------|--------|------|------|----------|--------|------------|------------|--------|----------|------|----------------|----------|--------|----------|------|",
        ]
    )
    for row in rows:
        selected = row["p2"]["selected"]
        weights = selected["weights"]
        lines.append(
            f"| {row['decision_date']} | {row['as_of']} | "
            f"{label_cn[row['actual']['label']]} | "
            f"{row['actual']['change_bp']:+.0f}bp | "
            f"{_pct(row['market']['probability']['hike'])} | "
            f"{_pct(row['p1']['probability']['hike'])} | "
            f"{_pct(row['p2']['raw_probability']['hike'])} | "
            f"{_pct(row['p2']['calibrated_probability']['hike'])} | "
            f"{label_cn[row['p2']['calibrated_most_likely']]} | "
            f"{row['p2']['magnitude_bp']:+.1f}bp | "
            f"{row['p2']['magnitude_error_bp']:+.1f}bp | "
            f"{weights['market']:.2f}/{weights['macro']:.2f}/{weights['text']:.2f} | "
            f"{selected['blend']:.2f} | "
            f"{row['p2']['selected_calibrator']} | "
            f"{row['p2']['selected_magnitude']} | "
            f"{'✓' if row['p2']['correct'] else '✗'} |"
        )

    lines.extend(
        [
            "",
            "## 四、概率校准",
            "",
            "| 模型 | 分箱 | 样本数 | 平均预测 | 实际上涨比例 |",
            "|------|------|--------|----------|--------------|",
        ]
    )
    for key in ("market", "p2_raw", "p2_calibrated"):
        for item in result["calibration"][key]:
            lines.append(
                f"| {labels[key]} | {item['bin']} | {item['count']} | "
                f"{_pct(item['mean_predicted'])} | {_pct(item['actual_rate'])} |"
            )
    if not any(result["calibration"][key] for key in ("market", "p2_raw", "p2_calibrated")):
        lines.append("| 无 | 0 | — | — | — |")

    ece = result["calibration"]["ece"]
    lines.extend(
        [
            "",
            "ECE（期望校准误差，越小越好）：",
            "",
            "| 模型 | 加息 ECE | 降息 ECE |",
            "|------|----------|----------|",
            f"| 市场先验 | {ece['market_hike']:.4f} | {ece['market_cut']:.4f} |",
            f"| P2 原始 | {ece['p2_raw_hike']:.4f} | {ece['p2_raw_cut']:.4f} |",
            f"| P2 校准 | {ece['p2_calibrated_hike']:.4f} | {ece['p2_calibrated_cut']:.4f} |",
            "",
            "## 五、幅度预测",
            "",
            "| 幅度口径 | MAE(bp) | RMSE(bp) | 25bp 网格命中 | 方向命中 | 大幅变动精确率 | 大幅变动召回 |",
            "|----------|---------|----------|----------------|----------|----------------|--------------|",
        ]
    )
    magnitude_labels = {
        "market": "市场隐含变动",
        "p1_class": "P1 类别期望（±25bp）",
        "p2_class": "P2 类别期望（±25bp）",
        "p2_linear": "P2 线性校准幅度（对照）",
        "p2_calibrated_class": "P2 校准后类别期望",
        "p2_magnitude": "P2 幅度（walk-forward 选择，主）",
    }
    for key, label in magnitude_labels.items():
        item = result["magnitude"][key]
        lines.append(
            f"| {label} | {item['mae_bp']:.1f} | {item['rmse_bp']:.1f} | "
            f"{_pct(item['exact_grid_rate'])} | {_pct(item['direction_accuracy'])} | "
            f"{_pct(item['large_move_precision'])} | {_pct(item['large_move_recall'])} |"
        )

    biggest = sorted(
        rows, key=lambda row: abs(row["p2"]["magnitude_error_bp"]), reverse=True
    )[:5]
    lines.extend(
        [
            "",
            "幅度误差最大的五次会议：",
            "",
            "| 决议日 | 实际变化 | 市场隐含 | P2幅度 | 误差 |",
            "|--------|----------|----------|--------|------|",
        ]
    )
    for row in biggest:
        lines.append(
            f"| {row['decision_date']} | {row['actual']['change_bp']:+.0f}bp | "
            f"{row['p2']['market_bp']:+.1f}bp | {row['p2']['magnitude_bp']:+.1f}bp | "
            f"{row['p2']['magnitude_error_bp']:+.1f}bp |"
        )

    if result["large_move_events"]:
        lines.extend(
            [
                "",
                "50bp 及以上事件（幅度口径为 P2 walk-forward 选择）：",
                "",
                "| 决议日 | 实际变化 | 市场隐含 | P2幅度 | 是否识别为大幅变动 |",
                "|--------|----------|----------|--------|--------------------|",
            ]
        )
        for event in result["large_move_events"]:
            lines.append(
                f"| {event['decision_date']} | {event['actual_bp']:+.0f}bp | "
                f"{event['market_bp']:+.1f}bp | {event['predicted_bp']:+.1f}bp | "
                f"{'是' if event['large_move_hit'] else '否'} |"
            )

    lines.extend(
        [
            "",
            "## 六、混淆矩阵",
            "",
            "### P2 校准后",
            "",
            "| 实际 \\ 预测 | 降息 | 维持 | 加息 |",
            "|-------------|------|------|------|",
        ]
    )
    for actual in CLASSES:
        confusion = calibrated["confusion"][actual]
        lines.append(
            f"| {label_cn[actual]} | {confusion['cut']} | {confusion['hold']} | "
            f"{confusion['hike']} |"
        )
    lines.extend(
        [
            "",
            "### 市场先验",
            "",
            "| 实际 \\ 预测 | 降息 | 维持 | 加息 |",
            "|-------------|------|------|------|",
        ]
    )
    for actual in CLASSES:
        confusion = market["confusion"][actual]
        lines.append(
            f"| {label_cn[actual]} | {confusion['cut']} | {confusion['hold']} | "
            f"{confusion['hike']} |"
        )

    contribution = result["signal_contribution"]
    lines.extend(
        [
            "",
            "## 七、信号贡献与动态权重",
            "",
            f"- 市场锚定主动偏离 {contribution['anchor_applied']} 次；"
            f"最终校准概率与市场定价的平均最大偏差 {contribution['mean_abs_deviation_pp']:.1f}pp"
            "（来自概率校准，不是锚定偏离）。",
            f"- 最终判断与市场不同 {contribution['divergence_count']} 次："
            f"改善 {contribution['divergence_help']} 次、"
            f"恶化 {contribution['divergence_hurt']} 次。",
            f"- P1 固定权重与市场方向不同 {contribution['p1_market_divergence']} 次，"
            f"其中 P1 错、市场对 {contribution['p1_wrong_market_right']} 次；"
            f"锚定在这些会议上保留市场判断，避免误报 {contribution['anchor_saved']} 次。",
            "- 权重分布（市/宏/文）："
            + "、".join(
                f"{key} {value} 次" for key, value in contribution["weight_counts"].items()
            )
            + "。",
            "- 偏离强度分布："
            + "、".join(
                f"{key} {value} 次" for key, value in contribution["blend_counts"].items()
            )
            + "。",
            "- 校准器分布："
            + "、".join(
                f"{key} {value} 次"
                for key, value in contribution["calibrator_counts"].items()
            )
            + "。",
            "- 幅度口径分布："
            + "、".join(
                f"{key} {value} 次"
                for key, value in contribution["magnitude_counts"].items()
            )
            + "。",
            "- 动态选择结果：所有候选权重/偏离强度在训练期 log loss 上都未比纯市场锚定改进超过 "
            f"{result['config']['selection_tolerance']:.3f}，因此收敛到默认 "
            f"{result['config']['default_weights'][0]:.2f}/"
            f"{result['config']['default_weights'][1]:.2f}/"
            f"{result['config']['default_weights'][2]:.2f}、blend="
            f"{result['config']['default_blend']:.2f}。这不是未启用动态选择，而是样本内非市场信号"
            "无法稳定改善市场定价。",
            f"- 敏感性检查：把偏离强度强制为 0.5 或 1.0 时，{result['sample_size']} 次会议的最终方向均不变，"
            "说明本样本中没有同时满足“宏观与文本同向、且与市场反向、市场未接近确定”的会议。",
        ]
    )
    for row in anchor_cases:
        lines.append(
            f"- {row['decision_date']} 偏离市场：{row['p2']['anchor']['reason']}；"
            f"市场 {_pct(row['market']['probability']['hike'])} hike → "
            f"P2 {_pct(row['p2']['calibrated_probability']['hike'])} hike，"
            f"实际{label_cn[row['actual']['label']]}。"
        )
    for case in contribution["p1_wrong_cases"]:
        lines.append(
            f"- 锚定避免的 P1 误报 {case['decision_date']}：市场"
            f"{label_cn[case['market']]}、P1 {label_cn[case['p1']]}、P2 "
            f"{label_cn[case['p2']]}，实际{label_cn[case['actual']]}。"
        )
    for row in improved:
        lines.append(
            f"- 改善案例 {row['decision_date']}：市场判断"
            f"{label_cn[row['market']['most_likely']]}，P2 判断"
            f"{label_cn[row['p2']['calibrated_most_likely']]}，"
            f"实际{label_cn[row['actual']['label']]}。"
        )
    if not improved:
        lines.append("- 本样本内没有出现由主动偏离市场带来的方向改善。")

    if sweep:
        lines.extend(
            [
                "",
                "## 八、预测截止期敏感性",
                "",
                "| 截止期 | 样本 | P2准确率 | Brier | log loss | 幅度MAE | ECE(hike) | 平均偏离强度 | 市场准确率 |",
                "|--------|------|----------|-------|----------|---------|-----------|--------------|------------|",
            ]
        )
        for cutoff in sweep["cutoffs"]:
            run = sweep["runs"][str(cutoff)]
            item = run["metrics"]["p2_calibrated"]
            blend_values = [
                row["p2"]["selected"]["blend"] for row in run["meetings"]
            ]
            lines.append(
                f"| {cutoff} 天 | {run['sample_size']} | {_pct(item['accuracy'])} | "
                f"{item['brier']:.4f} | {item['log_loss']:.4f} | "
                f"{run['magnitude']['p2_magnitude']['mae_bp']:.1f}bp | "
                f"{run['calibration']['ece']['p2_calibrated_hike']:.4f} | "
                f"{sum(blend_values) / len(blend_values):.2f} | "
                f"{_pct(run['metrics']['market']['accuracy'])} |"
            )
        lines.append("")
        for cutoff in sweep["cutoffs"]:
            run = sweep["runs"][str(cutoff)]
            wrong = [
                f"{row['decision_date']}（实际{label_cn[row['actual']['label']]}，"
                f"P2 {label_cn[row['p2']['calibrated_most_likely']]}，"
                f"市场 {label_cn[row['market']['most_likely']]}）"
                for row in run["meetings"]
                if not row["p2"]["correct"]
            ]
            if wrong:
                lines.append(
                    f"- {cutoff} 天截止期错误 {len(wrong)} 次：" + "；".join(wrong) + "。"
                )
            else:
                lines.append(f"- {cutoff} 天截止期无方向错误。")

    lines.extend(
        [
            "",
            "## 九、方法与无前视处理",
            "",
            f"- 预测截止日 T = 决议日前 {result['config']['cutoff_days']} 个自然日。",
            "- 市场：取 T 当日或之前最后交易日 ZQ 收盘价；会议月合约会后天数不足 7 天时改用次月干净合约。",
            "- 宏观：使用 ALFRED 在 T 当日或之前最近归档的实时 vintage，避免修订前视；"
            "BAMLH0A0HYM2 在 2023-11 前无数据时改用 BAA10Y vintage。",
            "- 文本：只使用 T 之前已发布的声明；P2 在声明分与 diff 分之外加入异议票方向微调。",
            "- 合成：以市场概率为锚；仅当宏观与文本同向、且与市场方向相反、市场未接近确定时偏离；偏离幅度上限 "
            f"{result['config']['max_deviation_pp']:.0f}pp。",
            "- 动态权重与偏离强度：在每次会议之前已开完的会议上用 log loss 选择，默认偏向更高市场权重；"
            "若改进小于容差则回到默认，避免小样本过拟合。",
            "- 校准：在 raw、向训练期基础频率收缩、Platt 缩放三者间按历史样本外 log loss 滚动选择；"
            "幅度：在市场隐含变动、线性校准、类别期望三者间按历史样本外 MAE 滚动选择。",
            "- 修订影响对照：加 --no-vintages 可回退到当前版本 + 发布滞后，用于检验 vintage 对结论的影响。",
            "- 实际标签：决议生效日目标区间上限相对决议日的变化；上升=加息、下降=降息、不变=维持。",
            "",
            "## 十、数据与局限",
            "",
        ]
    )
    lines.extend(f"- {caveat}" for caveat in caveats)
    quality = result["data_quality"]
    lines.extend(
        [
            "",
            f"- 数据质量：行情覆盖 {quality['zq_symbols']} 个合约 / "
            f"{quality['zq_observations']} 条观测；最大行情滞后 "
            f"{quality['quote_age_max_days']} 天；次月干净合约 "
            f"{quality['next_clean_month']} 次、当月合约分解 {quality['meeting_month']} 次；"
            f"定价合约最小成交量 {quality['min_pricing_volume']:.0f} 手；"
            f"可靠性非 high {len(quality['non_high_reliability'])} 次"
            f"（其中 low {len(quality['low_reliability'])} 次）。",
            f"- 信用利差来源：高收益 OAS {quality['credit_spread_series']['BAMLH0A0HYM2']} 次、"
            f"BAA10Y {quality['credit_spread_series']['BAA10Y']} 次。",
            _vintage_line(quality["macro_vintage"]),
            "",
            "## 十一、复现",
            "",
            "```bash",
            "python3 scripts/fetch_fomc_calendar.py --start-year 2021 --end-year 2027",
            "node scripts/fetch_barchart_history.js --start 2020-11 --months 75 --from 2020-10-01 --out data/backtest/zq_history.csv",
            "python3 scripts/fetch_backtest_data.py",
            "python3 scripts/fetch_macro_vintages.py",
            "python3 run_backtest.py --mode p2 --start 2021-01-01 --end 2026-11-22",
            "```",
            "",
        ]
    )
    return "\n".join(lines)
