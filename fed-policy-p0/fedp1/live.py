"""实时预测：用截至当前的数据快照运行 P2 锚定 + 校准 + 幅度模型。

与回测的区别：
1. 预测基准是「现在」，不是决议日前 7 天；使用今日可见的市场/宏观/文本。
2. 训练只使用已经开完且已有实际利率的会议，避免前视。
3. 输出带数据新鲜度、市场基准、模型 edge 与经验误差区间，便于人工决策。
"""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from fedp0.calendar import Calendar, Meeting
from fedp0.market import Quote, load_quotes
from fedp0.pipeline import build_baseline
from fedp1.backtest import (
    BacktestConfig,
    actual_decision,
    load_rates_long,
    load_statements,
    load_zq_history,
    text_signal_for,
)
from fedp1.calibration import fit_magnitude, fit_platt, fit_shrinkage
from fedp1.combine import expected_bp, most_likely
from fedp1.macro import (
    MacroConfig,
    build_macro_path,
    build_macro_signal,
    load_vintages,
)
from fedp1.p2 import (
    P2Config,
    _backtest_config,
    anchored_combination,
    run_p2_backtest,
    select_calibrator,
    select_hyperparameters,
    select_magnitude_method,
)

MACRO_MAX_AGE_DAYS = {
    # 年龄按「观测日期」计算，需覆盖发布滞后与发布周期，否则月末会误报过期。
    "PCEPILFE": 100,
    "CPILFESL": 75,
    "UNRATE": 75,
    "PAYEMS": 75,
    "GDPC1": 150,
    "NROU": 120,
    "NFCI": 10,
    "BAMLH0A0HYM2": 5,
    "BAA10Y": 5,
    "DGS2": 5,
    "DGS10": 5,
    "T10YIE": 5,
}


@dataclass
class LiveConfig:
    n_meetings: int = 3
    training_start: date = date(2021, 1, 1)
    edge_threshold_pp: float = 5.0
    market_max_age_days: int = 3
    rates_max_age_days: int = 7
    statement_max_age_days: int = 60
    fail_on_stale: bool = False
    p2: P2Config = field(default_factory=P2Config)


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _round_prob(probability: dict) -> dict:
    rounded = {name: round(float(probability.get(name, 0.0)), 6) for name in ("cut", "hold", "hike")}
    total = sum(rounded.values())
    if total <= 0:
        return {"cut": 0.0, "hold": 1.0, "hike": 0.0}
    return {name: round(value / total, 6) for name, value in rounded.items()}


def _completed_meetings(
    calendar: Calendar, rates: dict, as_of: date
) -> list[Meeting]:
    completed = []
    for meeting in calendar.policy_meetings():
        if meeting.decision_date >= as_of or meeting.effective_date >= as_of:
            continue
        try:
            actual_decision(meeting, rates)
        except ValueError:
            continue
        completed.append(meeting)
    return completed


def _fit_state(train_rows: list[dict], cfg: P2Config) -> dict:
    selected = select_hyperparameters(train_rows, cfg)
    raw_probs = [row["p2"]["raw_probability"] for row in train_rows]
    labels = [row["actual"]["label"] for row in train_rows]
    shrinkage = fit_shrinkage(
        raw_probs, labels, cfg.shrinkage_grid, cfg.min_calibration_n
    )
    platt = fit_platt(raw_probs, labels, cfg.min_calibration_n + 2)
    calibrator, calibrator_losses = select_calibrator(train_rows, cfg)
    magnitude_model = fit_magnitude(
        [row["market"]["change_bp"] for row in train_rows],
        [row["actual"]["change_bp"] for row in train_rows],
        cfg.magnitude_shrink,
        cfg.min_magnitude_n,
    )
    magnitude, magnitude_losses = select_magnitude_method(train_rows, cfg)
    residuals = []
    for row in train_rows:
        market_bp = row["market"]["change_bp"]
        if magnitude == "identity":
            prediction = market_bp
        elif magnitude == "linear":
            prediction = max(-100.0, min(100.0, magnitude_model.predict(market_bp)))
        else:
            prediction = expected_bp(row["p2"]["calibrated_probability"], cfg.step)
        residuals.append(row["actual"]["change_bp"] - prediction)
    return {
        "selected": selected,
        "shrinkage": shrinkage,
        "platt": platt,
        "calibrator": calibrator,
        "calibrator_losses": calibrator_losses,
        "magnitude_model": magnitude_model,
        "magnitude": magnitude,
        "magnitude_losses": magnitude_losses,
        "residual_low_bp": _quantile(residuals, 0.10),
        "residual_high_bp": _quantile(residuals, 0.90),
        "residual_n": len(residuals),
    }


def _apply_calibration(raw: dict, state: dict, cfg: P2Config) -> tuple[dict, dict]:
    candidates = {
        "raw": raw,
        "shrinkage": state["shrinkage"].apply(raw),
        "platt": state["platt"].apply(raw),
    }
    return candidates, candidates[state["calibrator"]]


def _write_market_snapshots(
    zq: dict,
    rates: dict,
    as_of: date,
    out_dir: Path,
    supplement: dict[str, Quote] | None = None,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    zq_path = out_dir / "zq_curve.csv"
    with zq_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["symbol", "as_of", "open", "high", "low", "close", "volume", "open_interest"]
        )
        for symbol in sorted(zq):
            rows = [row for row in zq[symbol] if row[0] <= as_of]
            if not rows:
                continue
            day, close, volume = rows[-1]
            quote = (supplement or {}).get(symbol)
            if quote is not None and as_of >= quote.as_of >= day:
                writer.writerow(
                    [
                        symbol,
                        quote.as_of.isoformat(),
                        "",
                        "",
                        "",
                        quote.close,
                        quote.volume,
                        quote.open_interest,
                    ]
                )
            else:
                writer.writerow([symbol, day.isoformat(), "", "", "", close, volume, 0])
        # 历史长表可能缺少最近上市的新合约，用独立快照补齐。
        for symbol, quote in sorted((supplement or {}).items()):
            if symbol in zq or quote.as_of > as_of:
                continue
            writer.writerow(
                [
                    symbol,
                    quote.as_of.isoformat(),
                    "",
                    "",
                    "",
                    quote.close,
                    quote.volume,
                    quote.open_interest,
                ]
            )

    rate_path = out_dir / "rate_snapshot.csv"
    with rate_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["series", "as_of", "value", "source"])
        for series in ("DFEDTARU", "DFEDTARL", "EFFR"):
            rows = [row for row in rates.get(series, []) if row[0] <= as_of]
            if not rows:
                continue
            day, value = rows[-1]
            writer.writerow([series, day.isoformat(), f"{value:.4f}", f"FRED:{series}"])
    return zq_path, rate_path


def _freshness(
    zq: dict,
    rates: dict,
    macro_dir: str | Path,
    statements: list[dict],
    as_of: date,
    cfg: LiveConfig,
    supplement: dict[str, Quote] | None = None,
) -> dict:
    market_latest = max(
        (day for rows in zq.values() for day, _, _ in rows), default=None
    )
    if supplement:
        supplement_latest = max(
            (quote.as_of for quote in supplement.values() if quote.as_of <= as_of),
            default=None,
        )
        if supplement_latest is not None and (
            market_latest is None or supplement_latest > market_latest
        ):
            market_latest = supplement_latest
    market_age = (as_of - market_latest).days if market_latest else None

    rate_report = {}
    for series in ("DFEDTARU", "DFEDTARL", "EFFR"):
        latest = max((day for day, _ in rates.get(series, [])), default=None)
        age = (as_of - latest).days if latest else None
        rate_report[series] = {
            "latest": latest.isoformat() if latest else None,
            "age_days": age,
            "ok": age is not None and age <= cfg.rates_max_age_days,
        }

    macro_report = {}
    raw_dir = Path(macro_dir) / "raw"
    for series, limit in MACRO_MAX_AGE_DAYS.items():
        path = raw_dir / f"{series}.csv"
        latest = None
        if path.exists():
            with path.open(newline="", encoding="utf-8") as fh:
                days = [
                    date.fromisoformat(row["observation_date"])
                    for row in csv.DictReader(fh)
                    if row.get("observation_date")
                ]
            latest = max(days, default=None)
        age = (as_of - latest).days if latest else None
        macro_report[series] = {
            "latest": latest.isoformat() if latest else None,
            "age_days": age,
            "max_age_days": limit,
            "ok": age is not None and age <= limit,
        }

    statement_latest = max((row["_day"] for row in statements), default=None)
    statement_age = (as_of - statement_latest).days if statement_latest else None
    statement_ok = (
        statement_age is not None and statement_age <= cfg.statement_max_age_days
    )

    stale = []
    if market_age is None or market_age > cfg.market_max_age_days:
        stale.append("market")
    for series, item in rate_report.items():
        if not item["ok"]:
            stale.append(f"rates:{series}")
    for series, item in macro_report.items():
        if not item["ok"]:
            stale.append(f"macro:{series}")
    if not statement_ok:
        stale.append("statements")

    critical = (
        market_age is None
        or market_age > cfg.market_max_age_days
        or not rate_report["EFFR"]["ok"]
    )
    return {
        "as_of": as_of.isoformat(),
        "market": {
            "latest": market_latest.isoformat() if market_latest else None,
            "age_days": market_age,
            "max_age_days": cfg.market_max_age_days,
            "ok": market_age is not None and market_age <= cfg.market_max_age_days,
        },
        "rates": rate_report,
        "macro": macro_report,
        "statements": {
            "latest": statement_latest.isoformat() if statement_latest else None,
            "age_days": statement_age,
            "max_age_days": cfg.statement_max_age_days,
            "ok": statement_ok,
        },
        "stale_sources": stale,
        "critical_stale": critical,
    }


def _file_hashes(paths: list[Path]) -> list[dict]:
    items = []
    for path in paths:
        if not path.exists():
            items.append({"path": str(path), "exists": False})
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        items.append(
            {
                "path": str(path),
                "exists": True,
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    return items


def predict_live(
    calendar_path: str | Path,
    zq_path: str | Path,
    rates_path: str | Path,
    macro_dir: str | Path,
    statements_path: str | Path,
    as_of: date,
    config: LiveConfig | None = None,
    vintages_path: str | Path | None = None,
    snapshot_dir: str | Path | None = None,
    quote_snapshot_path: str | Path | None = None,
) -> dict:
    cfg = config or LiveConfig()
    p2cfg = cfg.p2
    calendar = Calendar.load(calendar_path)
    zq = load_zq_history(zq_path)
    rates = load_rates_long(rates_path)
    statements = load_statements(statements_path)
    vintages = load_vintages(vintages_path) if vintages_path else None
    supplement = (
        load_quotes(quote_snapshot_path) if quote_snapshot_path is not None else None
    )

    freshness = _freshness(zq, rates, macro_dir, statements, as_of, cfg, supplement)
    if cfg.fail_on_stale and freshness["critical_stale"]:
        raise RuntimeError(
            "关键数据过期，拒绝预测：" + "、".join(freshness["stale_sources"])
        )

    completed = _completed_meetings(calendar, rates, as_of)
    if not completed:
        raise ValueError("没有可用于训练的已完成会议")
    train_end = completed[-1].decision_date
    training = run_p2_backtest(
        calendar_path=calendar_path,
        zq_path=zq_path,
        rates_path=rates_path,
        macro_dir=macro_dir,
        statements_path=statements_path,
        start=cfg.training_start,
        end=train_end,
        config=p2cfg,
        vintages_path=vintages_path,
    )
    train_rows = training["meetings"]
    state = _fit_state(train_rows, p2cfg)

    snapshot = Path(snapshot_dir) if snapshot_dir else Path(tempfile.mkdtemp())
    quote_path, rate_snapshot_path = _write_market_snapshots(
        zq, rates, as_of, snapshot, supplement
    )
    baseline = build_baseline(
        calendar_path=calendar_path,
        zq_path=quote_path,
        rates_path=rate_snapshot_path,
        as_of=as_of,
        n_meetings=cfg.n_meetings,
        step=p2cfg.step,
        min_post_days=p2cfg.min_post_days,
    )
    current_rate = baseline["policy_basis"]["current_rate"]
    macro_signal = build_macro_signal(
        macro_dir=macro_dir,
        as_of=as_of,
        current_rate=current_rate,
        config=MacroConfig(),
    )
    macro_path = build_macro_path(
        signal=macro_signal,
        market_path=baseline["path"],
        current_rate=current_rate,
        step=p2cfg.step,
    )
    text = text_signal_for(statements, as_of, _backtest_config(p2cfg))

    predictions = []
    cumulative = 0.0
    for market_row, macro_row in zip(baseline["path"], macro_path):
        anchor = anchored_combination(
            market_row["probability"],
            macro_row["probability"],
            text["text_score"],
            state["selected"]["weights"],
            state["selected"]["blend"],
            market_row["reliability"],
            p2cfg,
        )
        raw = _round_prob(anchor["probability"])
        candidates, calibrated = _apply_calibration(raw, state, p2cfg)
        candidates = {name: _round_prob(value) for name, value in candidates.items()}
        calibrated = _round_prob(calibrated)
        market_bp = market_row["change_bp"]
        if state["magnitude"] == "identity":
            magnitude_bp = market_bp
        elif state["magnitude"] == "linear":
            magnitude_bp = max(
                -100.0, min(100.0, state["magnitude_model"].predict(market_bp))
            )
        else:
            magnitude_bp = expected_bp(calibrated, p2cfg.step)
        cumulative += magnitude_bp
        market_prob = market_row["probability"]
        edge_pp = {
            name: round((calibrated[name] - market_prob[name]) * 100.0, 2)
            for name in ("hike", "hold", "cut")
        }
        anchor_edge_pp = {
            name: round((raw[name] - market_prob[name]) * 100.0, 2)
            for name in ("hike", "hold", "cut")
        }
        calibration_shift_pp = {
            name: round((calibrated[name] - raw[name]) * 100.0, 2)
            for name in ("hike", "hold", "cut")
        }
        top = max(calibrated.values())
        max_edge = max(abs(value) for value in edge_pp.values())
        max_shift = max(abs(value) for value in calibration_shift_pp.values())
        market_top = max(market_prob.values())
        market_most_likely = market_row["most_likely"]
        p2_most_likely = most_likely(calibrated)
        direction_flip = p2_most_likely != market_most_likely and not anchor["applied"]
        if anchor["applied"]:
            hint = "宏观/文本与市场反向，模型主动偏离"
        elif max_shift >= cfg.edge_threshold_pp:
            hint = f"市场锚定，校准上调概率约 {max_shift:.1f}pp，非方向 alpha"
        elif max_edge >= cfg.edge_threshold_pp:
            hint = "模型与市场存在分歧，需人工复核"
        else:
            hint = "模型与市场接近，无显著 edge"
        if direction_flip:
            hint += f"；校准使判断由 {market_most_likely} 翻为 {p2_most_likely}，需人工复核"
        if market_top >= 0.80 and market_row["reliability"] == "low":
            hint += "；市场接近确定但定价可靠性低"
        predictions.append(
            {
                "decision_date": market_row["meeting"]["decision_date"],
                "effective_date": market_row["meeting"]["effective_date"],
                "contract": market_row["contract"],
                "market": {
                    "probability": market_row["probability"],
                    "most_likely": market_row["most_likely"],
                    "expected_bp": market_row["expected_bp"],
                    "reliability": market_row["reliability"],
                },
                "macro_probability": macro_row["probability"],
                "text_score": text["text_score"],
                "anchor": {
                    "applied": anchor["applied"],
                    "reason": anchor["reason"],
                    "market_dir": anchor["market_dir"],
                    "macro_dir": anchor["macro_dir"],
                    "text_dir": anchor["text_dir"],
                },
                "raw_probability": raw,
                "calibrated_probability": calibrated,
                "calibrators": candidates,
                "most_likely": p2_most_likely,
                "market_most_likely": market_most_likely,
                "direction_flip_by_calibration": direction_flip,
                "direction_confidence": round(top, 4),
                "expected_bp": round(expected_bp(calibrated, p2cfg.step), 2),
                "magnitude_bp": round(magnitude_bp, 2),
                "magnitude_interval_bp": [
                    round(magnitude_bp + state["residual_low_bp"], 2)
                    if state["residual_low_bp"] is not None
                    else None,
                    round(magnitude_bp + state["residual_high_bp"], 2)
                    if state["residual_high_bp"] is not None
                    else None,
                ],
                "edge_pp": edge_pp,
                "anchor_edge_pp": anchor_edge_pp,
                "calibration_shift_pp": calibration_shift_pp,
                "cumulative_magnitude_bp": round(cumulative, 2),
                "action_hint": hint,
            }
        )

    input_paths = [
        Path(calendar_path),
        Path(zq_path),
        Path(rates_path),
        Path(statements_path),
        *(Path(macro_dir) / "raw").glob("*.csv"),
    ]
    if vintages_path:
        input_paths.append(Path(vintages_path))
    if quote_snapshot_path is not None:
        input_paths.append(Path(quote_snapshot_path))
    return {
        "schema_version": "live.p2.0.0",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(
            timespec="seconds"
        ),
        "as_of": as_of.isoformat(),
        "policy_basis": baseline["policy_basis"],
        "predictions": predictions,
        "model": {
            "weights": {
                "market": state["selected"]["weights"][0],
                "macro": state["selected"]["weights"][1],
                "text": state["selected"]["weights"][2],
            },
            "blend": state["selected"]["blend"],
            "calibrator": state["calibrator"],
            "magnitude": state["magnitude"],
            "calibration_losses": state["calibrator_losses"],
            "magnitude_losses": state["magnitude_losses"],
            "residual_interval_bp": [
                round(state["residual_low_bp"], 2)
                if state["residual_low_bp"] is not None
                else None,
                round(state["residual_high_bp"], 2)
                if state["residual_high_bp"] is not None
                else None,
            ],
            "edge_threshold_pp": cfg.edge_threshold_pp,
        },
        "training": {
            "start": cfg.training_start.isoformat(),
            "end": train_end.isoformat(),
            "n": len(train_rows),
            "metrics": training["metrics"]["p2_calibrated"],
        },
        "data_quality": {
            **freshness,
            "inputs": _file_hashes(input_paths),
        },
        "caveats": [
            "实时模式使用当前版本数据（可能含修订），不使用历史 vintage；vintage 仅用于训练。",
            "市场概率来自 ZQ 期货的第三方延迟日线，不是 CME 实时结算价。",
            "概率是风险中性定价 + 历史校准，不构成投资建议；实盘前需纸面交易与扣费验证。",
            "幅度区间为训练期历史残差的 10%-90% 分位，不是严格的预测置信区间。",
            "校准器在训练样本上选择并拟合；当锚定未触发时，edge 主要来自校准抬升而非方向 alpha。",
        ],
    }


def render_live_markdown(result: dict) -> str:
    lines = [
        "# 美联储利率决议实时预测",
        "",
        f"- 生成时间：{result['generated_at']}",
        f"- 数据截止：{result['as_of']}",
        f"- 训练样本：{result['training']['n']} 次会议（至 {result['training']['end']}）",
        f"- 模型：权重 {result['model']['weights']['market']:.2f}/"
        f"{result['model']['weights']['macro']:.2f}/"
        f"{result['model']['weights']['text']:.2f}，"
        f"校准 {result['model']['calibrator']}，幅度 {result['model']['magnitude']}",
        "",
        "## 逐会议预测",
        "",
        "| 决议日 | 市场 P(加/维/降) | P2 P(加/维/降) | P2 判断 | 预期幅度 | 80% 区间 | 对市场 edge | 校准抬升 | 提示 |",
        "|--------|------------------|----------------|---------|----------|----------|-------------|----------|------|",
    ]
    for row in result["predictions"]:
        market = row["market"]["probability"]
        prob = row["calibrated_probability"]
        low, high = row["magnitude_interval_bp"]
        edge = row["edge_pp"]
        shift = row["calibration_shift_pp"]
        lines.append(
            f"| {row['decision_date']} | "
            f"{market['hike']:.1%}/{market['hold']:.1%}/{market['cut']:.1%} | "
            f"{prob['hike']:.1%}/{prob['hold']:.1%}/{prob['cut']:.1%} | "
            f"{row['most_likely']} | {row['magnitude_bp']:+.1f}bp | "
            f"{low:+.1f}~{high:+.1f}bp | "
            f"{edge['hike']:+.1f}pp | {shift['hike']:+.1f}pp | "
            f"{row['action_hint']} |"
        )
    lines += ["", "## 数据新鲜度", ""]
    quality = result["data_quality"]
    lines.append(
        f"- 行情：{quality['market']['latest']}（{quality['market']['age_days']} 天，"
        f"{'正常' if quality['market']['ok'] else '过期'}）"
    )
    lines.append(
        f"- 声明：{quality['statements']['latest']}（{quality['statements']['age_days']} 天，"
        f"{'正常' if quality['statements']['ok'] else '过期'}）"
    )
    stale = quality["stale_sources"]
    lines.append(
        "- 过期来源：" + ("、".join(stale) if stale else "无")
    )
    lines += ["", "## 局限", ""]
    lines += [f"- {item}" for item in result["caveats"]]
    return "\n".join(lines) + "\n"
