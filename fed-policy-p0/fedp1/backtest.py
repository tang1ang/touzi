"""回测：逐次 FOMC 会议生成三路信号，与实际利率决议对比。

预测截止日 T = 决议日前 cutoff_days 天，市场先验只使用 T 当日或之前的 ZQ 收盘价，
宏观序列按发布滞后截断，文本只使用 T 之前已发布的声明，避免前视偏差。
"""

from __future__ import annotations

import bisect
import csv
import json
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from fedp0.calendar import Calendar, Meeting
from fedp0.contracts import contract_symbol
from fedp0.implied import estimate_step, implied_rate
from fedp0.market import Quote
from fedp0.probability import three_class_probabilities
from fedp1.combine import CLASSES, expected_bp, most_likely, pool
from fedp1.macro import build_macro_path, build_macro_signal, load_vintages
from fedp1.text import score_statement, score_statement_diff

DEFAULT_PUBLICATION_LAGS = {
    "PCEPILFE": 55,
    "CPILFESL": 40,
    "UNRATE": 35,
    "PAYEMS": 35,
    "GDPC1": 110,
    "NROU": 120,
    "NFCI": 7,
    "BAMLH0A0HYM2": 1,
    "BAA10Y": 0,
    "DGS2": 0,
    "DGS10": 0,
    "T10YIE": 0,
}


@dataclass
class BacktestConfig:
    cutoff_days: int = 7
    step: float = 0.25
    min_post_days: int = 7
    weights: tuple[float, float, float] = (0.60, 0.25, 0.15)
    text_scale: float = 2.0
    smoothing_eps: float = 0.005
    use_dissent: bool = False
    publication_lags: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_PUBLICATION_LAGS)
    )


def _next_month(year: int, month: int) -> tuple[int, int]:
    return (year + 1, 1) if month == 12 else (year, month + 1)


def _r(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def load_zq_history(path: str | Path) -> dict[str, list[tuple[date, float, float]]]:
    """读取 Barchart 长表行情：symbol -> [(date, close, volume)]，按日期升序。"""
    history: dict[str, list[tuple[date, float, float]]] = {}
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            symbol = row["symbol"].strip().upper()
            history.setdefault(symbol, []).append(
                (
                    date.fromisoformat(row["as_of"]),
                    float(row["close"]),
                    float(row.get("volume") or 0),
                )
            )
    for rows in history.values():
        rows.sort(key=lambda item: item[0])
    if not history:
        raise ValueError(f"行情文件为空: {path}")
    return history


def load_rates_long(path: str | Path) -> dict[str, list[tuple[date, float]]]:
    """读取 series/as_of/value 长表，返回 series -> [(date, value)]。"""
    series: dict[str, list[tuple[date, float]]] = {}
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = row["series"].strip()
            series.setdefault(name, []).append(
                (date.fromisoformat(row["as_of"]), float(row["value"]))
            )
    for rows in series.values():
        rows.sort(key=lambda item: item[0])
    return series


def load_statements(path: str | Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = sorted(payload["statements"], key=lambda row: row["date"])
    for row in rows:
        row["_day"] = date.fromisoformat(row["date"])
    return rows


def latest_on_or_before(
    rows: list[tuple[date, float]], day: date
) -> tuple[date, float] | None:
    days = [item[0] for item in rows]
    index = bisect.bisect_right(days, day) - 1
    if index < 0:
        return None
    return rows[index]


def quote_on(
    history: dict[str, list[tuple[date, float, float]]], symbol: str, as_of: date
) -> Quote | None:
    rows = history.get(symbol)
    if not rows:
        return None
    days = [item[0] for item in rows]
    index = bisect.bisect_right(days, as_of) - 1
    if index < 0:
        return None
    day, close, volume = rows[index]
    return Quote(symbol=symbol, as_of=day, close=close, volume=volume)


def market_signal(
    meeting: Meeting,
    as_of: date,
    calendar: Calendar,
    zq: dict[str, list[tuple[date, float, float]]],
    rates: dict[str, list[tuple[date, float]]],
    config: BacktestConfig,
) -> dict:
    symbol = contract_symbol(meeting.year, meeting.month)
    quote = quote_on(zq, symbol, as_of)
    if quote is None:
        raise ValueError(f"{meeting.decision_date} 缺少 {symbol} 在 {as_of} 前的行情")
    effr = latest_on_or_before(rates["EFFR"], as_of)
    if effr is None:
        raise ValueError(f"{as_of} 缺少 EFFR")
    pre_rate = effr[1]

    quotes = {symbol: quote}
    ny, nm = _next_month(meeting.year, meeting.month)
    next_symbol = contract_symbol(ny, nm)
    next_quote = quote_on(zq, next_symbol, as_of)
    if next_quote is not None:
        quotes[next_symbol] = next_quote

    estimate = estimate_step(
        meeting, quote, pre_rate, calendar, quotes, min_post_days=config.min_post_days
    )
    prob = three_class_probabilities(estimate.post_rate, estimate.pre_rate, config.step)
    probability = {name: round(prob[name], 6) for name in CLASSES}
    pricing_symbol = next_symbol if estimate.method == "next_clean_month" else symbol
    pricing_volume = (
        next_quote.volume
        if estimate.method == "next_clean_month" and next_quote is not None
        else quote.volume
    )
    next_rate = implied_rate(next_quote.close) if next_quote is not None else None
    shape_bp = (
        (next_rate - implied_rate(quote.close)) * 100.0
        if next_rate is not None
        else None
    )
    return {
        "contract": symbol,
        "pricing_symbol": pricing_symbol,
        "method": estimate.method,
        "quote_date": quote.as_of.isoformat(),
        "quote_age_days": (as_of - quote.as_of).days,
        "quote_volume": quote.volume,
        "pricing_volume": pricing_volume,
        "next_contract": next_symbol,
        "next_implied_rate": _r(next_rate, 6) if next_rate is not None else None,
        "pricing_shape_bp": _r(shape_bp, 2) if shape_bp is not None else None,
        "pre_rate": _r(estimate.pre_rate, 6),
        "post_rate": _r(estimate.post_rate, 6),
        "change_bp": _r(estimate.change_bp, 2),
        "probability": probability,
        "expected_bp": _r((estimate.post_rate - estimate.pre_rate) * 100.0, 2),
        "most_likely": most_likely(probability),
        "reliability": estimate.reliability,
    }


def macro_signal_for(
    meeting: Meeting,
    as_of: date,
    pre_rate: float,
    macro_dir: str | Path,
    config: BacktestConfig,
    vintages: dict | None = None,
) -> dict:
    signal = build_macro_signal(
        macro_dir=macro_dir,
        as_of=as_of,
        current_rate=pre_rate,
        publication_lags=config.publication_lags,
        vintages=vintages,
    )
    market_path = [
        {
            "index": 1,
            "meeting": {
                "decision_date": meeting.decision_date.isoformat(),
                "effective_date": meeting.effective_date.isoformat(),
            },
            "pre_rate": pre_rate,
        }
    ]
    path = build_macro_path(
        signal=signal, market_path=market_path, current_rate=pre_rate, step=config.step
    )[0]
    probability = {name: round(path["probability"][name], 6) for name in CLASSES}
    return {
        "probability": probability,
        "expected_bp": path["expected_bp"],
        "most_likely": most_likely(probability),
        "macro_target_rate": signal.taylor["macro_target_rate"],
        "taylor_rate": signal.taylor["taylor_rate"],
        "policy_gap_bp": signal.taylor["policy_gap_bp"],
        "scores": {key: round(value, 4) for key, value in signal.scores.items()},
        "features": signal.features,
        "sources": signal.sources,
    }


def text_signal_for(statements: list[dict], as_of: date, config: BacktestConfig) -> dict:
    eligible = [row for row in statements if row["_day"] <= as_of]
    if not eligible:
        raise ValueError(f"{as_of} 之前没有 FOMC 声明文本")
    current = eligible[-1]
    previous = eligible[-2] if len(eligible) > 1 else None
    statement_score = score_statement(current)
    diff = (
        score_statement_diff(current, previous)
        if previous is not None
        else {"diff_score": 0.0, "added": [], "removed": []}
    )
    # 回测窗口内没有逐次演讲语料，按 50/50 重新归一化声明与声明 diff。
    text_score = _clip(
        0.5 * statement_score["score"] + 0.5 * float(diff["diff_score"])
    )
    dissent_score = 0.0
    if config.use_dissent:
        # 异议票方向作为小幅修正项；不改变 P1 默认口径。
        dissent_score = _clip(
            (current.get("hike_dissents", 0) - current.get("cut_dissents", 0)) / 3.0
        )
        text_score = _clip(
            0.45 * statement_score["score"]
            + 0.45 * float(diff["diff_score"])
            + 0.10 * dissent_score
        )
    return {
        "as_of_statement": current["date"],
        "previous_statement": previous["date"] if previous is not None else None,
        "statement_score": statement_score["score"],
        "diff_score": diff["diff_score"],
        "dissent_score": _r(dissent_score, 4),
        "text_score": _r(text_score, 4),
        "action": current.get("action"),
        "vote": current.get("vote"),
        "hike_dissents": current.get("hike_dissents", 0),
        "cut_dissents": current.get("cut_dissents", 0),
        "notes": (
            "声明 0.45 / diff 0.45 / 异议票 0.10"
            if config.use_dissent
            else "省略演讲信号，声明与 diff 按 0.5/0.5 归一"
        ),
    }


def text_probabilities(text_score: float, scale: float) -> dict:
    """文本单信号的对称三分类分布：softmax([-s*scale, 0, +s*scale])。"""
    logits = [-scale * text_score, 0.0, scale * text_score]
    top = max(logits)
    exps = [math.exp(value - top) for value in logits]
    total = sum(exps)
    return {name: exps[index] / total for index, name in enumerate(CLASSES)}


def actual_decision(
    meeting: Meeting, rates: dict[str, list[tuple[date, float]]]
) -> dict:
    before_lower = latest_on_or_before(rates["DFEDTARL"], meeting.decision_date)
    before_upper = latest_on_or_before(rates["DFEDTARU"], meeting.decision_date)
    after_lower = latest_on_or_before(rates["DFEDTARL"], meeting.effective_date)
    after_upper = latest_on_or_before(rates["DFEDTARU"], meeting.effective_date)
    if None in (before_lower, before_upper, after_lower, after_upper):
        raise ValueError(f"{meeting.decision_date} 缺少目标区间数据")
    delta = after_upper[1] - before_upper[1]
    label = "hike" if delta > 0 else "cut" if delta < 0 else "hold"
    return {
        "label": label,
        "change_bp": _r(delta * 100.0, 2),
        "target_lower_before": before_lower[1],
        "target_upper_before": before_upper[1],
        "target_lower_after": after_lower[1],
        "target_upper_after": after_upper[1],
    }


def _class_report(probabilities: list[dict], labels: list[str], eps: float = 1e-6) -> dict:
    n = len(labels)
    predictions = [most_likely(prob) for prob in probabilities]
    correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
    brier = sum(
        sum(
            (prob[name] - (1.0 if name == label else 0.0)) ** 2 for name in CLASSES
        )
        for prob, label in zip(probabilities, labels)
    ) / n
    log_loss = -sum(
        math.log(max(prob[label], eps)) for prob, label in zip(probabilities, labels)
    ) / n

    confusion = {actual: {pred: 0 for pred in CLASSES} for actual in CLASSES}
    for pred, label in zip(predictions, labels):
        confusion[label][pred] += 1

    by_class = {}
    for name in CLASSES:
        support = sum(1 for label in labels if label == name)
        predicted = sum(1 for pred in predictions if pred == name)
        hit = sum(
            1 for pred, label in zip(predictions, labels) if pred == name and label == name
        )
        precision = hit / predicted if predicted else 0.0
        recall = hit / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        by_class[name] = {
            "support": support,
            "predicted": predicted,
            "precision": _r(precision, 4),
            "recall": _r(recall, 4),
            "f1": _r(f1, 4),
        }

    return {
        "n": n,
        "accuracy": _r(correct / n, 4),
        "correct": correct,
        "brier": _r(brier, 4),
        "log_loss": _r(log_loss, 4),
        "confusion": confusion,
        "by_class": by_class,
    }


def calibration_table(
    probabilities: list[dict], labels: list[str], bins: int = 10
) -> list[dict]:
    rows = []
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        members = [
            (prob["hike"], label)
            for prob, label in zip(probabilities, labels)
            if (low <= prob["hike"] < high)
            or (index == bins - 1 and prob["hike"] == 1.0)
        ]
        if not members:
            continue
        mean_pred = sum(item[0] for item in members) / len(members)
        actual = sum(1 for item in members if item[1] == "hike") / len(members)
        rows.append(
            {
                "bin": f"{int(low * 100)}-{int(high * 100)}%",
                "count": len(members),
                "mean_predicted": _r(mean_pred, 4),
                "actual_rate": _r(actual, 4),
            }
        )
    return rows


def run_backtest(
    calendar_path: str | Path,
    zq_path: str | Path,
    rates_path: str | Path,
    macro_dir: str | Path,
    statements_path: str | Path,
    start: date,
    end: date,
    config: BacktestConfig | None = None,
    vintages_path: str | Path | None = None,
) -> dict:
    cfg = config or BacktestConfig()
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

    rows: list[dict] = []
    for meeting in meetings:
        as_of = meeting.decision_date - timedelta(days=cfg.cutoff_days)
        market = market_signal(meeting, as_of, calendar, zq, rates, cfg)
        macro = macro_signal_for(
            meeting, as_of, market["pre_rate"], macro_dir, cfg, vintages
        )
        text = text_signal_for(statements, as_of, cfg)
        text_prob = {
            name: round(value, 6)
            for name, value in text_probabilities(text["text_score"], cfg.text_scale).items()
        }
        neutral = pool(
            market["probability"],
            macro["probability"],
            0.0,
            cfg.weights,
            cfg.text_scale,
            cfg.smoothing_eps,
        )["probability"]
        combined = pool(
            market["probability"],
            macro["probability"],
            text["text_score"],
            cfg.weights,
            cfg.text_scale,
            cfg.smoothing_eps,
        )["probability"]
        contributions = {
            "macro": {
                name: round((neutral[name] - market["probability"][name]) * 100.0, 2)
                for name in CLASSES
            },
            "text": {
                name: round((combined[name] - neutral[name]) * 100.0, 2)
                for name in CLASSES
            },
            "total": {
                name: round((combined[name] - market["probability"][name]) * 100.0, 2)
                for name in CLASSES
            },
        }
        actual = actual_decision(meeting, rates)
        combined_bp = expected_bp(combined, cfg.step)
        rows.append(
            {
                "decision_date": meeting.decision_date.isoformat(),
                "effective_date": meeting.effective_date.isoformat(),
                "as_of": as_of.isoformat(),
                "market": market,
                "macro": macro,
                "text": {
                    **text,
                    "probability": text_prob,
                    "expected_bp": _r(
                        (text_prob["hike"] - text_prob["cut"]) * cfg.step * 100.0, 2
                    ),
                    "most_likely": most_likely(text_prob),
                },
                "combined": {
                    "probability": combined,
                    "expected_bp": _r(combined_bp, 2),
                    "expected_post_rate": _r(
                        market["pre_rate"] + combined_bp / 100.0, 6
                    ),
                    "most_likely": most_likely(combined),
                },
                "actual": actual,
                "contributions_pp": contributions,
                "correct": most_likely(combined) == actual["label"],
                "market_correct": market["most_likely"] == actual["label"],
                "diverges_from_market": most_likely(combined)
                != market["most_likely"],
            }
        )

    labels = [row["actual"]["label"] for row in rows]
    combined_probs = [row["combined"]["probability"] for row in rows]
    market_probs = [row["market"]["probability"] for row in rows]
    macro_probs = [row["macro"]["probability"] for row in rows]
    text_probs = [row["text"]["probability"] for row in rows]
    hold_probs = [{"cut": 0.0, "hold": 1.0, "hike": 0.0} for _ in rows]

    metrics = {
        "combined": _class_report(combined_probs, labels),
        "market": _class_report(market_probs, labels),
        "macro": _class_report(macro_probs, labels),
        "text": _class_report(text_probs, labels),
        "always_hold": _class_report(hold_probs, labels),
    }

    divergence_cases = [row for row in rows if row["diverges_from_market"]]
    divergence_help = sum(
        1
        for row in divergence_cases
        if row["correct"] and not row["market_correct"]
    )
    divergence_hurt = sum(
        1
        for row in divergence_cases
        if not row["correct"] and row["market_correct"]
    )
    macro_hike_contrib = [
        row["contributions_pp"]["macro"]["hike"] for row in rows
    ]
    text_hike_contrib = [row["contributions_pp"]["text"]["hike"] for row in rows]

    actual_counts = {
        name: sum(1 for label in labels if label == name) for name in CLASSES
    }
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
        "min_pricing_volume": min(
            row["market"]["pricing_volume"] for row in rows
        ),
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
    }

    return {
        "schema_version": "bt.1.0.0",
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "config": {
            "cutoff_days": cfg.cutoff_days,
            "step": cfg.step,
            "min_post_days": cfg.min_post_days,
            "weights": {
                "market": cfg.weights[0],
                "macro": cfg.weights[1],
                "text": cfg.weights[2],
            },
            "text_scale": cfg.text_scale,
            "smoothing_eps": cfg.smoothing_eps,
            "publication_lags": cfg.publication_lags,
        },
        "sample_size": len(rows),
        "actual_counts": actual_counts,
        "metrics": metrics,
        "calibration": {
            "combined": calibration_table(combined_probs, labels),
            "market": calibration_table(market_probs, labels),
        },
        "signal_contribution": {
            "macro_hike_mean_pp": _r(
                sum(macro_hike_contrib) / len(macro_hike_contrib), 2
            ),
            "text_hike_mean_pp": _r(
                sum(text_hike_contrib) / len(text_hike_contrib), 2
            ),
            "divergence_count": len(divergence_cases),
            "divergence_help": divergence_help,
            "divergence_hurt": divergence_hurt,
        },
        "data_quality": quality,
        "meetings": rows,
        "caveats": [
            "市场先验使用 Barchart 第三方延迟 ZQ 日线，非 CME 直连结算价。",
            "宏观数据为 FRED 当前版本（含后续修订），不是实时 vintage；已按发布滞后截断，但仍可能存在修订偏差。",
            "文本信号只使用会前最近两份 FOMC 声明，未纳入演讲、纪要或新闻发布会。",
            "实际决议由 FRED DFEDTARU/DFEDTARL 在决议生效日的变化推导；50bp 降息仍记为 cut 一次。",
            "三分类概率只建模单次会议至多一个 25bp 步长；50bp 情形在类别上正确，但幅度会被低估。",
            "样本类别极不平衡（17 次维持 / 6 次降息 / 1 次加息），准确率会被维持类主导，应结合 Brier、log loss 与分类别精确率/召回率解读。",
            "样本为 24 次会议，统计功效有限，指标对个别会议敏感。",
        ],
    }


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _pp(value: float) -> str:
    return f"{value:+.1f}pp"


def render_markdown(result: dict) -> str:
    metrics = result["metrics"]
    rows = result["meetings"]
    combined = metrics["combined"]
    label_cn = {"hike": "加息", "cut": "降息", "hold": "维持"}
    worsened = [row for row in rows if not row["correct"] and row["market_correct"]]
    worsen_text = ""
    if worsened:
        details = "；".join(
            f"{row['decision_date']} 合成 {_pct(row['combined']['probability']['hike'])}、"
            f"市场 {_pct(row['market']['probability']['hike'])}，"
            f"实际{label_cn[row['actual']['label']]}"
            for row in worsened
        )
        worsen_text = f"方向恶化案例：{details}。"

    lines = [
        "# 美联储加息预测回测（2023-11 至 2026-11）",
        "",
        f"- 回测窗口：{result['window']['start']} 至 {result['window']['end']}",
        f"- 样本：{result['sample_size']} 次政策会议；实际为 "
        f"{result['actual_counts']['hold']} 次维持、"
        f"{result['actual_counts']['cut']} 次降息、"
        f"{result['actual_counts']['hike']} 次加息。",
        f"- 预测截止：每次决议日前 {result['config']['cutoff_days']} 天；"
        f"权重 市场 {result['config']['weights']['market']:.2f} / "
        f"宏观 {result['config']['weights']['macro']:.2f} / "
        f"文本 {result['config']['weights']['text']:.2f}。",
        "",
        "## 一、核心结论",
        "",
        f"- 合成模型准确率 **{_pct(combined['accuracy'])}**"
        f"（{combined['correct']}/{combined['n']}），"
        f"市场先验 {_pct(metrics['market']['accuracy'])}，"
        f"始终维持 {_pct(metrics['always_hold']['accuracy'])}。",
        f"- 合成 Brier **{combined['brier']:.4f}**（市场 {metrics['market']['brier']:.4f}），"
        f"log loss **{combined['log_loss']:.4f}**（市场 {metrics['market']['log_loss']:.4f}）。",
        f"- 加息类：精确率 {_pct(combined['by_class']['hike']['precision'])}、"
        f"召回率 {_pct(combined['by_class']['hike']['recall'])}、"
        f"F1 {combined['by_class']['hike']['f1']:.4f}；"
        f"市场召回率 {_pct(metrics['market']['by_class']['hike']['recall'])}。",
        f"- 与市场方向不一致 {result['signal_contribution']['divergence_count']} 次，"
        f"其中改善 {result['signal_contribution']['divergence_help']} 次、"
        f"恶化 {result['signal_contribution']['divergence_hurt']} 次。",
        "- 在本样本中，ZQ 市场先验已基本完全定价实际决议；规则型宏观与文本信号未提升准确率。"
        + worsen_text,
        "",
        "## 二、指标对比",
        "",
        "| 模型 | 准确率 | Brier | log loss | 加息召回 | 降息召回 | 维持召回 |",
        "|------|--------|-------|----------|----------|----------|----------|",
    ]
    labels = {
        "combined": "合成（市场+宏观+文本）",
        "market": "市场先验（ZQ）",
        "macro": "宏观反应函数",
        "text": "文本信号",
        "always_hold": "始终维持",
    }
    for key in ("combined", "market", "macro", "text", "always_hold"):
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
            "| 决议日 | 截止日 | 实际 | 变化 | 市场加息 | 宏观加息 | 文本分 | 合成加息 | 合成预期 | 合成判断 | 正确 |",
            "|--------|--------|------|------|----------|----------|--------|----------|----------|----------|------|",
        ]
    )
    for row in rows:
        text_score = row["text"]["text_score"]
        lines.append(
            f"| {row['decision_date']} | {row['as_of']} | "
            f"{label_cn[row['actual']['label']]} | "
            f"{row['actual']['change_bp']:+.0f}bp | "
            f"{_pct(row['market']['probability']['hike'])} | "
            f"{_pct(row['macro']['probability']['hike'])} | "
            f"{text_score:+.3f} | "
            f"{_pct(row['combined']['probability']['hike'])} | "
            f"{row['combined']['expected_bp']:+.1f}bp | "
            f"{label_cn[row['combined']['most_likely']]} | "
            f"{'✓' if row['correct'] else '✗'} |"
        )

    lines.extend(
        [
            "",
            "## 四、加息概率校准",
            "",
            "合成模型：",
            "",
            "| 预测加息概率区间 | 样本数 | 平均预测 | 实际上加息比例 |",
            "|------------------|--------|----------|----------------|",
        ]
    )
    for item in result["calibration"]["combined"]:
        lines.append(
            f"| {item['bin']} | {item['count']} | "
            f"{_pct(item['mean_predicted'])} | {_pct(item['actual_rate'])} |"
        )
    if not result["calibration"]["combined"]:
        lines.append("| 无 | 0 | — | — |")

    lines.extend(
        [
            "",
            "市场先验：",
            "",
            "| 预测加息概率区间 | 样本数 | 平均预测 | 实际上加息比例 |",
            "|------------------|--------|----------|----------------|",
        ]
    )
    for item in result["calibration"]["market"]:
        lines.append(
            f"| {item['bin']} | {item['count']} | "
            f"{_pct(item['mean_predicted'])} | {_pct(item['actual_rate'])} |"
        )
    if not result["calibration"]["market"]:
        lines.append("| 无 | 0 | — | — |")

    lines.extend(["", "### 混淆矩阵（合成）", "", "| 实际 \\ 预测 | 降息 | 维持 | 加息 |", "|-------------|------|------|------|"])
    for actual in CLASSES:
        confusion = combined["confusion"][actual]
        lines.append(
            f"| {label_cn[actual]} | {confusion['cut']} | {confusion['hold']} | "
            f"{confusion['hike']} |"
        )

    lines.extend(["", "### 混淆矩阵（市场先验）", "", "| 实际 \\ 预测 | 降息 | 维持 | 加息 |", "|-------------|------|------|------|"])
    for actual in CLASSES:
        confusion = metrics["market"]["confusion"][actual]
        lines.append(
            f"| {label_cn[actual]} | {confusion['cut']} | {confusion['hold']} | "
            f"{confusion['hike']} |"
        )

    signal = result["signal_contribution"]
    hike_rows = [row for row in rows if row["actual"]["label"] == "hike"]
    lines.extend(
        [
            "",
            "## 五、信号贡献",
            "",
            f"- 宏观对加息概率的平均贡献：{_pp(signal['macro_hike_mean_pp'])}；"
            f"文本：{_pp(signal['text_hike_mean_pp'])}（相对市场先验）。",
            f"- 合成与市场方向不一致 {signal['divergence_count']} 次："
            f"改善 {signal['divergence_help']} 次、恶化 {signal['divergence_hurt']} 次。",
        ]
    )
    for row in hike_rows:
        lines.append(
            f"- 样本内加息会议 {row['decision_date']}："
            f"实际 {row['actual']['change_bp']:+.0f}bp；"
            f"市场 hike 概率 {_pct(row['market']['probability']['hike'])}，"
            f"合成 {_pct(row['combined']['probability']['hike'])}，"
            f"宏观 {_pct(row['macro']['probability']['hike'])}，"
            f"文本分 {row['text']['text_score']:+.3f}。"
        )
    if not hike_rows:
        lines.append("- 样本内没有加息会议，加息类指标仅反映误报情况。")

    lines.extend(
        [
            "",
            "## 六、方法与无前视处理",
            "",
            f"- 预测截止日 T = 决议日前 {result['config']['cutoff_days']} 个自然日。",
            "- 市场先验：取 T 当日或之前最后交易日 ZQ 收盘价；会议月合约会后天数不足 7 天时改用次月干净合约。",
            "- 宏观：对每个 FRED 序列按发布滞后截断后再计算，滞后（天）："
            + "、".join(
                f"{key} {value}" for key, value in result["config"]["publication_lags"].items()
            )
            + "。",
            "- 文本：只使用 T 之前已发布的声明，使用最近两份声明做语句分与 diff 分，0.5/0.5 归一。",
            f"- 合成：log-odds 池化，文本 logit 缩放 {result['config']['text_scale']}，平滑 ε={result['config']['smoothing_eps']}。",
            "- 实际标签：决议生效日目标区间上限相对决议日的变化；上升=加息、下降=降息、不变=维持。",
            "",
            "## 七、数据与局限",
            "",
        ]
    )
    lines.extend(f"- {caveat}" for caveat in result["caveats"])
    lines.extend(
        [
            "",
            f"- 数据质量：行情覆盖 {result['data_quality']['zq_symbols']} 个合约 / "
            f"{result['data_quality']['zq_observations']} 条观测；"
            f"最大行情滞后 {result['data_quality']['quote_age_max_days']} 天；"
            f"次月干净合约 {result['data_quality']['next_clean_month']} 次、"
            f"当月合约分解 {result['data_quality']['meeting_month']} 次；"
            f"定价合约最小成交量 {result['data_quality']['min_pricing_volume']:.0f} 手；"
            f"可靠性非 high {len(result['data_quality']['non_high_reliability'])} 次"
            f"（其中 low {len(result['data_quality']['low_reliability'])} 次）。",
            "",
            "## 八、复现",
            "",
            "```bash",
            "python3 scripts/fetch_barchart_history.js  # 需 Node + puppeteer-core + Chrome",
            "python3 scripts/fetch_backtest_data.py",
            "python3 run_backtest.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)
