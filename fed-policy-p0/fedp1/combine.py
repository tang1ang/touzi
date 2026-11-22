"""P1 三路信号合成：市场先验 + 宏观反应函数 + 沟通文本，log-odds 池化。"""

from __future__ import annotations

import math

CLASSES = ("cut", "hold", "hike")
DEFAULT_EPS = 0.005
DEFAULT_TEXT_SCALE = 2.0
REVIEW_THRESHOLD_PP = 15.0


def _log_score(probability: float, eps: float) -> float:
    """三分类的多项 logit 得分：log(p_k)。

    等价于以“维持”为基准类的 log-odds；单一信号权重为 1 时，
    softmax 会严格还原该信号的三分类分布。
    """
    return math.log(max(probability, eps))


def _softmax(logits: list[float]) -> list[float]:
    top = max(logits)
    exps = [math.exp(value - top) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def _prob_dict(values: list[float]) -> dict:
    return {name: values[index] for index, name in enumerate(CLASSES)}


def _clean_probabilities(probabilities: dict, market: dict, macro: dict) -> dict:
    """市场与宏观都未给出的类别，不因对数平滑产生虚假尾部。"""
    values = dict(probabilities)
    for name in CLASSES:
        if (
            market.get(name, 0.0) <= 1e-9
            and macro.get(name, 0.0) <= 1e-9
            and values[name] < 0.01
        ):
            values[name] = 0.0
    total = sum(values.values())
    return {name: value / total for name, value in values.items()}


def _round_prob(values: dict) -> dict:
    return {name: round(values[name], 6) for name in CLASSES}


def most_likely(values: dict) -> str:
    return max(CLASSES, key=lambda name: values[name])


def expected_bp(values: dict, step: float) -> float:
    return (values["hike"] - values["cut"]) * step * 100.0


def validate_weights(weights: tuple[float, float, float]) -> tuple[float, float, float]:
    market, macro, text = weights
    if min(market, macro, text) < 0:
        raise ValueError("权重不能为负")
    total = market + macro + text
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"权重之和必须为 1，当前为 {total}")
    if market < 0.5:
        raise ValueError("下一次会议的市场权重不得低于 0.5")
    return market, macro, text


def weights_for(index: int, base: tuple[float, float, float]) -> tuple[float, float, float]:
    """文本信号随会议期限衰减，市场权重不低于 0.5。"""
    market, macro, text = base
    text_i = text * (0.5 ** (index - 1))
    market_i = 1.0 - macro - text_i
    if market_i < 0.5:
        market_i = 0.5
        macro = max(0.0, 1.0 - market_i - text_i)
    total = market_i + macro + text_i
    return market_i / total, macro / total, text_i / total


def pool(
    market: dict,
    macro: dict,
    text_score: float,
    weights: tuple[float, float, float],
    text_scale: float = DEFAULT_TEXT_SCALE,
    eps: float = DEFAULT_EPS,
) -> dict:
    market_logits = [_log_score(market[name], eps) for name in CLASSES]
    macro_logits = [_log_score(macro[name], eps) for name in CLASSES]
    text_logits = [
        market_logits[0] - text_scale * text_score,
        market_logits[1],
        market_logits[2] + text_scale * text_score,
    ]
    market_w, macro_w, text_w = weights
    logits = [
        market_w * market_logits[i] + macro_w * macro_logits[i] + text_w * text_logits[i]
        for i in range(len(CLASSES))
    ]
    probabilities = _prob_dict(_softmax(logits))
    probabilities = _clean_probabilities(probabilities, market, macro)
    return {
        "probability": probabilities,
        "logits": {name: logits[i] for i, name in enumerate(CLASSES)},
        "signal_logits": {
            "market": _prob_dict(market_logits),
            "macro": _prob_dict(macro_logits),
            "text": _prob_dict(text_logits),
        },
    }


def decompose(
    market: dict,
    macro: dict,
    text_score: float,
    weights: tuple[float, float, float],
    text_scale: float = DEFAULT_TEXT_SCALE,
    eps: float = DEFAULT_EPS,
) -> dict:
    market_w, macro_w, text_w = weights
    market_logits = [_log_score(market[name], eps) for name in CLASSES]
    macro_logits = [_log_score(macro[name], eps) for name in CLASSES]

    macro_only_logits = [
        (market_w * market_logits[i] + macro_w * macro_logits[i]) / (market_w + macro_w)
        for i in range(len(CLASSES))
    ]
    macro_only = _prob_dict(_softmax(macro_only_logits))
    macro_only = _clean_probabilities(macro_only, market, macro)
    full = pool(market, macro, text_score, weights, text_scale, eps)
    full_prob = full["probability"]

    macro_pp = {name: (macro_only[name] - market[name]) * 100.0 for name in CLASSES}
    text_pp = {name: (full_prob[name] - macro_only[name]) * 100.0 for name in CLASSES}
    total_pp = {name: (full_prob[name] - market[name]) * 100.0 for name in CLASSES}

    macro_logit = {
        name: macro_w * (macro_logits[i] - market_logits[i])
        for i, name in enumerate(CLASSES)
    }
    text_logit = {
        name: text_w * text_scale * text_score
        * (1.0 if name == "hike" else -1.0 if name == "cut" else 0.0)
        for name in CLASSES
    }
    abs_total = sum(
        abs(macro_logit[name]) + abs(text_logit[name]) for name in CLASSES
    )
    shares = {
        name: {
            "macro": (abs(macro_logit[name]) / abs_total) if abs_total else 0.0,
            "text": (abs(text_logit[name]) / abs_total) if abs_total else 0.0,
        }
        for name in CLASSES
    }

    return {
        "market": _round_prob(market),
        "macro": _round_prob(macro),
        "macro_pooled": _round_prob(macro_only),
        "combined": _round_prob(full_prob),
        "weights": {
            "market": round(market_w, 6),
            "macro": round(macro_w, 6),
            "text": round(text_w, 6),
        },
        "contributions_pp": {
            "macro": {name: round(macro_pp[name], 2) for name in CLASSES},
            "text": {name: round(text_pp[name], 2) for name in CLASSES},
            "total": {name: round(total_pp[name], 2) for name in CLASSES},
        },
        "logit_contributions": {
            "macro": {name: round(macro_logit[name], 4) for name in CLASSES},
            "text": {name: round(text_logit[name], 4) for name in CLASSES},
            "total": {
                name: round(macro_logit[name] + text_logit[name], 4) for name in CLASSES
            },
            "shares": {
                name: {
                    "macro": round(shares[name]["macro"], 4),
                    "text": round(shares[name]["text"], 4),
                }
                for name in CLASSES
            },
        },
    }


def build_combination(
    market_path: list[dict],
    macro_path: list[dict],
    text_score: float,
    weights: tuple[float, float, float],
    step: float = 0.25,
    text_scale: float = DEFAULT_TEXT_SCALE,
    eps: float = DEFAULT_EPS,
) -> dict:
    base_weights = validate_weights(weights)
    rows: list[dict] = []
    for market_row, macro_row in zip(market_path, macro_path):
        index = market_row["index"]
        meeting_weights = weights_for(index, base_weights)
        market_prob = market_row["probability"]
        macro_prob = macro_row["probability"]
        parts = decompose(
            market_prob, macro_prob, text_score, meeting_weights, text_scale, eps
        )
        combined_prob = parts["combined"]
        combined_bp = expected_bp(combined_prob, step)
        market_bp = expected_bp(market_prob, step)
        macro_bp = expected_bp(macro_prob, step)
        pre_rate = market_row["pre_rate"]
        divergence_pp = {
            name: round((combined_prob[name] - market_prob[name]) * 100.0, 2)
            for name in CLASSES
        }
        review_flag = any(abs(value) > REVIEW_THRESHOLD_PP for value in divergence_pp.values())
        rows.append(
            {
                "index": index,
                "meeting": market_row["meeting"],
                "weights": parts["weights"],
                "market": {
                    "pre_rate": market_row["pre_rate"],
                    "post_rate": market_row["post_rate"],
                    "change_bp": market_row["change_bp"],
                    "cumulative_bp": market_row["cumulative_change_bp"],
                    "probability": parts["market"],
                    "expected_bp": round(market_bp, 2),
                    "most_likely": most_likely(market_prob),
                },
                "macro": {
                    "pre_rate": macro_row["pre_rate"],
                    "post_rate": macro_row["post_rate"],
                    "change_bp": macro_row["change_bp"],
                    "cumulative_bp": macro_row["cumulative_bp"],
                    "probability": parts["macro"],
                    "expected_bp": round(macro_bp, 2),
                    "most_likely": most_likely(macro_prob),
                },
                "combined": {
                    "pre_rate": pre_rate,
                    "expected_bp": round(combined_bp, 2),
                    "expected_post_rate": round(pre_rate + combined_bp / 100.0, 6),
                    "cumulative_bp": round(
                        (
                            pre_rate
                            + combined_bp / 100.0
                            - market_path[0]["pre_rate"]
                        )
                        * 100.0,
                        2,
                    ),
                    "probability": combined_prob,
                    "expected_steps": round(combined_bp / (step * 100.0), 4),
                    "most_likely": most_likely(combined_prob),
                },
                "contributions_pp": parts["contributions_pp"],
                "logit_contributions": parts["logit_contributions"],
                "divergence_pp": divergence_pp,
                "review_flag": review_flag,
            }
        )

    first = rows[0]
    summary = {
        "next_meeting_date": first["meeting"]["decision_date"],
        "market": first["market"],
        "macro": first["macro"],
        "combined": first["combined"],
        "contributions_pp": first["contributions_pp"],
        "divergence_pp": first["divergence_pp"],
        "review_flag": first["review_flag"],
    }
    return {
        "weights": {
            "market": base_weights[0],
            "macro": base_weights[1],
            "text": base_weights[2],
        },
        "text_logit_scale": text_scale,
        "smoothing_epsilon": eps,
        "review_threshold_pp": REVIEW_THRESHOLD_PP,
        "summary": summary,
        "path": rows,
    }
