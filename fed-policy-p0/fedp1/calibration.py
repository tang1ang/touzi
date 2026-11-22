"""P2 概率校准与幅度预测。

小样本下优先使用「向基础频率收缩」（shrinkage）与 Platt 缩放；
isotonic 需要足够样本，仅作为报告中的对照说明，不参与主预测。
所有拟合函数都只接收训练期样本，调用方负责按会议时间做 walk-forward 切分。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

CLASSES = ("cut", "hold", "hike")


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize(probabilities: dict) -> dict:
    values = {name: max(float(probabilities.get(name, 0.0)), 0.0) for name in CLASSES}
    total = sum(values.values())
    if total <= 0:
        return {name: 1.0 / len(CLASSES) for name in CLASSES}
    return {name: values[name] / total for name in CLASSES}


def base_rates(labels: list[str], smoothing: float = 1.0) -> dict:
    counts = {
        name: smoothing + sum(1 for label in labels if label == name) for name in CLASSES
    }
    total = sum(counts.values())
    return {name: counts[name] / total for name in CLASSES}


def apply_shrinkage(probabilities: dict, alpha: float, base: dict) -> dict:
    alpha = _clip(alpha, 0.0, 1.0)
    return normalize(
        {
            name: (1.0 - alpha) * probabilities.get(name, 0.0) + alpha * base[name]
            for name in CLASSES
        }
    )


def mean_log_loss(probabilities: list[dict], labels: list[str], eps: float = 1e-9) -> float:
    return -sum(
        math.log(max(prob[label], eps)) for prob, label in zip(probabilities, labels)
    ) / len(labels)


def mean_brier(probabilities: list[dict], labels: list[str]) -> float:
    return sum(
        sum(
            (prob[name] - (1.0 if name == label else 0.0)) ** 2 for name in CLASSES
        )
        for prob, label in zip(probabilities, labels)
    ) / len(labels)


@dataclass
class ShrinkageCalibrator:
    alpha: float
    base: dict
    n: int
    raw_log_loss: float
    calibrated_log_loss: float

    def apply(self, probabilities: dict) -> dict:
        return apply_shrinkage(probabilities, self.alpha, self.base)

    def to_dict(self) -> dict:
        return {
            "method": "shrinkage",
            "alpha": round(self.alpha, 4),
            "base_rates": {name: round(self.base[name], 6) for name in CLASSES},
            "n": self.n,
            "raw_log_loss": round(self.raw_log_loss, 6),
            "calibrated_log_loss": round(self.calibrated_log_loss, 6),
        }


def fit_shrinkage(
    probabilities: list[dict],
    labels: list[str],
    grid: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
    min_n: int = 8,
) -> ShrinkageCalibrator:
    n = len(labels)
    base = base_rates(labels)
    if n == 0:
        return ShrinkageCalibrator(0.0, base, 0, 0.0, 0.0)
    raw_loss = mean_log_loss(probabilities, labels)
    if n < min_n:
        return ShrinkageCalibrator(0.0, base, n, raw_loss, raw_loss)
    best_alpha = 0.0
    best_loss = raw_loss
    for alpha in grid:
        calibrated = [apply_shrinkage(prob, alpha, base) for prob in probabilities]
        loss = mean_log_loss(calibrated, labels)
        if loss < best_loss - 1e-12:
            best_loss = loss
            best_alpha = alpha
    return ShrinkageCalibrator(best_alpha, base, n, raw_loss, best_loss)


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _logit(probability: float, eps: float = 1e-6) -> float:
    p = _clip(probability, eps, 1.0 - eps)
    return math.log(p / (1.0 - p))


@dataclass
class PlattModel:
    a: float = 1.0
    b: float = 0.0
    n: int = 0
    fitted: bool = False

    def predict(self, probability: float) -> float:
        if not self.fitted:
            return probability
        return _sigmoid(self.a * _logit(probability) + self.b)

    def to_dict(self) -> dict:
        return {
            "a": round(self.a, 4),
            "b": round(self.b, 4),
            "n": self.n,
            "fitted": self.fitted,
        }


def fit_binary_platt(
    scores: list[float],
    targets: list[int],
    l2: float = 1.0,
    learning_rate: float = 0.15,
    epochs: int = 1200,
    min_n: int = 10,
) -> PlattModel:
    """一维 Platt 缩放：p = sigmoid(a * logit(score) + b)，L2 向恒等映射收缩。"""
    n = len(targets)
    if n < min_n:
        return PlattModel(n=n, fitted=False)
    a, b = 1.0, 0.0
    for _ in range(epochs):
        grad_a = 0.0
        grad_b = 0.0
        for score, target in zip(scores, targets):
            error = _sigmoid(a * score + b) - target
            grad_a += error * score
            grad_b += error
        grad_a = grad_a / n + l2 * (a - 1.0) / n
        grad_b = grad_b / n + l2 * b / n
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b
    a = _clip(a, 0.25, 4.0)
    b = _clip(b, -2.0, 2.0)
    return PlattModel(a=a, b=b, n=n, fitted=True)


@dataclass
class PlattCalibrator:
    hike: PlattModel
    cut: PlattModel
    n: int

    def apply(self, probabilities: dict) -> dict:
        hike = self.hike.predict(probabilities.get("hike", 0.0))
        cut = self.cut.predict(probabilities.get("cut", 0.0))
        hold = max(0.0, 1.0 - hike - cut)
        return normalize({"cut": cut, "hold": hold, "hike": hike})

    def to_dict(self) -> dict:
        return {
            "method": "platt",
            "hike": self.hike.to_dict(),
            "cut": self.cut.to_dict(),
            "n": self.n,
        }


def fit_platt(
    probabilities: list[dict],
    labels: list[str],
    min_n: int = 10,
) -> PlattCalibrator:
    hike_scores = [_logit(prob["hike"]) for prob in probabilities]
    cut_scores = [_logit(prob["cut"]) for prob in probabilities]
    hike_targets = [1 if label == "hike" else 0 for label in labels]
    cut_targets = [1 if label == "cut" else 0 for label in labels]
    return PlattCalibrator(
        hike=fit_binary_platt(hike_scores, hike_targets, min_n=min_n),
        cut=fit_binary_platt(cut_scores, cut_targets, min_n=min_n),
        n=len(labels),
    )


def binary_ece(
    probabilities: list[dict],
    labels: list[str],
    positive: str = "hike",
    bins: int = 5,
) -> float:
    """按预测概率分箱的期望校准误差（ECE）。"""
    total = len(labels)
    if total == 0:
        return 0.0
    error = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        members = [
            (prob[positive], label)
            for prob, label in zip(probabilities, labels)
            if low <= prob[positive] < high
            or (index == bins - 1 and prob[positive] == 1.0)
        ]
        if not members:
            continue
        mean_pred = sum(item[0] for item in members) / len(members)
        actual = sum(1 for item in members if item[1] == positive) / len(members)
        error += len(members) / total * abs(mean_pred - actual)
    return error


@dataclass
class MagnitudeModel:
    intercept: float = 0.0
    slope: float = 1.0
    n: int = 0
    fitted: bool = False

    def predict(self, market_bp: float) -> float:
        return self.intercept + self.slope * market_bp

    def to_dict(self) -> dict:
        return {
            "intercept": round(self.intercept, 4),
            "slope": round(self.slope, 4),
            "n": self.n,
            "fitted": self.fitted,
            "note": "actual_bp = intercept + slope * market_expected_bp；向恒等映射收缩。",
        }


def fit_magnitude(
    market_bp: list[float],
    actual_bp: list[float],
    shrink: float = 0.5,
    min_n: int = 6,
) -> MagnitudeModel:
    n = len(actual_bp)
    if n < min_n:
        return MagnitudeModel(n=n, fitted=False)
    mean_x = sum(market_bp) / n
    mean_y = sum(actual_bp) / n
    sxx = sum((x - mean_x) ** 2 for x in market_bp)
    if sxx <= 1e-12:
        return MagnitudeModel(intercept=mean_y, slope=1.0, n=n, fitted=True)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(market_bp, actual_bp))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    slope = (1.0 - shrink) * slope + shrink * 1.0
    intercept = (1.0 - shrink) * intercept
    slope = _clip(slope, 0.5, 1.5)
    intercept = _clip(intercept, -15.0, 15.0)
    return MagnitudeModel(intercept=intercept, slope=slope, n=n, fitted=True)


def grid_round(value: float, step_bp: float = 25.0) -> float:
    return round(value / step_bp) * step_bp


def magnitude_metrics(
    predictions: list[float],
    actuals: list[float],
    step_bp: float = 25.0,
) -> dict:
    n = len(actuals)
    if n == 0:
        return {
            "n": 0,
            "mae_bp": 0.0,
            "rmse_bp": 0.0,
            "exact_grid_rate": 0.0,
            "direction_accuracy": 0.0,
            "large_move_precision": 0.0,
            "large_move_recall": 0.0,
        }
    errors = [pred - actual for pred, actual in zip(predictions, actuals)]
    mae = sum(abs(error) for error in errors) / n
    rmse = math.sqrt(sum(error**2 for error in errors) / n)
    exact = sum(
        1
        for pred, actual in zip(predictions, actuals)
        if abs(grid_round(pred, step_bp) - actual) < 1e-9
    ) / n
    direction = sum(
        1
        for pred, actual in zip(predictions, actuals)
        if (grid_round(pred, step_bp) > 0 and actual > 0)
        or (grid_round(pred, step_bp) < 0 and actual < 0)
        or (grid_round(pred, step_bp) == 0 and actual == 0)
    ) / n
    large_actual = [abs(actual) >= 2 * step_bp for actual in actuals]
    large_pred = [abs(grid_round(pred, step_bp)) >= 2 * step_bp for pred in predictions]
    true_positive = sum(
        1 for actual, pred in zip(large_actual, large_pred) if actual and pred
    )
    predicted_positive = sum(1 for pred in large_pred if pred)
    actual_positive = sum(1 for actual in large_actual if actual)
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / actual_positive if actual_positive else 0.0
    return {
        "n": n,
        "mae_bp": round(mae, 2),
        "rmse_bp": round(rmse, 2),
        "exact_grid_rate": round(exact, 4),
        "direction_accuracy": round(direction, 4),
        "large_move_precision": round(precision, 4),
        "large_move_recall": round(recall, 4),
        "large_move_support": actual_positive,
    }
