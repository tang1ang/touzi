"""三分类概率：把预期变动映射到 25bp 步长网格。"""

from __future__ import annotations

import math

DEFAULT_STEP = 0.25


def three_class_probabilities(
    expected_rate: float,
    baseline_rate: float,
    step: float = DEFAULT_STEP,
) -> dict:
    """返回 cut / hold / hike 概率。

    将预期变动 delta 映射到相邻两个步长结果上（线性权重）。
    当 |delta| <= step 时，等价于文档中的
    P(hike)=clamp(delta/step,0,1)、P(cut)=clamp(-delta/step,0,1)。
    超过一个步长时按多步结果分摊，再聚合为三分类。
    """
    delta = expected_rate - baseline_rate
    x = delta / step
    lo = math.floor(x)
    hi = lo + 1
    w_hi = min(max(x - lo, 0.0), 1.0)
    outcomes = {lo: 1.0 - w_hi, hi: w_hi}

    cut = sum(weight for k, weight in outcomes.items() if k < 0)
    hold = outcomes.get(0, 0.0)
    hike = sum(weight for k, weight in outcomes.items() if k > 0)
    total = cut + hold + hike

    return {
        "cut": cut / total,
        "hold": hold / total,
        "hike": hike / total,
        "expected_bp": delta * 100,
        "expected_steps": x,
        "beyond_one_step": abs(x) > 1.0,
    }


def most_likely(probabilities: dict) -> str:
    return max(("cut", "hold", "hike"), key=lambda k: probabilities[k])
