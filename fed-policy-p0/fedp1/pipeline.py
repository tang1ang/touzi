"""P1 管道：P0 市场先验 + 宏观反应函数 + 沟通文本 -> 联合概率与贡献分解。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from fedp0.pipeline import build_baseline

from .combine import DEFAULT_EPS, DEFAULT_TEXT_SCALE, build_combination
from .macro import MacroConfig, build_macro_path, build_macro_signal, macro_sources
from .text import build_text_signal


def _slim_market_path(baseline: dict) -> list[dict]:
    rows = []
    for row in baseline["path"]:
        rows.append(
            {
                "index": row["index"],
                "meeting": row["meeting"],
                "contract": row["contract"],
                "method": row["method"],
                "method_label": row["method_label"],
                "pre_rate": row["pre_rate"],
                "post_rate": row["post_rate"],
                "change_bp": row["change_bp"],
                "cumulative_change_bp": row["cumulative_change_bp"],
                "probability": row["probability"],
                "expected_bp": row["expected_bp"],
                "expected_steps": row["expected_steps"],
                "most_likely": row["most_likely"],
                "reliability": row["reliability"],
            }
        )
    return rows


def build_p1(
    calendar_path: str | Path,
    zq_path: str | Path,
    rates_path: str | Path,
    macro_dir: str | Path,
    text_dir: str | Path,
    as_of: date,
    n_meetings: int = 4,
    step: float = 0.25,
    min_post_days: int = 7,
    weights: tuple[float, float, float] = (0.60, 0.25, 0.15),
    text_scale: float = DEFAULT_TEXT_SCALE,
    smoothing_eps: float = DEFAULT_EPS,
    macro_config: MacroConfig | None = None,
) -> dict:
    baseline = build_baseline(
        calendar_path=calendar_path,
        zq_path=zq_path,
        rates_path=rates_path,
        as_of=as_of,
        n_meetings=n_meetings,
        step=step,
        min_post_days=min_post_days,
    )
    current_rate = baseline["policy_basis"]["current_rate"]
    market_path = _slim_market_path(baseline)

    macro_signal = build_macro_signal(
        macro_dir=macro_dir,
        as_of=as_of,
        current_rate=current_rate,
        config=macro_config,
    )
    macro_path = build_macro_path(
        signal=macro_signal,
        market_path=market_path,
        current_rate=current_rate,
        step=step,
    )
    text_signal = build_text_signal(text_dir=text_dir, as_of=as_of)
    combination = build_combination(
        market_path=market_path,
        macro_path=macro_path,
        text_score=text_signal["text_score"],
        weights=weights,
        step=step,
        text_scale=text_scale,
        eps=smoothing_eps,
    )

    provenance = list(baseline["provenance"])
    provenance.append(
        {
            "item": "宏观数据（通胀 / 就业 / 增长 / 金融条件）",
            "source": "FRED",
            "url": "https://fred.stlouisfed.org/",
            "as_of": as_of.isoformat(),
        }
    )
    provenance.append(
        {
            "item": "FOMC 声明与演讲文本",
            "source": "Federal Reserve",
            "url": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
            "as_of": text_signal["sources"].get("retrieved_at"),
        }
    )

    caveats = [
        caveat
        for caveat in baseline["caveats"]
        if "尚未叠加宏观反应函数" not in caveat
    ]
    caveats.extend(
        [
            "宏观反应函数是规则型（Taylor 规则 + 就业动量/金融条件修正），不是估计的有序 Probit；系数与阈值是透明先验，需用 walk-forward 回测校准。",
            "Taylor 规则使用 CBO 自然失业率（NROU）与 Okun 系数 2；产出缺口由失业率缺口推算。",
            "FRED 提供当前版本数据，可能包含修订，不是实时 vintage；as-of 仅表示取数截止日。",
            "文本评分采用可复现的鹰鸽词典，未使用 LLM；能稳定复现，但对上下文、反讽和条件句理解有限。",
            "声明 diff 仅比较最近两份声明；演讲样本为近 45 天、经政策相关性筛选，并按角色与时效加权。",
            "合成在 log-odds 空间进行，下一次会议市场权重 0.60（满足不低于 0.5 的约束），文本权重随会议期限衰减。",
            "三分类概率假设单次会议至多一个 25bp 步长；50bp 等大幅情形未单独建模。",
            "当合成概率与市场隐含偏离超过 15pp 时，review_flag 置为 true，需人工复核而非自动发布异见。",
        ]
    )

    return {
        "schema_version": "p1.0.0",
        "as_of": as_of.isoformat(),
        "policy_basis": baseline["policy_basis"],
        "market_prior": {
            "summary": baseline["summary"],
            "path": market_path,
            "data_quality": baseline["data_quality"],
        },
        "macro": {
            **macro_signal.to_dict(),
            "path": macro_path,
            "sources": macro_sources(macro_dir),
        },
        "text": text_signal,
        "combination": combination,
        "provenance": provenance,
        "caveats": caveats,
    }
