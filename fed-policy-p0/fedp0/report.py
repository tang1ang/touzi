"""P0 报告渲染：Markdown 版本。"""

from __future__ import annotations

LABELS = {"cut": "降息", "hold": "维持", "hike": "加息"}
METHOD_LABELS = {"next_clean_month": "次月干净合约", "meeting_month": "当月合约日权重"}
RELIABILITY_LABELS = {"high": "高", "medium": "中", "low": "低"}


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _rate(value: float) -> str:
    return f"{value:.4f}%"


def _bp(value: float) -> str:
    return f"{value:+.1f} bp"


def _num(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def render_markdown(baseline: dict) -> str:
    basis = baseline["policy_basis"]
    summary = baseline["summary"]
    path = baseline["path"]
    quality = baseline["data_quality"]
    first = path[0]

    lines: list[str] = []
    add = lines.append

    add("# 美联储政策雷达 · P0 市场基准")
    add("")
    add(f"- 基准日：**{baseline['as_of']}**")
    add(
        f"- 下一次 FOMC：**{first['meeting']['decision_date']}**"
        f"（{first['meeting']['start']} 至 {first['meeting']['end']}）"
    )
    add(
        f"- 当前政策基准：**{_rate(basis['current_rate'])}**（{basis['rate_basis']}，"
        f"目标区间 {_rate(basis['target_range']['lower'])}–{_rate(basis['target_range']['upper'])}）"
    )
    add("")
    add(
        f"> **结论：最可能{LABELS[summary['most_likely']]}"
        f"（{_pct(summary[summary['most_likely']])}）**"
        f"（降息 {_pct(summary['cut'])} / 维持 {_pct(summary['hold'])} / "
        f"加息 {_pct(summary['hike'])}），"
        f"市场隐含会后利率 {_rate(summary['expected_post_rate'])}，"
        f"较当前 {_bp(summary['expected_change_bp'])}。"
    )
    add("")

    add("## 1. P0 口径")
    add("")
    add("| 项目 | 定义 |")
    add("| --- | --- |")
    add("| 预测对象 | 下一次 FOMC 的降息 / 维持 / 加息三分类概率，并外推未来四次会议路径 |")
    add(f"| 合约 | {basis['contract']}，以月度平均 EFFR 结算 |")
    add(f"| 利率基准 | 当前取 {basis['rate_basis']} = {_rate(basis['current_rate'])}；EFFR 无法使用时回退目标区间中值 |")
    add("| 决议生效 | 决议日当天仍按旧利率计息，**次一自然日**生效 |")
    add("| 合约映射 | 会议所在月份的 ZQ 合约（如 2026-12 → ZQZ26） |")
    add(f"| 步长 | {basis['step_bp']} bp |")
    add(f"| 分解优先级 | {basis['method_priority']} |")
    add(f"| 最少会后天数 | {basis['min_post_days']} 天；低于该值时可靠性下调 |")
    add("")
    add("隐含利率与日权重分解：")
    add("")
    add("```text")
    add("r_month = 100 - 期货收盘价")
    add("r_post  = (r_month × D_month - r_pre × D_pre) ÷ D_post")
    add("```")
    add("")
    add(
        "概率把预期变动映射到 25bp 网格：`x = (r_post - r_pre) / 0.25`，"
        "在 `floor(x)` 与 `ceil(x)` 两档上按线性权重分摊，再聚合为降息 / 维持 / 加息。"
        "当 `|x| ≤ 1` 时，等价于 `P(hike)=clamp(x,0,1)`、`P(cut)=clamp(-x,0,1)`。"
    )
    add("")

    add("## 2. 输入快照")
    add("")
    add("| 数据 | 最新日期 | 相对基准日 | 来源 |")
    add("| --- | --- | --- | --- |")
    add(
        f"| ZQ 期货收盘价 | {quality['quote_date']} | {quality['quote_age_days']} 天 | "
        "Barchart（第三方延迟） |"
    )
    add(
        f"| 目标区间 | {quality['target_date']} | {quality['target_age_days']} 天 | FRED |"
    )
    if quality.get("effr_date"):
        add(
            f"| EFFR | {quality['effr_date']} | {quality['effr_age_days']} 天 | FRED |"
        )
    add("")
    if quality["warnings"]:
        add("数据质量提示：")
        add("")
        for warning in quality["warnings"]:
            add(f"- {warning}")
        add("")

    add("## 3. 市场隐含曲线")
    add("")
    add("| 合约 | 月份 | 收盘价 | 隐含利率 | 成交量 | 用途 |")
    add("| --- | --- | --- | --- | --- | --- |")
    path_symbols = {row["contract"] for row in path}
    check_symbols = {row["symbol"] for row in baseline["curve_cross_check"]}
    for row in baseline["market_curve"]:
        role = "路径" if row["symbol"] in path_symbols else (
            "校验" if row["symbol"] in check_symbols else "参考"
        )
        add(
            f"| {row['symbol']} | {row['month']} | {_num(row['close'], 4)} | "
            f"{_rate(row['implied_rate'])} | {row['volume']:,} | {role} |"
        )
    add("")
    add("> 路径列为实际进入四次会议推算的会议月合约；校验列为用于曲线一致性校验的合约。")
    add("")

    add("## 4. 下一次会议推算（逐步）")
    add("")
    meeting = first["meeting"]
    days = first["days"]
    add(f"**{meeting['decision_date']}**，决议于 **{meeting['effective_date']}** 生效。")
    add("")
    add(f"1. 合约 `{first['contract']}` 收盘价 {_num(first['contract_close'], 4)}，"
        f"隐含月均利率 `r_month = 100 - {_num(first['contract_close'], 4)} = "
        f"{_rate(first['contract_implied_rate'])}`。")
    if first["method"] == "meeting_month":
        add(
            f"2. 当月共 {days['month']} 天，会前 {days['pre']} 天、会后 {days['post']} 天，"
            f"会前利率取当前 {_rate(first['pre_rate'])}。"
        )
        add(
            f"3. `r_post = ({_rate(first['contract_implied_rate'])} × {days['month']} - "
            f"{_rate(first['pre_rate'])} × {days['pre']}) ÷ {days['post']} = "
            f"{_rate(first['post_rate'])}`。"
        )
    else:
        add(f"2. {first['note']}，会后利率直接取 {_rate(first['post_rate'])}。")
    add(
        f"4. 变动 `{_bp(first['change_bp'])}`，即 {_num(first['expected_steps'], 2)} 个 25bp 步长。"
    )
    add(
        f"5. 三分类概率：降息 **{_pct(first['probability']['cut'])}** / "
        f"维持 **{_pct(first['probability']['hold'])}** / "
        f"加息 **{_pct(first['probability']['hike'])}**。"
    )
    add("")

    add("## 5. 未来四次会议路径")
    add("")
    add("| # | 会议 | 合约 | 方法 | 会前 | 会后 | Δ | 降息 | 维持 | 加息 | 可靠性 |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in path:
        add(
            f"| {row['index']} | {row['meeting']['decision_date']} | {row['contract']} | "
            f"{METHOD_LABELS.get(row['method'], row['method'])} | {_rate(row['pre_rate'])} | "
            f"{_rate(row['post_rate'])} | {_bp(row['change_bp'])} | "
            f"{_pct(row['probability']['cut'])} | {_pct(row['probability']['hold'])} | "
            f"{_pct(row['probability']['hike'])} | "
            f"{RELIABILITY_LABELS.get(row['reliability'], row['reliability'])} |"
        )
    add("")
    add(
        f"累计：到 **{summary['path_end_meeting']}** 市场隐含利率 "
        f"**{_rate(summary['path_end_rate'])}**，较当前 {_bp(summary['path_cumulative_bp'])}，"
        f"约合 **{_num(summary['path_hikes_priced'], 2)} 次 25bp 加息**。"
    )
    add("")

    add("## 6. 曲线一致性校验")
    add("")
    add("用拟合出的阶梯利率重算各月平均，与市场合约隐含利率对比：")
    add("")
    add("| 月份 | 合约 | 市场隐含 | 模型隐含 | 偏差 | 结果 |")
    add("| --- | --- | --- | --- | --- | --- |")
    for row in baseline["curve_cross_check"]:
        add(
            f"| {row['month']} | {row['symbol']} | {_rate(row['observed_rate'])} | "
            f"{_rate(row['model_rate'])} | {_bp(row['diff_bp'])} | "
            f"{'通过' if row['pass'] else '偏离'} |"
        )
    add("")
    add("> 容差 ±2bp。偏差主要来自期货价格 0.005 的最小变动与月内资金面波动。")
    add("")

    add("## 7. 数据来源与刷新")
    add("")
    add("| 数据 | 来源 | 基准时点 | 链接 |")
    add("| --- | --- | --- | --- |")
    for row in baseline["provenance"]:
        add(
            f"| {row['item']} | {row['source']} | {row['as_of'] or '-'} | "
            f"{row['url']} |"
        )
    add("")
    add("刷新命令：")
    add("")
    add("```bash")
    add("python3 scripts/fetch_fomc_calendar.py")
    add("python3 scripts/fetch_fred.py --as-of 2026-11-22")
    add("node scripts/fetch_barchart.js --out data/market/zq_curve_latest.csv")
    add("python3 run_p0.py --as-of 2026-11-22 --zq data/market/zq_curve_latest.csv")
    add("```")
    add("")

    add("## 8. 限制与下一阶段")
    add("")
    for caveat in baseline["caveats"]:
        add(f"- {caveat}")
    add("")
    add("P0 只回答“市场当前如何定价”。P1 将在此先验上叠加宏观反应函数（通胀、就业、金融条件）与美联储沟通文本信号，形成可解释的联合概率。")
    add("")

    return "\n".join(lines)
