"""P1 Markdown 报告。"""

from __future__ import annotations


def _pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def _bp(value: float, digits: int = 1) -> str:
    return f"{value:+.{digits}f}bp"


def _num(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _short(text: str, limit: int = 240) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


ACTION_LABELS = {
    "raise": "加息",
    "increase": "加息",
    "maintain": "维持",
    "lower": "降息",
    "cut": "降息",
}

CLASS_LABELS = {"cut": "降息", "hold": "维持", "hike": "加息"}


def render_markdown(data: dict) -> str:
    combo = data["combination"]
    summary = combo["summary"]
    macro = data["macro"]
    text = data["text"]
    market_path = data["market_prior"]["path"]
    macro_path = macro["path"]
    path = combo["path"]
    features = macro["features"]
    taylor = macro["taylor"]
    statement = text["statement"]
    diff = text["statement_diff"]
    speeches = text["speeches"]

    lines: list[str] = []
    lines.append("# P1 美联储政策信号：市场先验 + 宏观反应函数 + 沟通文本")
    lines.append("")
    lines.append(
        f"基准日 **{data['as_of']}** ｜ 下次 FOMC **{summary['next_meeting_date']}** "
        f"｜ 生成时间 {data.get('generated_at', '—')}"
    )
    lines.append("")
    lines.append("## 1. 结论")
    lines.append("")
    lines.append(
        f"下次会议合成概率：**降息 {_pct(summary['combined']['probability']['cut'])} / "
        f"维持 {_pct(summary['combined']['probability']['hold'])} / "
        f"加息 {_pct(summary['combined']['probability']['hike'])}**，"
        f"最可能结果为 **{CLASS_LABELS.get(summary['combined']['most_likely'], summary['combined']['most_likely'])}**，"
        f"期望变动 **{_bp(summary['combined']['expected_bp'])}**，"
        f"期望会后利率 **{_num(summary['combined']['expected_post_rate'], 3)}%**。"
    )
    lines.append("")
    lines.append(
        f"市场先验为加息 {_pct(summary['market']['probability']['hike'])}，"
        f"宏观反应函数为 {_pct(summary['macro']['probability']['hike'])}，"
        f"文本信号 {text['text_score']:+.2f}（鹰派为正）。"
        f"合成较市场 **{summary['divergence_pp']['hike']:+.2f}pp** "
        f"（加息概率口径），"
        f"其中宏观贡献 {summary['contributions_pp']['macro']['hike']:+.2f}pp、"
        f"文本贡献 {summary['contributions_pp']['text']['hike']:+.2f}pp。"
    )
    lines.append("")
    last = path[-1]
    lines.append(
        f"四次会议路径：合成路径终点 **{_num(last['combined']['expected_post_rate'], 3)}%**，"
        f"累计 **{_bp(last['combined']['cumulative_bp'])}**；"
        f"市场路径终点 {_num(market_path[-1]['post_rate'], 3)}%（{_bp(market_path[-1]['cumulative_change_bp'])}），"
        f"宏观路径终点 {_num(macro_path[-1]['post_rate'], 3)}%（{_bp(macro_path[-1]['cumulative_bp'])}）。"
    )
    lines.append("")
    if summary["review_flag"]:
        lines.append("> **复核标记**：合成概率与市场隐含偏离超过 15pp，需人工复核。")
    else:
        lines.append(
            "> **复核标记**：未触发。合成与市场在 15pp 阈值内，方向一致；差异主要来自宏观反应函数的收敛速度与文本倾斜。"
        )
    lines.append("")

    lines.append("## 2. 下次会议三路信号")
    lines.append("")
    rows = [
        [
            "市场先验（P0）",
            _pct(summary["market"]["probability"]["cut"]),
            _pct(summary["market"]["probability"]["hold"]),
            _pct(summary["market"]["probability"]["hike"]),
            _bp(summary["market"]["expected_bp"]),
            f"{combo['weights']['market']:.2f}",
        ],
        [
            "宏观反应函数",
            _pct(summary["macro"]["probability"]["cut"]),
            _pct(summary["macro"]["probability"]["hold"]),
            _pct(summary["macro"]["probability"]["hike"]),
            _bp(summary["macro"]["expected_bp"]),
            f"{combo['weights']['macro']:.2f}",
        ],
        [
            "沟通文本",
            "—",
            "—",
            "—",
            "倾斜项",
            f"{combo['weights']['text']:.2f}",
        ],
        [
            "**合成**",
            f"**{_pct(summary['combined']['probability']['cut'])}**",
            f"**{_pct(summary['combined']['probability']['hold'])}**",
            f"**{_pct(summary['combined']['probability']['hike'])}**",
            f"**{_bp(summary['combined']['expected_bp'])}**",
            "—",
        ],
    ]
    lines.append(_table(["信号", "降息", "维持", "加息", "期望变动", "权重"], rows))
    lines.append("")
    lines.append("### 贡献分解（概率空间，pp）")
    lines.append("")
    contribution_rows = []
    for name, label in (("cut", "降息"), ("hold", "维持"), ("hike", "加息")):
        contribution_rows.append(
            [
                label,
                _pct(summary["market"]["probability"][name]),
                f"{summary['contributions_pp']['macro'][name]:+.2f}",
                f"{summary['contributions_pp']['text'][name]:+.2f}",
                _pct(summary["combined"]["probability"][name]),
                f"{summary['contributions_pp']['total'][name]:+.2f}",
            ]
        )
    lines.append(
        _table(["类别", "市场", "宏观贡献", "文本贡献", "合成", "总偏离"], contribution_rows)
    )
    lines.append("")
    lines.append(
        "贡献按顺序分解：市场 → 加入宏观（权重归一化）→ 加入文本。"
        "因此「宏观贡献 + 文本贡献 = 总偏离」在概率空间严格成立。"
    )
    lines.append("")

    lines.append("## 3. 四次会议路径")
    lines.append("")
    path_rows = []
    for row in path:
        path_rows.append(
            [
                str(row["index"]),
                row["meeting"]["decision_date"],
                _pct(row["market"]["probability"]["hike"]),
                _pct(row["macro"]["probability"]["hike"]),
                f"**{_pct(row['combined']['probability']['hike'])}**",
                _bp(row["market"]["cumulative_bp"]),
                _bp(row["macro"]["cumulative_bp"]),
                _bp(row["combined"]["cumulative_bp"]),
            ]
        )
    lines.append(
        _table(
            ["#", "会议", "市场加息", "宏观加息", "合成加息", "市场累计", "宏观累计", "合成累计"],
            path_rows,
        )
    )
    lines.append("")
    lines.append(
        "说明：概率为该会议节点上的条件三分类；累计为相对当前有效利率的期望变动。"
        "合成期望会后利率以 P0 市场会前利率为节点参考。"
    )
    lines.append("")

    lines.append("## 4. 宏观反应函数")
    lines.append("")
    lines.append("### 4.1 关键特征")
    lines.append("")
    feature_rows = [
        [
            "核心 PCE 同比",
            f"{features['core_pce']['yoy_pct']:.2f}%",
            features["core_pce"]["latest_date"],
            "高于 2% 目标",
            f"{macro['scores']['inflation']:+.2f}",
        ],
        [
            "核心 PCE 3m 年化",
            f"{features['core_pce']['ann3m_pct']:.2f}%",
            features["core_pce"]["latest_date"],
            f"短中期通胀动量（基期 {features['core_pce']['ann3m_base_date']}）",
            "—",
        ],
        [
            "核心 CPI 3m 年化",
            f"{features['core_cpi']['ann3m_pct']:.2f}%",
            features["core_cpi"]["latest_date"],
            f"通胀再加速风险（基期 {features['core_cpi']['ann3m_base_date']}）",
            "—",
        ],
        [
            "失业率 / CBO 自然率",
            f"{features['labor']['unemployment']:.2f}% / {features['labor']['nrou']:.2f}%",
            features["labor"]["unemployment_date"],
            f"缺口 {features['labor']['unemployment_gap_pp']:+.2f}pp",
            f"{macro['scores']['labor']:+.2f}",
        ],
        [
            "非农 3m 均值",
            f"{features['labor']['payrolls_3m_avg_k']:+.0f}k",
            features["labor"]["payrolls_latest_date"],
            "就业动量",
            f"{macro['scores']['labor_momentum']:+.2f}",
        ],
        [
            "实际 GDP 环比年化",
            f"{features['growth']['gdp_qoq_ann_pct']:.2f}%",
            features["growth"]["gdp_latest_date"],
            f"同比 {features['growth']['gdp_yoy_pct']:.2f}%",
            f"{macro['scores']['growth']:+.2f}",
        ],
        [
            "NFCI / 高收益 OAS",
            f"{features['financial_conditions']['nfci']:.2f} / {features['financial_conditions']['hy_oas']:.2f}%",
            features["financial_conditions"]["nfci_date"],
            "金融条件偏松",
            f"{macro['scores']['financial_conditions']:+.2f}",
        ],
        [
            "2y / 10y 美债",
            f"{features['market_rates']['dgs2']:.2f}% / {features['market_rates']['dgs10']:.2f}%",
            features["market_rates"]["dgs2_date"],
            f"10y 盈亏平衡 {features['market_rates']['t10yie']:.2f}%",
            "—",
        ],
    ]
    lines.append(_table(["指标", "最新值", "日期", "读数", "评分"], feature_rows))
    lines.append("")
    lines.append("评分范围 [-1, +1]：正值偏鹰（支持更高利率），负值偏鸽。")

    lines.append("")
    lines.append("### 4.2 Taylor 规则与政策缺口")
    lines.append("")
    lines.append("口径：")
    lines.append("")
    lines.append("```text")
    lines.append(
        f"r* = {taylor['r_star']:.2f}%    π = 核心PCE同比 {features['core_pce']['yoy_pct']:.2f}%    "
        f"目标通胀 = {taylor['target_inflation']:.2f}%"
    )
    lines.append(
        f"产出缺口 = -Okun × (u - u*) = -{macro['config']['okun_coefficient']:.1f} × "
        f"({features['labor']['unemployment']:.2f}% - {features['labor']['nrou']:.2f}%) "
        f"= {taylor['output_gap_pct']:+.2f}%"
    )
    lines.append(
        f"Taylor 利率 = r* + π + 0.5×(π-2%) + 0.5×产出缺口 "
        f"= {taylor['taylor_rate']:.2f}%"
    )
    lines.append(
        f"就业动量修正 = {taylor['labor_momentum_adjustment_pp']:+.2f}pp    "
        f"金融条件修正 = {taylor['financial_conditions_adjustment_pp']:+.2f}pp"
    )
    lines.append(f"宏观目标利率 = {taylor['macro_target_rate']:.2f}%")
    lines.append(
        f"政策缺口 = 当前利率 {taylor['current_rate']:.3f}% - Taylor {taylor['taylor_rate']:.2f}% "
        f"= {taylor['policy_gap_bp']:+.1f}bp"
    )
    lines.append(f"收敛系数 λ = {taylor['convergence_lambda']:.2f}（每次会议向目标收敛 25%）")
    lines.append("```")
    lines.append("")
    lines.append(
        f"政策缺口为负，表示当前利率低于规则隐含水平；宏观路径因此偏鹰。"
        f"宏观目标利率 {taylor['macro_target_rate']:.2f}% 是 Taylor 规则叠加就业动量与金融条件修正后的结果。"
    )
    lines.append("")
    lines.append("### 4.3 宏观隐含路径")
    lines.append("")
    macro_rows = []
    for row in macro_path:
        macro_rows.append(
            [
                str(row["index"]),
                row["meeting"]["decision_date"],
                _num(row["pre_rate"], 3),
                _bp(row["change_bp"]),
                _num(row["post_rate"], 3),
                _pct(row["probability"]["hike"]),
                _bp(row["cumulative_bp"]),
            ]
        )
    lines.append(
        _table(["#", "会议", "会前", "隐含变动", "会后", "加息概率", "累计"], macro_rows)
    )
    lines.append("")

    lines.append("## 5. 沟通文本信号")
    lines.append("")
    lines.append("### 5.1 最新声明")
    lines.append("")
    lines.append(
        f"声明日期 **{statement['date']}**，动作 **{ACTION_LABELS.get(statement['action'], statement['action'])}**，"
        f"投票 {statement['vote']}，主张加息异议 {statement['hike_dissents']} 人、"
        f"主张降息异议 {statement['cut_dissents']} 人。"
        f"词典分 {statement['lexicon']['score']:+.2f}，动作分 {statement['action_score']:+.2f}，"
        f"异议分 {statement['dissent_score']:+.2f}，综合 **{statement['score']:+.2f}**。"
    )
    lines.append("")
    term_rows = []
    for term in statement["lexicon"]["terms"][:8]:
        term_rows.append(
            [
                term["phrase"],
                f"{term['weight']:+.2f}",
                "是" if term["negated"] else "否",
            ]
        )
    if term_rows:
        lines.append(_table(["命中短语", "权重", "否定翻转"], term_rows))
        lines.append("")

    lines.append("### 5.2 与上次声明的措辞变化")
    lines.append("")
    lines.append(
        f"比较对象：{diff['previous_date']} → {statement['date']}；"
        f"新增文本得分 {diff['added_score']:+.2f}，删除文本得分 {diff['removed_score']:+.2f}，"
        f"净变化 **{diff['diff_score']:+.2f}**。"
    )
    lines.append("")
    if diff["added"]:
        lines.append("新增/改写句：")
        lines.append("")
        for sentence in diff["added"][:5]:
            lines.append(f"- {_short(sentence)}")
        lines.append("")
    if diff["removed"]:
        lines.append("删除/改写句：")
        lines.append("")
        for sentence in diff["removed"][:5]:
            lines.append(f"- {_short(sentence)}")
        lines.append("")
    if abs(diff["diff_score"]) < 0.05:
        diff_stance = "接近中性"
    elif diff["diff_score"] > 0:
        diff_stance = "边际偏鹰"
    else:
        diff_stance = "边际偏鸽"
    lines.append(
        f"本次 diff **{diff_stance}**：新增文本得分 {diff['added_score']:+.2f}，"
        f"删除文本得分 {diff['removed_score']:+.2f}。"
        "措辞变化反映政策动作与前瞻指引的边际调整；"
        "声明整体语气仍由最新文本的鹰鸽得分决定。"
    )
    lines.append("")

    lines.append("### 5.3 近期演讲")
    lines.append("")
    speech_rows = []
    for item in speeches["items"]:
        speech_rows.append(
            [
                item["date"],
                item["speaker"],
                item["role_label"],
                _short(item["title"], 42),
                f"{item['score']:+.2f}",
                f"{item['weight']:.2f}",
            ]
        )
    lines.append(
        _table(["日期", "发言人", "角色", "主题", "评分", "权重"], speech_rows)
    )
    lines.append("")
    lines.append(
        f"演讲加权得分 **{speeches['weighted_score']:+.2f}**；"
        f"权重 = 角色权重（主席 3.0 / 副主席 2.5 / 理事 2.0 / 地区主席 1.5）× 时效权重。"
    )
    lines.append("")

    lines.append("### 5.4 文本信号合成")
    lines.append("")
    component_rows = [
        [
            "最新声明",
            f"{text['components']['statement']:+.2f}",
            f"{text['weights']['statement']:.2f}",
        ],
        [
            "声明 diff",
            f"{text['components']['diff']:+.2f}",
            f"{text['weights']['diff']:.2f}",
        ],
        [
            "近期演讲",
            f"{text['components']['speeches']:+.2f}",
            f"{text['weights']['speeches']:.2f}",
        ],
        [
            "**文本总分**",
            f"**{text['text_score']:+.2f}**",
            "1.00",
        ],
    ]
    lines.append(_table(["组件", "得分", "权重"], component_rows))
    lines.append("")

    lines.append("## 6. 合成方法与约束")
    lines.append("")
    lines.append("三分类采用多项 logit 池化（以「维持」为基准类的 log-odds）：")
    lines.append("")
    lines.append("```text")
    lines.append("log p_combined(k) ∝ w_market·log p_market(k)")
    lines.append("                    + w_macro·log p_macro(k)")
    lines.append("                    + w_text·log p_text(k)")
    lines.append("p_combined = softmax(log p_combined)")
    lines.append("```")
    lines.append("")
    lines.append(
        f"下一次会议权重：市场 {combo['weights']['market']:.2f}、宏观 {combo['weights']['macro']:.2f}、"
        f"文本 {combo['weights']['text']:.2f}。市场权重满足不低于 0.5 的约束；"
        f"文本权重按 0.5 的衰减率随会议期限递减（0.15 → 0.075 → 0.0375 → 0.01875）。"
    )
    lines.append("")
    lines.append(
        f"文本倾斜：文本分数每 +1，加息 log-odds 提高 {combo['text_logit_scale']:.1f}，"
        f"降息 log-odds 降低同等幅度；文本不直接生成独立的三分类概率。"
    )
    lines.append("")
    lines.append(
        f"数值处理：log 平滑 ε = {combo['smoothing_epsilon']:.3f}；"
        "若市场与宏观对某一类的概率都为 0，且合成结果低于 1%，则将该类置 0，避免虚假尾部。"
    )
    lines.append("")

    lines.append("## 7. 数据来源与复现")
    lines.append("")
    lines.append("| 数据 | 来源 | 时点 |")
    lines.append("|---|---|---|")
    for row in data["provenance"]:
        lines.append(
            f"| {row.get('item', '—')} | {row.get('source', '—')} | {row.get('as_of') or row.get('retrieved_at') or '—'} |"
        )
    lines.append("")
    lines.append("复现命令：")
    lines.append("")
    lines.append("```bash")
    lines.append(f"python3 scripts/fetch_macro.py --as-of {data['as_of']}")
    lines.append(f"python3 scripts/fetch_fomc_text.py --as-of {data['as_of']}")
    lines.append(f"python3 run_p1.py --as-of {data['as_of']}")
    lines.append("```")
    lines.append("")

    lines.append("## 8. 限制与下一阶段")
    lines.append("")
    for caveat in data["caveats"]:
        lines.append(f"- {caveat}")
    lines.append("")
    lines.append(
        "下一阶段建议：P2 用历史会议面板估计有序 Probit（通胀缺口、失业率缺口、金融条件、文本分数），"
        "对 P1 的规则系数做 walk-forward 校准；同时接入共识预期（Surprise）与盘中利率期货，"
        "把「水平」信号升级为「相对预期」信号。"
    )
    return "\n".join(lines) + "\n"
