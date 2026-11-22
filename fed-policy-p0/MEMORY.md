# 项目记忆：Fed Policy Radar

最后更新：2026-11-22

## 目标与边界

- 目标：用市场定价 + 宏观 + FOMC 文本，输出美联储利率决议的加息/维持/降息概率、幅度与市场 edge，辅助人工投资决策。
- 不做自动下单；实盘前必须先完成纸面交易验证。
- 不虚构数据；外部数据源抓取失败时保留本地快照并显式标注新鲜度。
- 回测训练使用 ALFRED vintage；实时预测使用当前可见数据（vintage 仅用于训练）。

## 常用命令

```bash
cd /workspace/fed-policy-p0
python3 run_live.py --as-of 2026-11-22      # 本地最新数据
python3 run_live.py --refresh               # 先刷新再预测
python3 run_live.py --fail-on-stale         # 关键数据过期则拒绝输出
python3 run_backtest.py                     # P1/P2 回测
python3 -m unittest discover -s tests -q    # 当前 109 项通过
```

产物：`output/live/prediction.json`、`prediction.md`、`manifest.json`、`history.jsonl`。

## 当前状态（2026-11-22）

- 实时 P2 已可用：下一会议 2026-12-09，训练至 2026-10-28（47 次）。
- 模型权重 0.85/0.10/0.05，锚定偏离强度 `blend=0`，校准器 Platt，幅度 identity。
- 当前 P2 与市场方向一致，edge 全部来自概率校准（约 8-10pp），不是方向 alpha。
- 2027-03-17 出现校准把 hold 翻成 hike，已标记为需人工复核。

## 待优化项（按优先级）

1. **行情数据源与合约覆盖**
   - 现状：ZQ 为 Barchart 第三方延迟日线；历史长表缺新合约，需静态快照补齐；`--refresh` 的行情步骤依赖 Node + puppeteer-core，当前环境未安装。
   - 目标：接入 CME 结算价或付费实时源；自动合约滚动；缺失合约校验与告警。

2. **校准层防止“虚假 edge”**
   - 现状：Platt 在训练样本上拟合，把 90% 抬到 98.6%，并可能翻转方向。
   - 目标：只用 walk-forward 前向样本拟合；限制单次校准幅度；增加校准翻转告警；报告中固定展示 raw 与 calibrated。

3. **方向信号增强**
   - 现状：宏观以 Taylor 规则 + 少数金融条件为主，文本只有声明 diff。
   - 目标：加入通胀 3m/6m 动量、就业动量、实际利率、期限利差、声明措辞变化、SEP/点阵图、发布会文本；做消融与样本外检验。

4. **决策与风控层**
   - 现状：只有概率、幅度和提示，没有仓位与交易成本。
   - 目标：风险预算/Kelly 上限、ZQ 流动性与到期检查、滑点与手续费、事件风险规则（SEP/发布会前后）。

5. **纸面交易与监控**
   - 现状：`history.jsonl` 已记录预测，但没有回看评估。
   - 目标：滚动 Brier/log loss/命中率；数据过期告警；模型漂移告警；每日定时运行。

6. **数据发布日历**
   - 现状：新鲜度按观测日期判断，月末可能误报。
   - 目标：接入 CPI/PCE/就业/GDP 发布日历，区分“未发布”和“已过期”。

7. **回测与实时一致性**
   - 现状：回测用 vintage，实时用修订后当前值。
   - 目标：同时记录“当时可见值”和“修订后值”，评估修订对决策的影响。

## 已知问题

- `.trae-html-share-packages/fed-policy-radar/fed-policy-radar.html.zip` 有未提交的二进制改动，与本项目无关。
- 实时行情刷新目前无法在缺少 puppeteer-core 的环境中完成，只能使用本地快照。
- 概率校准器在训练样本上选择，样本少时容易高估 edge。
