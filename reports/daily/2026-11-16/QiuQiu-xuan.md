# 每日交付：2026-11-16 / QiuQiu-xuan

对应 [逐日计划](https://github.com/Zqqq-qq/webshop-stepwise-rl/blob/main/docs/DAILY_PLAN.md) 的准备日，以及 11/17（手算指标、独立核对样例）、11/18（评估输入与重复 task 检查）、11/20（指标 v1、评估配置）里能在 CPU 上完成的部分。11/19 的"人工复核 3 条真实轨迹"**没有完成**，原因是仓库里没有本组 rollout 日志，见第 4 节。

## 今天要解决的问题

1. 指标口径能不能被第二套代码独立复算，而不只是"跑通脚本"。
2. 一次评估开跑之前，怎么在 CPU 上发现 config 漂移、adapter 缺失、ID 重复、fingerprint 未冻结、两臂不可比。
3. 仓库自带的 78 条演示轨迹能不能当失败案例用（结论：不能，且要留下可复现的依据）。

## 学到并能解释的内容

**一条真实购物轨迹里，agent 看见什么、能改什么（L1/L2）。** 环境每一步给一段文本观测（`rollout.py` 把它作为 `user` 回合），agent 只能生成一条 `Action: search[...]` 或 `Action: click[...]`；动作由 `parse_action` 解析、`is_executable` 对照当前页面可点击项。环境返回 `observation/reward/done`，这些是环境给的，不是 agent 生成的 token。

**购买结束不等于成功。** WebShop 在买到商品时就返回 `done=True`，但 `final_reward` 可能是 1、0.6，甚至正好 0（买了但没有任何必需属性命中）。所以 `rollout.py` 单独记录 `termination='purchase'` 与 `success=done and reward>=1-1e-6`，指标主分母仍是全部尝试的 episode。这也是原包审计里"购买即成功"的问题所在。

**SFT / 奖励模型 / 策略更新更新的是不同东西。** SFT 只更新 LoRA 参数、监督来自已有对话的 assistant token；奖励模型/进度估计器拟合终局奖励；策略更新用采样轨迹的回报改策略参数。当前仓库里后两者都还没有实机验证，不能混着说。

## 实际完成与证据

- 分支/PR：`feature/unified-eval-qiuqiuxuan`，PR #8（本日两轮提交都在该 PR 上），Issue #5。
- 本轮新增（全部 CPU、标准库）：
  - `webshop-lab preflight`：config 必备键、adapter 目录与文件、ID 列表类型/重复/空、两臂 parity、登记表表头与数值自洽；未冻结 fingerprint 只告警，`--require-frozen` 才阻塞。
  - `webshop-lab compare`：只接受同一批 `(task, repeat)`，输出配对四格、McNemar 精确 p、配对任务簇 bootstrap 的差值区间。
  - `webshop-lab results-table`：登记表转 Markdown 结果表，只取 `status=complete`。
  - `scripts/verify_metrics.py`：**不 import 本包**的第二套实现，复算 10 个字段并与 `evaluate` 输出逐项比对。
  - `scripts/check_upstream_demo_data.py` + `results/upstream-demo-data-check.json`：演示轨迹可用性检查。
  - `results/metrics-definition.json`：指标 v1，机器可读；测试会检查它与代码字段一致。
  - 失败分诊修正：购买奖励为 0 归入 `specification`；购买结果优先于格式错误，`format` 进次要信号，新增 `any_counts`。
- 执行命令与结果：
  - `python -m unittest discover -s tests` → **43 项通过**（第一轮 32 + 本轮 11）。
  - `python -m compileall -q src scripts tests`、`python scripts/check_repository.py` → 通过。
  - `python scripts/verify_metrics.py --jsonl <合成 b0> --evaluation <evaluate 输出>` → 10/10 字段一致；手改一个字段后返回 1。
  - `preflight --a-config configs/eval-base.json --b-config configs/eval-sft.json` → `ready=false`，唯一阻塞项是 `adapter directory does not exist`（SFT 产物还没落地），fingerprint 未冻结只是 warning。
  - `compare` 在 6 条合成任务上（两条完全相同的 run）→ `risk_difference=0`、`mcnemar_exact_p=1`；任务集不同则拒绝。
  - `results-table` 在只有表头的登记表上 → 输出"暂无 complete 行"，不产生假数字。
- 未验证：任何真实模型输出、真实失败案例、test 集结果。合成 fixture 的数字不写进 proposal/report。

## 观察、错误与判断

1. **规则 bug（已修）**：第一版分诊只在 `0<reward<1` 时判预算/规格，但 WebShop 买错时奖励经常正好是 0，这类轨迹会被错分到 loop/retrieval。检查真实观测文本时发现，已改为"已购买且奖励<1"即进入购买结果判定，并加了两个回归用例。
2. **优先级 bug（已修）**：一次"先格式错误、最后买错"的轨迹原来只被记成 `format`，购买结果被掩盖。现在主标签给购买结果，`format` 保留在 `secondary`，并用 `any_counts` 统计"出现过该信号"的 episode。
3. **我自己的误判（已纠正）**：我用临时脚本统计演示轨迹时把正则转义写错，先得出"没有一次购买前有价格"的错误结论；重写成 `scripts/check_upstream_demo_data.py` 后真实数字是 **79/100 次 Buy Now 前置观测含价格**、803/1170 个 assistant 回合不可解析、奖励只有 0/1。结论修正为"可校准解析，不可当案例/成功率"。
4. 演示轨迹的观测大量是 `Invalid format` 回放，说明其中很多 assistant 回合本身是截断生成；这也解释了为什么不能拿它算失败分布。

## 交接

- 给 HowseaonLink / Jun040504：跑完 B0/B1 后请回传 **`.jsonl` + 同名 `.meta.json` + 实际 config + 所用 `*_ids.json`**；并在配置里填 `/health` 的 `goal_fingerprint`。只给汇总数字无法核对分母。
- 给 Zqqq-qq：`reports/proposal-review-evaluation.md` 里列了 proposal 提交前必须改的 4 处（R-TG/R-PG 同 grounding、可检查的公平性字段、指标分母与区间、划分规模）；`metrics-definition.json` 可直接作为指标 v1 引用。
- 给 Yuxiao-Liu233：图表口径——方法对比只画 `status=complete` 的登记行，图注必须写任务数与区间；失败案例只能用人工确认过的本组日志。

## 六个问题的当前答案

1. agent 能看见当前页面的文本观测与可执行动作列表，看不见环境内部商品属性、评分计算和未来页面。
2. agent 只生成 assistant 回合里的那条 `Action:`；`Observation:`、`Instruction:` 是环境/数据给的。
3. 购买结束 ≠ 成功：`done=True` 只说明买了，成功还要 `final_reward>=1-1e-6`；从原始 JSONL 用 `verify_metrics.py` 重算可复核。
4. SFT 更新 LoRA、监督是对话里的 assistant token；奖励模型拟合终局奖励；策略更新用轨迹回报改策略。三者在本仓库的完成度不同。
5. 一次 generate、一次 reward-export、一次 loss 下降都只说明某个环节跑通；RL 有效性需要"更新前后策略不同 + 新策略重新采样 + 独立任务对照"。
6. 本组实测目前只有数据审计、划分和工具自检；原论文数字只能引用；RL 与 B0/B1 结果都还是计划。

## 下一个学习问题

`max_steps` 截断时 GAE 该不该 bootstrap、本环境怎么区分"真实终止"与"时间上限截断"？这直接影响 12/01 那次策略更新里优势估计的写法，我想在下一轮 L7 精读时确认。
