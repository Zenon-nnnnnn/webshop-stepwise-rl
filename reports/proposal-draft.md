# Proposal 草案：资源受限条件下 WebShop 智能体的逐步奖励学习

> 草案，待填课程名、五名组员与学号、rubric、页数要求。截止暂按 2026-11-22；不是已提交版本。

详细设计与可复用实验段落见 [实验协议](../docs/EXPERIMENT_PROTOCOL.md)。奖励辅助项、容量与运行配置待复核冻结，本文不代表实验已经运行。

## 1. Problem and Motivation

LLM agent 需要多次搜索、查看商品、选择规格后才完成购物。终局奖励稀疏，可能难以指导早期决策。我们基于 SPA-RL 的逐步进度归因思路，研究它在单张 16 GB GPU、小型模型和有限轨迹条件下的适用性。

## 2. Research Questions

RQ1：参数高效 SFT 能否改善小模型在独立 WebShop 任务上的成功率？

RQ2：仅用轨迹最终奖励，能否学到具有独立验证预测能力的可加进度估计？

RQ3：在相近的采样与更新预算下，逐步奖励相对最终奖励能否改善 agent 行为？RQ3 的完成依赖 RL 集成通过实机验收，不能以奖励模型误差替代策略评估。

## 3. Data and Environment

使用 WebShop 模拟购物环境与现有 SFT 轨迹。本次材料含 1,624 条 SFT 记录；审计发现默认 train/test 有 12 个 ID 重叠，附带进度模型 train/val 相同。因此首先建立任务/底层购物指令分组划分，使用实际环境映射进一步排除交叉。

训练用 train，方法开发用 val，最终配置冻结后才用 test。记录资产 SHA、split manifest 与环境 fingerprint。

## 4. Proposed Method

基于同一个 1.5B 级 instruct 模型建立无任务微调基线，采用 QLoRA 进行 SFT。通过真实 WebShop 交互收集多样成功/失败轨迹。进度估计器以各步分数总和拟合最终奖励；先建立低成本可加线性基线，再评估原神经 SPA 实现的适配可行性。

在可行预算内，将进度分数与动作可执行信号融合，比较 terminal-reward 与 stepwise-reward 策略优化。所有模型与上下文、提示、任务列表和训练预算保持可比。

## 5. Evaluation

主要指标为满奖励任务成功率；辅助指标为平均最终奖励、交互步数、无效动作率、资源开销。进度估计用独立任务上的总分 MSE，并检查逐步归因案例。报告不确定性与失败类别；不保证任何方法必然优于基线。

## 6. Expected Contribution

预期交付包括可复核的数据审计与独立划分、资源受限 SFT 基线、逐步奖励预测分析，以及条件允许时的 RL 对照实验。复用 SPA-RL 算法与代码会明确署名；新增工具、实验与观察分别说明。

## 7. Resources, Risks and Contingency

硬件：i5-13400F、32 GB RAM、RTX 5060 Ti 16 GB。新 GPU 与旧训练依赖存在兼容性风险；采用 GPU/CPU 环境隔离和短运行检查。PPO 存在额外显存、数据策略一致性与调试风险。

优先跑通 PPO；若旧依赖持续阻塞，准备明确标注的 on-policy 策略梯度分支，两种奖励臂使用同一算法，保留真实 RL 实践并说明与论文差异。若真实 RL 仍未完成，须如实报告并对照课程 rubric 处理，不能预先认定仅 SFT 与进度估计足以满足要求。

## 8. Schedule and Team

11/22 proposal 为规划日期；11/27 pre、12/15 report 已确认。11/25 首次真实 RL 更新尝试，12/01 多轮闭环硬验收，12/07 冻结最终配置、12/08–10 最终测试。HowseaonLink 负责环境与主算力，Zqqq-qq 负责数据/奖励/RL 与技术整合，Jun040504 负责 SFT，QiuQiu-xuan 负责评估，Yuxiao-Liu233 负责 PPT/pre；各人写自己的报告模块。详细验收见团队及每日计划。

## References

Wang et al., 2025. [SPA-RL: Reinforcing LLM Agents via Stepwise Progress Attribution](https://arxiv.org/abs/2505.20732)。提交前补充实际使用的 WebShop、基础模型及 QLoRA 文献，按课程规定格式排版。
