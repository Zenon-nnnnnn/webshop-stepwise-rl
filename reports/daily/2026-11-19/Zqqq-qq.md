# Zqqq-qq · 2026-11-19：学习与实际检查

## 本人回答与 Tutor 示范的区别

用户亲自回答了优势/策略更新、GAE 和环境 token mask 的概念问题；计算、代码核验、下载和划分操作由 Codex Tutor 按用户要求执行。不写成学生独立运行 PPO 或独立实现训练。

- 用户解释优势的原话：“优势大只是表明这次的动作可以强化……指导参数更新 → 参数变化使概率变化。”Tutor 补充：方向由正负决定，绝对值影响样本推动强度；不是把优势直接加到概率上。
- 用户解释 GAE 的原话：“局部td残差表现为负，但是后续……td仍可能为正数，结合后续结果这一步的优势依然是正的。”Tutor 纠正即时奖励与 TD 残差的区别。该概念回答正确，但不是用户独立完成全部数值计算的证据。
- 用户解释环境 mask 的原话：“这些是外部环境，不算 agent 动作，不能混入策略损失。”Tutor 补充：环境文字作为输入可读，只是对应位置不直接计入策略损失。
- 已讲解 PPO 正/负优势下的概率比和 clipping；用户反馈理解，尚无独立 clipping 数值作答或真实 PPO 运行。

## 已执行的数值练习

详见 [E1/E2 学习材料](../../../learning/2026-11-19-e1-e2.md) 和 [计算结果](../../../learning/2026-11-19-e1-e2-results.json)。简化单 logit 更新验证正/负/零优势方向；GAE=[0.174605,0.251,0.2]。合成 CPU 示例，不是购物结果或 PPO 训练。

## 今日实际工作

1. 网络恢复后取得 How 的真实映射原件，SHA 与 Release/环境 manifest 一致，12087 个任务。
2. 使用该映射在 Mac 重跑 split：1451 train、160 val、200 test、排除13。独立检查 12 项全部通过，包括完整实际指令的三组隔离、保留记录不变及与环境侧 ID/manifest blob 一致。见 [映射复核](../../../results/mapped-split-20261119-zqq/README.md)。
3. 在独立 CPU 工具 venv 使用 transformers 4.56.2 和固定 Qwen tokenizer revision，实际核对 SFT 样本0。三个 target 分别58/79/61 tokens，前缀131/874/1060 tokens全部 labels=-100；target包含 Thought、Action 和结束模板。见 [token 检查](../../../results/learning-20261119-zqq/README.md)。

## 未完成与后续

没有下载/加载模型权重，没有 GPU、SFT forward/backward、Trainer collator 或真实 RL 更新。token检查依据 main eb76f4e 的预处理逻辑；Jun 的训练分支若改变模板/版本，须复核对应版本。

数据的完整映射依赖已解除，静态划分已独立复现；目标机正式运行前仍须使用对应 IDs、完整资产和同一 goal fingerprint 通过 live health。环境 fingerprint 来自目标机报告，未在 Mac 重新生成全环境 goals。

后续优先：组员复核并确认本次映射版划分的采用记录；实现/验收终局奖励契约（progress/reward 仍直接读 final_reward）；准备11/20 的预算、SFT和RL接入清单。数据 Issue #2 不因本日报自动关闭；未自动合并其他 PR。
