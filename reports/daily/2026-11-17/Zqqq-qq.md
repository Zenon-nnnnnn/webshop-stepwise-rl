# 每日交付：2026-11-17 / Zqqq-qq（Tutor 协助示范）

## 执行与学习边界

用户要求“你帮我完成，我看着就行”，因此本次审计和验证由 Codex 在用户 Mac 上实际执行，用户旁观。以下为操作证据与 Tutor 解释，不写成学生独立完成、已掌握全部方法的声明。策略梯度 CPU 练习和 RL 集成尚未执行。

## 今天解决的问题

验证组内共享的原 SFT，重跑审计和 seed=42 初步划分，核对 ID/指令交叉、记录保真与重复性；为环境负责人提供可复核交接。

## 实际执行与证据

- 源码基于 `eb76f4efe35f71842ae80a8020875148ff10cd1f`；macOS / Python 3.13.16；只使用标准库 CPU 工具，不代表团队 Python 3.10 GPU 环境验证。
- 实际原始输入：`/Users/sschy/Downloads/webshop_sft.json`，9,312,587 字节；SHA-256 `721e46dba410cff900e187bccea91a456c38b105b6d14d068d7762afb7791de3`，与私有 Release 一致。
- 运行 `PYTHONPATH=src python -m webshop_lab.cli audit --sft /Users/sschy/Downloads/webshop_sft.json --output artifacts/zqq-data-20261117/audit.json`。
- 运行两次 `split --sft /Users/sschy/Downloads/webshop_sft.json --seed 42`，分别写入 `data/processed/offline-20261117-zqq` 与 `artifacts/zqq-data-20261117/repeated-split`，比较全部输出一致。
- 独立核对 11 项条件均通过；现有 8 项 CPU unittest 通过。
- 可提交的摘要、三份 ID、指纹与核对证据见 [离线证据目录](../../../results/offline-split-20261117-zqq/README.md)。完整对话未提交。

## 结果及解释

1. 原始 1,624 条 SFT，默认 train/test 有 12 个相同任务 ID。训练中见过测试任务会干扰对独立能力的判断，不能直接沿用。
2. 新候选为 1,451 train / 160 val，保留 200 个测试任务 ID；共排除 13 条。额外的任务 1746 与测试任务 5413 只是预算不同，说明 ID 不同不等于底层购物请求不同。
3. train/val ID、两者对 test ID、train/val 归一化指令、与已知测试指令的交叉均为零。
4. `reward != true_reward` 有 8 条；本次只记录而未改标签。SFT 工具没有按这些分数字段做 loss 加权，但数据质量及筛选规则仍需在正式实验协议解释。
5. 这只是离线候选：未见实际环境完整目标映射，不能声称测试请求全部无交叉，也不能称为冻结版。

## 交接与尚未完成

- HowseaonLink：从 data-v1 获取完整资产，启动环境并提供实际 goal-map/fingerprint；之后重新核对、划分和冻结。
- Jun：可复现候选划分和检查 SFT 数据，但正式训练须等待最终协议确认。
- Qiu：可复核 ID、摘要和输入/输出哈希，评估不使用这些计数冒充模型成绩。
- 不关闭 Issue #2，不合并为“最终训练数据”；不声称执行了 GPU、真实 rollout、SFT 或 PPO。

## Tutor 的理解检查（未替用户作答验收）

- 为什么删掉 12 个相同 ID 后还需要删任务 1746？
- 为什么多次运行得到相同划分，不等于这个划分已排除全部泄漏？

理解确认后，下一步继续 CPU 策略梯度练习，或取得实际环境 goal-map 后完成数据最终冻结。
