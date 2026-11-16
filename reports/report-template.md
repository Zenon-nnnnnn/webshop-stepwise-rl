# Final Report 模板

> 2026-12-15 提交。课程名、成员、学号、语言、字数与格式待填。此文件没有实验结论。

## Abstract

填实际问题、方法、数据规模、主要实测结果与限制。所有实验完成后再写，不预填提升。

## 1. Introduction

任务、动机、研究问题、贡献边界；区分复用和新增。

## 2. Related Work

SPA-RL、WebShop、参数高效微调、过程奖励与相关 agent 训练。引用原始来源。

## 3. Data and Experimental Protocol

来源与条款、原始审计、去重、train/val/test、环境 goal 映射、资产 SHA、冻结规则。

## 4. Methods

基础模型与 prompt、量化/LoRA、训练标签、历史处理；实际完成的进度估计和奖励公式。若没有策略优化实验，不把仅奖励导出写成 PPO。

## 5. Experimental Setup

硬件、驱动、软件版本、模型 revision、超参、seed、任务数、采样/训练预算和资源成本。说明与原论文的差异。

## 6. Results

从已核实的结果登记表填入，附任务数与不确定性；说明每个比较控制了哪些变量。表格不得引用模板数字。

## 7. Analysis

数据量/奖励分布、失败案例、预测误差与 agent 性能之间的区别。进度分解的非唯一性及表面特征风险。

## 8. Limitations

单卡预算、样本量、任务相关性、模型替换、未完成模块、旧 PPO 适配状态。明确负结果。

## 9. Conclusion

仅根据实际实验回答研究问题，不超出证据。

## Contribution Statement

逐人写环境、数据、训练、奖励、评估、报告中的具体工作和 PR/实验记录。

## References

原论文、环境、模型、LoRA/QLoRA 及实际使用工具。

## Reproducibility Appendix

仓库 commit、配置、资产和划分哈希、环境 fingerprint、模型/日志共享位置、复现实验命令与预计资源。
