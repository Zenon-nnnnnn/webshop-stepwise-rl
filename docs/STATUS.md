# 当前状态与下一步

## 已完成的仓库建设

- 原始 SPA 可复用源码快照与清单；移除缓存、平台二进制和一处硬编码 credential。
- 新 CLI、资产导入、上游审计、任务/指令分组划分。
- GPU 诊断、单卡 SFT 实现、统一配置。
- 原 WebShop CPU 环境桥、真实 rollout 实现与指标。
- 可加线性进度估计器、预测与奖励导出。
- CPU 测试、GitHub Actions、中文上手文档、分工和交付草案。

## 验证分层

| 层 | 当前证据 |
|---|---|
| 标准库工具逻辑 | 本地 unittest 与端到端合成轨迹检查，见 validation 记录 |
| 真实材料审计 | 原始 SFT 和示例轨迹条数、重叠、分组划分 |
| 代码静态可导入/语法 | 新 src 与 scripts 编译检查 |
| 依赖存在性 | 候选版本需按官方包源核对；存在不等于互相兼容 |
| GPU kernel / 量化 / SFT | 尚未在目标 5060 Ti 运行 |
| 完整 WebShop / Java / 索引 | 尚未在目标机端到端运行 |
| 真实 agent 成功率 | 尚无结果 |
| 原神经 SPA / PPO | 代码保留，现代 GPU 适配与正确性验证待完成 |

## 更新规则

首次目标机成功后，提交 evidence 路径和命令，再修改本表。不能因文件存在、脚本启动或合成测试通过就宣称训练/论文复现成功。详细验收见 TEAM 与 EXPERIMENTS。

## 线上验收

首次完整提交的 [GitHub Actions CPU checks](https://github.com/Zqqq-qq/webshop-stepwise-rl/actions/runs/43848833958) 已全部通过。五人分工与交付拆成 [6 个任务](https://github.com/Zqqq-qq/webshop-stepwise-rl/issues)，已按已知 GitHub 账号制定第一轮模块分工，具体见 TEAM.md；真实姓名与学号由组内补充。
