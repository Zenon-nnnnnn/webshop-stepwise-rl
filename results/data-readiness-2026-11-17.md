# 数据准备进展与 SFT 文件待办（2026-11-17）

## 当前状态

**当前工作目录缺少 `webshop_sft.json`，完整 SFT 审计与 train/val 初步划分尚未完成。** 原作者专家轨迹归档尚未下载成功，请全组协助确认已有材料、下载负责人和共享位置。

本次已完成仓库附带材料的离线审计。审计对象为提交 `9df3063321ccda2a3a96e1a5d7d9d720bdd296e0`；以下计数是数据材料检查结果，不是训练或模型性能结果。

## 已完成的材料审计

| 项目 | 结果 |
|---|---|
| 上游训练任务 ID 数 | 1,624 |
| 上游测试任务 ID 数 | 200 |
| 原始训练/测试重叠 ID 数 | 12 |
| 附带探索轨迹 | 78 条，覆盖 26 个任务 |
| 附带探索满奖励轨迹 | 1 条 |
| 附带进度模型训练/验证材料是否相同 | 是 |
| 附带 PPO 示例轨迹 | 41 条 |

重叠 ID 为：818、978、2640、2933、4161、5413、5900、7475、7603、7624、7999、8573。需要按项目数据协议重新划分，并在正式训练前对齐真实环境指令。

复验命令：

```bash
PYTHONPATH=src .venv-tools/bin/python -m unittest discover -s tests -v
.venv-tools/bin/python -m compileall -q src scripts
.venv-tools/bin/python scripts/check_repository.py
PYTHONPATH=src .venv-tools/bin/python -m webshop_lab.cli audit --output artifacts/pr-audit.json
```

结果：8 项测试通过，编译和仓库检查通过，审计确认上述计数。本次完整 SFT 文件缺失，因此审计报告不含对该文件的重新验证。测试样例的成功率等输出不作为真实实验结果。

## 需要组内提供 webshop_sft.json

[已有历史审计](material-audit.json) 记录过以下文件：

- SFT 记录数：1,624；
- SHA-256：`721e46dba410cff900e187bccea91a456c38b105b6d14d068d7762afb7791de3`。

这是历史材料记录，不能视为本次已取得该文件。请保留材料的组员确认持有人、来源与组内共享方式。收到文件后将核对 SHA、结构、条数和任务/指令重复；若 SHA 不同，应先解释差异再统一版本。

## 原作者下载来源及阻塞

来源已与 [数据指南](../docs/DATA.md) 和 [上游 README](../upstream/README.md) 核对：

| 材料 | 原作者链接 | 当前状态 |
|---|---|---|
| 专家轨迹归档 | [Google Drive](https://drive.google.com/file/d/1_tBMDixZcIjKuv-LExNllha-YIRxhKIq/view) | Drive 主域名及官方 usercontent 下载域名均连接超时，未下载 |
| 商品数据 data.zip | [Google Drive](https://drive.google.com/file/d/1G_0ccLWn5kZE5rpeyAdh_YuoNzvBUjT9/view) | Drive 页面连接超时，未下载 |
| 搜索索引 indexes.zip | [Google Drive](https://drive.google.com/file/d/11zOUDkJSgGhYin9NxQtG8PVpDsika86y/view) | 来源已核对，未下载；未单独验证该文件可用性 |

连接在取得 HTTP 下载响应前超时，因此不能据此认定链接失效、文件不存在或权限不足。

同时检查了原作者 GitHub 默认分支的完整目录树（提交 `b1ccd776d990278fee5f42b2050bb35638f5ac1b`，未截断），没有发现 `webshop_sft.json` 或对应 ZIP 的直接入口。

注意：专家轨迹与环境资产可能都叫 `data.zip`，应分目录保存。专家轨迹归档的内部文件和本组 SFT 是否一致，仍需取得后核验。

## 请全组讨论与评审

- 是否已有组员持有 `webshop_sft.json` 或原始 SPA 材料？请确认文件来源、SHA 和组内共享位置。
- 是否有组员可以访问原作者 Drive，协助下载专家轨迹、商品和索引？请在 PR 中明确负责人。
- 请审阅审计结论、材料来源和后续划分流程，统一数据版本及来源/许可记录。

不要将原始数据或模型权重提交到 Git；请通过组内共享存储交接材料。

## 数据到位后的操作

仅取得 SFT 时，先放到 `data/raw/webshop_sft.json`，即可做离线审计与初步划分：

```bash
webshop-lab audit --output artifacts/data-audit.json
webshop-lab split
```

如工具导入受执行环境影响，可按上述复验方式设置 `PYTHONPATH=src` 并使用 `python -m webshop_lab.cli` 调用。

三份材料齐全后，按数据指南使用 `webshop-lab import-assets --source <本地数据目录>` 导入，保存 SHA 与 manifest。完整环境就绪后，导出 goal map、重新划分并核对任务指令；通过环境对齐后再冻结正式训练/评估划分。

目前未生成真实 train/val 划分，未进行模型训练或 PPO，也没有可宣称的性能提升。
