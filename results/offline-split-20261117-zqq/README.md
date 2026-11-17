# 2026-11-17 离线审计与初步划分（Zqqq-qq / Tutor 示范）

状态：**provisional_offline_only**。由 Codex 在用户明确要求下运行，用户旁观；不是学生独立运行或理解程度的评估。没有 GPU 训练、环境映射或 agent 性能结论。

## 结果

- 原始 SFT：1,624 条 / 1,624 个不同任务 ID，底层归一化购物指令 1,599 种。
- 默认训练/测试 ID 交叉：12 个。
- 初步划分：train 1,451 条、val 160 条、reserved test 200 个环境任务 ID。
- 排除 13 条：12 条是测试 ID；额外排除任务 1746，因为它与测试任务 5413 是同一购物要求的预算变体（380 vs 370 美元）。
- 原字段 reward 与 true_reward 有 8 条不一致，已记录，未擅自改标签。969 条 reward=1，其余为部分奖励，不能把全部材料称为满分专家演示。
- 附带探索样例 78 条 / 26 个任务 / 1 条满奖励；进度训练/验证内容相同；旧 PPO 示例 41 条。

## 本目录证据

- `audit.json`：完整审计计数与原 SFT SHA。
- `validation.json`：11 项检查、各输出文件 SHA、输入/源码版本、未验证边界。
- `train_ids.json` / `val_ids.json` / `test_ids.json`：仅 ID，不含完整对话。
- `split-manifest.json`：seed、源文件哈希、归一化指令指纹、候选划分说明；goal_map_sha256 为 null。

完整 train.json/val.json 仅保存在本地 `data/processed/offline-20261117-zqq/`，重复运行产物在 `artifacts/zqq-data-20261117/repeated-split/`，两者均被 Git 忽略。原始数据统一从私有 data-v1 Release 获取。

## 如何复现

先按 [数据共享指南](../../docs/SHARED_DATA.md) 获取同一 SFT。在仓库根目录，将下列路径替换为实际下载位置；选择尚不存在的输出目录，避免覆盖自己已有数据：

```bash
PYTHONPATH=src python -m webshop_lab.cli audit --sft /path/to/webshop_sft.json --output artifacts/zqq-data-20261117/audit.json
PYTHONPATH=src python -m webshop_lab.cli split --sft /path/to/webshop_sft.json --output data/processed/offline-20261117-zqq --seed 42
PYTHONPATH=src python -m webshop_lab.cli split --sft /path/to/webshop_sft.json --output artifacts/zqq-data-20261117/repeated-split --seed 42
```

同一 seed 两次输出的 train/val 对话、三份 ID 和 manifest 均逐字节一致。另使用独立指令提取/归一化计算验证已知指令交叉、ID 交叉、排除原因和全量记录核算；代码与结果逻辑见 validation.json 的检查项。原保留记录与源记录逐结构比较相等，未修改对话/奖励内容。

## 为什么仍不能冻结

200 个 test ID 不代表 SFT 内有 200 条测试演示。当前只能检查 SFT 已出现的测试请求；SFT 中未出现的实际测试指令、环境 goal 排序仍未知。须取得 HowseaonLink 的实际 `goal-instructions.json` 与环境 fingerprint，再用 `split --goal-map` 重划分、检查 health 中 alignment 和 split_status 后冻结。不要直接把本目录 IDs 复制到默认 processed/ 并绕过该验证。

不对数据任务 Issue #2 宣称全部完成，不把重复性检查或 CPU 测试当成 GPU 训练成功。
