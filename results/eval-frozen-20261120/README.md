# 开发 val 冻结件（2026-11-20）

这是评估侧对**开发集 val 清单与评估配置**的冻结记录。它只回答"用哪些任务、用哪个配置"，不含任何模型结果。

## 冻结内容

| 项 | 值 |
|---|---|
| val 任务数 | 160（互不重复的整数任务 ID） |
| `val_ids.json` sha256 | `45440d724b33e69e13916d0adbe243cc63ce3bd67677b8220feff9f30b866c36` |
| 环境 `goal_fingerprint` | `e69c9ae2a0aad88977c44a7baf49f1f5ef7918ba8317923c4ae24dcc1f5638d8` |
| 划分 `goal_map_sha256` | `6b5d9dfceee913828871b8dff62fd59720a40306738a92a496d1b0fb6838fa88`（12,087 任务） |
| 模型 revision | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| `configs/eval-b0-frozen.json` sha256 | `91d3f8ad5150a724dfd76eaf478a7d1aa68a8ba8f03235740b62436abddb07ac` |
| `configs/eval-b1-frozen.json` sha256 | `03cce78c6f2a208b59e14e923927968fdfcc43d8f9dda3b0c4d70a652702990f` |

两臂配置只有 `adapter` 不同；`check-parity --require-frozen` 必须为 true。完整字段见 `manifest.json`。

## 来源与交叉核对

- ID 列表来自 PR #10 的 `results/env-2026-11-17/val_ids.json`；PR #15 的 `results/mapped-split-20261119-zqq/split-manifest.json` 与环境侧 `split-manifest.json` **全字段一致**（seed、`source_sha256`、`test_ids_sha256`、`goal_map_sha256`、1451/160/200、排除 13、train/val 指令哈希）。
- `goal-instructions.json` 从 Release `data-v1` 下载复算：1,969,461 字节、SHA `6b5d9dfc…`、12,087 任务，与上面的 `goal_map_sha256` 一致。

## 怎么用

```bash
# 跑一臂（示例：B0）
webshop-lab rollout --config configs/eval-b0-frozen.json \
  --ids results/eval-frozen-20261120/val_ids.json --output artifacts/b0-val.jsonl

# 出评估与登记行
webshop-lab evaluate --method B0 --meta artifacts/b0-val.meta.json --jsonl artifacts/b0-val.jsonl \
  --config configs/eval-b0-frozen.json \
  --split-ids results/eval-frozen-20261120/val_ids.json \
  --output artifacts/b0-val-evaluation.json --register-row artifacts/b0-val-row.csv
```

## 复核命令

```bash
sha256sum results/eval-frozen-20261120/val_ids.json          # 45440d72…
sha256sum configs/eval-b0-frozen.json configs/eval-b1-frozen.json
webshop-lab check-parity --a-config configs/eval-b0-frozen.json \
  --b-config configs/eval-b1-frozen.json --require-frozen     # parity=true
webshop-lab preflight --a-config configs/eval-b0-frozen.json \
  --b-config configs/eval-b1-frozen.json \
  --ids results/eval-frozen-20261120/val_ids.json --require-frozen
```

## 边界

- **不是 test**：200 个保留测试任务没有放进这个目录，也不应在这里出现；test 只在 12/07 冻结配置后使用一次。
- **不含结果**：这里没有任何成功率、奖励或失败案例；B0/B1 的数字必须来自真实 rollout 并登记到 `results/experiment-register.csv`。
- **改动规则**：一旦正式 run 开始，这份列表和两份配置的哈希都不能再改；如确需更换划分或 adapter 路径，必须重新冻结并作废旧哈希，不能混用。
