# PPO 主对照的运行输入（2026-11-21 冻结）

对应 `docs/RL_EXECUTION_PROTOCOL_20261121.md`：两臂 `terminal` / `option_potential`，种子 42、43，
每臂每种子 4 次更新 × 每更新 4 个固定 train 任务 = 16 episode，上限由
`ppo_comparison.common_budget()` 从 `max_steps`、`max_new_tokens` 与批次数推出（本配置为
128 策略轮 / 128 次实际 step / 16384 生成 token，每臂每种子）。

## 文件

| 文件 | 用途 |
|---|---|
| `configs/ppo-qwen15b-b1-op-seed42.json` | 种子 42 的基础配置（`ppo-compare` 只用它构造两臂） |
| `configs/ppo-qwen15b-b1-op-seed43.json` | 种子 43，仅 `seed` 不同 |
| `configs/ppo-rounds/round-{1..4}.json` | 四个 4 任务批次，顺序即执行顺序 |

## 设备上的两条硬要求

1. **模型路径必须是快照目录本身**。`ppo.py` 校验 `Path(cfg["model"]).name == cfg["revision"]`，
   所以不能写模型名，也不能写 `snapshots/` 的父目录。约定：

   ```bash
   mkdir -p artifacts/models
   ln -sfn "$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306" \
           artifacts/models/989aa7980e4cf806f80c7fef2b1adb7bc71aa306
   ```

2. **adapter 目录与哈希**。`artifacts/b1-pr18/adapter` 指向原 PR #18 的一 epoch LoRA：

   ```bash
   mkdir -p artifacts/b1-pr18
   ln -sfn "<实际 adapter 目录>" artifacts/b1-pr18/adapter
   sha256sum artifacts/b1-pr18/adapter/adapter_model.safetensors
   # 必须等于 c39e2a78e7c5973328e9d2c23fb6320e566352e63cc089562488f3dd46ae5f9c
   ```

## 执行顺序

```bash
# 1) 环境服务必须带 option-state
webshop-lab env-server --port 8766 --option-state

# 2) CPU 预检（不训练），核对两臂身份、预算与来源
webshop-lab ppo-compare --config configs/ppo-qwen15b-b1-op-seed42.json \
  --split-dir data/processed --goal-map artifacts/goal-instructions.json \
  --round-ids configs/ppo-rounds/round-1.json --round-ids configs/ppo-rounds/round-2.json \
  --round-ids configs/ppo-rounds/round-3.json --round-ids configs/ppo-rounds/round-4.json \
  --candidate option_potential --output artifacts/ppo-compare-seed42

# 3) 预检通过后再加 --execute 执行；种子 43 换配置与输出目录
```

## 边界

- `test_authorization=false`：本配置不用于 test，测试集仍另行门禁放行。
- 两臂共享 episode 数与上限，**不强制实际消耗相同**；每 run 的决策数、实际 `/step`、生成 token 与墙钟时间都要单独记录。
- 第四轮 endpoint 固定评估，不按 val 增加更新、不挑 checkpoint。
