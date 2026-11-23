# 神经 PRM 设备侧验收记录（2026-11-24）

范围：本机（RTX 5060 Ti 16GB，WSL2）上对 PRM 适配链路的真实执行证据。**不声称泛化验收通过**；`test` 未放行。

## 1. 数据（B1 train 池探索轨迹）

- 采样配置：`configs/prm-data-b1-4bit-sample.json`（B1 adapter SHA `c39e2a78…`，4-bit，temperature 1.0、top_p 1.0、max_steps 10）
- 采样规模：200 个冻结 `train_ids` 任务各 1 条，输出 `b1-train-200-full.jsonl`
- 说明：先用 FP32 采过，约 70 秒/条、200 条约 4 小时，改为 4-bit 采样（约 15 秒/条）；**PRM 训练本身仍为 FP32**，这一点作为偏差披露
- 门禁：`train_ids` 1451、禁止任务 360、禁止指令 359、与 val/test 指令重叠 0；goal-map SHA `6b5d9dfceee9…`、val_ids SHA `45440d724b33…`
- 切分：fit 160 任务（可用 121 条）、dev 40 任务（可用 29 条），指令重叠 0
- 超长拒绝（`max_length=2048`，显式拒绝不截断）：fit 39 条、dev 11 条（合计 **25%**），最长轨迹 6,926 token
- 标签分布：fit full 37 / partial 69 / zero 15；dev full 7 / partial 18 / zero 4

## 2. GPU 冒烟（真实 Qwen 1.5B FP32）

| 项 | 结果 |
|---|---|
| 设备 | `cuda`，torch 2.8.0+cu128，CUDA 12.8 |
| 后端 | matmul `highest`、TF32 关、cudnn TF32 关、autocast 关 |
| 步数 | 5 |
| 参数变化 | head_changed=true、lora_changed=true |
| 峰值显存 | 7.29 GiB allocated / 8.48 GiB reserved（≤16GB） |
| 耗时 | 41.5 秒（5 步，含加载与预测） |
| 保存 | adapter + head.pt + `prm-manifest.json` + SHA256SUMS |
| 独立进程重载 | 身份校验通过，34 个样本总分与训练时 reference **最大差 0.0** |

## 3. 学习率发现（重要）

评分头是「151936 维词表 logits → 线性 → 逐回合求和」的聚合，totals 量纲随回合数放大，FP32 下优化很敏感：

| lr | totals 序列（5 步） | 结果 |
|---|---|---|
| 5e-6（上游默认） | 0 → 6.77 → 19.42 → 3.79 → 0.02 | 发散，loss 到 377 |
| 1e-6 | 0 → 1.36 → 3.89 → 0.76 → 0.004 | 仍不稳，loss 到 15 |
| **1e-7** | 0 → 0.135 → 0.815 → 0.31 → 0.376 | **稳定，loss 递减** |

原因：上游实现把头放在 **bfloat16**（`nn.Linear(vocab,1).to(torch.bfloat16)`），5e-6 的更新在 bf16 下几乎不动；本项目按设备契约用 FP32，因此需要把 lr 降到 1e-7 量级。这属于对齐设备精度，不是改评分定义。

## 4. 20 步训练与 dev 对比

- 配置 `configs/prm/qwen25-b1-run-001.json`，lr 1e-7、weight_decay 0.01、20 步
- 训练：loss 1.0 → 0.002–0.23（区间波动但无发散），121 条样本，120.8 秒，峰值 7.29 GiB
- **dev（29 条）**：PRM MSE **0.1744**；常数基线 0.1196；长度基线 0.1135 → **未通过泛化门槛**

判断：当前只证明链路可跑，训练量与数据量都远不足（20 步 / 121 条）。下一步扩到全量 1451 任务与数百步后再评。

## 5. 与 #42 的 GPU 交接

`#42` 已按 Zqqq 口径在当前批次存档后停止：实际 34/1452 更新（round-34 完整落盘），GPU 释放后用于本记录的全部运行。

## 6. 复现入口

```
# 采样（约 50 分钟）
python -m webshop_lab.cli rollout --config configs/prm-data-b1-4bit-sample.json \
  --ids artifacts/prm-data-20261123/b1-train-ids-200.json \
  --output artifacts/prm-data-20261123/b1-train-200-full.jsonl

# 切分
python -m webshop_lab.prm.data --rollout ... --config configs/prm/qwen25-b1-run-001.json \
  --split-dir data/processed --goal-map artifacts/goal-instructions.json \
  --output artifacts/prm-data-20261123/dataset-200 --fit-fraction 0.8 --seed 42 \
  --training-jsonl fit=.../fit200.training.jsonl --training-jsonl dev=.../dev200.training.jsonl

# 训练
python -m webshop_lab.prm.train --config configs/prm/qwen25-b1-run-001.json \
  --samples .../fit200.training.jsonl --train-ids data/processed/train_ids.json \
  --steps 20 --device cuda --learning-rate 1e-7 --weight-decay 0.01
```
