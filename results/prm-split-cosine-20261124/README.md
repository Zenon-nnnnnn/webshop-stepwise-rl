# PR #54 证据归档：分离评分头/LoRA 学习率与余弦退火三组对照（2026-11-24）

本目录归档 PR #54（代码提交 `07bcdcbd75fb289272c2054702afa32059e3c98e`）对应的全部非权重原始 JSON 证据、独立新进程重载报告、初始本地 5 步短测原始日志以及零依赖离线复算脚本 [recompute.py](recompute.py)。

> **范围与边界声明**：
> 1. `rtx3090-fp32-three-arms/` 是在独立 NVIDIA RTX 3090（24GB）上按 `webshop_lab.prm.train` 与 `webshop_lab.prm.infer` 完整执行的三组 960 次轨迹呈现对照，**与 PR #52 在目标 RTX 5060 Ti 上执行的单一恒定 `1e-5` 续训链完全独立、互不混写**；本记录不替代目标 RTX 5060 Ti 实机复核，也不代表独立 `dev` 泛化已通过，`test_authorization=false`。
> 2. `local-mac-5step-probe.log` 为首次提交 PR #54 草稿时用于验证首步无超调及第 3–5 步 LoRA 介入无二次发散的 Apple M2 Pro 本地 CPU 5 步探针原始输出（采用 `bfloat16` 常驻权重 + 逐层 `float32` 算子计算，非正式 CLI 产物）；在配齐独立 CUDA 环境后，8 条记忆门槛的完整验证以 `rtx3090-fp32-three-arms/` 三组标准 FP32 CLI 原件为准。

---

## 一、 一键离线校验与指标复算

在仓库根目录直接运行（仅使用 Python 标准库，无需 GPU 或模型权重）：

```bash
python3 results/prm-split-cosine-20261124/recompute.py
# 若本地持有原 8 条 samples.jsonl 原件（SHA-256: 9c88beff...），可同时交叉核对原始文件：
python3 results/prm-split-cosine-20261124/recompute.py --samples /path/to/samples.jsonl
```

`recompute.py` 自动执行以下五项核验：
1. 校验仓库 `src/webshop_lab/prm/` 核心文件 SHA-256 与三组 `prm-manifest.json` 记录完全一致（且 `train.py` SHA-256 为 `d71a30626cb4a747837a09a9e3814abc4bcccb2fbad2d2da6fe98febf4ab0ac8`，与提交 `07bcdcbd` 逐字节相等）；
2. 校验三组检查点目录内所有跟踪文件（`summary.json`、`reference-predictions.json`、`config.json`、`adapter/adapter_config.json`、`adapter/README.md`）及 `prm-manifest.json` 自身与 `SHA256SUMS` 哈希逐字节匹配；
3. 校验每组 `reference-predictions.json` 与独立新进程 `*-reload.json` 的 8 条样本顺序、`turn_end_indices`、逐回合 `step_scores`、跨回合求和 `total` 一致性（实测三组最大逐步与总分重载差均为 `0.0`）；
4. 校验每组 `summary.json` 中全部优化步（120 / 120 / 960 步）记录的 `learning_rate` 与 `head_learning_rate` 严格符合 `constant` 或 `cosine` 公式；
5. 按原 8 条轨迹标签独立复算 `MSE`、最大绝对误差、3 回合满分组均值（`0, 6225`）与 3 回合零分组均值（`95, 5031`）。

---

## 二、 执行身份、源码同步说明与实际命令

- **硬件与软件环境**：NVIDIA GeForce RTX 3090（24576 MiB，驱动 `580.105.08`），Python `3.12.3`、`torch==2.5.1+cu124`（CUDA `12.4`）、`transformers==4.56.2`、`peft==0.17.1`、`safetensors==0.6.2`、`numpy==1.26.4`。
- **后端设置**（见各组 `summary.json` 的 `backend` 字段）：`float32_matmul_precision="highest"`、`cuda_matmul_allow_tf32=false`、`cudnn_allow_tf32=false`、`autocast_enabled=false`，全模型参数 `torch.float32`。
- **源码身份说明**：执行机通过 `git archive 07bcdcbd75fb289272c2054702afa32059e3c98e` 离线同步并在机内执行 `git init && git commit` 以满足工作区干净检查，因此各组 `prm-manifest.json` 中记录的本地根提交号为 `d25e6164894c87fdea796bb73e3ee546807483d8`（`source_tracked_clean=true`）；该检出的 Git tree 对象（`a6da83c56a0ced0cbb6081c94f5eaf0f0bf305f6`）及全部源码文件 SHA-256 与提交 `07bcdcbd` 100% 相同。
- **锁定输入 SHA-256**：
  - `base_weights_sha256`: `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`（revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`）
  - `sft_adapter` (`adapter_model.safetensors`): `c39e2a78e7c5973328e9d2c23fb6320e566352e63cc089562488f3dd46ae5f9c`
  - `sft_adapter` (`adapter_config.json`): `18993c388788ec14de72fd4f6f9ff86ce761ffc3aad5d0923b33f4365be03d97`
  - `samples.jsonl` (8 条训练轨迹): `9c88beffaef277f519712ab4a07c45766212579770d6292572c7f189bca4d8fe`
  - `results/env-2026-11-17/train_ids.json`: `8c85e56b3ac2e1f0a09b54560ad0f9bf77eedddb241d73617c7ff32416522e20`
- **实际执行命令**：
  ```bash
  # Arm 1: hidden-lr-low-acc8
  python -m webshop_lab.prm.train \
    --config /root/autodl-tmp/runs/issue48-verify-v1/hidden-lr-low-acc8.json \
    --samples /root/autodl-tmp/inputs/samples.jsonl \
    --train-ids results/env-2026-11-17/train_ids.json \
    --device cuda --seed 42 --steps 120 \
    --learning-rate 1e-7 --lr-scheduler constant \
    --weight-decay 0.01 --gradient-accumulation-steps 8
  python -m webshop_lab.prm.infer \
    --config /root/autodl-tmp/runs/issue48-verify-v1/hidden-lr-low-acc8.json \
    --checkpoint /root/autodl-tmp/runs/issue48-verify-v1/hidden-lr-low-acc8 \
    --samples /root/autodl-tmp/inputs/samples.jsonl \
    --reference /root/autodl-tmp/runs/issue48-verify-v1/hidden-lr-low-acc8/reference-predictions.json \
    --output /root/autodl-tmp/runs/issue48-verify-v1/hidden-lr-low-acc8-reload.json

  # Arm 2: hidden-split-cos-acc8
  python -m webshop_lab.prm.train \
    --config /root/autodl-tmp/runs/issue48-verify-v1/hidden-split-cos-acc8.json \
    --samples /root/autodl-tmp/inputs/samples.jsonl \
    --train-ids results/env-2026-11-17/train_ids.json \
    --device cuda --seed 42 --steps 120 \
    --learning-rate 2e-4 --head-learning-rate 1e-5 --lr-scheduler cosine \
    --weight-decay 0.01 --gradient-accumulation-steps 8
  python -m webshop_lab.prm.infer \
    --config /root/autodl-tmp/runs/issue48-verify-v1/hidden-split-cos-acc8.json \
    --checkpoint /root/autodl-tmp/runs/issue48-verify-v1/hidden-split-cos-acc8 \
    --samples /root/autodl-tmp/inputs/samples.jsonl \
    --reference /root/autodl-tmp/runs/issue48-verify-v1/hidden-split-cos-acc8/reference-predictions.json \
    --output /root/autodl-tmp/runs/issue48-verify-v1/hidden-split-cos-acc8-reload.json

  # Arm 3: hidden-split-cos-acc1
  python -m webshop_lab.prm.train \
    --config /root/autodl-tmp/runs/issue48-verify-v1/hidden-split-cos-acc1.json \
    --samples /root/autodl-tmp/inputs/samples.jsonl \
    --train-ids results/env-2026-11-17/train_ids.json \
    --device cuda --seed 42 --steps 960 \
    --learning-rate 2e-4 --head-learning-rate 1e-5 --lr-scheduler cosine \
    --weight-decay 0.01 --gradient-accumulation-steps 1
  python -m webshop_lab.prm.infer \
    --config /root/autodl-tmp/runs/issue48-verify-v1/hidden-split-cos-acc1.json \
    --checkpoint /root/autodl-tmp/runs/issue48-verify-v1/hidden-split-cos-acc1 \
    --samples /root/autodl-tmp/inputs/samples.jsonl \
    --reference /root/autodl-tmp/runs/issue48-verify-v1/hidden-split-cos-acc1/reference-predictions.json \
    --output /root/autodl-tmp/runs/issue48-verify-v1/hidden-split-cos-acc1-reload.json
  ```

---

## 三、 与 PR #52 的实验关系及 `acc=8 / 120` vs `acc=1 / 960` 退火对齐说明

1. **与 PR #52 单一学习率实验的对照层级**：
   - 低学习率常数基线（本目录 `hidden-lr-low-acc8`，`lr=1e-7, constant, acc=8, 120 updates`）：`MSE = 0.303530243`，`max_abs_err = 0.873627`（欠拟合）；
   - 中间共同常数学习率（PR #52 在 5060 Ti 独立跑出的 `lr=1e-5, constant, acc=8, 120 updates`）：`MSE = 0.002860`，`max_abs_err = 0.094438`（最大误差过线，但受共同恒定步长限制，`MSE` 距 `< 0.001` 仍差 `2.86×`）；
   - 分离头/LoRA 学习率 + 余弦退火（本目录 `hidden-split-cos-acc8`，`lora_lr=2e-4, head_lr=1e-5, cosine, acc=8, 120 updates`）：`MSE = 0.000000715`，`max_abs_err = 0.001441`（两项均过线）。
2. **`acc=8 / 120 更新` 与 `acc=1 / 960 更新` 在相同轨迹呈现次数下的进度对齐**：
   - 两臂总轨迹呈现次数均为 `E = 960`（遍历同一份 8 条样本 `120` 轮）。
   - 在已消耗 `e = 8k` 条轨迹（`k = 0, 1, ..., 119`）的对齐边界上，`acc=8` 处于第 `k` 次更新（余弦相位 `k / 120`），`acc=1` 处于第 `m = 8k` 次更新（余弦相位 `8k / 960 = k / 120`），**两者在每一个整轮 8 样本边界上的余弦衰减乘子严格相等**，从而在相同起止学习率曲线与相同数据曝光量下纯粹比较“8 轨迹均值梯度一次裁剪更新（裁剪 `114/120`）”与“单轨迹逐步裁剪更新（裁剪 `818/960`）”。
3. **二进制权重载荷说明**：
   - 按仓库 `.gitignore` 与 `CONTRIBUTING.md` 规范，二进制 `.safetensors` / `.pt` 不纳入 Git 历史；各组 `adapter/adapter_model.safetensors`、`head.pt`、`training-state.pt` 的精确 SHA-256 已完整记录于对应 `prm-manifest.json` 与 `SHA256SUMS` 中（原件保留于执行机 `/root/autodl-tmp/runs/issue48-verify-v1/`，如需上传至主仓私有/公开 Release 可随时打包提供）。
