# PRM 表示诊断：评估侧复算契约

对应 [Issue #48 最新分工](https://github.com/Zqqq-qq/webshop-stepwise-rl/issues/48#issuecomment-7127949072)。负责人 QiuQiu-xuan。本交付是只读 CPU 复算入口、验收层级与原件清单，不执行训练、不批准新的预算或 test。

## 输入与命令

分析环境需要 NumPy；真实模板复核可选 Transformers 4.56.2。目标训练环境不增加依赖。可在独立分析环境安装 NumPy 1.26.4 后执行数值测试，不把普通无依赖 CI 的 skip 当作通过。本次GitHub连接不具备修改workflow权限，因此没有新增CI工作流；本地16项数值测试均实际执行。

```bash
python scripts/audit_prm_representation.py \
  --features /evidence/features.npz \
  --samples /evidence/samples.jsonl \
  --positions /evidence/positions.json \
  --manifest /evidence/export-manifest.json \
  --tokenizer /local/pinned-qwen-tokenizer \
  --rcond 1e-10 --expected-samples 8 \
  --output /new/audit.json
```

所有输入须位于 manifest 目录内，文件原样保留；输出须是新路径。只使用本地 tokenizer，禁止为审计隐式下载或加载模型。How 如已有其他原件格式，保留原件并增加一个注明来源/转换脚本 SHA 的审计副本，不要求改写历史文件。

`features.npz` 使用无 pickle 的数字数组：

| 数组 | 形状 | 定义 |
|---|---|---|
| `F` | `[8, V+1]` | 每行 `[sum_t z_it, N_i]`，不再加轨迹截距；float32/float64，记录实际求和精度 |
| `y` | `[8]` | 同顺序终局任务奖励，与 samples 核对；数值有限且在 `[0,1]` |
| `step_logits` | `[sum_i N_i, V]` | 可选但完整复算需要：实际 FP32 raw logits，按样本/回合顺序连接，不做 softmax |
| `offsets` | `[9]` | 可选但须与上项同时给：从 0 开始的累计回合数 |

`samples.jsonl` 是原8条训练输入，每行至少有 `task_id`、可选 `repeat=0`、`messages`、`terminal_reward`。不得只交挑选后的成功子集。`positions.json` 是同顺序的8个对象：`task_id`、`repeat`、完整 `input_ids`、`turn_end_indices`。真实 tokenizer 从原 messages 独立重编码，核完整输入与各 assistant 前缀末位，不只检查该 token 恰好是 EOS。

manifest 契约样式如下；SHA 占位符必须替换为真实值，不能直接执行：

```json
{
  "schema_version": "prm-representation-export-v1",
  "vocab_size": 151936,
  "im_end_token_id": 151645,
  "max_length": 2048,
  "feature_definition": "concat(sum_step_raw_logits,turn_count)",
  "score_position": "assistant_im_end",
  "eval_mode": true,
  "dropout_enabled": false,
  "labels_in_forward": false,
  "parameter_dtype": "float32",
  "aggregation_dtype": "float32",
  "model_identity": {
    "base_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "b1_adapter_sha256": "c39e2a78e7c5973328e9d2c23fb6320e566352e63cc089562488f3dd46ae5f9c",
    "base_files_sha256": {},
    "tokenizer_files_sha256": {},
    "template_sha256": "REPLACE_WITH_ACTUAL_SHA256",
    "controller_sha256": "REPLACE_WITH_ACTUAL_SHA256",
    "core_source_sha256": "REPLACE_WITH_ACTUAL_SHA256",
    "extractor_sha256": "REPLACE_WITH_ACTUAL_SHA256"
  },
  "files_sha256": {
    "features.npz": "REPLACE_WITH_ACTUAL_SHA256",
    "samples.jsonl": "REPLACE_WITH_ACTUAL_SHA256",
    "positions.json": "REPLACE_WITH_ACTUAL_SHA256"
  }
}
```

tokenizer 文件与模板 SHA 在 CLI 加载前核对。模型身份、eval/no-dropout 与 R 未进入 forward 的声明，还须人工核提取脚本/命令/实际运行记录；文件上的声明不是独立重跑证明。

## 输出能证明什么

输出记录任务/标签顺序与文件 SHA、`rcond`/截断阈值/奇异值/数值秩、保留非零谱条件数、最小 L2 解范数/最大元素、残差、CPU NumPy FP32 重算误差。若有逐步特征，另核 `F` 重建及“先逐步评分再求和”的误差。

- 退出 0：原始逐步特征与真实 tokenizer 边界检查均执行，状态 `representation_inputs_checked`。不意味着实际模型前向已独立重跑或精度/源码声明已全部验收。
- 退出 2：已完成可做的数值复算，但缺逐步特征或真实 tokenizer，状态明确为 partial；不能声称已核每回合特征或真实边界。
- 退出 1：拒绝缺失/变造 SHA、NaN/Inf、标签错配、任务错序/重复、错误偏置列、额外截距、评分位置错误等输入。已有输出不覆盖。

满行秩8不等于 LM 输出矩阵满列秩。条件数仅描述给定特征和缩放下的保留谱，不能据单个数断言所有优化都容易。最小 L2 解不是最小 L∞ 解；其最大权重不是所有拟合解都必须达到的下界。`steps × lr` 也不是 AdamW 的通用参数位移上限。因此这里不会生成“至少860步”或“1500步足以排除预算原因”的结论。

## 分层签收，禁止跨级

| 层级 | 必要证据 | 当前边界 |
|---|---|---|
| 数值风险 | 参数/梯度/优化器状态与写回 dtype；真实更新幅度 | BF16 构造风险与已归档 FP32 实验分开登记 |
| 8条记忆训练 | 原样本 SHA、每64步预测、真实更新数、checkpoint、独立逐步重载 | 沿用 MSE<0.001 且每条绝对误差<0.1；线性求解成功不能代替梯度训练通过 |
| 固定表示拟合 | 原位特征、提取身份、SVD、残差、FP32复算 | 只支持该8条是否可拟合，不证明泛化或根因唯一 |
| 总分泛化 | 独立任务/指令组确认集、fit拟合的常数/长度基线、配对误差与区间 | 已反复查看的dev继续作为开发集；不重新包装为盲验 |
| 逐步归因 | 同状态动作分叉、规格、循环/无效动作、恢复轨迹、前缀因果检查 | 总分低MSE不能代替过程验收；可执行性与增益分别定义 |
| 策略效果 | 同B1/交互/更新预算与种子，TG vs learned-progress PPO | 前置层级通过、执行单冻结后才开展；test另行批准 |

当前具体设备预算、优化器和停止标准由 Zqqq 整合 Jun/How 的实际原件后冻结。此文档不自动批准1200–1500步，也不修改已有模型协议。

## How 待交原件

1. 原8条 inputs/labels/评分位和原512更新控制器、核心源码、配置/命令、日志、每64步预测、最终checkpoint/reference、新进程输出、optimizer/RNG。
2. 表示诊断的原提取脚本、实际模型/tokenizer身份、原始逐步特征或F、SVD求解命令与完整数值结果。
3. Release明确链接与每文件 SHA；缺项如实列出。现有三臂384更新的归档不能替代这两次新实验。

测试夹具均是显式合成，不包含用户对话、真实模型特征或实验结果。实际数据保留集复核和旧证据复算另列在 `results/prm-evaluation-20261124/`，不能混写为新的8样本诊断已经验收。
