# 11/23 评估精度更正与批次证据补交

关联 [#37 独立统计复核](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/37)、[#4 执行交接](https://github.com/Zqqq-qq/webshop-stepwise-rl/issues/4)。

## 核验范围与更正

六组存档开发轨迹均覆盖相同的 159/160 个任务，唯一缺失为 6700。28 个实际批次的 meta 已从设备找到并补交。复算脚本检查 run ID、配置/任务文件哈希、已完成任务前缀、环境指纹、无重复覆盖和配对统计。原整批失败运行的顶层 meta 不能描述后来拼接的批次。

本轮核对发现比打包缺项更重要的精度偏差。原全量运行脚本直接执行 CLI rollout；当时 `rollout.run()` 调用 `load(cfg)`，未传 `dtype`。`model.load()` 对未指定 dtype 的非量化模型默认选择支持时 BF16、否则 FP16。配置 `load_in_4bit=false` 仅关闭量化。历史 batch meta 未记录实际参数精度，因此原来“已完成 FP32 同设置主验证”的标签撤回，保留为**非量化默认精度开发验证，历史实际 dtype 未记录**。

该问题不改变 PPO 训练显式 `dtype=float32` 的调用记录，也不改变早先 8-task 检查的 FP32 wrapper。当前设备支持 BF16，但当前检测不能补作历史在内存中的精度证明。

## 可复算结论

- 成功数仍为 B0 17、B1 52、TG42/43 各 52、OP42/43 各 50，分母均 159。
- B1−B0 成功率差 +22.01pp，按明确统一的重采样实现区间 +15.09～+29.56pp。
- TG 与 B1 每个种子有 10 个成功变失败、另 10 个失败变成功。总数相同不能写“逐项相同”或“没有策略变化”。
- OP−TG 的成功率差区间跨 0，仅支持本预算与已记录评估路径下未观察到成功率改善。
- 终局任务奖励以未购买归零，不能直接拿页面最后匹配分求均值差。seed42/43 的 OP−TG 均值差为 −0.00933 / −0.01426。
- 不再用历史 4-bit 与本批默认精度运行的差异，断言单独的精度因果效应。

## 私有原件与复算

[补充证据包](https://github.com/Zqqq-qq/webshop-stepwise-rl/releases/download/data-v1/webshop-nov23-eval-metadata-precision-correction.zip)

SHA-256：`fa0d9eb53f1b0215e3c3e0a7b8bce615d91c63747ffbc1ef06aa8191283d7068`。111 个载荷和 SHA256SUMS，不含权重或专家 SFT 对话。包含原始批次 JSONL/IDs/meta、执行脚本/日志、执行仓库 aa77f44 的当前干净源码归档、当前模型文件哈希及修复后的加载检查。脚本与源码用于审计，历史 shell 脚本不是建议直接执行的启动器。

```bash
unzip webshop-nov23-eval-metadata-precision-correction.zip -d evidence
cd evidence
sha256sum -c SHA256SUMS
python3 audit_nov22_evaluations.py . --output audit-recomputed.json
```

Git 中的 [audit.json](audit.json) 不含页面/回复原文。重采样为 Python `random.Random(42).choices`，10,000 次，按冻结 val ID 顺序对齐，线性分位数。

## 加载器修复与下一次评估

加载器现在支持 `policy_dtype=float32`，拒绝与量化或显式 dtype 冲突。未指定时保持旧 auto 行为，避免改变历史 SFT 默认设置。每个新 rollout 的 meta 记录选定加载精度、实际参数/可训练参数的 dtype 分布，以及 rollout/model/common 源码 SHA。

六份 `configs/eval-qwen25-*-fp32.json` 提供显式精度、固定任务设置的重评输入。B1 的实际加载检查已通过：`torch.float32` 共 1,544,803,840 个参数，未执行购物回合或优化步骤。该检查证明修复后的入口，不证明任何旧结果的精度。

在运行仓库的同版本隔离检出中，准备同一个模型快照/B1 与四个第四轮 adapter，并先核对文件身份后，可用以下命令收集单个固定批次：

```bash
python -m webshop_lab.cli rollout --config configs/eval-qwen25-b1-fp32.json --ids /path/to/frozen-batch-ids.json --output /path/to/new-run/rollout.jsonl
```

每次使用新输出路径，保留实际批次 meta，不覆盖历史记录。不用旧 source 的 CLI 搭配新配置，旧加载器会忽略 `policy_dtype`。重评必须使用含本次修复的源码，并检查新 meta 的 precision/source 字段。

## 尚未关闭的门槛

- 任务 6700 的技术超时、分批偏差及是否接受 159 分母仍由 Zqqq/Qiu 决定。当前不能登记为完整 160-task 主验证。
- 原历史 meta 缺少加载时参数精度与权重哈希。本次当前文件检查明确标记采集时间，不能补造当时的载入证明。
- 正式 FP32 重评尚未执行。这里只修复入口和证据/表述，不新增训练，也不增加研究预算。
- `test_authorization=false`。正式登记与 test 放行仍待负责人验收。
