# Qiu：PRM 原件与保留集独立核验（2026-11-24）

对应 [Issue #48 分工](https://github.com/Zqqq-qq/webshop-stepwise-rl/issues/48#issuecomment-7127949072)。本次独立做 CPU 算术、源码/文件身份及真实 tokenizer 复编码，没有加载1.5B模型、重跑GPU训练或放行test。

## 新8样本原件：已核与未核

[How发布的新包](https://github.com/Zqqq-qq/webshop-stepwise-rl/releases/download/data-v1/prm-evidence-20261124.zip)，SHA `b711fc49d9934adc321236a8837932406b6d1f3c8b81f8952d1f2c4a382453c8`。94个ZIP文件，顶层清单列出的85个payload逐一核SHA通过；另外9个是顶层/各检查点的校验清单本身。

完整机器结果：[memory-review.json](memory-review.json)。

- 原8条与前一归档64-fit的选中记录逐字段一致，全部属于冻结train。样本SHA `9c88beff…`与计划一致；标签顺序一致。
- 固定真实Qwen tokenizer四文件SHA与checkpoint一致。独立重编码8/8成功，长度902–1052、回合数3或10；所有8个检查点的每个评分位均与重新编码的assistant im_end一致。
- 合并各summary后更新记录恰为1–512，无缺失/重复。每64步的8条预测、MSE、最大误差均独立复算一致。
- 512步MSE **0.1385312484**、最大绝对误差 **0.6487203240**，未过MSE<0.001且每条绝对误差<0.1。8个检查点全部未过。最低已保存MSE在256步，约0.13330；这条曲线不能解释为已证明只需更多预算。
- 原件中的reference与另存独立重载预测逐步完全相同，差0；这是核对既有保存记录，本机未独立运行模型重载。
- 8个checkpoint的LoRA safetensors头均为F32；head.pt安全检查pickle指令包含FloatStorage，没有执行pickle。运行中的参数/梯度/optimizer状态精度不能只据此完全签收；按Zqqq最新分工由Yuxiao承接，保留Jun已有方案来源。
- 核心5文件SHA与旧包中a67db6a源码一致。原8个checkpoint manifest均引用training-state.pt，但包里均缺失；文件缺失和校验不匹配分开记录，未将其当完整可恢复归档。

## 表示诊断：仍缺原F，不能签收独立SVD

已核报告的任务、标签、token/回合数与原8条一致；已复算报告内奇异值首尾比、逐样本拟合值所给残差的算术一致性。**没有F/逐步logits，尚未独立重算奇异值、秩、解范数、FP32误差。**

已读提取脚本：加载base合并B1、eval/inference_mode；选原回合末位raw logits，先在FP32求和再转float64，最后追加N，无额外轨迹截距。它没有挂新的PRM LoRA；若要代表“初始PRM”的特征，应把零初始化adapter的功能等价与身份另外写清。脚本读取的实际 `configs/prm/qwen25-b1-run-001.json` 不在包内，命令也未绑定提取当时实际core SHA；须补身份，而非仅凭当前同名文件重现。

报告中的 `max_abs_step_budget_512_lr1e7` 不是AdamW通用位移上限；`ratio_weight_norm_to_step_budget` 用的是L2范数/单坐标步长量，更不能直接作为必需更新数。最小L2解最大坐标也不是所有可拟合解的必要下界。因此不签收“至少860步”或“1500步失败即可排除预算”的推断。

已交 [CPU复算入口](../../scripts/audit_prm_representation.py) 和 [原件格式/判据](../../docs/PRM_EVALUATION_AUDIT.md)。How只需先补同次原F（及样本顺序/列定义/聚合精度/SHA），可先完成聚合表示复算；若有原逐步特征则可进一步核F构造。若原数组未保留，须明示，重新提取应另记身份与执行批准，不能冒充同次原件。

## #45 保留集：可验收的范围已完成

固定head `0e1a15e46d2419e4451e6863934a6eafdd1ad3f7`，31项原数据/CLI回归独立执行通过。另用真实tokenizer对原911条逐条重编码，复现旧manifest的501 fit/127 dev；超长分别228/55，没有静默截断。实际新保留集经retained_episodes/write_training_jsonl导出后，编码与训练文件键序、标签、步骤数、来源SHA与行为策略声明一致；完整CLI保留集控制流由上述合成回归覆盖。没有将真实重编码说成完整模型跨PR验收。

完整机器结果：[data-review.json](data-review.json)。

| 项 | fit | dev |
|---|---:|---:|
| 过滤前 | 729 | 182 |
| 保留 | 501 | 127 |
| 超长拒绝 | 228 | 55 |
| 满奖励/部分/零 | 154/292/55 | 36/77/14 |

任务/指令fit-dev交叉0，val/test指令交叉0；train/val/test文件与goal-map SHA均和原manifest一致。仅实际911/1451任务，整体保留628/911。行为策略SHA是轨迹来源声明与文件身份核对，不是从文本证明采样实际权重。新代码目标GPU与模型重载仍不在本次签收范围。

## 复核命令与环境

复核Python3.12.10、Transformers4.56.2、Tokenizers0.22.2、Jinja2 3.1.6；数值脚本NumPy2.5.4，未使用Torch。目标设备依赖保持原契约。

```bash
python scripts/audit_prm_memory_evidence.py \
  --bundle /evidence/prm-evidence-20261124.zip \
  --previous-bundle /evidence/prm-diagnostic-evidence-20261124.zip \
  --tokenizer /local/pinned-qwen-tokenizer \
  --train-ids results/env-2026-11-17/train_ids.json \
  --output /new/memory-review.json
python -m unittest discover -s tests -v
python -m compileall -q src scripts
python scripts/check_repository.py
webshop-lab audit --output artifacts/prm-evaluation-source-audit.json
```

本PR保留原始未完成状态：BF16数值风险、已归档FP32结果、8条记忆失败、尚缺F的固定表示诊断、泛化、归因分别登记；不能跨级签收。
