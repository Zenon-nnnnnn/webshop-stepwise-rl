# 11/21 有限预算 RL 主对照执行协议

关联 [Issue #4](https://github.com/Zqqq-qq/webshop-stepwise-rl/issues/4)、[PR #30](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/30)、[机器执行单](../results/execution-handoff-20261121/proposed-execution.json)。用户明确授权 Tutor 审核并落实路线/预算；本文件由 Tutor 代为记录，不冒充学生独立完成方法验收。

## 决定与研究边界

采用 **TG：终局任务奖励 vs OP：规则型规格势函数塑形**，进行有限预算、两种子的真实 on-policy PPO 对照。学习型进度估计器未通过基线/循环检查，保留为失败诊断；本轮不采用其权重作为主奖励，也不新增神经估计器训练。

这改变了原“学习型进度奖励”的主问题。proposal、图注和 report 统一写规则势函数对照，不能称神经 SPA 精确复现。训练预算有限，效果是否改善是待检验问题，负结果也交付。

以下是执行授权范围，**满足预检后 How 可按执行单启动，不必再等一次笼统确认**；并非宣称实验已完成，也不代替 Jun/Qiu 的模型与统计验收。不合并任何工程 PR，不启动 GPU 作业于本次文档更新。

## 起点、任务与算法

- 共同起点为 #18 原 SFT B1：Qwen2.5-1.5B-Instruct revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`；adapter 文件 SHA `c39e2a78e7c5973328e9d2c23fb6320e566352e63cc089562488f3dd46ae5f9c`，adapter_config SHA `18993c388788ec14de72fd4f6f9ff86ce761ffc3aad5d0923b33f4365be03d97`。完整身份见 [#14 锁定记录](https://github.com/Zqqq-qq/webshop-stepwise-rl/blob/ebb8e2846344f261fbfa53a8b769d937311130e3/configs/identities/b1-pr18.json)。设备读取核验后使用，不以两步 smoke 或第三轮 PPO 模型替代。
- 每种子每臂从同一 B1 独立初始化；后续只恢复本臂自己的 checkpoint。种子 **42、43 都预先安排**，不能看 seed42 成绩好坏再挑是否做 seed43。
- 每臂每种子 **4 更新，每更新 4 个 train 任务，共 16 episode**；四批具体 ID 已在 JSON 固定，并独立核对均属于 train、无 val/test 交叉。同一任务批次不等于要求两臂产生相同轨迹。
- 训练 FP32、非 4-bit，context=2048、max_new_tokens=128、max_steps=8；采样 temperature=1/top_p=1/top_k=0，禁用 dropout。每次完整 assistant 回复是一个动作，逐 token 保存概率并求和形成联合动作 ratio。
- gamma=lambda=1、clip=0.2、AdamW lr=1e-5/weight_decay=0、value coefficient=0.5、梯度裁剪 1.0；不标准化优势。每批一次 optimizer step，批内回复累积梯度，old logp/advantage/return target 固定。本预算不是多 epoch clipping 效果研究。
- 两臂 **beta=0、KL coefficient=0**；不加每合法动作固定奖励，reference-KL 不作为 loss。记录它作为范围限制，不声称检验了共同非零 grounding 的作用。
- 使用已审阅 #28@b3024cf 的代码路线或明确记录、经检查的集成后继。实际 commit/tree、环境 wrapper 和补丁必须保存；PR approved 不等于可以不记录实际运行源码。

## 奖励与势函数

购买完成取环境评分 R，否则有限任务结束取 R=0。页面匹配分只作诊断。

TG：`r_t = 1[t是最后一步] * R`。

OP：`r_t = 1[t是最后一步] * R + Phi(s_(t+1)) - Phi(s_t)`。

Phi 使用 #27 的固定规则：选中规格值分词后，与去除指定停用词的指令词集合匹配；值为 `0.25 * 匹配词数 / 规格词数`，无规格词为 0。不使用隐藏目标商品答案或环境购买评分构造 Phi。记录 option_state，但不把它额外加入策略文本输入。环境需开启 option-state，回报按完整 episode 计算；购买或 max_steps 结束时下一 Phi 强制为 0。

在 gamma=1、初始 Phi=0、终止 Phi=0 时，差分首尾抵消，总训练奖励等于 R。循环回到同势值的净塑形奖励为 0。**这不保证每步都分配正确，也不保证有限更新一定提升。** action-level GAE 的后缀回报是 `R-Phi(s_t)`，critic 必须学习对应奖励下的价值。若 critic 恰好补偿势值变化，优势也可能接近；这不构成算法错误。

## 预算和模型选择

| 口径 | 每臂每种子 | 两臂两种子合计 |
|---|---:|---:|
| optimizer 更新 | 4 | 16 |
| 训练 episode | 16 | 64 |
| 策略决策上限 | 128 | 512 |
| 实际 /step 上限 | 128 | 512 |
| 生成 token 上限 | 16,384 | 65,536 |

预算定义为相同 episode/更新数和共同上限，不强迫实际步骤/token 相同。实际消耗逐 run 报告，reset/health 不算已应用 /step。不得把此设计写成严格相同实际交互量实验。1–4 小时/种子对只是含评估的资源预留；How 记录实际训练、加载、服务、评估时间，不能用文件时间差代替精确 GPU 计时。

跑完四批即停，固定选择第四次更新的 endpoint；不用 val 选最优中间 checkpoint。先 seed42 双臂，再 seed43 双臂，单 GPU/环境串行。若资源或故障使第二种子未完成，公开部分结果与原因，不改变预定种子并不声称两种子结论。

## 执行前门槛与异常

1. Zqqq 记录采用真实映射划分版本：1451 train/160 val 对话、200 reserved test 任务 ID、排除13；goal-map SHA `6b5d9dfceee913828871b8dff62fd59720a40306738a92a496d1b0fb6838fa88`，fingerprint `e69c9ae2a0aad88977c44a7baf49f1f5ef7918ba8317923c4ae24dcc1f5638d8`；每次启动仍检查 live health。采用记录不自动登记旧实验或放行 test。
2. Jun 核实际 B1 文件/配置/底模/tokenizer 与恢复记录；Qiu 核任务隔离、指标、配对和实际预算。现有六轮原件的一致性复核见 [Tutor 摘要](../results/execution-handoff-20261121/tutor-verification-20261121.json)，GPU 数值重放、真实权重载荷未在审阅机执行，不能扩大核验声明。
3. How 为两个种子生成具体两臂计划和配置/hash、训练任务文件；确认 option-state 服务、可用内存/显存、依赖/源码和私有输出入口。先完成门禁，不要求把 draft 配置直接改名运行。
4. 新 B1 同批 8-task 基线的 config/meta/IDs/check/逐任务记录目前仅本机路径；须补私有入口及原件 SHA，Qiu 复算后才能称为独立核验。无需因此重训。该基线证据交接与执行预检可以并行。

概率重放超限、NaN/Inf、身份错配、环境故障立即停止受影响 run，保留日志和 checkpoint。修复/重跑须记原因及范围；不能丢弃困难任务或失败种子。若改奖励/算法/批次，建立新协议版本，不能沿用本执行单声称预注册。

## 统一验证与 test 边界

开发 val：固定同一 160 个任务（清单 SHA `45440d724b33e69e13916d0adbe243cc63ce3bd67677b8220feff9f30b866c36`）、每任务一次、FP32、贪心、context2048/max_new128/max_steps10。评估原 B1 和每个种子两臂第四轮 endpoint。所有配置除 adapter/方法身份外保持相同；使用固定 FP32 runner 字节 SHA `ff91196a99813656469c769c2bb2562026e16b717101094b9f5ffb0aa136a89e`，其导入的 rollout/model 源码也要记录核对。

旧 4-bit 的 50/160 不替代新 FP32 同设置基线；若展示 B0/B1 同表，B0 也需同设置重评，否则单列历史表。小样本 5/8 不能与全量31.25%直接比较。

主指标是全部有效完成评估 episode 上的满分购买成功率；同时报告购买数、平均终局分、非法动作/总决策、步数与资源。正常策略失败保留；技术中断单列，按事先规则补跑后补足固定任务。每种子分别给配对差、任务 bootstrap（预定10,000次/seed42/线性分位数），单臂 Wilson；不把两个种子的320条轨迹假装成320个独立任务，也不由两个种子声称普遍稳定性。

本次 **test_authorization=false**。11/23 计划中的 test 前检查不自动授权：实际模型哈希、协议/选择规则、最终评估配置、任务覆盖和故障规则由 Zqqq/Qiu 核对后另行放行。看 test 后不调参、不追加训练、不挑 checkpoint；不足以完成就如实报告未完成。

## 交接与日期

How：满足门槛即按固定预算执行，交每轮原件/模型身份、最终轨迹、实际计数/耗时和失败记录。Jun：独立核模型/恢复。Qiu：独立统计、登记判断和 test 门禁。Zqqq：协议/数据/结论统筹。Yuxiao：按新研究问题做方法图和约5分钟展示，不把负结果隐藏。

目标11/24夜间完成必要实跑与核对，11/25验收及PPT冻结，11/26 pre，12/15 report。日期不覆盖技术验收；不为赶工伪造完成或用SFT替代RL。
