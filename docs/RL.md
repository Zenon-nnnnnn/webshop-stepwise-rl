# 进度估计、奖励和 PPO 的具体边界

## 论文思路

轨迹有 T 个步骤。估计每步贡献 `p_t`，使 `sum(p_t)` 接近最终奖励 `R`。策略训练使用 `alpha * p_t + beta * grounding_t`。仅有总分监督并不唯一确定每步贡献，因此低总分预测误差不等于因果正确归因。

原神经估计器在 `upstream/prm/`，PPO 在 `upstream/ppo/`。默认新工具提供一个无依赖的可加线性基线，帮助在有限算力下先检查数据、监督信号和奖励接口。

## 低成本线性基线：已实现

对当前观测与模型回复做哈希词频特征 `phi_t`，使用线性权重输出 `p_t = w · phi_t`。以轨迹特征之和预测 `R`，优化平方误差并以独立验证集选择 epoch。特征不包含最终奖励、目标商品答案或未来步骤。模型在观测上可能学到表面线索，不能据此宣称理解了真实进展。

```bash
webshop-lab progress-train --train artifacts/explore-train-small.jsonl --val artifacts/explore-val-small.jsonl --output artifacts/progress-linear.json
webshop-lab progress-predict --model artifacts/progress-linear.json --input artifacts/explore-train-small.jsonl --output artifacts/explore-train-annotated.jsonl
```

输出 JSON 模型：权重、维度、最佳 epoch、val MSE、输入 SHA、train/val ID、goal fingerprint。预测将 `progress` 加入每步。训练/验证任务或底层指令重叠时拒绝拟合；训练奖励完全没有变化也拒绝拟合。

注意：annotate 训练轨迹的结果是训练集预测，不应拿其误差当泛化表现。要报告预测误差，使用独立 val 轨迹；如需更严格的奖励标注，考虑按任务交叉拟合，不能把自己训练过的轨迹效果当成 out-of-sample 证据。

可用 `progress-train --guarded` 拟合第二种工程候选。无效动作、相同搜索、同一页面的相同动作和立即反向翻页预测为零。截距只在第一条可计分动作上使用，避免每多走一步就重复增加固定奖励。训练、离线预测与 PPO 使用同一特征规则。原线性权重保留其原有语义。此规则仍需在独立验证集上验证总分误差和动作分配。

## 奖励导出：已实现

```bash
webshop-lab reward-export --input artifacts/explore-train-annotated.jsonl --output artifacts/rewards-spa.json --mode spa
webshop-lab reward-export --input artifacts/explore-train-small.jsonl --output artifacts/rewards-terminal.json --mode terminal
webshop-lab reward-export --input artifacts/explore-train-small.jsonl --output artifacts/rewards-uniform.json --mode uniform
```

- `terminal`：前 T−1 步为 0，最后一步为实际 R。
- `uniform`：每步为 R/T，是简单奖励分配对照。
- `spa`：每步为 alpha × progress + beta × grounding，默认 alpha=1、beta=0.5；这里指接口公式，估计器类型必须另报。

SPA 模式缺少任意一步预测即报错；不允许把全 0 或最终奖励偷偷当作进度模型预测。grounding 是 0/1，而 beta 控制其权重。加入 grounding 后奖励和不一定等于最终 R，这是融合奖励的设计结果。

导出字段：`task_id`、`repeat`、`instruction`、`observations`、`responses`、`rewards`、`progress_sum`、`final_reward`、`goal_fingerprint`。

## 原神经进度模型：代码保留，目标机未验证

| 文件 | 作用 |
|---|---|
| `upstream/prm/data_org.py` | 将旧格式探索输出转换为进度模型输入 |
| `train_our_progress_model.py` | 全参数/原始版本参考 |
| `train_our_progress_model_lora.py` | LoRA 版本参考 |
| `train_our_progress_model_lora_fp16.py` | 带 fp16 调整的本地版本 |
| `inference_prm.py` | 预测各回合贡献 |
| `rl_data_org.py` | 融合奖励并构造旧 PPO 输入 |

这些脚本不是新 CLI 格式的直接消费者。原始探索 JSON 字段是 `agent_conversations` 等，新工具是逐步 JSONL。移植时必须显式转换，并测试 turn 数、assistant mask、终止奖励、ID 和奖励长度是否一致。

原脚本会使用相同/包含关系的 train/val 数据，不能照运行后直接报告验证结果。必须按本项目划分收集独立 train/val 轨迹，替换路径并检查 token 截断对每步分数的影响。

## 终局奖励 PPO 工程短跑

`ppo-smoke` 从已存在的 LoRA 初始化，采样 1–8 个冻结 train 任务，使用每条完整助手回复的联合概率做一次 PPO 更新。未购买的购物回合得 0。beta 和 KL 系数为 0，gamma 和 lambda 为 1。逐条回复计算梯度并在整批完成后更新一次。未记录的采集截断与正式实验不在该命令的范围内。指定 `--resume` 可从已成功重载的本地输出继续，恢复 adapter、critic、AdamW 和 PyTorch RNG，然后重新采样。续跑使用相同配置和数据划分，可传新的训练 ID 清单，并写入一个新目录。

```bash
webshop-lab ppo-smoke --config artifacts/ppo-smoke-config.json \
  --ids artifacts/ppo-smoke-train-ids.json --split-dir data/processed \
  --goal-map artifacts/goal-instructions.json --output artifacts/ppo-smoke-run
```

配置必填项包括本地固定快照目录 `model`、`revision`、`base_weights_sha256`、LoRA 路径 `adapter`、`adapter_sha256`、`environment_url`、`goal_fingerprint`、`context_length`、`max_steps` 和 `max_new_tokens`。使用已验证的 Qwen2.5 单文件权重快照。可指定 `load_in_4bit` 和 `seed`。采样固定为 temperature=1、top_p=1、top_k=0。建议用少量训练任务起步，输出目录必须为新目录。命令会先核对 live health、目标文本和三个数据划分。

成功时生成 `summary.json`，包括概率重放范围、梯度和参数哈希，以及本进程的 adapter/critic 保存重载结果。运行中断可能留下部分文件，以该摘要是否存在作为一次更新通过的判据。训练轨迹保留在本机。首次真实 GPU 验收另核对长回复概率重放和独立进程重载。

### 逐步奖励接入

工程对照可在同一初始 SFT adapter 上分别启动两臂。默认 `reward_mode=terminal` 在购买结束时发放终局奖励。逐步臂显式指定 `reward_mode=progress`、`progress_model` 和 `progress_model_sha256`，固定线性进度估计器并对照训练、验证划分。两臂 beta 为 0。逐步臂每轮只发放一次预测进度，购买终局分照常为评估保留。续跑仍要求与父轮配置逐字节相同。

另有 `reward_mode=option_potential` 工程候选，使用 `env-server --option-state`。它按当前已选规格与购物指令的词匹配定义势值，逐轮增加 `Phi(next)-Phi(current)`，购买或步数耗尽时下一状态势值为 0。训练数据记录原任务奖励、势值差与合成奖励。在 gamma=1 时，从零势值出发的完整轨迹总奖励仍等于购买终局奖励，回到同一状态的循环净塑形奖励为零。它是一个规则基线，独立于拟合的逐步进度模型。

### 双臂预检

`webshop-lab ppo-compare` 默认只做 CPU 预检与计划生成。提交一份共同的 FP32 PPO 配置（不含奖励模式），并通过 `--round-ids` 按顺序提供每次更新的训练 ID 文件。共同参数为 `--config`、`--split-dir`、`--goal-map`、`--output` 和 `--candidate`。学习型候选还需 `--progress-model` 与 `--progress-model-sha256`；规格势函数候选执行时需要支持 `--option-state` 的桥接。普通桥接上的 CPU 预检可生成计划，并报告接口尚未就绪。

计划保存各臂配置哈希、相同任务顺序、共同预算上限和逐轮命令。使用 GPU Python、冻结源码和独立新目录添加 `--execute` 才会启动两组。每组首轮从同一 SFT 初始化，随后只续跑本组 checkpoint。实际决策、环境调用和生成 token 可以因为提前结束而不同。训练摘要分别记录实际消耗，正式实验若要求严格匹配实际消耗，须另定停止协议。

## 原 PPO：可用参考源码，不是已完成的新主线

入口：`upstream/ppo/train_ppo.sh` → `step_ppo.py` → `step_ppotrainer.py`；配置 `upstream/config/StepTool_ppo.json`；合并 `upstream/ppo/merge.py`。

原包提供 `upstream/ppo/requirements.txt`，指定 `torch==2.2.0`、`transformers==4.43.1`、`trl==0.10.1`、`peft==0.5.0`；根 requirements 又指定 `trl==0.7.7`，两个环境不能混装。这些旧版 torch 配置不能直接用于 5060 Ti，PPO 接口也不能无条件替换成最新版 TRL。现代 GPU 环境与旧 PPO 需要单独适配，不能用原 requirements 覆盖已经验证的 SFT 环境。

原 PPO 从 JSON 加载固定 prompt/response/reward，再对这些响应更新；这不自动保证它们来自当前待更新策略。正式实验要检查采样策略、保存的行为概率、旧策略概率和 PPO 多轮更新之间的一致性。仅把离线数据送入名为 PPO 的类不足以证明是正确的 on-policy PPO。

同时检查：

1. tokenizer 与实际模型一致；原 `step_ppo.py` tokenizer 路径硬编码为 Llama。
2. 上下文截断后，fragment mask 与逐步奖励仍一一对应；原默认上下文 512 很容易截断长轨迹。
3. 相同初始化、采样预算与更新步数下比较 terminal / stepwise。
4. value head、reference policy、adapter、KL、优势估计与 EOS mask 均经过小样本检查。
5. batch=1、16 GB 显存能跑过一次完整 forward/backward/save/reload，记录真实峰值。
6. 训练后的真实 WebShop 成功率和行为变化在独立 val 上评估。

**当前仓库不提供“现代环境下一键完整 PPO 成功”的虚假承诺。** Zqqq-qq 需要完成上述集成验收后，才把 R0/R1 状态改成可运行；相关任务在团队计划中有明确截止和备选路线。

## 最新学习与验收安排

以 [逐日计划](DAILY_PLAN.md) 和 [学习指南](LEARNING_GUIDE.md) 为准：11/25 首次真实策略更新尝试、11/28 缓冲、12/01 多轮闭环硬验收。主对照使用相同 grounding 的终局臂与逐步臂；当前 terminal 导出不含 grounding，新增 TG 路径属于待实现任务。若 PPO 兼容性阻塞，明确采用 on-policy 策略梯度学习分支并同步修改实验描述，不能将其称为 PPO 或仅用 SFT 替代 RL。
