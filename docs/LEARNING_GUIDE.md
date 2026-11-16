# 边做边学：Agentic RL 资源、练习与代码导读

[逐日安排](DAILY_PLAN.md) · [当前负责人](TEAM.md) · [实验命令](EXPERIMENTS.md) · [RL 实现边界](RL.md)

目标是把阅读与可检查的操作连起来。没有每日工时上限，也不要求先看完一整门课才动手。每天先能说明今天改的张量/代码/指标，再交一个真实产物。以下链接在 2026-11-16 查验；版本化文档用于理解相应实现，不代表应把旧版依赖直接装到新显卡上。

## 学习顺序与精读资源

| 编号 | 资源与精读位置 | 对应任务 / 要交的理解证据 | 优先成员 / 日期 |
|---|---|---|---|
| L1 | [ReAct 原论文](https://arxiv.org/abs/2210.03629)：方法图、WebShop 交互案例 | 画出 instruction→observation→action→next observation；说明 ReAct 提示不是 RL 更新 | 全员，11/16–17 |
| L2 | [WebShop 原论文](https://arxiv.org/abs/2207.01206)：环境、动作、奖励与评估 | 手动解释一条购买但未满分的轨迹，说明商品/规格/预算如何影响结果 | 全员，11/17–19 |
| L3 | [Transformers 4.56.2 chat templates](https://huggingface.co/docs/transformers/v4.56.2/chat_templating) 与 [PEFT 量化训练](https://huggingface.co/docs/peft/developer_guides/quantization)：apply_chat_template、4-bit + adapter | 展示一条 token/label mask；解释基础权重冻结、LoRA 更新与量化的关系 | Jun/HowseaonLink 精读，你/Qiu 复核；11/17–19 |
| L4 | [SPA-RL 原论文](https://arxiv.org/html/2505.20732v1)：§3 方法、§4 实验设置、附录实现 | 写出进度和拟合终局的目标、融合奖励、grounding；列出我们与原论文模型/估计器/规模的差异 | 你精读、全员方法图；11/18–24 |
| L5 | [Spinning Up：Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)：simplest policy gradient、reward-to-go、baseline | 完成 E1；解释为什么奖励影响动作概率，为什么不直接对环境 reward.backward() | 你/Jun 精读，其他人理解示例；11/17–20 |
| L6 | [Hugging Face：PPO clipping 教程](https://huggingface.co/learn/deep-rl-course/unit8/clipped-surrogate-objective) 与 [PPO 原论文](https://arxiv.org/abs/1707.06347)：ratio、clipped objective、训练循环 | 完成 E3，指出行为/旧策略、新策略、参考策略分别是什么 | 你/Jun 精读，Qiu 审查日志；11/19–25 |
| L7 | [GAE 原论文](https://arxiv.org/abs/1506.02438)：TD residual 与 advantage 的构造 | 完成 E2，检查 done、bootstrap、gamma、lambda 和 padding | 你/Jun，11/19–24 |
| L8 | [TRL 0.10.1 PPO Trainer 文档](https://huggingface.co/docs/trl/v0.10.1/ppo_trainer)：rollout→evaluation→optimization 和 trainer 接口 | 对照上游 trainer 的输入输出，列出固定旧轨迹与当前策略采样不一致的风险 | 你/Jun/HowseaonLink，11/20–25 |
| L9 | [PyTorch autograd 官方教程](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)：requires_grad、backward、梯度图 | 检查哪些参数有梯度、何时 detach、optimizer.step 前后哪些参数改变 | 你/Jun，按需先补，再做 E1/E4 |
| L10 | [LoRA 原论文](https://arxiv.org/abs/2106.09685) 与 [QLoRA 原论文](https://arxiv.org/abs/2305.14314)：低秩更新与量化基础权重 | 手写 W+BA，检查 rank/可训练参数；明确量化不是把所有训练状态都缩到 4-bit | Jun/HowseaonLink 精读，其他人看图与方法；11/17–22 |
| L11 | [PPO 的 37 个实现细节](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) 与 [CleanRL PPO 源码](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py) | 按需查 GAE、advantage normalization、entropy、gradient clipping；对照日志解释一次异常 | 你/Jun，11/24–12/04；进阶，不要求全组逐行读完 |
| L12 | [Colab 官方 FAQ](https://research.google.com/colaboratory/faq.html)：资源限制、持久化、Drive I/O | 若用 Colab，先保存 runtime/GPU/版本证据，设计 checkpoint 与日志备份 | HowseaonLink/Jun，需要该路线时再看 |

资源用途限制：Spinning Up/CleanRL 的小环境示例用于理解更新，不是用 CartPole 成绩代替 WebShop。TRL 文档固定在上游 PPO requirements 对应的 0.10.1 接口；它不是本组现代 GPU 环境的已验证安装方案。原论文数字只能作为引用，不作为本组成绩。

## 第一轮所有人需要答出的六个问题

1. 一条真实购物轨迹中，agent 能看见什么、不能看见什么？
2. 哪些 token 是 agent 生成的动作，哪些只是环境给的观察？
3. 购物结束是否等于购物成功？怎样从原始日志重算？
4. SFT、奖励模型训练、策略强化学习分别更新什么、监督来自哪里？
5. 为什么一次模型 generate、一次 reward-export 或一次 loss 下降不能独立证明 RL 有效？
6. 哪个结果是本组实测，哪个是原论文，哪个仍是计划？

Yuxiao 不需要承担 PPO 开发，但要能用自己的真实轨迹案例回答这些问题；展示本身就是理解闭环与证据的练习。

## E1：先在 CPU 看见“奖励改变概率”

读 L5 和 L9。构造一个两动作策略，使用 logits/softmax 或 log_softmax；选一个动作，给它一个与计算图分离的 advantage，用 `loss = - advantage * log_prob(action)` 做梯度更新。

要求交：

- 正 advantage 与负 advantage 两种情况的更新前后动作概率，以及实际梯度。
- 说明梯度下降为什么会分别增加/减少该动作概率。
- 说明环境奖励可以作为常数信号进入 loss，但不需要环境可微。
- 说明这个 CPU 小练习只是梯度理解，不能当作完整 Agentic RL 实验。

主责：Zqqq-qq；Jun 检查；其他人至少复核一次数值与方向。

## E2：手算三步 GAE，再检查代码

设三步奖励 `[0, 0, 1]`，全部 value 为 0，最后一步真实终止，gamma=0.99、lambda=1。

应得到 advantage `[0.9801, 0.99, 1]`。再改变 value/lambda，并分别讨论真实终止与 time-limit truncation 的 bootstrap。写清本环境 max_steps 的建模选择，不把所有截断机械当成同一类终止。

要求交手算、一个数值检查，以及解释：稀疏奖励不等于 PPO 永远只能更新最后一个动作；早期动作仍可能通过 return/advantage 得到信号，长时程有效信用分配则更困难。

主责：Zqqq-qq；Jun/Qiu 复核。

## E3：PPO clipping 与概率对齐

先手算：epsilon=0.2、ratio=1.4、advantage=1 时，clipped surrogate 为 `min(1.4,1.2)=1.2`；再试负 advantage，解释 `min` 与负号。

再检查一批实际样本：

- ratio 的分子/分母使用同一条响应、同一上下文与动作 mask。
- 未更新时，如果模型状态和概率定义一致，ratio 应接近 1；不接近时先排查 dropout、量化、padding、temperature/top-p 与 token 边界，而非直接调学习率。
- 旧策略是这批轨迹的行为策略快照；reference policy 是 KL 约束的参照，二者不能混为一谈。
- `do_sample`、temperature、top-p 会影响实际采样分布，记录并让 probability 计算与算法假设一致；不要无解释地混用。

交：手算、概率对齐检查、至少一个主动发现并解释的失败条件。

## E4：真实 WebShop 最小 RL 闭环

前提是任务映射、基础模型与 SFT 已验收。在少量 **训练任务** 上：

1. 从指定当前策略采新轨迹，保存行为策略 ID、概率所需信息及采样参数。
2. 从环境取得真实最终奖励，构造已定义的优势/回报。
3. 按选定算法更新 LoRA 参数。核对 mask、有限 loss/有效梯度与参数变化。
4. 保存并在独立加载后检查对应策略；新策略再次采轨迹。
5. 重复至少 3 个循环，让一位非作者核对轨迹确实来自各轮当前策略。

若目标 PPO 未实现，明确标记 on-policy 策略梯度分支的算法与限制。没有 clipping/value/GAE 就不要把该分支称为 PPO；不同奖励对照固定同一算法。

交：代码/配置、旧/新 checkpoint 标识、至少一段参数差异、日志、更新后新轨迹、第三人验收。小样本成功率升降仅作行为观察，不作性能结论。

## E5：你们自己的逐步奖励研究

先检查终局预测，再判断步骤分数：

- 独立 train/val 任务，终局奖励有变化；无变化先排查/采集，不“拟合全零”后宣布成功。
- 输出轨迹 `sum(progress)`、真实终局奖励、误差、每步 grounding/融合奖励。
- 用一条成功、一条部分奖励、一条失败轨迹做解释；若当前没有某类，明确缺失，不伪造。
- 低 MSE 不证明每步分数有唯一或因果意义，检查偏置、步骤长度和投机模式。
- 主对照为相同 grounding 的 terminal+G vs stepwise+G；额外奖励尺度/长度变化单独报告。

交：独立验证表、逐步分数可视化、局限和同预算 RL 对照。现有线性模型与原神经 SPA 估计器必须分别命名。

## 代码导读：每个人需要跟到哪

| 人员 | 第一轮精读文件 | 需要亲手追踪/修改并验证的环节 |
|---|---|---|
| HowseaonLink | [doctor.py](../src/webshop_lab/doctor.py)、[environment.py](../src/webshop_lab/environment.py)、[rollout.py](../src/webshop_lab/rollout.py) | 模型/环境启动到一条任务日志，错误与资源诊断 |
| Zqqq-qq | [common.py](../src/webshop_lab/common.py)、[progress.py](../src/webshop_lab/progress.py)、[rewards.py](../src/webshop_lab/rewards.py)、[旧 PPO trainer](../upstream/ppo/step_ppotrainer.py) | 奖励来源、动作概率、优势与参数更新；新 RL 集成仍待实现与实测 |
| Jun040504 | [model.py](../src/webshop_lab/model.py)、[sft.py](../src/webshop_lab/sft.py)、[旧 PPO 入口](../upstream/ppo/step_ppo.py) | chat template、assistant label、LoRA 和 checkpoint；支持 RL 载入 |
| Yuxiao-Liu233 | [PROJECT](PROJECT.md)、[ARCHITECTURE](ARCHITECTURE.md)、[UPSTREAM_AUDIT](UPSTREAM_AUDIT.md)，跟读一条真实 JSONL | 自己标注 observation/action/reward，做准确流程图与失败案例页 |
| QiuQiu-xuan | [metrics.py](../src/webshop_lab/metrics.py)、[rollout.py](../src/webshop_lab/rollout.py)、[核心测试](../tests/test_core.py) | 从原始轨迹重算成绩、检查分母/重复任务、结果可信性 |

“亲手参与”不要求所有人从零写一个 PPO。每个人应交一个自己理解、可被问答、经验证的贡献；环境/数据/评估同样是实际 Agentic RL 工作。

## Tutor 评审怎么进行

每天提交 [日报](../reports/daily-update-template.md)；训练节点提交 [实验评审包](../reports/experiment-review-template.md)。我会优先检查当前阻塞的最小证据，说明错误为何发生、相关原理和下一步验证，避免不理解地反复换依赖/参数。

不把未提供的日志视作已经成功；不承诺自动在某天后台完成实验或提醒。需要我审查时，把对应 PR、issue、运行 ID 和日志发到聊天即可。
