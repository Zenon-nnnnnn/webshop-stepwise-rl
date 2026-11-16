# 五人分工、里程碑与验收

日期基于 Asia/Shanghai，2026 年。这是按模块依赖制定的第一轮分工，未假定成员已有特定经验，也未确认谁拥有此前提到的 5060 Ti。账号与角色先按下表执行；如果组内调整，需同时更新对应 Issue 和本表。

## 负责人

| 角色 / GitHub | 主责 | 具体产物 | 任务 |
|---|---|---|---|
| A：HowseaonLink | 环境与集成、Colab 适配 | 可从头执行的 notebook/安装步骤；依赖版本、GPU 检查、WebShop health、启动日志 | [#1](https://github.com/Zqqq-qq/webshop-stepwise-rl/issues/1) |
| B：Zqqq-qq（你） | 数据协议、实验设计、进度协调 | 数据审计、split manifest、研究问题与对照表、proposal 主稿、提交核对 | [#2](https://github.com/Zqqq-qq/webshop-stepwise-rl/issues/2)、[#6](https://github.com/Zqqq-qq/webshop-stepwise-rl/issues/6) |
| C：Jun040504 | 基础策略与 SFT | 基础模型运行、2-step 短训练、正式 SFT adapter、config/run.json、资源记录 | [#3](https://github.com/Zqqq-qq/webshop-stepwise-rl/issues/3) |
| D：Yuxiao-Liu233 | 进度模型、奖励与 RL | 论文方法说明、奖励测试、独立进度验证、PPO 兼容与策略更新实验 | [#4](https://github.com/Zqqq-qq/webshop-stepwise-rl/issues/4) |
| E：QiuQiu-xuan | 评估与实验分析 | 指标核对、固定评估任务、结果表与图、真实失败案例、实验登记 | [#5](https://github.com/Zqqq-qq/webshop-stepwise-rl/issues/5) |

你使用 Mac，可以直接承担审计、划分、实验设计和写作；完整环境资产解压/运行由 A 协助。GPU 任务由 A/C/D 协调使用已验证的组员机器或 Colab，不要求每人自备 GPU。当前仓库尚无已验证的一键 Colab notebook，A 的适配属于待完成任务；谁拥有 5060 Ti 不影响模块负责人安排。

## 本周：proposal 前每个人交什么

**11/20 前，每人提交一个可 review 的 PR 和一小段负责章节的草稿。** 没有 GPU 的工作先做，避免大家等模型训练。

| 人员 | 这周先做 | 交给谁 / 怎么验收 |
|---|---|---|
| A | 确认实际可用计算资源；准备 Colab/本机运行路线，做 CUDA kernel 与模型加载短检查；尝试 WebShop 启动 | C 能按步骤重新加载模型；B 能拿到环境任务映射。未跑通则提交错误日志和明确阻塞 |
| B | 整理课程 rubric、研究问题、原论文引用和对照实验；在 Mac 重算 audit 与离线 split；与 A 核对完整环境后冻结划分 | C/E 使用同一份 train/val/test ID、split SHA 和环境 fingerprint；合并全组 proposal 段落 |
| C | 阅读 sft.py 和模型配置，先检查数据/标签与短训练流程；资源可用后跑 2 个优化步骤并保存重载 adapter | E 能加载同一个基础模型及 adapter；完整训练依赖 A 的环境和 B 的正式划分 |
| D | 读懂 SPA 的逐步归因、grounding 和 PPO；用小样例核对奖励公式；列出旧 PPO 迁移点与最小策略更新验收 | 全组看懂方法图和公式；C/E 知道训练数据、checkpoint 与奖励接口，不把奖励导出当作策略训练 |
| E | 明确 success/reward/steps/invalid action 定义；手工核对合成/示例日志，准备评估命令与空白结果表 | B 确认实验设计可比较；真实模型可用后能直接执行评估，示例数字不进正式成绩表 |

**11/21 汇总并互审，11/22 提交 proposal。** B 负责整合，A/C/D/E 各写自己的技术段落，不能只报口头进度。

## pre 前与最终报告

- 11/25 前：A/C 交可运行系统与首轮 SFT；B/E 核对同一 val 上的基础模型与 SFT；D 交进度模型/奖励初步分析及 RL 接入状态。
- 11/26：B 整合 PPT；A 提供备用演示，C 提供训练配置与证据，D 提供方法图，E 提供真实图表/失败案例。全组排练。
- 11/27：B 讲问题与实验设计，A 讲系统，C 讲 SFT，D 讲逐步奖励/RL，E 讲结果与分析。实际分段随课程时长调整。
- 12/03 前：D 主责、A/C 支持，完成 PPO 可行性检查及一次完整策略更新/保存/重载；E 核对真实交互评估。未完成时明确剩余问题与调整范围。
- 12/10 前：固定配置、完成最终评估。12/11–14 每人撰写自己模块的方法/结果，B 负责总稿，E 核对图表与证据，全组通读；12/15 提交。

## 依赖和交接

A（运行环境）与 B（正式数据协议）→ C（SFT checkpoint）→ D（采集/奖励/RL）→ E（同协议评估）；E 的评估工具、D 的方法核对、B 的审计与写作从第一天就可以并行。

交接至少包含：Git commit、配置、输入 SHA、split/goal fingerprint、命令、日志、产物位置与已知限制。检查通过再给下游，不只发一句“跑好了”。模型和大日志放团队共享存储，Git 提交代码与脱敏汇总。

代码互审建议：A↔C、B↔E，D 的奖励与策略更新由 C/E 共同核对。每个人需要能解释自己的实现与结果。

## 截止与阶段产物

| 日期 | 阶段产物 | 验收 |
|---|---|---|
| 11/16–11/20 | 环境、数据、kernel、3 任务 rollout、2-step SFT | 有日志和真实运行证据；发现问题及时缩范围 |
| 11/21–11/22 | proposal | 引用清楚、实验矩阵、五人分工、风险与备选；不虚构结果 |
| 11/23–11/25 | 首轮 SFT 与 B0/B1 val 比较 | 同任务/模型/提示词；记录任务数和失败 |
| 11/26 | 冻结 pre、录备用演示、排练 | 每人知道方法与贡献；demo 不依赖临场训练 |
| 11/27 | presentation | 展示实际完成部分、初步结果与后续计划 |
| 11/28–12/03 | 进度与 RL 集成检查 | 独立 val、奖励对齐、一次策略更新与保存重载 |
| 12/04–12/06 | 决定主线 | RL 不稳定则明确降级为 SFT/进度分析；核对 rubric |
| 12/07–12/10 | 固定配置并最终评估 | 冻结测试集、记录完整性、必要多种子 |
| 12/11–12/14 | report、引用、图表、复现清单 | 所有结果可追踪到 run / config / commit |
| 12/15 | report 提交 | 按课程实际提交规则执行，不由本仓库自动提交 |

## GPU 预约建议

在 GitHub issue 评论里认领时间段，写预计开始/结束、任务和显存需求。任何长任务先通过短检查。不要同时加载环境 CUDA torch 与训练模型；环境使用 CPU torch。训练结束立即登记日志与 adapter 路径，避免只有一个人知道模型存在哪里。

## 周会 15 分钟

每人只汇报：已完成并有证据的内容、阻塞、下一项可验收产物。不要用“代码基本好了”代替命令与输出。proposal/pre/report 的内容以证据状态为准。
