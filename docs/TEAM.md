# 五人分工、里程碑与验收

日期基于 Asia/Shanghai，2026 年。此版按组内最新安排：**Zqqq-qq 接手 Yuxiao-Liu233 原来的进度模型、奖励与 RL 任务；Yuxiao-Liu233 负责 PPT 和 pre。** 旧版技术主责分配以此版为准。

已知算力顺序：HowseaonLink > Yuxiao-Liu233 > Jun040504 > Zqqq-qq ≈ QiuQiu-xuan。算力执行与模块责任分开；具体任务规模取决于实机显存/内存检查。

**细化执行入口：** [按 11/27 pre、12/15 report 倒排的每日任务表](DAILY_PLAN.md)、[边做边学资源与练习](LEARNING_GUIDE.md)。每日细化计划的阶段验收优先于旧版粗略时间安排；这是计划，不是已完成记录。

## 当前负责人

| 人员 | 主责 | 明确交付 | Issue |
|---|---|---|---|
| HowseaonLink | 环境集成、主算力与正式作业执行 | 稳定运行环境、WebShop 启动、正式 SFT/重推理/采集日志与模型产物；必要的 RL 作业执行 | [#1](https://github.com/Zqqq-qq/webshop-stepwise-rl/issues/1) |
| Zqqq-qq（你） | 数据协议、实验设计、进度模型、奖励与 RL | 数据审计/划分、进度预测验证、奖励实现、PPO 接入和 RL 对照分析；proposal 技术整合 | [#2](https://github.com/Zqqq-qq/webshop-stepwise-rl/issues/2)、[#4](https://github.com/Zqqq-qq/webshop-stepwise-rl/issues/4) |
| Jun040504 | 基础模型与 SFT 实现 | 数据标签检查、SFT 配置与短验证、模型重载、训练结果解释；正式长训练配置交主设备执行 | [#3](https://github.com/Zqqq-qq/webshop-stepwise-rl/issues/3) |
| Yuxiao-Liu233 | PPT 制作、pre 内容整合与展示组织 | 幻灯片结构与制作、展示讲稿、素材整合、演示串联、排练与问答准备 | [#6](https://github.com/Zqqq-qq/webshop-stepwise-rl/issues/6) |
| QiuQiu-xuan | 统一评估与实验分析 | 指标/评估代码、结果表与图、日志核对、失败案例、实验登记 | [#5](https://github.com/Zqqq-qq/webshop-stepwise-rl/issues/5) |

**Yuxiao-Liu233 不再负责进度模型训练、奖励代码、PPO 适配或 RL 调参。** 这些由 Zqqq-qq 主责，Jun040504 支持模型接入，QiuQiu-xuan 支持效果核对。Zqqq-qq 原先承担的 PPT 整合也交给 Yuxiao-Liu233；全组各提供自己的准确素材和技术说明。

最终报告仍由每个人撰写自己模块；Yuxiao-Liu233 撰写展示/系统说明及自己的实际工作，不能把全组 report 或技术实验全部交给某一人。Zqqq-qq 负责 proposal 技术整合及最终报告技术一致性，全组共同审阅和确认提交。

## 计算如何安排

| 人员 / 设备层级 | 优先任务 | 边界 |
|---|---|---|
| HowseaonLink：主算力 | 稳定环境、正式 SFT、批量推理、轨迹采集、最终评估；按 Zqqq-qq 配置执行可容纳的 RL 作业 | 不包办 SFT/RL 开发与结果解释 |
| Yuxiao-Liu233：第二算力 | 在展示工作不受影响且资源可用时，可协调借用其设备执行冻结作业 | 设备支持不等于重新承担 RL 模块责任；当前主责是 PPT/pre |
| Jun040504：第三算力 | 小规模推理、SFT 短检查、模型/标签验证 | 正式长训练可交主设备执行，仍负责 SFT 方法与配置 |
| Zqqq-qq：Mac | 数据审计/划分、奖励与进度代码、CPU 测试、小样例分析、RL 方案与结果解释 | 不要求 Mac 本地跑 CUDA；重训练按配置交主设备，Colab 可作备选 |
| QiuQiu-xuan：相近设备 | 评估实现、日志解析、CPU 统计、图表与失败分类 | 批量 GPU 推理交主设备执行，自己核对返回日志和指标 |

每个重作业由模块负责人提供 commit、冻结配置、输入/划分 SHA、环境 fingerprint、命令与产物要求；执行人返回日志、模型路径、版本和资源记录；模块负责人检查并解释结果。

优先使用组内现有稳定环境。Colab 为补充路线，当前仓库尚无已验证的一键 Colab notebook。若确需适配，由环境/训练负责人协调；不因 Zqqq-qq 接手 RL 就默认其 Mac 能完成完整训练。

## 本周：proposal 前交付

**11/20 前，每人交可检查的产物及自己负责部分的草稿。** 没有 GPU 的工作先做，不等全量训练才启动。

| 人员 | 本周先做 | 验收和交接 |
|---|---|---|
| HowseaonLink | 主设备 CUDA kernel、模型加载、WebShop 启动与版本记录；接收短训练配置 | Jun 能重载模型，Zqqq-qq 能核对环境映射；失败则给出完整日志 |
| Zqqq-qq | 数据审计/离线划分、研究问题/对照表；读懂 SPA 奖励公式并做小样例检查；列出 PPO 接入点 | 和 HowseaonLink 完成实际环境对齐后冻结数据；交奖励接口和 RL 验收清单 |
| Jun040504 | 检查 SFT 标签/配置，跑短训练与保存重载；准备正式作业命令 | 正式训练产物可交 Zqqq-qq 做后续奖励/RL、交 QiuQiu-xuan 做评估 |
| Yuxiao-Liu233 | 先确定 pre 叙事、PPT 结构、页面骨架、素材清单与讲解分段 | 给各模块负责人明确素材需求；先留结果占位，不填虚构成绩 |
| QiuQiu-xuan | 指标定义、示例日志人工核对、评估脚本与空白结果表 | 真实模型可用后将固定评估配置交主设备执行，核对日志和分母 |

11/21 全组互审；11/22 提交 proposal。Zqqq-qq 汇总技术稿，各成员提交自己的内容；Yuxiao-Liu233 可复用已核实材料建立展示结构。

## pre 与 report

- **11/25 前**：环境/SFT 负责人交可运行系统与首轮 SFT；评估负责人交同一 val 上的真实结果；Zqqq-qq 交奖励方法及 RL 当前状态。各人把图、配置说明、案例和限制交给 Yuxiao-Liu233。
- **11/26**：Yuxiao-Liu233 完成 PPT、讲稿、演示串联和排练组织；HowseaonLink 提供备用演示；全组核对自己模块的技术准确性和问答。
- **11/27**：Yuxiao-Liu233 负责 pre 主讲/串联与展示组织，其他成员按课程要求和组内讲解安排参与；各模块负责人支持技术问答。
- **11/25 首次尝试、11/28 缓冲、12/01 硬验收**：Zqqq-qq 主责真实 RL 多轮闭环，Jun 支持模型接入，HowseaonLink 执行重作业，QiuQiu-xuan 独立核对；12/02–03 执行匹配预算的正式对照。
- **12/04–06**：根据 RL 验收完善主对照和预定消融；若 PPO 仍阻塞则优先完成明确命名的 on-policy 策略梯度分支，保留真实 RL 实践，不以 SFT 充数；范围变化需对照课程 rubric。
- **12/10 前**：冻结最终配置与实验结果；测试集不用于调参。
- **12/11–14**：每人写自己模块，核对引用、图表、日志和复现信息，全组通读。
- **12/15**：按课程要求提交 report。

## 工作流与排队

环境＋数据 → Jun 的 SFT 实现/配置＋主设备正式训练 → Zqqq-qq 的进度/奖励/RL → QiuQiu-xuan 的统一评估与分析。Yuxiao-Liu233 从第一周并行准备展示结构，再逐步纳入真实结果。

主 GPU 优先：环境短检查 → 正式 SFT → 基础模型/SFT 固定 val 对照 → 正式采集 → RL/最终评估。每个作业写预计时间、配置、资源需求和依赖；同一块 GPU 默认一次一个重任务，同一环境服务只允许一个 rollout 客户端。

Review：环境与 SFT 由 HowseaonLink/Jun 互审；Zqqq-qq 的奖励/RL 由 Jun 和 QiuQiu-xuan 核对；PPT 由技术负责人分别检查。结果必须可追踪到配置和日志。
