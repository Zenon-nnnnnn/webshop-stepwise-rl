# Proposal 草案：资源受限 WebShop 智能体的终局奖励与规则势函数塑形对照

> 2026-11-21 路线更新，待填课程名、学号、rubric、篇幅/引用格式并由全组定稿。11/22 组内定稿、11/23 提交；不是已提交版本。详细执行条件见 [有限预算 RL 协议](../docs/RL_EXECUTION_PROTOCOL_20261121.md)。

## 1. Problem and Motivation

购物 agent 需要搜索、查看商品、选择规格并购买。我们研究小型 LLM 在单张 16 GB GPU 下的真实策略更新，以及固定预算内改变中间奖励是否影响购物表现。项目借鉴 SPA-RL 的逐步奖励研究问题，但不宣称精确复现论文模型或性能。

前期学习型可加进度模型未通过独立预测/循环检查，因此主对照调整为可解释的规则型规格势函数与终局奖励。学习型估计器的负结果仍作为方法分析，而不悄悄替换其名称或删除失败证据。

## 2. Research Questions

RQ1：任务 SFT 相对基础模型是否改善购物执行和成功率？历史验证与新同设置评估分开报告。

RQ2：仅用终局监督的线性可加估计器能否泛化、避免无效动作和循环获利？比较独立验证上的常数/长度基线，并承认总分预测准确不等于逐步因果贡献正确。

RQ3（主 RL 问题）：相同 SFT 起点、算法、任务批次/更新数与共同上限下，终局奖励与规则型规格势函数塑形是否产生不同的独立验证表现？不预设有提升，不以 loss 或模型预测误差替代真实策略评估。

## 3. Data and Environment

材料含1624条SFT对话。默认划分发现12个train/test ID重叠；真实环境goal-map校验后采用1451 train /160 val对话，保留200个test任务ID，排除13条，seed42。按ID和真实指令核对集合隔离；保留test任务ID不代表有200条test专家演示。

train用于SFT/探索/RL更新，val用于开发及固定协议评估。记录原始资产、split manifest、映射SHA和环境fingerprint，每次运行检查live health。最终test只能在模型选择/协议/配置冻结、负责人另行放行后使用，本次执行单不授权test。

## 4. Method and Actual Scope

共同基模为固定revision的Qwen2.5-1.5B-Instruct。使用#18原始1 epoch SFT LoRA作为两臂共同起点，记录实际续跑及同步补丁限制；不把该运行改写为执行了未冻结的正式SFT草案。策略/critic保存重载、on-policy采样、token概率/mask/奖励和结束边界按工程清单验收。

设R为购买结束评分，未购买结束为0；页面匹配分只诊断。TG只在最后动作给R。OP在TG上加Phi(s_next)-Phi(s)，Phi是选中规格词与需求词的固定匹配势值，范围[0,0.25]，不使用隐藏目标答案。gamma=1、初始势值为0，购买/有限步数终止时下一势值清零，总奖励首尾抵消保持为R。循环回到同势值不产生净塑形收益；这不证明逐步因果归因或必然提升。

两臂共享beta=0、KL coefficient=0，FP32非量化训练，assistant-turn联合回复概率比PPO；gamma=lambda=1、clip0.2、AdamW lr1e-5、value系数0.5、梯度裁剪1.0。每批只更新一次，不把它描述为多epoch裁剪效果研究。

两个预定训练种子42/43；每臂每种子4更新，每更新4个冻结train任务，共16 episode。共享context2048、生成128 token上限、训练8动作上限。比较固定episode/更新数和共同预算上限，真实环境步/token/GPU时间另报，不声称实际消费被强制相同。固定选择第四轮endpoint，不按val挑checkpoint或追加训练。学习型进度模型与规则势函数明确区分，原神经SPA未复现。

## 5. Evaluation

固定同一160-task val，每任务一次、FP32贪心、context2048/生成128/最多10动作，比较原B1和各种子两臂endpoint；实际配置/runner及导入源码版本均留档。旧4-bit全量指标与新FP32不同设置结果不混作同一对照。

主指标为全部有效评估episode中的购买完成且满奖励成功率；未购买终局分为0。辅助指标为购买数、平均终局任务分、非法动作率、步数、循环和资源。正常策略失败不能删除，技术中断单列并按预定规则补跑。每臂列任务数/episode数/分母；单臂Wilson、每种子配对任务bootstrap区间；两个种子不当成双倍独立任务，不以小样本持平证明等价。

候选历史160-task B0/SFT结果为19/160与50/160，属于SFT验证而非RL提升。三轮工程候选在8-task val上两臂均5/8；新同批B1反馈同为5/8，基线原始文件的独立复核仍待补交。阶段结果不替代正式全量对照或最终test。

## 6. Deliverables, Risks and Limitations

交付真实数据核验、SFT起点、可追踪的多轮策略更新/恢复、限定预算奖励对照、统一评估和失败分析。代码归因、执行来源和个人贡献分别记录。有限样本/少种子、规则势值的词面局限、原进度模型泛化失败、历史恢复证据缺项、KL关闭和量化/精度差异均披露。

核心验收以真实闭环和可复核结果为准，不要求正提升；若运行未完成，记录缺口并核对课程rubric，不以SFT或奖励导出冒充RL。扩大预算或切换算法必须版本化，不能为了得到显著结果一直训练。

## 7. Schedule and Team

11/22 proposal组内定稿，11/23提交；11/24夜间完成必要训练/评估/核对；11/25验收及PPT/讲稿/演示冻结排练；11/26 pre；12/15 report。HowseaonLink负责环境/GPU执行，Zqqq-qq负责数据/奖励/RL和技术统筹，Jun040504负责SFT/模型/恢复核对，QiuQiu-xuan负责独立评估/统计，Yuxiao-Liu233负责PPT/pre。各人写本人report模块；详见[逐日计划](../docs/DAILY_PLAN.md)。

## References

Wang et al., 2025. [SPA-RL: Reinforcing LLM Agents via Stepwise Progress Attribution](https://arxiv.org/abs/2505.20732)。相关WebShop、PPO、势函数塑形、基础模型与LoRA文献需全组逐条核对后按课程格式补齐；此草案不伪造未核查引用。
