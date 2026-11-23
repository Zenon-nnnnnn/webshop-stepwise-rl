# Jun：神经PRM模型、损失与重载交接

本交接对应草稿PR #43，不替代How负责的Qwen集成、数据转换、独立进程推理与GPU执行。原始参考为upstream/prm/train_our_progress_model_lora.py的prm_model.forward及零初始化LN，不修改upstream。当前新增的是可集成核心代码，尚非完整可运行的Qwen训练管线。

## 已实现及验证范围

- src/webshop_lab/prm/model_core.py：读取真实backbone.config.vocab_size，在明确的assistant回合末位取词表logits，V到1线性头，按轨迹求和后MSE；标签只进入loss。
- src/webshop_lab/prm/train_core.py：真实PyTorch backward、有限梯度检查与optimizer.step。调用方提供已隔离、已分词、未截断的完整轨迹。
- 保存PRM专用adapter与独立head.pt，并记录初始化、tokenizer、模板、源码和文件清单SHA。加载先verify_prm，再恢复相同基础初始化和PRM adapter，最后restore_head；仅调用restore_head不构成完整重载。
- CPU测试使用小型因果backbone；已验证参考求和损失、padding排除、非法评分位置拒绝、标签不进入预测、未来后缀不影响早期位置及head文件身份/损坏拒绝。测试里的adapter文件是明确的契约占位，不能用于真实推理；没有验证真实PEFT保存、Qwen模板或GPU容量。

运行CPU测试需要PyTorch；现有无torch的CPU CI不可自动导入本测试。请将tests/test_neural_prm_core.py放入明确安装torch的PRM专用测试任务；不要加入现有仅标准库的通用discover作业而使CI缺依赖失败。

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_neural_prm_core.py -v
```

## 与How适配入口的张量契约

input_ids与attention_mask：[B,L]。turn_end_indices：[B,T]，long；turn_mask：[B,T]，bool。有效评分位置严格递增、在非padding范围内，无效位置为-1。labels：[B]，有限且位于[0,1]；未购买正常终止的R为0，技术中断不混为正常失败。输出step_scores：[B,T]，无效步骤严格为0；totals：[B]，loss为按轨迹平均MSE。

评分位置采用包含当前动作的因果末位定义，不向前移位套用PPO概率切片。真实模板必须保证assistant之外的用户/网页/角色/padding位置不被标记。模型核心只能核位置形状和padding，无法从任意token ID推断真实角色，必须由模板模块独立验证。

## 两个容易误验收的点

1. 原评分头零初始化时，首个反向步骤的backbone/LoRA梯度为0属于数学预期，head应有梯度并更新。随后在非零head和非零标签数据上检查PRM LoRA的梯度/参数变化；不能因为第一步LoRA为0就更改原初始化或宣称断梯度，也不能只测第一步就宣布LoRA训练通过。LoRA自身的零初始化也会影响各子矩阵何时出现梯度，按模块和实际更新记录判断。
2. 本核心使用完整[B,L,V]参考路径，再取位；不宣称减少其峰值显存。等价节省路径须在LM head之前先选末位hidden states，再经过原LM head和V到1评分头；核对相同输出与梯度后才能采用。直接hidden-state到1是架构变体。GPU显存不足应保留尺寸、日志和峰值记录，不静默改结构。

## GPU最小实训与独立进程交接

1. How补明确的Qwen构建入口：核基础/B1/tokenizer文件身份；若使用B1，先加载原SFT adapter并合并冻结，再挂独立PRM LoRA，不覆盖原B1。当前配置给出此方案为待验证选择，未声明已经完成合并。
2. 用策略train范围的真实完整轨迹，人工解码每个评分位置。不得把旧六策略160项val记录切作fit/dev。记录原件SHA、任务/规范化指令组、成功/部分/失败分布、超长拒绝数量。
3. 按执行方设备契约固定FP32、关闭TF32/autocast、离线本地快照、batch size 1；记录上下文、依赖、短训批次/步数和容差；两步只是起始探针，若不足以产生可观察LoRA更新须明确追加诊断范围，不冒称正式PRM训练。保存实际loss、head/LoRA梯度、参数前后摘要、峰值显存和耗时。
4. 保存adapter、评分头、完整初始化/模板身份；续训另存optimizer/scheduler及CPU/CUDA RNG、步数。当前save_prm只交付推理检查点，不提供完整续训状态，集成入口须另存。
5. 原进程eval、关闭dropout，对固定无标签输入保存逐步分数及总分；退出后由新进程重新加载基础/B1初始化、PRM adapter和head，逐项比较步骤对齐、分数及总分，采用运行前冻结的容差。不能复用内存模型，不能拿策略critic替代PRM head。
6. GPU未执行、来源未完整或重载不一致就保持待验收，不把CPU toy、文件保存成功、低MSE当完整模块完成。后续常数/长度基线与循环/规格分叉诊断由Qiu独立评估。

GPU仍需由How在#42完成当前批次的保存与audit后交接，并提供实际最后完整run和停派发状态。本交接没有操作GPU进程，也不声称已经腾卡。神经PRM通过后再冻结learned-progress PPO的新协议；旧规则OP结果单独保留。

目标执行环境按[设备契约](PRM_DEVICE_CONTRACT.md)：Python3.10.22/torch2.8.0+cu128/transformers4.56.2/peft0.17.1，不新增依赖；本轮CPU契约测试在torch2.15.0执行，目标版本与真实Qwen仍待How实测，不视为设备兼容验收。最终集成还须写config.json、summary.json、SHA256SUMS及训练续跑状态；save_prm仅提供adapter/head/manifest子契约。
