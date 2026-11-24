# 故障排查

## CUDA 可用但首次运算报 no kernel image / unsupported architecture

这是新 GPU 与 PyTorch/扩展编译架构不匹配的典型方向。先检查当前激活环境、`doctor` 中的 torch build 和 compute capability，按 SETUP 的候选 cu128 版本做独立 kernel 测试。不要先修改模型训练代码，也不要直接套用原包 torch 2.5 配置。

新入口使用 PyTorch SDPA，不依赖原包 flash_attn / xformers / vLLM。若你额外安装这些扩展，需单独检查它们是否支持目标显卡；它们不在主线安装要求内。

## bitsandbytes 量化失败

确认当前是 NVIDIA GPU 环境、torch kernel 已成功、bnb 版本与 CUDA 路径没有被另一环境污染。先用候选版本运行小模型加载。若 1.5B 不量化 LoRA 经实测可用，可以复制配置将 `load_in_4bit=false`，但训练/评估比较应保持一致，并记录修改。不要将失败的候选组合标成稳定环境。

## CUDA OOM

先关闭占显存进程与重复模型实例。按顺序尝试：较小模型、短一点的上下文、减少生成长度、确认 gradient checkpointing 和 micro-batch=1、关闭不必要 checkpoint。上下文变短会改变任务可见信息，需在全部对照中一致使用并报告 skipped 数量；不能仅为了跑通而忽略大量复杂任务。

PPO 还可能持有价值头与参考模型，SFT 能跑不代表 PPO 必然能跑。16 GB 是否足够必须实测。

## java / pyserini / Lucene 报错

在 `webshop-env` 检查 `java -version` 和 JAVA_HOME；依赖按本项目 CPU 文件安装。索引目录需要直接含 `segments_1` 和 `_0.*` 文件，不能多套 `indexes/indexes`。不要用不匹配的 Java、Pyserini 或索引版本强行继续生成结果。

## spaCy 报找不到 en_core_web_lg

在实际启动服务的同一个 `webshop-env` 运行 `python -m spacy download en_core_web_lg`。GPU 环境里安装过并不代表 CPU 环境可见。

## 原始压缩包 CRC 不匹配 / 磁盘不足

保留原包，先核对文件大小和 SHA，必要时重新获取。导入工具不会覆盖坏的已有文件；确认没有需要保留的数据后将指定坏文件移到备份目录，再重试。剩余 `.partial` 是中断写入，不作为完整资产使用。

## alignment_mismatches 非 0

说明 SFT 的任务 ID 与当前环境底层指令不匹配，不能直接比较。查看 `artifacts/environment-alignment.json`，核对：完整商品文件是否正确、human_goals 是否启用、goal shuffle 是否沿用原实现、是否用过子集数据或另一份索引/指令文件。

只有预算不同会记在 `budget_variant_ids`，不算底层请求映射错误；冻结环境 fingerprint 后保持预算一致。若底层 ID 仍不匹配，应由 B 负责人基于实际指令构建显式映射并经过 review，不能简单删掉守卫继续跑。当前工具不自动猜测或重排冲突 ID。

## split_status.verified 为 false

先 `webshop-lab split --goal-map artifacts/goal-instructions.json`，再查看 health。若仍失败，检查 train/val/test 底层指令重叠是否因映射变化造成。修复划分后重新训练受影响模型，不要用已泄漏模型继续评估。

## 模型总是空回复 / Invalid format

确认 adapter 与 base model 对应；检查 tokenizer chat template 和 EOS；查看实际 response；增加合理的生成长度；先比较基础模型是否也失败。不要用“去掉 Action: 检查”掩盖格式问题。动作格式本身是 agent 能力的一部分，应计入无效动作率。

## 当前观测超过上下文预算

工具保留任务系统提示和最近完整交互，不截断当前观测内部文本。若当前观测本身过长会报错/在 SFT 中记录 skipped。应选择合理上下文或设计并统一评估观察压缩策略，再记录为明确方法改动。

## 连接被拒绝 / 运行中断

检查 CPU 服务还在、端口匹配、只有一个 rollout 客户端。网络故障不能当作模型失败。中断日志保留作调试；换新输出路径重新运行完整任务列表，当前没有自动断点续跑。正式汇总必须核对 selected_ids 是否全部完成。

## loss 下降但成功率不升

这完全可能发生。验证 token loss 测的是模仿，不是任务完成；进度 MSE 测的是总分预测，也不是策略改善。用真实 rollout 检查商品选择、规格、预算、循环和动作格式，报告实际结果。
