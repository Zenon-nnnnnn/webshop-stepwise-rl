# Qiu：#52 GPU 短测独立验收与 #51 交接核对

2026-11-24。按 [#52 最新分工](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/52#issuecomment-7132933687)，本次核对已保存GPU预测、裁剪记录、覆盖和重载输出，不执行新的模型前向、GPU训练或设备续训。#49补包/CPU后续由Yuxiao承接，原Qiu/Tutor成果保留。

## 已发布三档16步原件

[短测包](https://github.com/Zqqq-qq/webshop-stepwise-rl/releases/download/data-v1/prm-evidence-hidden-lrprobe-20261124.zip) SHA `aa33e5ca0b62e06a43801ea6416326256ae0d389f8ee8e6c3d46cbf571e75b08`；[特征/源码包](https://github.com/Zqqq-qq/webshop-stepwise-rl/releases/download/data-v1/prm-evidence-hidden-features-20261124.zip) SHA `aa35ec78c4149dc81414e0a44b49740abe5b55103b2fb7fdc57383bc67e5615d`。分别29/16个载荷SHA全部匹配，第16步完整检查点另有8个载荷逐项匹配。

机器结果：[gpu-probes.json](gpu-probes.json)。同8条原输入SHA `9c88beff…`、固定tokenizer四文件及模板SHA一致；独立重编码确认8条、38个assistant im_end评分位。每档16次实际更新/128次轨迹呈现、8/8保留，各16次都触发clip1。每条训练loss由当次8条预测/标签重算，最大差小于2.4e-8。三档reference与另存独立重载的逐步/总分差均为0；这是核保存记录，不是本机再次运行1.5B重载。

| LR / acc8 | 末次8条 MSE | 最大绝对误差 | 满分组−零分组均值 | 记忆门槛 |
|---|---:|---:|---:|---|
| 1e-7 | 0.346186244 | 0.980564242 | +0.000005769 | 未过 |
| 1e-6 | 0.306178403 | 0.849675998 | −0.000186510 | 未过 |
| 1e-5 | 0.206815201 | 0.720792741 | −0.003560647 | 未过 |

均值差定义为 `(p_0+p_6225)/2 − (p_95+p_5031)/2`。四条均为3回合，前两条标签1，后两条标签0。在1e-5下：

| task | 标签 | 总分 |
|---|---:|---:|
| 0 | 1 | 0.287688971 |
| 6225 | 1 | 0.279207259 |
| 95 | 0 | 0.287153423 |
| 5031 | 0 | 0.286864102 |

满分组均值0.283448115，零分组均值0.287008762，当前未形成有效排序。不将该现象或高余弦解释为“特征必然没有信息”。常数MSE **0.187500**、按回合数分组均值MSE **0.129629630**，仅是这8条上的样本内参照，不能当独立dev结果。后续dev基线只用fit拟合。

实际保存身份仍为 `5a529686c2a5a26da8893b67ded98ac63a208d12`；提供的7个核心源码文件逐字节SHA与执行单 `5ae6d83553ced1a74db32d750ada6f9d9b6786b1` 一致，覆盖train/infer/runtime/model_core/train_core/config/tokenize。未独立恢复本地执行的完整Git树，因此只签收这7文件一致，不改历史manifest身份或宣称两个提交完全同树。

## #51 独立核对与交接

固定 #51 `cfb10c9e681ff21090b610504da7f0c6d1424dcf`。已核 [Actions run 45379955682](https://github.com/Zqqq-qq/webshop-stepwise-rl/actions/runs/45379955682) 的实际head、工作流、成功状态；下载artifact14912463941，ZIP SHA `50a10570b28cd2e057328578fff717c81e32ef3eb6c320fb5a21ca66e38e8229`，其results.json与仓库跟踪的JSON逐字段一致。

独立从保存的8条预测/标签复算4组共32个检查点，MSE/max error/轨迹呈现/优化器步数/裁剪计数一致；512呈现的batch1为512更新、batch8为64更新。四组均未通过记忆门槛，不包装为同优化器步数比较或dev调参。本机没有复跑PyTorch训练，也没有反序列化optimizer；安全加载逻辑、114状态等检查通过的证据来自已核验的固定Actions运行，不能写成Qiu本机再次检查了运行中精度。

同时用补包的重提F独立执行NumPy最小二乘：磁盘(8,151936)，显式加N后(8,151937)，rank8、condition205.1749698264、解L2约0.00701406445、FP32最大标签误差约7.56e-7。结果在平台浮点容差内一致；保留“重提特征”来源等级，不改称首次保存原F。完整机器记录：[pr51-replay.json](pr51-replay.json)。

对 #49 `d9d2f6c` 与 #51 做了 `git merge-tree --write-tree` 整合检查：无冲突且两方记录可同时保留。没有执行分支合并、覆盖#49或改main。建议以保留双方源码/结果的方式纳入，避免用新的CPU JSON替换旧记录或将不同数值实现说成逐bit复现。

## 下一检查点

这些附件没有累计48/80/120步结果，本次不预先签收。收到第48步后，按原标签和相同来源核对：新增32/累计48更新、累计384次呈现、原8条完整覆盖，分段裁剪比例和全部8条误差，重点4条的标签/总分/组间差，检查点SHA及独立重载。随后80/120仍各自验收，设备执行按Zqqq现行单子，不由本报告追加预算。正式记忆门槛仍为MSE<0.001且每条误差<0.1，泛化/逐步归因/PPO和test不在此层签收。

## 可复核入口

```bash
python scripts/audit_prm_hidden_results.py \
  --probe-archive /evidence/prm-evidence-hidden-lrprobe-20261124.zip \
  --features-archive /evidence/prm-evidence-hidden-features-20261124.zip \
  --tokenizer /local/pinned-qwen-tokenizer --repo /local/repository \
  --output /new/gpu-probes.json
python -m unittest discover -s tests -p test_prm_hidden_results_audit.py -v
```

需NumPy的旧SVD复算由#49/#51保留；本入口的指标/覆盖检查使用标准库，真实模板复核使用Transformers4.56.2/Tokenizers0.22.2，不加载Torch。`--repo`须已取得固定#52提交以核七个源码文件。审计输出不覆盖输入或已有文件。测试中的11个构造反例分别核缺样本/缺步骤、任务重复、NaN、评分位置、裁剪标记、标签/呈现数/记录loss，以及续训分段计数；不把它们当真实GPU结果。本地完整107项CPU测试通过，compileall、仓库检查和来源审计通过。
