# 11/19：真实映射下的独立划分复核

HowseaonLink 已在 PR #11 共享完整 [goal-instructions.json](https://github.com/Zqqq-qq/webshop-stepwise-rl/releases/download/data-v1/goal-instructions.json)。Tutor 在 Mac 下载原字节并运行，不是用户独立运行。

原件1969461字节，12087条，SHA-256 `6b5d9dfceee913828871b8dff62fd59720a40306738a92a496d1b0fb6838fa88`，与环境记录一致；本机重新生成 train1451/val160/test200，排除13。12项独立检查全部通过，详见 [validation.json](validation.json)。manifest见 [split-manifest.json](split-manifest.json)。

三份ID和manifest与目标环境 commit `d1d160bcfa44c7caa2c92609d8e5012662a23de6` 的 Git blob 完全一致；完整实际环境指令的train/val/test归一化集合不相交，1624条SFT底层指令全部对齐。原始保留对话记录未改变。静态验证已从“只见部分测试指令”提升为“完整映射复核通过”。

## 复现命令

原始 SFT 与 goal-map 从私有 data-v1 Release 下载，完整对话与映射不入 Git。仓库根目录执行：

```bash
git fetch origin env/setup-howseaonlink
PYTHONPATH=src python -m webshop_lab.cli split --sft /path/to/webshop_sft.json --goal-map /path/to/goal-instructions.json --output data/processed/mapped-20261119-zqq --seed 42
python scripts/verify_mapped_split.py /path/to/webshop_sft.json /path/to/goal-instructions.json
```

输出目录用于独立复核，未覆盖默认 data/processed 的运行输入。正式训练须在目标机采用同一版本文件，检查服务 health 的指令对齐/划分状态与指纹。参考环境 goal_fingerprint 为 `e69c9ae2a0aad88977c44a7baf49f1f5ef7918ba8317923c4ae24dcc1f5638d8`，来自实机证据，本机没有启动完整环境或重新计算全 goals 指纹。

此记录提供可采用的映射版候选证据；不自动合并 PR、不自动关闭 Issue #2，也不代表SFT/RL已验收。三份 IDs 已在 #11 和 #10 中提交且一致，本目录不再重复存储。
