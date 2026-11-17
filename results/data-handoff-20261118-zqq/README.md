# 11/18 数据与环境交接复核

执行者：Codex Tutor，用户授权代为操作；不是学生独立运行。状态：`handoff_artifacts_cross_checked_raw_goal_map_pending`，未冻结最终数据。

## 实际执行与结果

运行 `python scripts/verify_data_handoff.py`，以固定 Git commit 读取原始 blob，避免工作区换行转换干扰。核对 15 项全部通过；详细检查项、SHA 与来源见 [validation.json](validation.json)。

- 环境证据：PR #10 commit `d1d160bcfa44c7caa2c92609d8e5012662a23de6`。
- 离线证据：PR #11 commit `12ce2b11dffb5e0956b8a0d6750c9e05ad62cb41`。
- train/val/test 三份 ID 的完整字节相同，分别 1451 / 160 / 200，ID 不相交。
- 两份 manifest 除 goal_map_sha256 外字段全部相同，环境侧记有映射 SHA。
- 已提交 health 的 ready、1624 条对齐、0 mismatch、split verified 及零指令交叉，与对齐报告一致；环境 manifest、health、rollout meta 的 fingerprint 一致。
- 三条 rollout 均来自 val，未购买，按 `terminal_reward = final_reward if done else 0` 重算均为 0；末次页面分均值为 0.9166667，只作诊断。
- 终局补充件记录的 JSONL SHA/字节数与源 blob 一致，任务级均值也与独立重算一致。

## 复现

仓库根目录执行。先确保两个源提交已 fetch 到本地（来自本仓库 PR 分支）：

```bash
git fetch origin env/setup-howseaonlink data/zqq-offline-audit-20261117-demo
python scripts/verify_data_handoff.py
```

脚本读取固定快照，生成本目录 validation.json，不使用 Python/GPU/环境服务的第三方依赖。它是证据交叉检查，不是完整环境重现。

## 剩余依赖与验收边界

完整 `goal-instructions.json` 未出现在已提交材料或本地已知下载中；预期 SHA-256 为 `6b5d9dfceee913828871b8dff62fd59720a40306738a92a496d1b0fb6838fa88`。需要环境负责人将本次实机生成的原文件共享到私有仓库/Release 或组员可访问位置，并保留原始字节。不能用手写重建的映射替代。

拿到后按顺序：核对文件 SHA → 覆盖全部测试 ID → 用该映射重跑 `split --goal-map` 到新目录 → 与环境侧三份 IDs/manifest 核对 → 确认对应环境 health/fingerprint → 再记录正式数据版本。若出现任何差异，先排查，不覆盖现有候选或宣布冻结。

本次没有本地启动 WSL 服务、没有 CUDA/SFT/RL 训练，也不修改原始 rollout 或旧验证记录。
