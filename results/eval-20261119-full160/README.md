# Full-160 评估证据：B0 与本地 LoRA（2026-11-19）

这是 [PR #18](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/18) 的 160 任务验证运行的**仓库内可核对副本**。原始证据包在私有 Release `data-v1` 的 `webshop-pr18-qwen25-sft-and-validation-evidence.zip`（77,820 字节，SHA-256 `c62a1a97ac259a4a0f5ffb7d70e054ca2523c14d9866f53a40b56562c6aaae3b`）。

**这不是正式冻结实验**：B1 是本地 1 epoch LoRA 候选，采样 commit 尚未入仓库，执行配置也不是 `configs/eval-*-frozen.json`。可用于 proposal/pre 的"初步真实结果"，但**暂不进入** `results/experiment-register.csv`（原因见第 4 节）。

## 1. 独立核对做了什么

| 检查 | 结果 |
|---|---|
| 证据包 SHA256SUMS（60 个文件，按原始字节） | 全部通过 |
| `ids.json` 与 `results/eval-frozen-20261120/val_ids.json` | 内容与 sha256 完全一致（`45440d72…`） |
| 两臂 meta：`selected_ids` 顺序、`ids_sha256`、`config_sha256`、health fingerprint | 全部与文件一致；160 个任务、repeats=1 |
| per-task 记录 | 每臂 160 条、任务 ID 与 `ids.json` 同序、无重复 |
| 两臂配对（按任务） | `check-parity` 通过：同模型、同 revision、同采样设置、同环境指纹，仅 adapter 不同 |
| 训练 `run.json` 的 train/val 哈希 | 与冻结划分的 `train.json`（`00e9a068…`）/ `val.json`（`2ac4dcb4…`）一致 |
| LoRA 重载 | 训练后与独立进程重载的 LoRA 张量哈希相同（`93760afb…`，112 个张量） |

各项原始数值见 `verification.json`。

## 2. 结果（160 个冻结 val 任务，每任务 1 条）

| 方法 | 满奖励成功 | 购买结束 | 成功率（Wilson 95%） | 平均终局奖励 | 平均步数 | 非法动作率 |
|---|---:|---:|---|---:|---:|---:|
| B0（基座） | 19/160 | 57 | 11.875%（7.74–17.80%） | 0.2389 | 7.62 | 47.99% |
| B1 本地 LoRA | 50/160 | 158 | 31.250%（24.58–38.80%） | 0.7059 | 3.48 | 3.23% |

按任务配对：成功率差 **+19.38 个百分点**，任务簇 bootstrap 95% 区间 **+13.11 至 +26.25 个百分点**；仅 B0 成功 3 项、仅 B1 成功 34 项、两边都成功 16 项。终局奖励差 +0.4671，区间 [0.4029, 0.5263]。

`mean_terminal_reward` 是任务奖励（未购买记 0），不是页面匹配分；页面上限分数已按 `terminal_reward` 归零。

## 3. 训练侧（摘要，不是购物表现）

- 1 epoch、656 步、训练样本 5,241（skipped 18）、验证样本 605、`eval_loss=0.4003`；
- 峰值 allocated 5.78 GiB、reserved 29.31 GiB（reserved 是分配器保留量，**不能**当物理显存需求或 OOM 保证）；
- 训练输入与冻结划分一致（见第 1 节哈希）；LoRA 重载一致。

## 4. 为什么暂不正式登记

登记表要求每行能追到"真实 run + config + commit + split SHA + fingerprint"。这批证据有三处不满足：

1. **采样 commit `dd9797e34aae0a7a0c050fcae154a41383b7c1a2` 不在仓库**（GitHub API 返回 422）。评估用的是 `main@21f3c83`，但 rollout 代码是本地 commit；
2. 两臂 meta **没有 `model_source`**（PR #14 尚未合并时产生）；
3. 执行配置与 `configs/eval-b0-frozen.json` 不同：`model` 写成本地快照路径、`environment_url` 是 8766、另有 `model_id` / `weights_sha256` 字段；采样参数、revision、fingerprint、任务列表都相同。

解除条件（任一即可）：

- 把 `dd9797e` 推到可访问分支，并在 README 说明它对应哪次采样；
- 或从 `main`（含 PR #14）重采 160 个任务，使用冻结配置（或同步重冻结并记录新哈希）。

满足后我可以直接用 `evaluate` 生成 summary 与登记行，并在 `results/summary.md`、`results/success-by-method.svg` 里出表和图。

## 5. 文件与复核

| 文件 | 内容 |
|---|---|
| `b0.per-task.jsonl` / `b1.per-task.jsonl` | 每任务：task_id、purchase、full_reward_success、terminal_reward、steps、逐步 grounding 布尔值（不含指令、页面文本、动作与原始响应） |
| `b0.config.json` / `b1.config.json` | 实际执行配置 |
| `b0.meta.json` / `b1.meta.json` | 原 metadata（含 run_id、config/ids 哈希、health 快照） |
| `ids.json` | 160 个任务 ID，顺序即采样顺序 |
| `verification.json` | 上述独立核对的机器可读结果 |

```bash
sha256sum results/eval-20261119-full160/ids.json     # 45440d72…（WSL/Linux 检出）
git cat-file blob HEAD:results/eval-20261119-full160/ids.json | sha256sum
```
