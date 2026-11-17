# 环境验收证据：2026-11-17 / HowseaonLink

本目录是 Issue #1 第 1–4 条验收的可核对原文，供评审人直接查阅。对应日报
[`reports/daily/2026-11-17/HowseaonLink.md`](../../reports/daily/2026-11-17/HowseaonLink.md) 与 PR #10。
按 `docs/DATA.md`「将三份 ID 列表、去标识的 split manifest、环境 fingerprint 放入本次实验的 `results/` 子目录供 review」放置。

## 被测源码与运行环境

- 本地 checkout 基于 main `9df3063321ccda2a3a96e1a5d7d9d720bdd296e0`。main 其后前进到 `eb76f4e`，该提交只修改 `README.md`、`docs/DATA.md`，新增 `docs/SHARED_DATA.md`、`data/shared-assets-manifest.json`、`results/dataset-release-verification.json`、`scripts/verify_dataset_assets.py`，**未触及 `src/`、`requirements/`、`configs/`**，故本目录证据所测代码与 `eb76f4e` 等价。
- 主机 Windows 11 build 26200.9974；WSL 3.0.1.0，Linux 内核 6.18.40.1-1，Ubuntu 22.04.5 LTS。
- GPU NVIDIA GeForce RTX 5060 Ti，16311 MiB，驱动 610.88，compute capability 12.0（sm_120）。
- WSL 内存配额：首次启动时为默认 15 GiB；随后经 `C:\Users\lhx\.wslconfig` 设为 `memory=24GB`、`swap=8GB`，重启后 `free` 显示 23 GiB。两次启动的读数分别记录于下。
- 两套 conda 环境均为 Python 3.10.22：`webshop-gpu`（CUDA）与 `webshop-env`（CPU 模拟器）。

## 文件清单

| 文件 | 内容 |
|---|---|
| `doctor-gpu.json` | GPU 环境 doctor 原文：`cuda_available`、`gpu`、`compute_capability`、`vram_gb`、`torch_cuda_build`、`cuda_kernel_smoke`、包版本、资产存在性 |
| `doctor-env.json` | 环境侧 doctor 原文：torch CPU 构建、pyserini/spacy/gym 版本、`java -version` 输出 |
| `gpu-pip-freeze.txt` | GPU 环境完整版本锁定（44 项） |
| `env-pip-freeze.txt` | 环境侧完整版本锁定（111 项） |
| `base-model-manifest.json` | 基础模型 repo、revision、逐文件字节数与 LFS SHA-256、下载 endpoint |
| `model-load-smoke.json` | 4bit NF4 加载与贪心生成实测：加载秒数、显存占用/峰值、token 吞吐、样本输出 |
| `health-frozen.json` | goal-map 重划后 `/health` 原文 |
| `environment-manifest.json` | `goal_fingerprint`、`n_goals`、seed |
| `environment-alignment.json` | 1624 条 SFT 与真实环境指令的逐条核对结果、预算变体 ID 列表 |
| `split-manifest.json` | 冻结划分清单：seed、`source_sha256`、`test_ids_sha256`、`goal_map_sha256`、行数、train/val 指令哈希 |
| `train_ids.json` / `val_ids.json` / `test_ids.json` | 三份任务 ID 列表（1451 / 160 / 200） |
| `rollout-base-3val.json` | 本次 rollout 实际使用的配置（`configs/eval-base.json` 副本，仅填入 `goal_fingerprint`） |
| `rollout-3val-ids.json` | 本次选用的 3 个 val 任务 ID |
| `rollout-3val.jsonl` | 3 个真实 episode 的逐步原文：observation、available、action、grounding、error、environment_reward |
| `rollout-3val.meta.json` | run_id、config 与 ids 的 SHA-256、selected_ids、repeats、health 快照、torch 版本、起始 UTC |
| `rollout-3val-summary.json` | `summarize` 输出的指标原文 |

`goal-instructions.json`（12087 条任务指令映射，1.97 MB）未入库，其 SHA-256 记于 `split-manifest.json` 的 `goal_map_sha256`；需要原文可向环境负责人索取。

## 数据来源与校验

三份原始材料来自 Zqqq-qq 共享的私有 Release `data-v1`，校验值见 `docs/SHARED_DATA.md`。

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `webshop_sft.json` | 9,312,587 | `721e46dba410cff900e187bccea91a456c38b105b6d14d068d7762afb7791de3` |
| `data.zip` | 1,242,271,168 | `9d96abd8c6626d6d4d9a9552cee70ef4a21a5c72b2edc3efde246bc8a510cb91` |
| `indexes.zip` | 2,819,730,135 | `9c6db284bef87d7ab3a108c978825c9ccd71578b1c41ba0dfa15a570ec6c3759` |

`data.zip` 与 `indexes.zip` 由环境负责人从原作者 Google Drive 链接独立下载，其整包 SHA-256 与 Release 附件 digest 一致；`indexes.zip` 另按 Release 的 1,500,000,000 / 1,319,730,135 字节切分点分别核算，两片哈希与 `indexes.zip.part01` / `part02` 的 digest 逐一相同。`scripts/verify_dataset_assets.py`（取自 main `eb76f4e`）对三份文件全部返回 OK。

## 关键读数

**环境启动（两次）**

| | 第一次（WSL 15 GiB） | 第二次（WSL 23 GiB） |
|---|---|---|
| 启动至 `/health` 就绪 | 100 秒 | 110 秒 |
| 进程 RSS 峰值 | 15,640,200 KB ≈ 14.92 GiB | 18,819,232 KB ≈ 17.95 GiB |
| 就绪后可用内存 | 1.3 GiB | 8.3 GiB |

第一次峰值已接近 15 GiB 上限、余量不足 1%，无法与训练进程并存；提高配额后第二次峰值升到 17.95 GiB，说明原配额确实不足。两次 `goal_fingerprint` 完全相同，环境状态可复现。

**冻结划分**：seed 42，train 1451 / val 160 / test 200，排除 13 行；`alignment_checked=1624`、`alignment_mismatches=0`、`split_status.verified=true`、三组指令交叉均为 0。1624 条中 1012 条仅预算不同（原环境预算随机生成），已单独记录于 `environment-alignment.json`。

**真实 rollout（n=3，val 任务 16 / 75 / 104，repeats=1）**

| 指标 | 值 |
|---|---|
| episodes / unique_tasks | 3 / 3 |
| successes / success_rate | 0 / 0.0 |
| success_rate_wilson_95 | [0.0, 0.5615] |
| mean_final_reward | 0.9167 |
| mean_steps | 10.0 |
| invalid_action_rate | 0.2667 |
| grounding_accuracy | 0.7333 |
| termination_counts | `max_steps: 3` |
| 3 个 episode 总耗时 | 65 秒 |

`final_reward` 是环境逐步返回的累计属性匹配分，购买并非其前提；三个 episode 的 `done` 均为 false，即**均未点击 `buy now`**，尽管该值在每步 available 列表中。三者末段均重复同一动作（任务 16 `click[sea change]`、任务 75 `click[size: 4 fl. oz (pack of 3)]`、任务 104 `click[white]`），属 loop 类失败；任务 75 的动作因自行添加 `size:` 前缀且 `fl. oz` 与环境值 `fl oz` 不符而全部非法，同时属 grounding/format 类。

## 边界

- **n=3，成功率 0 的 Wilson 上界为 0.5615**，只能表述为「基座模型在这 3 个 val 任务上未完成购买」，不得推广为模型能力结论。这是未经 SFT 的 B0 基线行为。
- 未执行 SFT、未执行 PPO、无任何训练后性能结果。Issue #1 第 5 条（2-step SFT 与 adapter 重载）仍待 Issue #3 的冻结配置。
- `rollout-3val.jsonl` 是**智能体自身生成的交互轨迹**，不是原始专家 SFT 对话，也不含 MTurk worker 信息。
- 本目录不含模型权重、原始数据集或完整大日志。