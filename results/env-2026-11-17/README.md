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
| `rollout-3val-summary.json` | 固定版评估器 `evaluate` 输出，包含完整性、终局购物奖励、页面匹配分和失败分类 |
| `rollout-3val-terminal-supplement.json` | 独立派生件：逐 episode 的页面分与 `terminal_reward`、`purchase_terminated`、任务级均值、episode 耗时合计，附定义、上游行号引用与源 JSONL 的 SHA-256 |
| `run-model-revision.json` | run_id 到模型 revision 的旁证件：repo、revision、权重 SHA-256、下载 endpoint、原 meta 的 SHA-256。原 meta 未修改，此件供结果登记行取用 |

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
| success_rate_wilson_95 | [0.0, 0.5615]（旧 `summarize` 输出下界为 `5.55e-17`，是 0.0 的浮点残差，见下文口径说明） |
| **purchase_terminated** | **0** |
| mean_final_reward（**页面匹配分，非任务奖励**） | 0.9167 |
| **mean_terminal_reward（任务级）** | **0.0** |
| mean_steps | 10.0 |
| invalid_action_rate | 0.2667 |
| grounding_accuracy | 0.7333 |
| termination_counts | `max_steps: 3` |
| **3 个 episode 耗时合计（JSONL `seconds` 字段之和）** | **52.39 秒** |

### `final_reward` 与 `terminal_reward` 的区别（重要，勿误读）

`rollout.py` 记录的 `final_reward` 是**环境最后一次返回的奖励**，而上游 `upstream/envs/webshop/src/webshop/web_agent_site/envs/web_agent_text_env.py` 在搜索页把 `self.reward` 设为 `search_max`（第 478 行）、在商品页设为 `maxr`（第 558 行），只有购买路径（第 630 行）才产生任务分，并经第 733 行作为 `status['reward']` 返回。**所以 `final_reward` 是页面匹配分，不是任务奖励。**

本次三个 episode 全部 `done=false`、`termination=max_steps`，即**从未点击 `buy now`**，但任务 75/104 的页面分已达 1.0，使 `mean_final_reward` 呈 0.9167。**这个数不能被读成「几乎做成了」**。按 `terminal_reward = reward if done else 0.0` 计算，三个 episode 的任务级奖励**全为 0.0**，`mean_terminal_reward = 0.0`，`purchase_terminated = 0`。

逐 episode 的两种奖励、耗时与推导定义见 [`rollout-3val-terminal-supplement.json`](rollout-3val-terminal-supplement.json)，其中记录了源 JSONL 的 SHA-256 与字节数以便追溯。**原始 `rollout-3val.jsonl` 与 `rollout-3val.meta.json` 未作任何修改**，以保持评审人已核对的哈希链；补充文件是独立派生件，不替代评估器输出。

### 耗时口径

**52.39 秒**是 JSONL 中三个 episode `seconds` 字段之和（12.866 + 20.571 + 18.953），即纯 episode 交互时间。先前口头/评论中提到的 **65 秒是外层 shell 两次 `date` 之间的墙钟差**，**另含模型加载与进程启动**，不是 episode 时间。两者不可混用；引用 episode 耗时请用 52.39 秒。

### 指标文件版本说明

`rollout-3val-summary.json` 由 [PR #8](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/8) 的固定审阅版本 `b19b87fdc402c4a3c3d5d32cfeddd71cd5b2e1d3` 重新生成。原始轨迹、元数据和配置保留原字节。评估器核对该三任务样本的完整性和溯源，输出购买 0、满分成功 0、平均终局购物奖励 0。区间下界的原始浮点值约为 0。摘要仅用于三任务工程验证，未写入实验登记表。

在该评估器源码快照中，将本目录原始输入放在同名相对路径，运行下面的命令。

```bash
PYTHONPATH=src python -m webshop_lab.cli evaluate \
  --jsonl results/env-2026-11-17/rollout-3val.jsonl \
  --meta results/env-2026-11-17/rollout-3val.meta.json \
  --config results/env-2026-11-17/rollout-base-3val.json \
  --output results/env-2026-11-17/rollout-3val-summary.json
```

### 失败模式

三个 episode 的 `done` 均为 false，即**均未点击 `buy now`**，尽管该值在每步 available 列表中。三者末段均重复同一动作（任务 16 `click[sea change]`、任务 75 `click[size: 4 fl. oz (pack of 3)]`、任务 104 `click[white]`），属 loop 类失败；任务 75 的动作因自行添加 `size:` 前缀且 `fl. oz` 与环境值 `fl oz` 不符而全部非法，同时属 grounding/format 类。评估负责人已用其 triage 独立复核，给出三条全 `loop`、任务 75 另加 `format`，与此处定性描述一致。

## 读表注意事项

- **两份 doctor JSON 里 `assets` 六项全为 `false`，那是导入前的状态快照**，不代表运行时资产缺失。doctor 分别在 `import-assets` 之前执行；导入后 `data/raw/webshop_sft.json`、`upstream/envs/webshop/data/items_*.json`、`upstream/envs/webshop/search_index/indexes/segments_1` 均已就位，环境启动与 `/health` 的 `alignment_checked=1624` 即为运行时资产可用的证据。
- **`doctor-env.json` 里 `transformers` 为 5.20.2，与 GPU 环境的 4.56.2 不同版**。它由 pyserini 0.17.0 的 `transformers>=4.6.0` 间接带入，CPU 环境不使用 transformers 做训练或推理，故无害；但**不要据此认为两套环境的钉版一致**。GPU 环境的钉版以 `gpu-pip-freeze.txt` 与 `requirements/gpu.txt` 为准。
- **`rollout-3val.meta.json` 只记 `torch_version`，不含模型 revision**，因为 `src/webshop_lab/model.py` 的 `load()` 不接受也不透传 revision（已在 Issue #3 提出）。revision 见 `run-model-revision.json` 与 `base-model-manifest.json`。原 meta 保持未修改，以维持评审人已核对的 `config_sha256` / `ids_sha256` 哈希链。

## 与 PR #11 的独立交叉验证

评估负责人比对后指出：**本目录的冻结划分与 PR #11（Zqqq-qq 在 macOS 上的离线候选划分）逐字节相同**。我已独立复核，用 `git hash-object` 计算本目录三份 ID 文件的 blob 哈希，与 PR #11 所提交的 blob SHA 逐个对照：

| 文件 | 本目录 blob SHA | PR #11 blob SHA | 一致 |
|---|---|---|---|
| `train_ids.json` | `448e74dcee7d771ef14e4c5ac6723c8d5791d582` | `448e74dcee7d771ef14e4c5ac6723c8d5791d582` | 是 |
| `val_ids.json` | `dbb1ec46ca488e47fe3ce0e550ad545a3a60cd78` | `dbb1ec46ca488e47fe3ce0e550ad545a3a60cd78` | 是 |
| `test_ids.json` | `a77927f148020866d25ab55c03a7e7d09a6cd0cf` | `a77927f148020866d25ab55c03a7e7d09a6cd0cf` | 是 |

含义有两层：一是**两台不同机器、不同操作系统（WSL2 Ubuntu 22.04 / macOS）、不同 Python 版本（3.10.22 / 3.13.16）独立跑出了逐字节一致的划分**，说明 seed=42 的分组划分是可复现的；二是**我做的 goal-map 重划没有改变任何成员归属**，即离线预检查的分组在真实环境指令核对后依然成立。

注意口径：PR #11 自身标注为 `provisional_offline_only`、不是最终冻结；本目录的划分是在真实环境 goal-map 对齐后重划并通过 `/health` 校验的版本。**逐字节相同是交叉验证结果，不等于两者验收等级相同**；正式冻结仍待 Issue #2 负责人确认。

## 边界

- **n=3，成功率 0 的 Wilson 上界为 0.5615**，只能表述为「基座模型在这 3 个 val 任务上未完成购买」，不得推广为模型能力结论。这是未经 SFT 的 B0 基线行为。
- **`mean_final_reward=0.9167` 是页面匹配分均值，不是任务奖励均值**；任务级 `mean_terminal_reward=0.0`、`purchase_terminated=0`。任何报告、图表或结果登记行引用奖励时必须标明是哪一种，不得用页面分暗示任务接近完成。
- 本目录仅覆盖 11/17 环境和三任务 rollout。后续 [PR #12](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/12#issuecomment-7008040535) 已单独交付两步 SFT 与独立重载证据。[PR #14](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/14#issuecomment-7007874484) 记录后续长样本容量和加载补测。正式 1 epoch SFT 仍待配置和划分确认。
- `rollout-3val.jsonl` 是**智能体自身生成的交互轨迹**，不是原始专家 SFT 对话，也不含 MTurk worker 信息。
- 本目录不含模型权重、原始数据集或完整大日志。