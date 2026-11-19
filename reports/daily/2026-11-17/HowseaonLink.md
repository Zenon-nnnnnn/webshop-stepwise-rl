# 每日交付：2026-11-17 / HowseaonLink

## 今天要解决的问题

对应 DAILY_PLAN 11/17 行「安装隔离环境；交 doctor、版本锁定和 kernel 证据」，以及 Issue #1 验收第 1、2 条。拆成三个具体问题：

1. 主算力机器是 Windows 11（build 26200.9974），`docs/SETUP.md` 以 Ubuntu 22.04 / WSL2 为例，需要先确定运行位置并把 WSL2 搭起来。
2. 两套 Python 3.10 环境能否按 `requirements/` 的候选版本装成，且互不污染。
3. RTX 5060 Ti 是 Blackwell（sm_120），原包 torch 2.5 / CUDA 12.1 的思路不适用。要验证 torch 2.8.0 cu128 能真正在这块卡上启动 kernel，而不只是 `import torch` 成功。

## 学到并能解释的内容

**为什么必须两套环境（L1 / SETUP.md「环境原则」）。** `webshop-env` 要 spaCy 3.6.1、gym 0.24.0、pyserini 0.17.0 这批 2022 年的老包，`webshop-gpu` 要 transformers 4.56.2、peft 0.17.1、bitsandbytes 0.48.2。装在一起会互相拉版本。隔离是否真的生效，可以直接在 doctor 输出里读到：GPU 环境的 `gym`/`pyserini`/`spacy`/`java` 全为 `null`，env 环境的 `torch_cuda_build` 为 `null`。两者不是同一个进程，靠 `127.0.0.1:8765` 的 HTTP 桥通信，桥在 `src/webshop_lab/environment.py`，只用标准库，所以跨平台。

**`cuda_kernel_smoke` 和「import 成功」差在哪。** `src/webshop_lab/doctor.py` 第 32–35 行在 cuda 上建一个 32×32 张量、做矩阵乘、把标量取回主机再 `synchronize()`。`import torch` 成功只说明包装上了；这一步才说明 wheel 里带着能在本机这块卡上执行的 kernel。这也是 `docs/SETUP.md` 强调「只有 import torch 成功不够」的原因。

**sm_120 不是靠兼容兜底。** `torch.cuda.get_arch_list()` 返回 `['sm_70','sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']`，本机算力 12.0 在列表里，说明是原生编译目标而非 PTX 向前兼容。

## 实际完成与证据

- **Commit / PR / Issue**：Issue #1；分支 `env/setup-howseaonlink`；PR #10（`env/setup-howseaonlink` → `main`，评审人 Jun040504、Zqqq-qq）。
- **环境 / 模型 / 配置 / 输入与划分 SHA**：
  - 主机 Windows 11 build 26200.9974；WSL 3.0.1.0，Linux 内核 6.18.40.1-1，默认版本 2；发行版 Ubuntu 22.04.5 LTS。
  - Ubuntu rootfs 来源 `https://mirrors.ustc.edu.cn/ubuntu-cloud-images/wsl/jammy/current/ubuntu-jammy-wsl-amd64-ubuntu22.04lts.rootfs.tar.gz`，341,130,963 字节，SHA-256 `1483cc5c1dce13064f774834cbffdff226559fd522a67a381a8ea77d63fb4109`，gzip 完整性校验通过，用 `wsl --import` 导入。
  - spaCy 模型 `en_core_web_lg-3.6.0-py3-none-any.whl`，587,689,479 字节，SHA-256 `9012584a8fcf5bc61bedb13a4b511fb81ffff9f770958ee06366671be10632e9`，zip 内全部条目 CRC 通过；字节数与 GitHub 官方 release 资产一致（该 release 未发布 digest 字段），并已在 spaCy 中实际加载验证。
  - 显卡 NVIDIA GeForce RTX 5060 Ti，16311 MiB，驱动 610.88，`compute_cap` 12.0。
  - **未训练任何模型，未执行 PPO，无 split hash、无 goal fingerprint**：三份数据仍缺失，doctor 中六个数据资产全为 `false`。
- **执行命令**（均在 WSL Ubuntu 内、仓库根目录执行）：

  ```bash
  # GPU 环境
  conda create -n webshop-gpu python=3.10 -y
  ~/miniconda3/envs/webshop-gpu/bin/python -m pip install torch==2.8.0 \
      --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu128
  ~/miniconda3/envs/webshop-gpu/bin/python -m pip install -r requirements/gpu.txt
  ~/miniconda3/envs/webshop-gpu/bin/python -m pip install -e .
  ~/miniconda3/envs/webshop-gpu/bin/python -m webshop_lab.cli doctor --output artifacts/doctor-gpu.json
  ~/miniconda3/envs/webshop-gpu/bin/python -m pip freeze > artifacts/gpu-pip-freeze.txt

  # 环境侧
  conda create -n webshop-env python=3.10 -y
  conda install -n webshop-env -c conda-forge openjdk=11 -y
  ~/miniconda3/envs/webshop-env/bin/python -m pip install pyjnius==1.7.0
  ~/miniconda3/envs/webshop-env/bin/python -m pip install torch==2.8.0 \
      --index-url https://download.pytorch.org/whl/cpu
  ~/miniconda3/envs/webshop-env/bin/python -m pip install -r requirements/environment.txt
  ~/miniconda3/envs/webshop-env/bin/python -m pip install faiss-cpu==1.7.4
  ~/miniconda3/envs/webshop-env/bin/python -m pip install -e upstream/envs/webshop --no-deps
  ~/miniconda3/envs/webshop-env/bin/python -m pip install -e .
  ~/miniconda3/envs/webshop-env/bin/python -m pip install /root/en_core_web_lg-3.6.0-py3-none-any.whl

  # JVM 环境变量：conda 版 OpenJDK 的布局与 pyjnius 默认猜测不符，必须显式设置。
  # 每次新开 shell 都要重设，或写进 conda 环境的 activate 脚本；doctor 只跑 java -version，
  # 不启动 PyJNIus 的 JVM，所以必须另外做 LuceneSearcher 导入验证。
  export JAVA_HOME=~/miniconda3/envs/webshop-env/lib/jvm
  export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"
  export PATH="$JAVA_HOME/bin:$PATH"
  java -version
  ~/miniconda3/envs/webshop-env/bin/python -c \
      "from pyserini.search.lucene import LuceneSearcher; print('LuceneSearcher import OK')"
  ~/miniconda3/envs/webshop-env/bin/python -c \
      "import spacy; n=spacy.load('en_core_web_lg'); print(n.meta['name'], n.meta['version'], n('rubber sole shoes').vector.shape)"
  ~/miniconda3/envs/webshop-env/bin/python -m webshop_lab.cli doctor --output artifacts/doctor-env.json

  # 提交前验证
  PYTHONPATH=src ~/miniconda3/envs/webshop-gpu/bin/python -m unittest discover -s tests
  ~/miniconda3/envs/webshop-gpu/bin/python -m compileall -q src scripts
  ~/miniconda3/envs/webshop-gpu/bin/python scripts/check_repository.py
  PYTHONPATH=src ~/miniconda3/envs/webshop-gpu/bin/python -m webshop_lab.cli audit --output artifacts/pr-audit.json
  ```

- **日志与产物位置**：原始产物在 `artifacts/`（被 `.gitignore` 忽略，不入库）。**供评审人核对的原文已提交到 [`results/env-2026-11-17/`](../../../results/env-2026-11-17/README.md)**，含两份 doctor JSON、两份 pip freeze（44 / 111 项）、模型 manifest 与加载 smoke、`/health`、对齐结果、冻结划分 manifest、三份 ID 列表、以及首批真实 rollout 的 jsonl / meta / config / 指标，附 README 注明被测源码 commit、运行环境、数据来源校验与口径边界。该目录按 `docs/DATA.md`「三份 ID 列表、去标识 split manifest、环境 fingerprint 放入 `results/` 供 review」放置。仓库克隆在 WSL 原生目录 `~/webshop-stepwise-rl`，未放 `/mnt/c`，避免跨文件系统 I/O 影响后续训练。
- **检查结果与尚未验证部分**：

  | 检查项 | 结果 |
  |---|---|
  | `cuda_available` | `true` |
  | `gpu` | NVIDIA GeForce RTX 5060 Ti |
  | `compute_capability` | `[12, 0]` |
  | `vram_gb` | 15.93 |
  | `torch_cuda_build` | 12.8 |
  | **`cuda_kernel_smoke`** | **`passed`** |
  | bitsandbytes 0.48.2 `Linear4bit` 前向（sm_120） | 通过，输出 `(4, 64)` fp16，已 `synchronize` |
  | `torch.cuda.get_arch_list()` | 含 `sm_120` |
  | GPU 环境版本 | torch 2.8.0+cu128 / transformers 4.56.2 / peft 0.17.1 / accelerate 1.10.1 / bitsandbytes 0.48.2，与 `requirements/gpu.txt` 钉版一致 |
  | env 环境版本 | torch 2.8.0+cpu / OpenJDK 11.0.30 / pyserini 0.17.0 / pyjnius 1.7.0 / spacy 3.6.1 / gym 0.24.0 / numpy 1.26.4 / scikit-learn 1.2.2 |
  | `pyserini.search.lucene.LuceneSearcher` 导入 | 通过（JVM 实际启动） |
  | `en_core_web_lg` 加载 | 通过，`core_web_lg 3.6.0`、`lang=en`、300 维向量、六段流水线齐全 |
  | unittest | `Ran 8 tests` / `OK` |
  | compileall | 通过 |
  | `check_repository.py` | 文档相对链接与 credential 模式检查通过 |
  | `pip check`（GPU 环境） | No broken requirements found |
  | `audit` | 原始 train/test 任务 ID 重叠 12 个，与 `results/material-audit.json` 一致 |

  **当日后续已完成（数据到位后追加，证据同在 `results/env-2026-11-17/`）**：三份材料经 `scripts/verify_dataset_assets.py` 全部校验 OK 并 `import-assets` 导入；完整 WebShop `env-server` 启动成功，`/health` 返回 `ready=true`、`n_goals=12087`、`alignment_checked=1624`、`alignment_mismatches=0`、`split_status.verified=true`、三组指令交叉全 0；`split --goal-map` 重划得 train 1451 / val 160 / test 200、排除 13，与离线预检查一致；启动耗时与 RAM 峰值两次实测见下文观察四；3 个 val 任务（16/75/104）的真实 rollout 已产出完整 jsonl 与 meta。**Issue #1 第 1–4 条据此完成。**

  **已验证的证据等级**：CUDA kernel 真实执行、`Linear4bit` 前向、组件导入与 JVM 启动、模型 4bit 加载与单条贪心生成、环境服务启动与指令对齐、单条串行 rollout。**这些都不等于训练就绪。**

  **尚未验证**：SFT 的反向传播、优化器步、adapter 保存与重载、量化训练、批量 rollout、B1 臂与 `check-parity` 比对、任何 RL/PPO 更新、以及任何模型性能结论。此段记录 11/17 当时的证据范围。11/19 补记见文末。

## 观察、错误与判断

**一、`requirements/environment.txt` 缺 `faiss-cpu`（建议补）。** 预期：按该文件装完即可导入 pyserini。实际：`from pyserini.search.lucene import LuceneSearcher` 抛 `ModuleNotFoundError: No module named 'faiss'`，调用链是 `pyserini/search/lucene/_impact_searcher.py:28` → `pyserini/encode/__init__.py:17` → `pyserini/encode/_base.py:19` 的 `import faiss`。上游 `upstream/envs/webshop/setup.py` 里 `faiss-cpu==1.7.4` 那行是被注释掉的，而本仓库用 `--no-deps` 安装上游包，所以这条依赖没有任何来源。补装 `faiss-cpu==1.7.4` 后导入通过。判断：这是候选依赖清单的缺口，不是本机配置问题，任何人按文档装都会撞到。

**二、`pyjnius` 钉版：可复现建议成立，但我原先给的故障原因是错的（已收回）。** 我最初写的是「pyserini 只写 `>=1.4.0`，pip 会解析到 1.9.0，而 1.9.0 没有 cp310 wheel，于是转源码编译」。Zqqq-qq 在 PR #10 评审中指出这与官方元数据不符，我复算后确认**他是对的**：PyPI 上 1.9.0 的 `requires_python` 为 `>=3.11`，因此使用官方元数据的 Python 3.10 pip **本来就会排除 1.9.0**，不会选中它，也就谈不上转源码编译。我只查了 wheel 标签、漏查了 `requires_python`，据此做出的推断不成立。

需要分清两件事：

- **已证实的**：本机 Python 3.10.22 上 `pyjnius==1.7.0` 安装成功、`LuceneSearcher` 导入成功、JVM 实际启动。cp310 的预编译包在 1.4.1–1.7.0 区间存在（Linux 与 Windows 均有），1.8.0 只提供 Android wheel。
- **未证实的**：我**从未观察到**不钉版时的解析结果或编译失败——1.7.0 是我**预防性钉的**，不是故障后修的。所以「不钉版就会失败」这个因果我没有证据。

**修正后的建议**：文档可以把 `pyjnius==1.7.0` 记为**本机已验证的可用版本**（可复现钉版建议），但不应写成「已证实的故障原因」。若要把不钉版的行为定论，需要补当时实际的解释器版本、pip 与 index 来源、以及脱敏错误日志；使用镜像元数据或特殊参数时行为可能与官方元数据不同，这一点我尚未验证。

**三、conda 版 OpenJDK 的 `JAVA_HOME` 布局与 pyjnius 默认猜测不符。** 预期：`JAVA_HOME` 指向环境根目录即可。实际：pyjnius 报 `Unable to find libjvm.so`，它试的三条路径都不对。真实位置是 `~/miniconda3/envs/webshop-env/lib/jvm/lib/server/libjvm.so`（`bin/java` 是指向 `../lib/jvm/bin/java` 的符号链接）。解法是 `JAVA_HOME` 指向 `.../webshop-env/lib/jvm`，并显式给 `JVM_PATH` 指向该 `libjvm.so`。判断：`docs/TROUBLESHOOTING.md` 的 java/pyserini 段可以补这一条。

**四、WSL 默认只分配一半主机内存，实测证明不足以让环境与训练并存。** 主机 32 GB，WSL 默认 `free` 显示 15 Gi。`docs/SETUP.md` 提醒的「完整 WebShop 商品加载的 RAM 峰值尚未实测，32 GB 是否足够应在首次启动记录」，当日已两次实测：

| | 第一次（默认 15 GiB） | 第二次（`.wslconfig` 设 `memory=24GB`、`swap=8GB` 后，`free` 显示 23 GiB） |
|---|---|---|
| 启动至 `/health` 就绪 | 100 秒 | 110 秒 |
| 进程 RSS 峰值 | 15,640,200 KB ≈ **14.92 GiB** | 18,819,232 KB ≈ **17.95 GiB** |
| 就绪后可用内存 | **1.3 GiB** | **8.3 GiB** |

第一次峰值已吃掉配额的 99%，余量不足 1%，无法再容纳训练进程；提高配额后峰值升到 17.95 GiB，说明**原 15 GiB 确实不足**，不是我在夸大风险。两次 `goal_fingerprint` 逐字符相同（`e69c9ae2…5638d8`），环境状态可复现。判断：主机 32 GB 够用，但**WSL 默认配额是真正的墙**；建议把这条与 `.wslconfig` 写法补进 `docs/SETUP.md`，其他 Windows 组员会撞到同样的问题。不要把这个现象直接归因为商品数据太大。

**五、`wsl --install` 在本机网络下不可用，改用 winget 商店源。** 预期：`wsl --install` 一条命令装好。实际：连续三次卡死无输出。抓进程连接发现 `wsl.exe` 连的是 `185.199.110.133`（GitHub），即新版 `wsl --install` 从 GitHub releases 取包，本机该方向实测 33 KB/s。另：仅用 DISM 启用 `Microsoft-Windows-Subsystem-Linux` 功能（返回 3010，重启后生效）**不足以**得到可用 WSL——重启后 `wsl.exe` 仍是 274 KB 占位程序，`C:\Program Files\WSL` 不存在，说明该 build 上运行时必须来自商店包。最终 `winget install --id 9P9TQF7MRM4R --source msstore` 数分钟内装成 WSL 3.0.1.0，自带内核，无需另装 kernel MSI。Ubuntu 则用中科大镜像的官方 rootfs 离线 `wsl --import`，绕开国际带宽。判断：这条路径对组内其他 Windows 机器可复用。

**六、`src/webshop_lab/model.py` 的 `load()` 不支持 revision，钉版与离线工作流会断。** 预期：按 CONTRIBUTING「正式结果建议记录 HuggingFace 具体 model revision」钉住 revision 下载后即可加载。实际：跑真实 rollout 时 `load()` 第 17 行 `AutoTokenizer.from_pretrained(name, trust_remote_code=False)` 抛 `OSError: We couldn't connect to 'https://hf-mirror.com' to load the files, and couldn't find them in the cached files.`。原因：`load()` 第 15–25 行只读 `config['model']`，**不读也不透传 `revision`**；用 `snapshot_download(revision=<commit>)` 下载后缓存只有该 commit 的 snapshot、没有 `refs/main`，而 `load()` 只解析默认分支。最小日志即上述报错，触发位置 `model.py:17`。

绕过办法：把缓存的 `refs/main` 写成已校验的 commit sha（`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`），**未改仓库代码**，rollout 随后跑通。判断：这有两层影响——一是钉版/离线路径直接断；二是即使联网，`load()` 每次解析上游 `main` 的当前状态，上游更新后两次实验权重可能不同而配置里仍是同一模型名，与 CONTRIBUTING 记录实际 revision 的意图冲突。`configs/sft-qwen15b.json` 与 `eval-base.json` 目前都没有 `revision` 字段，故 SFT 与 rollout 两条路径都受影响。建议给 `load()` 加 `config.get('revision')` 并透传给 tokenizer 与 model 两处、在 meta 落盘实际 revision；属 Issue #3 模块，已在该 issue 提出，由 Jun040504 决定。

**七、我先前两条实现提议被证伪，已收回。** 我曾据自己手写的加载 smoke 出现的警告，提议给 `rollout.py`/`sft.py` 补 `attention_mask`、并把 `generation_config` 采样参数当作问题。QiuQiu-xuan 在 Issue #3 核实并回复：`rollout.py:82` 已显式传 `attention_mask=torch.ones_like(input_ids)`，`sft.py:43` 存 mask、`54–55` 用 0 补齐，**仓库代码本来就是对的**；那条警告来自我自己的 smoke 调用未传 mask。贪心下 `generation_config` 的 `temperature/top_p/top_k` 失效是设计如此，她已把 `top_k` 加进 `check-parity` 并在 `docs/EVALUATION.md` §3 写清口径。我复核过这些行号，确认她是对的，两条提议撤回。教训：**自己脚本产生的警告不等于仓库缺陷**，提意见前应先读实现。她指出的真正潜在风险是 rollout 若改批量，`ones_like` 会失效，需改用 `input_ids != tok.pad_token_id` 构造 mask。

**下载源实测速率**（同一时段、同一机器，供后续排期参考）：中科大 ubuntu-cloud-images 1.65 MB/s；上交 pytorch-wheels 16.9 MB/s；阿里 PyPI 约 0.5–0.75 MB/s；`download.pytorch.org` 7.3 MB/s；GitHub releases 直连 33 KB/s。pip 同时挂 `--index-url` 与 `--extra-index-url` 时会在两源间挑到慢的那个，本次 cudnn 706 MB 因此掉到 828 KB/s；改为单源后恢复正常。

## 交接

- **给 Jun040504（Issue #3）**：`webshop-gpu` 可用，模型 4bit 加载与单条贪心生成已实测通过，真实环境与冻结划分就绪，**串行 rollout 路径已跑通**。但**这不等于 2-step SFT 就绪**——反向传播、优化器步、adapter 保存与重载都还没验过。以上为 11/17 交接时的状态，11/19 实测见文末。请注意上面第一条依赖缺口（`faiss-cpu`）与第六条 revision 缺口；第二条 pyjnius 请按修正后的口径理解（钉版是建议，不是已证实故障）。另：`configs/eval-sft.json` 指向的 adapter 目录仍不存在，QiuQiu 的 `preflight` 会因此报 `ready=false`，这是预期行为。
- **给 Zqqq-qq（Issue #2 / #4）**：三份材料已导入并校验，完整 WebShop 已启动，goal-map 对齐与重划已完成，`alignment_mismatches=0`、`split_status.verified=true`。冻结划分的关键值与全部原文已提交到 `results/env-2026-11-17/`，并在 Issue #2 请你**正式确认冻结**（按 `docs/DATA.md` 这是 B 负责人的决定）。`goal_fingerprint` 已同步给 QiuQiu-xuan。
- **给 QiuQiu-xuan（Issue #5）**：你 11/19 的「人工复核 3 条真实轨迹」**阻塞已解除**——首批真实 rollout 日志（3 个 val 任务、含逐 step observation/available/action/grounding/error/reward）与你要的 meta、config、ids 已入库，`goal_fingerprint` 可直接填评估配置。详见 Issue #5 我的评论。B1 臂仍等 Jun 的配置，故 `check-parity` 暂无法比对。
- **验收条件**：Issue #1 第 1–4 条以本文件表格与 `results/env-2026-11-17/` 的原文为证据；第 5 条的工程短训证据见文末 11/19 补记。
- **阻塞时的最小检查**：SFT 配置到位后先跑 `webshop-lab preflight`（QiuQiu 提供）确认配置与 adapter 路径，再用独立输出目录做 2-step smoke，检查 loss 有限、assistant-only mask 与 skipped 数量、adapter 可重载；不要跳过 `alignment_mismatches=0` 与 `split_status.verified=true` 直接产出成绩。

## 下一个学习问题

`max_steps` 截断一个 episode 时，GAE 该不该 bootstrap？`src/webshop_lab/rollout.py` 第 109–110 行已经把 `done` 与 `termination`（`purchase` / `max_steps`）分开记了，所以「真正终止」和「只是步数用完」在日志里是可区分的。我不清楚的是下游：算 advantage 时该不该对 `termination=max_steps` 的末尾用价值函数 bootstrap，而对 `purchase` 的末尾按真终止处理（不 bootstrap）？如果两者都当成真终止，截断的 episode 会被系统性低估。这直接影响 12/01 策略更新验收里 advantage 的正确性，也是 Zqqq-qq 在 Issue #4 要接的 RL 接口的一部分。

## 11/19 补记

[PR #12](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/12#issuecomment-7008040535) 的工程短训已在目标机执行，两步更新、保存 adapter 和新进程重载均有原始证据。[PR #14](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/14#issuecomment-7007874484) 另有固定版本离线加载、长样本两步压测及新进程生成补测。正式配置与数据冻结仍待负责人确认。本日报的三任务 B0 测试没有加载这些 adapter。
