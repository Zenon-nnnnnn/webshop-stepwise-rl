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

- **Commit / PR / Issue**：Issue #1；分支 `env/setup-howseaonlink`；PR 编号创建后补记于本行。
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
  ~/miniconda3/envs/webshop-env/bin/python -m webshop_lab.cli doctor --output artifacts/doctor-env.json

  # 提交前验证
  PYTHONPATH=src ~/miniconda3/envs/webshop-gpu/bin/python -m unittest discover -s tests
  ~/miniconda3/envs/webshop-gpu/bin/python -m compileall -q src scripts
  ~/miniconda3/envs/webshop-gpu/bin/python scripts/check_repository.py
  PYTHONPATH=src ~/miniconda3/envs/webshop-gpu/bin/python -m webshop_lab.cli audit --output artifacts/pr-audit.json
  ```

- **日志与产物位置**：`artifacts/doctor-gpu.json`、`artifacts/doctor-env.json`、`artifacts/gpu-pip-freeze.txt`（44 个包）、`artifacts/env-pip-freeze.txt`（111 个包）、`artifacts/pr-audit.json`。`artifacts/` 被 `.gitignore` 忽略，按 CONTRIBUTING「原始日志与权重放组内共享存储」不入库；本文件只记脱敏读数。仓库克隆在 WSL 原生目录 `~/webshop-stepwise-rl`，未放 `/mnt/c`，避免跨文件系统 I/O 影响后续训练。
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

  **尚未验证**：完整 WebShop 服务启动（`env-server`）、`/health` 的 `alignment_mismatches` 与 `split_status.verified`、真实 rollout、2-step SFT 与 adapter 重载、量化训练、RAM 峰值与启动耗时。全部阻塞于三份数据缺失，不是环境问题。Issue #1 验收第 3、4、5 条因此仍未完成。

## 观察、错误与判断

**一、`requirements/environment.txt` 缺 `faiss-cpu`（建议补）。** 预期：按该文件装完即可导入 pyserini。实际：`from pyserini.search.lucene import LuceneSearcher` 抛 `ModuleNotFoundError: No module named 'faiss'`，调用链是 `pyserini/search/lucene/_impact_searcher.py:28` → `pyserini/encode/__init__.py:17` → `pyserini/encode/_base.py:19` 的 `import faiss`。上游 `upstream/envs/webshop/setup.py` 里 `faiss-cpu==1.7.4` 那行是被注释掉的，而本仓库用 `--no-deps` 安装上游包，所以这条依赖没有任何来源。补装 `faiss-cpu==1.7.4` 后导入通过。判断：这是候选依赖清单的缺口，不是本机配置问题，任何人按文档装都会撞到。

**二、`pyjnius` 必须钉版本，且这不是 Windows 专属问题。** pyserini 0.17.0 的 requirements 只写 `pyjnius>=1.4.0`，pip 会解析到最新的 1.9.0，而 1.9.0 只提供 cp311–cp314 的 wheel，**没有 cp310**，于是转源码编译。查 PyPI 官方元数据：cp310 的预编译包在 1.4.1–1.7.0 区间存在（Linux 与 Windows 均有）。先装 `pyjnius==1.7.0` 再装 pyserini 即可满足 `>=1.4.0` 且不触发编译。判断：文档应显式钉这一条。

**三、conda 版 OpenJDK 的 `JAVA_HOME` 布局与 pyjnius 默认猜测不符。** 预期：`JAVA_HOME` 指向环境根目录即可。实际：pyjnius 报 `Unable to find libjvm.so`，它试的三条路径都不对。真实位置是 `~/miniconda3/envs/webshop-env/lib/jvm/lib/server/libjvm.so`（`bin/java` 是指向 `../lib/jvm/bin/java` 的符号链接）。解法是 `JAVA_HOME` 指向 `.../webshop-env/lib/jvm`，并显式给 `JVM_PATH` 指向该 `libjvm.so`。判断：`docs/TROUBLESHOOTING.md` 的 java/pyserini 段可以补这一条。

**四、WSL 默认只分配一半主机内存。** 主机 32 GB，WSL 内 `free -g` 显示 15 Gi。`docs/SETUP.md` 已提醒「完整 WebShop 商品加载的 RAM 峰值尚未实测，32 GB 是否足够应在首次启动记录」——在 WSL 下这个上限实际是 15 Gi，不是 32 GB。数据到位后首次启动必须记录峰值；若不足，用 Windows 侧 `.wslconfig` 调整，不要直接归因为商品数据太大。

**五、`wsl --install` 在本机网络下不可用，改用 winget 商店源。** 预期：`wsl --install` 一条命令装好。实际：连续三次卡死无输出。抓进程连接发现 `wsl.exe` 连的是 `185.199.110.133`（GitHub），即新版 `wsl --install` 从 GitHub releases 取包，本机该方向实测 33 KB/s。另：仅用 DISM 启用 `Microsoft-Windows-Subsystem-Linux` 功能（返回 3010，重启后生效）**不足以**得到可用 WSL——重启后 `wsl.exe` 仍是 274 KB 占位程序，`C:\Program Files\WSL` 不存在，说明该 build 上运行时必须来自商店包。最终 `winget install --id 9P9TQF7MRM4R --source msstore` 数分钟内装成 WSL 3.0.1.0，自带内核，无需另装 kernel MSI。Ubuntu 则用中科大镜像的官方 rootfs 离线 `wsl --import`，绕开国际带宽。判断：这条路径对组内其他 Windows 机器可复用。

**下载源实测速率**（同一时段、同一机器，供后续排期参考）：中科大 ubuntu-cloud-images 1.65 MB/s；上交 pytorch-wheels 16.9 MB/s；阿里 PyPI 约 0.5–0.75 MB/s；`download.pytorch.org` 7.3 MB/s；GitHub releases 直连 33 KB/s。pip 同时挂 `--index-url` 与 `--extra-index-url` 时会在两源间挑到慢的那个，本次 cudnn 706 MB 因此掉到 828 KB/s；改为单源后恢复正常。

## 交接

- **给 Jun040504（Issue #3）**：`webshop-gpu` 已可用，2-step SFT 的执行环境就绪。你冻结配置后我按配置执行。请注意上面第一、二条依赖缺口——如果你在自己机器上装 `webshop-env`，会撞到同样的 `faiss` 与 `pyjnius` 问题。另：`configs/eval-sft.json` 指向的 adapter 目录仍不存在，QiuQiu 的 `preflight` 会因此报 `ready=false`，这是预期行为。
- **给 Zqqq-qq（Issue #2 / #4）**：环境侧已就绪但 `env-server` 起不来，缺 `data.zip` 与 `indexes.zip`。你一旦提供 `webshop_sft.json`（见 PR #9 第 1 条，`results/material-audit.json` 与 `results/validation.md` 指向你 11/16 在 macOS 上跑审计时用的那份），我可以立即做离线 audit 与初步 split；两个环境资产包到位后才能做 goal-map 对齐与冻结。`goal_fingerprint` 产出后我会同步给 QiuQiu-xuan。
- **给 QiuQiu-xuan（Issue #5）**：你 11/19 的「人工复核 3 条真实轨迹」仍阻塞，依赖链是「环境资产包 → env-server → rollout 日志」，卡在第一环。我这边环境已不是瓶颈。
- **验收条件**：Issue #1 第 1、2 条以本文件表格与 `artifacts/doctor-*.json` 为证据；第 3、4、5 条待数据到位后另行提交。
- **阻塞时的最小检查**：数据到位后先跑 `webshop-lab import-assets --source <本地目录>` 核对 SHA 与 manifest，再 `env-server` + `curl /health`，确认 `ready=true` 后才做对齐；不要跳过 `alignment_mismatches=0` 直接 rollout。

## 下一个学习问题

`max_steps` 截断一个 episode 时，GAE 该不该 bootstrap？`src/webshop_lab/rollout.py` 第 109–110 行已经把 `done` 与 `termination`（`purchase` / `max_steps`）分开记了，所以「真正终止」和「只是步数用完」在日志里是可区分的。我不清楚的是下游：算 advantage 时该不该对 `termination=max_steps` 的末尾用价值函数 bootstrap，而对 `purchase` 的末尾按真终止处理（不 bootstrap）？如果两者都当成真终止，截断的 episode 会被系统性低估。这直接影响 12/01 策略更新验收里 advantage 的正确性，也是 Zqqq-qq 在 Issue #4 要接的 RL 接口的一部分。