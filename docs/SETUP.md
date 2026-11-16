# 安装与 RTX 5060 Ti 实机检查

## 环境原则

使用两个 Python 3.10 环境：`webshop-gpu` 负责模型，`webshop-env` 负责旧版模拟器。两者通过 `127.0.0.1:8765` 通信，避免 spaCy / gym / pyserini 与现代 GPU 依赖混装。新入口不依赖 vLLM、DeepSpeed 或 FlashAttention。

依赖文件是选定的**候选版本组合**，尚未在你们的 5060 Ti 上验证。第一次成功运行后保存 `pip freeze`，作为团队实测锁定文件；不要把候选环境标为 verified。

## 系统准备

推荐 Linux；Windows 可使用 WSL2 Ubuntu。下面以 Ubuntu 22.04 / WSL2 为例，要求已有 Conda。安装 Conda 请使用其官方发行渠道。GPU 驱动由 NVIDIA 官方渠道安装；WSL 使用 Windows 主机驱动，不要在 WSL 中再装一套 Linux 显卡驱动。

先运行：

```bash
nvidia-smi
git --version
conda --version
```

记录显卡名称、显存和驱动版本。`nvidia-smi` 的 CUDA Version 是驱动能力，不代表当前 PyTorch 自带运行时版本。

预留至少 50 GB SSD 空间供两套环境、模型缓存、原始压缩包和约 9.3 GB 解压资产使用；若保留多个模型 checkpoint，需要更多空间。完整 WebShop 商品加载的 RAM 峰值尚未实测，32 GB 是否足够应在首次启动记录，避免环境与训练同时用尽内存。

## GPU 环境

```bash
conda create -n webshop-gpu python=3.10 -y
conda activate webshop-gpu
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/gpu.txt
python -m pip install -e .
webshop-lab doctor --output artifacts/doctor-gpu.json
python -m pip check
```

GPU 检查需要同时出现 `cuda_available=true`、正确显卡名称和 `cuda_kernel_smoke=passed`。只有 `import torch` 成功不够。PyTorch 官方已在 2.7 + CUDA 12.8 加入 Blackwell 支持，因此原包的 torch 2.5 / CUDA 12.1 思路不适合直接照搬。此处选定 2.8.0 cu128 作为兼容候选，而非建议无条件升级所有组件到最新版。

```bash
python -m pip freeze > artifacts/gpu-pip-freeze.txt
```

如模型需要访问许可，先在 Hugging Face 接受相应模型的访问条件，再用官方 CLI 登录。本仓库默认不需要付费模型 API；不要将 token 放入配置文件或 Git URL。

## WebShop CPU 环境

另开终端，从同一仓库根目录执行：

```bash
conda create -n webshop-env python=3.10 -y
conda activate webshop-env
conda install -c conda-forge openjdk=11 -y
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/environment.txt
python -m pip install -e upstream/envs/webshop --no-deps
python -m pip install -e .
python -m spacy download en_core_web_lg
java -version
python -m pip check
```

`--no-deps` 是为了使用本仓库明确列出的候选环境依赖，不让上游 `setup.py` 额外装入未使用的 `env` 包等依赖。导入环境模块仍会加载 selenium，但文本模式不需要 ChromeDriver。原包中随附的过时平台二进制没有上传。

按 [DATA](DATA.md) 导入数据后，再执行：

```bash
webshop-lab env-server
```

启动可能较慢：加载商品、spaCy、JVM 与检索索引。另开终端检查：

```bash
curl http://127.0.0.1:8765/health
```

没有 curl 可用 Python：

```bash
python -c "import json,urllib.request; print(json.load(urllib.request.urlopen('http://127.0.0.1:8765/health')))"
```

`ready=true` 不代表划分已验证；完成 DATA 中的 goal-map 划分后，`split_status.verified` 也必须为 true。环境服务是单状态、单客户端；不要同时开两个 rollout 作业。

## 锁定环境与验收

A 负责人保存两套 `doctor`、`pip freeze`、驱动信息、服务 health、启动时长和内存峰值。B 负责人核对任务指令映射。C 负责人跑 2-step SFT 并确认 adapter 可再次加载。

这些证据完成前，不在 README 或 proposal 中写“已复现成功”。故障处理见 [TROUBLESHOOTING](TROUBLESHOOTING.md)。

参考：[PyTorch Blackwell 支持](https://pytorch.org/blog/pytorch-2-7/)、[PyTorch 历史版本](https://pytorch.org/get-started/previous-versions/)、[PEFT 量化训练](https://huggingface.co/docs/peft/developer_guides/quantization)。
