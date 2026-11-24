# 快速上手：组员第一小时

## 你需要先知道什么

这是一个模拟购物 agent 的实验项目，不连接真实购物支付。输入是购物要求和网页文字，输出是 `Thought:` 与 `Action:`。执行器只接受 `search[...]` 或 `click[...]`，由模拟器返回状态及最终奖励。

普通组员无需同时安装 CUDA、Java、FastChat 和全部旧依赖。先完成本页的 CPU 检查；只有训练负责人和环境负责人需要 [SETUP](SETUP.md)。所有命令默认在仓库根目录执行。

## 1. 获取代码

GitHub 账号需要先获得私有仓库访问权。使用 GitHub Desktop 或 HTTPS clone 均可；不要把 token 写到 URL。

```bash
git clone https://github.com/Zqqq-qq/webshop-stepwise-rl.git
cd webshop-stepwise-rl
python -m venv .venv-tools
source .venv-tools/bin/activate
python -m pip install -e .
```

Windows PowerShell 激活改用 `.venv-tools\Scripts\Activate.ps1`；如果执行策略阻止激活，也可直接使用 `.venv-tools\Scripts\python.exe -m webshop_lab.cli --help`。

## 2. 跑检查与审计

```bash
python -m unittest discover -s tests -v
python scripts/check_repository.py
webshop-lab doctor --output artifacts/doctor-tools.json
webshop-lab audit --output artifacts/upstream-audit.json
```

测试应通过；`doctor` 报 CUDA 不可用在普通电脑属于正常情况。`audit` 结果包含 12 个重叠任务 ID。它们是**原材料审计结果**，不是训练效果。

## 3. 定位你的任务

- A：从 [安装指南](SETUP.md) 开始；交 GPU kernel 与环境 health 检查证据。
- B：从 [数据指南](DATA.md) 开始；交 SHA、划分和环境指令对齐证据。
- C：从 [实验手册 SFT](EXPERIMENTS.md#3-sft-训练) 开始；先跑 2 个更新步骤。
- D：读 [奖励与 PPO 边界](RL.md)，区分线性基线、原神经 SPA 和策略更新。
- E：读 [指标定义](EXPERIMENTS.md#5-指标与结果记录)，为每个实验检查分母、任务列表和日志。

## 4. 第一次提交

```bash
git switch -c docs/your-topic
# 修改你负责的文件
python -m unittest discover -s tests -v
python scripts/check_repository.py
git add docs/your-file.md
git commit -m "docs: clarify data preparation"
git push -u origin docs/your-topic
```

在 GitHub 建 PR，请另一位组员 review；不要直接将几 GB 数据 `git add .`。训练配置和结果一并进入 PR，模型权重仅共享存储。

## 完成标准

你能向另一位组员说明：数据如何来、原项目是什么、我们新增了什么、哪个实验还没跑。能在自己的电脑运行 tests 和 audit，并知道自己的验收条件。
