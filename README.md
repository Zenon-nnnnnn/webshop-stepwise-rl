# WebShop Stepwise RL：五人 LLM 课程项目

[![CPU checks](https://github.com/Zqqq-qq/webshop-stepwise-rl/actions/workflows/checks.yml/badge.svg)](https://github.com/Zqqq-qq/webshop-stepwise-rl/actions/workflows/checks.yml)

面向 **RTX 5060 Ti 16 GB / i5-13400F / 32 GB RAM** 的购物智能体训练与评估。基于 SPA-RL 的研究问题，先完成可信的 SFT 基线，再研究逐步奖励。项目计划日期：2026-11-22 proposal、2026-11-27 presentation、2026-12-15 report。课程 rubric 尚未提供，最终范围需与 rubric 对齐。

> **当前状态：项目代码与团队文档已建立，CPU 工具已验证；目标 GPU、完整 WebShop 环境、训练成功率及 PPO 尚未完成验证。没有已训练模型，没有可宣称的性能提升。**
>
> `upstream/` 是用户提供的 SPA 源码快照；`src/webshop_lab/` 是新增的课程工具。默认 Qwen2.5-1.5B 是资源受限适配实验，**不是原论文 Llama-3.2-3B 的精确复现**。线性进度估计器是新增低成本基线，不冒充论文神经估计器。

## 新组员从这里开始

**Tutor 计划：** [每天五人的任务与验收](docs/DAILY_PLAN.md) · [学习资源与动手练习](docs/LEARNING_GUIDE.md) · [每日报告模板](reports/daily-update-template.md)。按组内要求，以完成质量验收，不设每日工时上限。

1. 阅读 [项目定位与方法](docs/PROJECT.md)，知道研究问题及哪些结果尚未产生。
2. 按 [快速上手](docs/QUICKSTART.md) 安装 CPU 工具并运行测试，无需显卡或大数据。
3. GPU 机器负责人按 [安装指南](docs/SETUP.md) 配置两个隔离环境。
4. 数据负责人按 [数据指南](docs/DATA.md) 导入材料、审计并冻结划分。
5. 训练与评估负责人按 [实验手册](docs/EXPERIMENTS.md) 运行真实实验。
6. 评估负责人按 [统一评估](docs/EVALUATION.md) 核对分母、生成登记行、失败案例与图表。
7. 全组按 [分工和时间表](docs/TEAM.md)、[协作规则](CONTRIBUTING.md) 提交工作；到 [任务清单](https://github.com/Zqqq-qq/webshop-stepwise-rl/issues) 查看五人分工与各自交付任务。

## 先在任何电脑跑起来

```bash
git clone https://github.com/Zqqq-qq/webshop-stepwise-rl.git
cd webshop-stepwise-rl
python -m venv .venv-tools
# Linux / macOS / WSL
source .venv-tools/bin/activate
# Windows PowerShell 改用：.venv-tools\Scripts\Activate.ps1
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/check_repository.py
webshop-lab audit --output artifacts/upstream-audit.json
webshop-lab --help
```

Python 3.10+ 可运行这些标准库工具；**GPU 与旧环境统一使用 Python 3.10**。私有仓库需要已获访问权限的 GitHub 登录。组员邀请由仓库拥有者在 Settings → Collaborators 操作。

## 仓库地图

| 路径 | 内容 | 负责人入口 |
|---|---|---|
| `src/webshop_lab/` | 安全导入、数据审计、分组划分、SFT、环境桥、rollout、指标、奖励导出、线性进度基线 | 新实验优先使用 |
| `configs/` | 默认 1.5B / 可选 3B 的 SFT 与推理配置 | C / D |
| `requirements/` | GPU 与环境 CPU 依赖候选组合 | A |
| `upstream/` | SPA 原始可复用代码、FastChat、三个任务相关组件及演示轨迹 | 论文复现参考 |
| `docs/` | 安装、数据、架构、实验、评估、故障排查、代码审计、分工 | 所有人 |
| `reports/` | proposal 草案、pre 内容提纲、report 模板 | 所有人各写负责模块 |
| `results/` | 实验登记表、审计证据、经核实的结果；当前无性能结果 | E |
| `tests/` | 无 GPU 的有意义逻辑测试 | 所有人 |
| `artifacts/` | 本地模型、日志、诊断、轨迹；默认不入 Git | 每次运行 |
| `data/` | 本地 SFT 与划分；默认不入 Git | B |

## 可用命令与验证边界

| 命令 | 功能 | 验证边界 |
|---|---|---|
| `doctor` | GPU kernel、版本、资产检查 | CPU 路径可运行；GPU 待实机 |
| `audit` / `split` | 审计、ID 与购物指令分组、预算变体隔离 | 标准库逻辑与本地真实 SFT 已检查 |
| `import-assets` | 校验 ZIP 路径、空间、CRC 并解压 | 路径保护有测试；完整大包解压待目标机 |
| `env-server` | 单客户端本地 WebShop 文本环境桥 | 完整 Java / Lucene / spaCy 待实机 |
| `sft` | assistant-only loss、LoRA / QLoRA、保存日志和 adapter | GPU 待实机 |
| `rollout` / `summarize` | 真实任务交互与指标 | 指标逻辑有测试；真实 rollout 待实机 |
| `check-run` / `evaluate` | 完成度核对、购买/成功分开统计、失败案例与 SVG 图 | 合成 fixture 端到端测试；真实日志待实机 |
| `check-parity` / `check-register` | B0/B1 公平性核对、结果登记表校验 | 逻辑测试通过；无真实登记行 |
| `preflight` / `compare` / `results-table` | 跑前闸门、配对比较（McNemar+配对 bootstrap）、登记表转结果表 | 合成 fixture 测试；真实 run 待实机 |
| `progress-train` / `progress-predict` | 低成本可加线性估计器 | 合成轨迹端到端测试；未宣称真实效果 |
| `reward-export` | terminal / uniform / progress+grounding 奖励 | 逻辑测试通过；**不会执行 PPO** |
| `upstream/ppo/` | 论文来源中的 PPO 实现 | 旧依赖、轨迹策略一致性、50 系显卡兼容性待验证 |

详细 CLI、输入输出和命令顺序见 [实验手册](docs/EXPERIMENTS.md)。

## 已知问题先读

原包有 12 个 train/test 任务 ID 重叠；附带进度模型训练/验证数据相同；附带探索只有 78 条轨迹、26 个任务、1 条满奖励轨迹；PPO 示例仅 41 条。另有 `ckt/ckpt` 路径混用、前台 controller 阻塞、购买即成功等问题。原包还包含一处硬编码 API credential，已在上传前清空，原值不会进入仓库。

新增工具绕开部分旧入口并修正数据/指标流程；**没有把保留的所有上游脚本都改造成已验证版本**。完整细节见 [原包审计](docs/UPSTREAM_AUDIT.md) 和 [当前验证状态](docs/STATUS.md)。

## 大文件与引用

约 4 GB 的原始压缩包、约 9.3 GB 的解压资产、SFT 数据和模型权重不放入 Git。下载来源与本地导入方法见 [数据指南](docs/DATA.md)。原始教程视频未上传；本项目提供独立编写的说明。

- 原论文：[SPA-RL: Reinforcing LLM Agents via Stepwise Progress Attribution](https://arxiv.org/abs/2505.20732)
- 原始仓库：[WangHanLinHenry/SPA-RL-Agent](https://github.com/WangHanLinHenry/SPA-RL-Agent)
- 署名、许可与快照变更：[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
- 引用：[CITATION.cff](CITATION.cff)
