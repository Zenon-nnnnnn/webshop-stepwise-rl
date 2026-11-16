# 验证记录

日期：2026-11-16。执行机器：macOS，Python 3.13.16；不是目标 RTX 5060 Ti 主机。

## 已运行

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python scripts/check_repository.py
python -m compileall -q src scripts
PYTHONPATH=src python -m webshop_lab.cli audit --sft /path/to/webshop_sft.json --output results/material-audit.json
PYTHONPATH=src python -m webshop_lab.cli split --sft /path/to/webshop_sft.json --output /temporary/split-validation
```

8 个测试通过：动作解析/可执行性、ZIP 路径穿越拒绝、历史保留、同 ID/指令/预算变体分组、奖励公式与非有限值拒绝、购买与成功区分、线性进度训练/标注/奖励导出端到端以及泄漏拒绝。

文档本地相对链接与 credential 模式检查通过。新代码语法编译通过。真实 SFT 离线划分：1451 train、160 val、200 reserved test，排除 13 条；ID 和 train/val 底层指令哈希交叉均为 0。完整环境的额外指令对齐仍待完成。

## 未运行

完整大包解压、Java/Lucene/WebShop 启动、5060 Ti CUDA kernel、量化加载、SFT 训练、真实 rollout、原神经 SPA、PPO。无真实成功率或性能提升结论。候选版本的兼容性以目标机 smoke 为准。

## GitHub Actions

首次完整提交 `e105f0e84fc47fd9f3454d56cd5a77933df75df6` 的 [CPU checks](https://github.com/Zqqq-qq/webshop-stepwise-rl/actions/runs/43848833958) 已通过：Ubuntu / Python 3.10 上安装项目、8 项测试、语法编译、链接/credential 检查和审计均成功。线上检查仍不包含 GPU 或真实模型训练。

## 2026-11-16 统一评估工具本地检查

执行机器：Windows 11，Python 3.12.10；CPU only，无 GPU、无模型权重、无 WebShop 服务。以下是**工具验证**，不是模型结果。

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m webshop_lab.cli check-run --meta <meta> --jsonl <jsonl> --config <config>
PYTHONPATH=src python -m webshop_lab.cli evaluate --method B0 --meta <meta> --jsonl <jsonl> \
    --config <config> --split-ids <val_ids> --output <eval.json> --cases <cases.md> \
    --figures-dir <figures> --register-row <row.csv> --peak-vram-gb 7.5
PYTHONPATH=src python -m webshop_lab.cli check-parity --a-meta <b0.meta> --b-meta <b1.meta>
PYTHONPATH=src python -m webshop_lab.cli check-register --register <row.csv>
PYTHONPATH=src python -m webshop_lab.cli figures --register <two-row.csv> --output <methods.svg>
```

32 项测试通过（原有 8 项 + 新增 24 项）：完成度核对（完整/缺条/多余/重复/repeat 缺失/run_id 与 fingerprint 不一致/config 事后被改）、购买与成功分开、重复任务改用簇 bootstrap、五类失败分诊与 engineering、案例工作表、公平性（adapter 可不同，context 与任务集不同则拒绝）、登记表（表头、数值自洽、区间、config 哈希、planned 行可空）以及端到端 evaluate。`python -m compileall -q src scripts tests` 与 `python scripts/check_repository.py` 通过。

用 6 条**合成 episode**（非模型输出）跑通整条链路：`check-run` 判定 complete；B0 汇总购买结束 3、满奖励成功 1、`success_rate=0.166667`、Wilson `[0.030053, 0.563503]`；`--cases` 生成 5 类各 1 例的工作表；B0/B1 两个 meta 仅 adapter 不同时 `check-parity` 通过（同一任务顺序与 fingerprint），改 context 后拒绝；`--register-row` 生成的登记行通过 `check-register`；两行登记表生成 `methods.svg`。合成数字不得写入报告或 proposal。

真实 B0/B1 日志、真实案例与 test 集评估仍未执行；汇总口径与人工核对清单见 [统一评估](../docs/EVALUATION.md)。
