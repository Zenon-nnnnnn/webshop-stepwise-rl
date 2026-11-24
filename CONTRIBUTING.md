# 团队协作

## 分支与 PR

主分支保存可 review 的稳定状态。建议分支前缀 `env/`、`data/`、`train/`、`rl/`、`eval/`、`docs/`。一个 PR 围绕一个可解释的改动，填写 PR 模板并由另一人检查。仓库创建者没有默认对所有组员开启写权限；先完成私有仓库邀请。

```bash
git switch main
git pull --ff-only
git switch -c data/validate-goal-mapping
# 编辑、测试
git add src/webshop_lab/data.py tests/test_core.py docs/DATA.md
git commit -m "data: validate environment goal mapping"
git push -u origin data/validate-goal-mapping
```

不要把本地训练产物直接合并进 Git。若 GitHub 提示大文件，检查是否误加了数据、模型或 checkpoint；不要直接 force push 覆盖同学提交。

## 提交前验证

```bash
python -m unittest discover -s tests -v
python -m compileall -q src scripts
python scripts/check_repository.py
webshop-lab audit --output artifacts/pr-audit.json
```

涉及 GPU 行为还要提供目标机 smoke 或实际实验；CI 是 CPU 测试，不能证明 GPU 能训练。涉及指标需人工检查一个成功、一个部分奖励、一个非法动作案例。

## 实验记录

每个结果必须能追踪到 Git commit、配置、数据 SHA、split、goal fingerprint、随机种子、模型版本、环境版本、完成任务列表。为了控制输入变化，正式结果建议记录 Hugging Face 具体 model revision；当前配置采用模型名，首次实验时应在日志中补记实际下载 commit。

原始日志与权重放组内共享存储；Git 提交配置和脱敏汇总，记录共享位置与 SHA。不得将模板示例填成真实测量结果，不把论文数字当成我们实测。

## 修改上游代码

优先在新增模块中适配。确需修改 `upstream/` 时，在 `docs/UPSTREAM_AUDIT.md` 记录文件、原因、验证；保留来源与版权。不要反复运行上游旧安装脚本污染已锁定环境。

## 写作与引用

明确区分原论文算法、复用代码、本组改动和计划工作。每人交付自己负责章节的可读草稿与证据，最终由全组核对。课程 rubric 未录入前，不假设复现天然满足原创要求。
