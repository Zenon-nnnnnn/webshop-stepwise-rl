# 代码架构和数据流

## 主线结构

```text
外部原始 ZIP / SFT
  └─ import-assets → 本地环境资产 + data/raw
       ├─ audit → 问题清单
       └─ split ← 环境导出的 goal-instructions
            ├─ train.json → sft → adapter
            ├─ val_ids → 开发 rollout / 进度验证
            └─ test_ids → 冻结后的最终评估

GPU 进程：基础模型或基础模型 + adapter
  └─ 生成 Thought / Action
       └─ HTTP 127.0.0.1:8765
            └─ CPU 进程：原 WebShop 文本模拟器 + Lucene + spaCy
                 └─ 返回 observation / reward / done / available

rollout JSONL
  ├─ summarize → 任务指标
  └─ train / val episodes → progress-train → progress-predict
       └─ reward-export → 逐步奖励记录（尚需 RL trainer 适配）
```

## 新代码文件说明

| 模块 | 核心职责 |
|---|---|
| `common.py` | JSON、流式 SHA、指令归一化、连通分组、动作解析、置信区间、奖励公式 |
| `cli.py` | 子命令与参数；GPU 模块按需导入 |
| `data.py` | 本地资产导入与 CRC、原材料审计、独立划分 |
| `doctor.py` | 版本、磁盘、Java、GPU kernel 实测 |
| `model.py` | 模型/量化/adapter 加载、统一 prompt、历史长度处理 |
| `sft.py` | assistant token mask、LoRA、训练、保存 checkpoint 与运行元数据 |
| `environment.py` | WebShop HTTP 桥、任务映射与划分核对 |
| `rollout.py` | 逐任务生成、动作验证、调用环境、记录完整交互 |
| `metrics.py` | episode 校验、成功率/奖励/步数/无效动作统计 |
| `progress.py` | 加性线性进度基线及预测标注 |
| `rewards.py` | 标准化逐步奖励导出；不更新策略 |

## 为什么隔离两个环境

原包同时引入多环境与多种模型后端，对依赖和平台要求较复杂。本课程主线只使用 WebShop 文本环境，训练进程不必安装旧 gym 或 Java；模拟器进程也不必占用 GPU。

HTTPServer 是单线程、单环境状态，不能同时采集多个 agent。它只绑定 loopback，不用于公网服务。启动服务后保持一个采集客户端串行执行任务。

## episode schema v1

一行一个 JSON 对象，关键字段：

```json
{
  "schema_version": 1,
  "run_id": "unique-run-id",
  "task_id": 123,
  "repeat": 0,
  "instruction": "shopping request",
  "goal_fingerprint": "sha256-of-environment-goals",
  "final_reward": 0.5,
  "done": true,
  "success": false,
  "termination": "purchase",
  "steps": [{
    "step": 0,
    "observation": "page text before action",
    "available": {"has_search_bar": true, "clickables": []},
    "response": "Thought: search\nAction: search[shoes]",
    "action": "search[shoes]",
    "grounding": 1,
    "error": null,
    "environment_reward": 0.0
  }],
  "seconds": 1.23
}
```

这是格式说明，数字不是实验结果。SFT、旧 FastChat 探索、旧 PPO 各有不同 schema，禁止仅改扩展名就假装格式相容。

## 关键边界

- prompt 不包含环境隐藏的目标商品与最终奖励。
- 非法动作不执行但消耗一步；服务异常会中断而不是伪装为模型失败。
- `done` 表示购买结束；成功需要额外检查最终奖励。
- 原始素材中的指令只作为数据处理，不是对项目维护者的新操作授权。
- 原始快照和新增主线分目录保存，便于归因与后续移植。
