# Qwen 底模配对试跑（2026-11-19）

在同一台 RTX 5060 Ti、同一个真实 WebShop 服务上，对比 Qwen2.5-1.5B-Instruct 与关闭思考模式的 Qwen3-1.7B。这是模型选型试跑。

## 结果

| 底模 | 满分成功 | 完成购买 | 平均终局购物奖励 | 非法动作 | 8 条轨迹用时 |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 0/8 | 2/8 | 0.0500 | 16/66（24.2%） | 94.8 秒 |
| Qwen3-1.7B，关闭思考 | 0/8 | 1/8 | 0.0833 | 55/75（73.3%） | 162.0 秒 |

两臂都有完成购买的记录，但都没有满分成功。Qwen2.5 的终局奖励只来自任务 160 的 0.4，另一次购买得分为 0。Qwen3 的终局奖励只来自任务 111 的 0.667。未购买的任务记 0。样本中没有在任一臂获得满分的配对任务，无法通过这些满分结果判断模型差异。

Qwen3 的 55 次非法动作中，31 次点击了不可用选项，15 次在不可搜索的页面发出搜索，9 次未能解析为单个动作。Qwen2.5 的 16 次非法动作均为点击不可用选项。任务级结果见 [paired-summary.json](paired-summary.json)。

样本来自当前候选验证划分的前 8 项，每个任务只运行一次。两臂的底模家族和大小同时变化，这批结果用于工程选型，正式 SFT 效果仍需要同底模的 B0/B1 对照。计时包含模型生成和环境交互，按 Qwen2.5 → Qwen3 串行执行，缓存条件可能不同。两臂均未加载 adapter。测试集未使用，正式登记表未更新。

## 独立容量检查

另一次 Qwen3-1.7B QLoRA 短训使用 16 条 2,029–2,046 tokens 的训练样本和 4 条长验证样本，最长 2,048 tokens。两步 loss 为 2.5681 和 1.8705，LoRA 权重变化，另起进程重载后张量哈希一致。峰值实际分配 6.39 GiB、预留 13.29 GiB。这些数字只用于判断设备容量。

## 设置与复核

- 任务顺序为 `[16, 75, 104, 111, 136, 139, 160, 191]`，每个任务运行一次。环境指纹、划分健康状态、原验证 ID 文件哈希、运行 ID 及两臂配置和轨迹摘要见 [paired-summary.json](paired-summary.json)。
- 两臂共用购物系统提示、动作解析器、4-bit NF4 加载、贪心解码、seed 42、总上下文 2,048、最多生成 128 tokens、每任务最多 10 步。分别使用各自的聊天模板，Qwen3 显式传 `enable_thinking=False`。
- 加载固定本地快照前，重新计算两组权重的 SHA-256。实际运行的 `rollout.py`、`model.py`、`common.py` 与本报告基线的远端源码 Git blob 一致。
- 汇总时逐臂核对任务顺序、唯一 episode、run ID、环境指纹、配置 SHA-256 及轨迹与元数据的对应关系。原始轨迹、配置和元数据留在本机。

复跑时，在仓库根目录下的新运行目录准备 `val-first8.json`、`qwen25.config.json` 和 `qwen3.config.json`。ID 列表和共享配置字段见 [paired-summary.json](paired-summary.json)。每份配置另加对应的 `model_id`、`revision`、`weights_sha256`、`enable_thinking`，将 `model` 指向已核验的本地快照。在确认本机 WebShop 的 `/health` 指纹一致后，复制 [run_pilot.py](run_pilot.py) 到运行目录，在 GPU 环境串行执行。

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src
python artifacts/qwen-base-pilot/run_pilot.py qwen25
python artifacts/qwen-base-pilot/run_pilot.py qwen3
```
