# 数据来源、导入、划分与环境对齐

## 组内共享下载（data-v1）

三份原始材料已准备为私有仓库 Release 附件，沿用仓库成员访问权限。下载入口和校验步骤见 [团队数据共享指南](SHARED_DATA.md)。数据不写入 Git 代码历史；`data/shared-assets-manifest.json` 只保存大小与哈希。

## 本地需要的三个文件

| 文件 | 实际大小约 | 用途 |
|---|---:|---|
| `data.zip` | 1.24 GB | 商品、属性、人工指令；解压约 5.68 GB |
| `indexes.zip` | 2.82 GB | Lucene 商品索引；解压约 3.60 GB |
| `webshop_sft.json` | 9.31 MB | 1,624 条对话轨迹 |

将它们放在仓库之外的同一个本地目录，例如 WSL 的 `/mnt/d/llm-assets`。共享时使用组内共享盘；GitHub 仓库保留代码与准备步骤。原始 SFT 来源及授权需要在最终报告中说明，不能自动把根目录 MIT 视为全部数据的许可。

若没有本地材料，可参考原作者 [README](../upstream/README.md) 的下载链接。上游有两个都叫 `data.zip` 的不同归档：一个是 WebShop 环境资产，另一个是专家轨迹。不要混用。

原作者环境资产链接（可用性需要下载时确认）：

- [WebShop data.zip](https://drive.google.com/uc?id=1G_0ccLWn5kZE5rpeyAdh_YuoNzvBUjT9)
- [WebShop indexes.zip](https://drive.google.com/uc?id=11zOUDkJSgGhYin9NxQtG8PVpDsika86y)
- [专家轨迹归档](https://drive.google.com/uc?id=1_tBMDixZcIjKuv-LExNllha-YIRxhKIq)：下载后先检查目录与条数，不能假设与本组 SFT 完全相同。

可选安装 `gdown` 并按上游链接下载；新工具不会静默下载几 GB 内容。视频材料不作为运行依赖。

## 导入资产

在工具或 CPU 环境中，从仓库根目录：

```bash
webshop-lab import-assets --source /mnt/d/llm-assets
webshop-lab audit --output artifacts/data-audit.json
```

工具先检查 ZIP 路径与结构，防止路径穿越；检查磁盘空间，按原有布局解压。已有文件会核对长度和 CRC；若发现坏文件会报错，不会静默覆盖。新文件先写 `.partial`，完成后替换。完整 ZIP 的 SHA-256 写入 `data/asset-manifest.json`，供团队比对。

最终布局：

```text
data/raw/webshop_sft.json
upstream/envs/webshop/data/items_shuffle.json
upstream/envs/webshop/data/items_ins_v2.json
upstream/envs/webshop/data/items_human_ins.json
upstream/envs/webshop/search_index/indexes/segments_1
```

不要把 `indexes` 多套一层，也不要将完整版索引与 1,000 商品子集混用。当前工具按完整商品与完整索引设计。环境资产 zip 中虽有 1,000 商品文件，但那不是对应的小型索引。

## 第一轮划分：离线预检查

```bash
webshop-lab split
```

输出：`data/processed/train.json`、`val.json`、三份 `*_ids.json`、`split-manifest.json`。

默认保留上游 `test_indices.json` 的 200 个测试任务，移除 SFT 中与测试任务相同 ID 及已知相同指令的记录。训练/验证按任务 ID 与归一化购物指令构造连通分组，再用 seed=42 划分。预算变体会归为同一购物请求，不允许跨 train/val。

**这一步尚不能检查 SFT 中未出现的测试指令。正式训练前必须继续做环境映射核对。**

## 第二轮划分：对齐真实环境

在 CPU 环境启动服务：

```bash
webshop-lab env-server
```

它输出：

- `artifacts/environment-alignment.json`：SFT ID 与实际环境购物指令的匹配情况。
- `artifacts/environment-manifest.json`：包含商品目标与预算信息的 goal fingerprint。
- `artifacts/goal-instructions.json`：任务 ID 到购物指令的映射，**不导出目标商品答案**。

预算在原环境中由随机过程生成。对齐检查区分“底层购物请求改变”和“只有预算不同”；前者必须先解决，后者单独记录并冻结环境 fingerprint。

利用完整环境指令重新划分：

```bash
webshop-lab split --goal-map artifacts/goal-instructions.json
curl http://127.0.0.1:8765/health
```

服务每次 health 请求会重新检查三组 ID 及底层指令是否交叉。需要 `alignment_mismatches=0`、`alignment_checked>0`、`split_status.verified=true`。否则 rollout 会拒绝运行，避免产生看似正常但无效的比较。

测试集内部可能有相似指令；指标按任务统计。进行置信区间解释时应说明任务相关性限制，不能将结果过度推广。

## 冻结与共享

B 负责人保存资产 SHA、split manifest、环境 fingerprint，团队使用同一版本。正式 SFT 后不要再随意修改划分；修改则必须重跑受影响训练与结果。

将三份 ID 列表、去标识的 split manifest、环境 fingerprint 放入本次实验的 `results/` 子目录供 review；不要把完整原始对话或 MTurk worker 信息提交进 Git。测试集仅用于最终确认；调参和 pre 初步展示使用 val。

## SFT 字段

每条记录有 `id`、`conversations`、`reward`、`true_reward` 等字段。对话使用 `from=human/gpt`、`value=文本`。有少量 `reward != true_reward` 的记录；审计会给出数量。新 SFT 入口不使用这些数值当 loss 权重；不要擅自把所有记录称为“满分专家数据”。
