# 团队数据共享与校验

## 访问权限和入口

下载入口：**[WebShop team datasets v1](https://github.com/Zqqq-qq/webshop-stepwise-rl/releases/tag/data-v1)**。

附件位于本项目的私有 GitHub 仓库，沿用仓库权限。组员须先接受仓库邀请并登录相应 GitHub 账号；未登录或无权限时显示 404 并不代表文件不存在。原始材料没有进入 Git 提交历史，也没有建立公开分享链接。

本次共享只包含 `data.zip`、`indexes.zip` 的二进制分片、`webshop_sft.json`、校验清单和导入说明，不上传带硬编码 credential 的原始 SPA.zip 或教程视频。数据用于本组课程复现；来源与使用边界仍按 [DATA.md](DATA.md) 和原来源处理。

## 附件清单

| 附件 | 字节数 | 用途 |
|---|---:|---|
| `webshop_sft.json` | 9,312,587 | 1,624 条 SFT 对话，可先下载解除审计阻塞 |
| `data.zip` | 1,242,271,168 | 商品、属性与人工购物指令 |
| `indexes.zip.part01` | 1,500,000,000 | 索引原包前半分片 |
| `indexes.zip.part02` | 1,319,730,135 | 索引原包后半分片 |
| `dataset-manifest.json` | 见附件 | 三份原始文件及索引分片的 SHA-256/长度 |
| `verify_and_join.py` | 见附件 | 标准库校验和字节级合并脚本 |
| `UPLOAD_README.md` | 见附件 | 独立于代码仓库的快速说明 |

**不要下载 GitHub 自动生成的 Source code.zip 来代替这些数据附件。** 索引两片不是可独立解压的 ZIP，须按顺序合并，合并后 `indexes.zip` 应为 2,819,730,135 字节。

## 方法一：网页下载

打开 Release，将以上附件下载到同一个本地目录。不要让浏览器将分片自动改名为带 `(1)` 的重复文件名。下载完成后，在仓库根目录：

```bash
python scripts/verify_dataset_assets.py --directory /absolute/path/to/data-assets
webshop-lab import-assets --source /absolute/path/to/data-assets
webshop-lab audit --output artifacts/data-audit.json
webshop-lab split
```

若不在代码仓库中，可直接执行附件中的脚本：

```bash
cd /absolute/path/to/data-assets
python verify_and_join.py --directory .
```

该脚本先检查 SFT/商品包，再检查分片，流式重建索引并核对原包 SHA-256；若原包已经存在，只验证而不覆盖。校验失败时停止，不用损坏数据继续训练。

## 方法二：已登录的 GitHub CLI

需要已安装并通过 `gh auth login` 登录有仓库访问权的账号。不要把 token 写进命令、Notebook 输出或 Git URL。

```bash
gh release download data-v1 --repo Zqqq-qq/webshop-stepwise-rl --dir data-assets
python scripts/verify_dataset_assets.py --directory data-assets
webshop-lab import-assets --source data-assets
```

CLI 下载目录中的 `Source code` 自动归档不是本流程依赖。Colab 使用时须另行完成私有 GitHub 认证，不能假设登录浏览器就自动让云端 shell 拥有权限。

## 只需先做 SFT 审计的成员

无需等待下载约 4 GB 的完整资产。只下载 `webshop_sft.json` 后先校验：

```bash
python -c "import hashlib; from pathlib import Path; p=Path('data-assets/webshop_sft.json'); h=hashlib.sha256(p.read_bytes()).hexdigest(); print(h); assert h=='721e46dba410cff900e187bccea91a456c38b105b6d14d068d7762afb7791de3'"
webshop-lab audit --sft data-assets/webshop_sft.json --output artifacts/sft-audit.json
webshop-lab split --sft data-assets/webshop_sft.json
```

也可用 CLI 单独下载：

```bash
gh release download data-v1 --repo Zqqq-qq/webshop-stepwise-rl --pattern webshop_sft.json --dir data-assets
```

此处的 split 只是离线初步划分。实际训练和评估前仍须由环境负责人导出完整 goal-map，再按 [DATA.md](DATA.md) 对齐、排除任务指令交叉并冻结。

## 原始文件校验值

| 文件 | SHA-256 |
|---|---|
| `webshop_sft.json` | `721e46dba410cff900e187bccea91a456c38b105b6d14d068d7762afb7791de3` |
| `data.zip` | `9d96abd8c6626d6d4d9a9552cee70ef4a21a5c72b2edc3efde246bc8a510cb91` |
| `indexes.zip` | `9c6db284bef87d7ab3a108c978825c9ccd71578b1c41ba0dfa15a570ec6c3759` |

机器可读副本： [shared-assets-manifest.json](../data/shared-assets-manifest.json)。原始 SFT 哈希与此前材料审计记录一致。

## 存储与验证边界

附件下载合计约 4.07 GB；重建索引另占约 2.82 GB；导入解压另占约 9.28 GB，再预留模型/环境/缓存空间。组员可在合并校验成功且无需保留分片后自行清理分片。

本次上传已按 GitHub 返回的附件状态、长度和可用的服务端 SHA-256 核对，最终附件元数据见 [dataset-release-verification.json](../results/dataset-release-verification.json)。合并工具已用小型样例检查字节一致重建、重复校验和损坏分片拒绝；这不代表已在组员主设备完成 9 GB 解压或 WebShop 启动。

数据共享解除文件缺失阻塞，但不等于完成环境 health、任务映射、正式数据划分、SFT、RL 或模型性能验收。各模块继续按原验收清单执行。
