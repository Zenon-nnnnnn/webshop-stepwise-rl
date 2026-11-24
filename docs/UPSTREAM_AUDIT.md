# 原始 SPA 压缩包审计

审计基于本次用户提供的 SPA.zip 和 webshop_sft.json，不代表原作者当前 main 分支均有相同问题。`source-manifest.json` 保存原包 SHA 与各文件原始 SHA，供追溯；没有冒称原始压缩包对应某个已确认的 Git commit。

## 数据事实

| 项目 | 本地材料中的结果 | 对实验的影响 |
|---|---:|---|
| SFT 记录 | 1,624 | 可用于初始监督训练；并非全部满奖励 |
| 默认 train/test ID 交叉 | 12 | 必须排除或重新划分 |
| 示例探索轨迹 | 78 | 流程样例规模 |
| 示例探索不同任务 | 26 | 同一任务多次探索 |
| 最终奖励为 1 的探索轨迹 | 1 | 数据高度偏失败，不适合直接宣称学到有效奖励 |
| 进度训练/验证文件 | 内容相同 | 不能估计泛化误差 |
| 处理后的 PPO 示例 | 41 | 示例输入不等于有效 RL 实验 |

重叠 ID：818、978、2640、2933、4161、5413、5900、7475、7603、7624、7999、8573。

用 `webshop-lab audit` 可重新计算。`results/material-audit.json` 是本次真实本地审计产物；不含模型性能结果。

## 代码问题与处置

| 问题 | 原位置 | 本项目处置 |
|---|---|---|
| 任务集重叠 | `eval_agent/data/webshop/*indices.json` | 新 split 保留 test、排除 ID/指令交叉，并核对环境映射 |
| 验证文件取自训练前 100 条 | `prm/data_org.py`、各 progress train 脚本 | 新主线要求独立 val；原脚本保留作对照 |
| controller 前台运行阻塞 | `exploration/webshop/my_generate_response_webshop.sh` | 新主线不用此脚本，环境服务和模型直接分进程 |
| 路径 `ckt` / `ckpt` 混用 | SFT、探索、PRM、eval 等 | 新配置统一在 `artifacts/`；旧入口需要自行适配 |
| 购买完成就 `success=True` | `eval_agent/envs/webshop_env.py` | 新 success 要求 done 且满奖励 |
| 非法动作异常分支 reward 可能未初始化 | 同上 | 新入口先验证动作并独立记录反馈 |
| 原根依赖旧且跨后端 | `requirements.txt` | 新入口分 CPU/GPU 候选依赖；不直接整体安装 |
| 根依赖与 PPO 专用依赖版本不同 | `requirements.txt` / `ppo/requirements.txt` | 分清两个旧环境；PPO 的 torch 2.2 组合需要适配新显卡 |
| 硬编码 API key | `eval_agent/configs/model/openai.json` | 上传前清空；保留 manifest 变更记录，不保留原密钥 |
| 陈旧 ChromeDriver 二进制 | 文本环境旁 | 排除，文本主线不使用 |
| 缓存、egg-info、notebook checkpoints | 多处 | 排除，保留清单 |

## 快照保留范围

保留所有非缓存可复用源码、配置、原 README、LICENSE、框架图以及小型示例轨迹。未带入 `.DS_Store`、`__pycache__`、`.ipynb_checkpoints`、构建元数据、陈旧平台浏览器二进制。完整清单见 [source-manifest.json](source-manifest.json)。密钥清空是对原快照的明确例外。

FastChat 部分包含第三方源码；源文件中的版权头保留。更多来源见 [第三方说明](../THIRD_PARTY_NOTICES.md)。

## 尚未验证

未在 RTX 5060 Ti 上执行完整 SFT、神经进度训练或 PPO；未验证论文指标能复现；未完整运行另外两个环境；未审计所有上游后端组合。保留源码不等于承诺所有脚本开箱即用。
