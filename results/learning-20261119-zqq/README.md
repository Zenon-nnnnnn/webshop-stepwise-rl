# 真实 tokenizer 与 SFT mask 检查

CPU Tutor 示范，不是训练。Qwen/Qwen2.5-1.5B-Instruct tokenizer revision 固定为 `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，与环境侧模型 manifest 对齐；transformers=4.56.2。仅 tokenizer 下载，不加载模型权重。

| assistant 回合 | 序列 token 数 | 屏蔽前缀 | 监督目标 |
|---|---:|---:|---:|
| 搜索 | 189 | 131 | 58 |
| 打开商品 | 953 | 874 | 79 |
| 购买 | 1121 | 1060 | 61 |

输入为原 SFT 的任务0，实际调用 training_messages 和 fit_history（2048上下文），复用 sft.py 的 prefix 检查和 labels 公式。各次只监督当前回复，旧 assistant 回复留在前缀并屏蔽。真实目标包含 Thought 和 Action，以及模板的 im_end/换行；不是仅动作字符串。详情及目标token IDs见 [token-mask-check.json](token-mask-check.json)，不含完整对话。

复现（Python3.10+，独立工具环境，非GPU训练环境）：

```bash
python -m venv .venv-tokenizer-check
.venv-tokenizer-check/bin/python -m pip install transformers==4.56.2 jinja2==3.1.6
PYTHONPATH=src .venv-tokenizer-check/bin/python scripts/verify_learning_token_mask.py /path/to/webshop_sft.json
```

padding检查只复现 labels=-100 的构造，不是实际 torch collator/Trainer 执行；没有验证梯度、SFT重载或RL token mask。只检查一个样本，不声称全部训练数据已验收。
