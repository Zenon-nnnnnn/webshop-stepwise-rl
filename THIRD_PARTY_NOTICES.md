# 来源、署名与许可

## SPA-RL

`upstream/` 来自用户提供的 `SPA.zip`，其 README 指向 [WangHanLinHenry/SPA-RL-Agent](https://github.com/WangHanLinHenry/SPA-RL-Agent) 与论文 [SPA-RL](https://arxiv.org/abs/2505.20732)。原包 LICENSE 是 MIT，版权为 2025 Hanlin Wang。原 LICENSE 保留在根目录和 `upstream/LICENSE`。

本仓库不是作者官方仓库，也未确认该压缩包对应哪个 Git commit。原 zip 与文件 SHA 见 `docs/source-manifest.json`。

## 快照调整

移除生成缓存、构建元数据、macOS 元数据、notebook checkpoints 和陈旧 ChromeDriver 二进制；清空 `upstream/eval_agent/configs/model/openai.json` 中嵌入的 API credential。未上传原视频教程、完整商品/索引归档或模型权重。清单列出每个排除项和 credential 配置变更，原敏感值不保留。

## FastChat、WebShop 与其他继承组件

上游 README 声明基于 [ETO](https://github.com/Yifan-Song793/ETO) 和 [StepTool](https://github.com/yuyq18/steptool)。`upstream/fastchat/` 包含 FastChat 派生代码，保留其原文件版权头；FastChat 官方 Apache-2.0 许可文本放在 `licenses/FastChat-Apache-2.0.txt`。

WebShop 源码来自上游内嵌环境。环境、数据和预训练模型的使用应分别遵守各自来源的条款；本仓库根 LICENSE 不改变第三方代码、数据、模型或论文的条款。新增 `src/webshop_lab/` 和团队文档以根目录 MIT 许可分发，继承文件保留原有声明。

## 报告引用建议

```bibtex
@article{wang2025spa,
  title={SPA-RL: Reinforcing LLM Agents via Stepwise Progress Attribution},
  author={Wang, Hanlin and Leong, Chak Tou and Wang, Jiashuo and Wang, Jian and Li, Wenjie},
  journal={arXiv preprint arXiv:2505.20732},
  year={2025}
}
```

另外引用实际使用的 WebShop、基础模型、LoRA / QLoRA 等原始工作。课程报告只报告本组运行结果；原论文指标须显式标为引用。
