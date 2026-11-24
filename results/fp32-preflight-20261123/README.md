# 11/23 FP32 重评预检与任务6700诊断

关联 [PR #38](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/38)。本次不追加策略训练，不改 TG/OP 奖励公式，不使用 test。先补环境评分瓶颈和模型身份预检，完整六策略160任务重评仍待验收。

## 6700 的可复现瓶颈

实际 upstream 搜索结果页会枚举所有商品规格组合，并逐次调用 get_reward。get_type_reward 对同一商品和目标重复做两次 NLP 解析，虽然类型分不依赖所选规格。

此前独立离线对照对该任务参考页的37,344个组合逐项比较了 reward 和分项，单页缓存后全部一致；参考worker合计629.15秒、缓存路径约1.01秒。这是历史对照，不冒充本次新跑的uncached性能数据。

本次将早期已用wrapper的思路做成显式 CLI 开关：

```bash
python -m webshop_lab.cli env-server --port 8766 --option-state --cache-type-reward
```

默认关闭，health 明确记录 type_reward_cache。缓存按商品/目标对象身份区分，只保留一个页面调用期间的类型评分，原规格枚举、其它评分项和计算顺序不变，返回字典副本，异常时恢复原评分函数。

仅适用于已有单客户端HTTPServer，不适用于并发服务。开关按新进程启动生效，不支持同进程多次调用serve后动态关闭。该优化未消除笛卡尔积本身，不能保证任何规格规模都快速返回。

## 本次实际诊断

- 原任务6700、同一长搜索查询，连续三次返回约1.95/0.82/0.91秒，页面奖励均0.6286764705882353，观测哈希相同。环境fingerprint与冻结值相同，alignment 1624/0、split verified、option_state=true。
- 显式FP32的B1自行执行任务6700：63.36秒、10次决策，check-run complete/provenance_ok，未发生环境超时。
- **这是正常策略失败**：未购买、max_steps、success=false。最后页面匹配分0.75不能当作任务奖励，终局任务奖励应为0。
- 新诊断不并入旧159条轨迹，不把一次任务诊断变成160项覆盖证明。全部页面评分一致性和长时间服务稳定性仍未重新验证。

## 模型与计算设置

当前六策略的底模/B1/四终点文件哈希已生成；底模和B1与锁定值一致。四终点实际safetensors张量摘要分别等于各自第四轮summary.actor_after，固定选择规则不变。这是本次设备预检，不证明旧评估当时的在内存身份。

本次B1诊断加载前后核对输入权重/config SHA。meta记录全部参数为torch.float32；审计sidecar记录：

| 设置 | 本次值 |
|---|---|
| float32_matmul_precision | highest |
| CUDA matmul TF32 | false |
| cuDNN TF32 | false |
| autocast | false |
| 任务上限 | 10动作，生成128 token，上下文2048 |

这些是参数精度和后端设置记录，不扩大成每个GPU算子内部精度的证明。prepared-configs是未来重评输入，尚未执行全量重评。

[六策略当前身份](identity-preflight.json) · [单任务审计](single-task-check.json) · [手动查询诊断](task6700-regression.json) · [执行源码身份](execution-source.json)

## 原件与验证

[私有诊断包](https://github.com/Zqqq-qq/webshop-stepwise-rl/releases/download/data-v1/webshop-nov23-fp32-task6700-preflight.zip)，SHA-256 `d812843146a3f530f74fa8e4dcba3503396436df33a68e378890fbb4c943964e`。36载荷及SHA256SUMS，包含实际单任务JSONL/meta/check、输入配置和驱动、身份/后端审计、源码及历史参考记录；不含权重与专家SFT数据。

CPU测试96项通过，新增5项覆盖调用减少、逐页隔离、对象身份、返回副本和异常恢复。独立代码复核也通过96项测试，未发现阻塞提交的问题。

## 下一步边界

先完成PR38复审及此修复的环境/统计验收。正式重评应从相同冻结起点加载六策略，用新输出目录和同一160-task清单（含6700），逐批保存真实meta、运行源码、输入文件身份与后端设置。若还有技术故障，保留中断原件并按负责人确认的规则处理，不能根据结果排除任务或增加训练。

Zqqq/Qiu负责实际采用、完整登记和test门禁。`test_authorization=false`。本机目前仅留标记清楚的缓存诊断环境服务，未运行训练或全量评估。
