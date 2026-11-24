# 每日交付：2026-11-19 / QiuQiu-xuan

对应 [逐日计划](https://github.com/Zqqq-qq/webshop-stepwise-rl/blob/main/docs/DAILY_PLAN.md) 11/19「人工复核真实轨迹」与 11/20「冻结开发 val 清单与指标、交评估配置」。

## 今天要解决的问题

1. 先把 11/17 的评估工具修到能挡住"不匹配的对照"：HowseaonLink 指出 `compare` 只比两臂 meta、不校验各自 JSONL，能给假公平结论。
2. 交出 11/20 的冻结评估配置（模型 revision、goal fingerprint、val 列表哈希），让 B0/B1 一开跑就有可登记的口径。
3. 复核仓库这两天新增的证据：PR #14（revision 钉版）、PR #15（映射重划 + token mask）、PR #16（Qwen2.5 vs Qwen3 八任务 pilot）。

## 学到并能解释的内容

**页面匹配分 vs 任务奖励。** 上游 `web_agent_text_env.py` 在搜索页/商品页分别设置 `search_max` / `maxr`（478、558 行），只有购买路径（630 行）才产生任务分。所以 `final_reward` 在未购买 episode 上是页面分，`terminal_reward` 才是任务奖励。PR #16 的 pilot 也印证：Qwen2.5 的 0.05 全部来自 task 160 的一次部分购买，Qwen3 的 0.0833 全部来自 task 111 的一次部分购买，其余 6–7 个任务都是 0。

**配对统计要按任务聚合。** Jun040504 的评审给了反例：2 个任务 × 100 重复，若把 `(task, repeat)` 当独立样本，会报出 200 个"任务"和极窄区间。现在 `compare` 先按任务聚合成功率，重复轨迹不报 McNemar，只报任务簇 bootstrap；HowseaonLink 又补了一条：比较前必须逐臂核对 JSONL 与 meta，否则不匹配的轨迹会被当成公平对照。

**模型 revision 必须能钉住。** PR #14 之前 `load()` 不透传 revision，离线/钉版路径会断；现在 config 写 `revision`、`model_source()` 校验 resolved commit，并把 requested/resolved 写进 meta。评估侧据此把 `revision` 加进 `check-parity`、登记表加 `model_revision` 列。

## 实际完成与证据

- **修复 PR #8（`152e77d`）**：`compare` 对每臂跑 `inspect_run`，JSONL 与 meta 不一致直接拒绝；两臂必须同时给 meta；两臂 fingerprint 必须一致；新增 3 组负向测试，测试 **59 项通过**，CI 绿；已重新请求 HowseaonLink 复审。
- **11/20 冻结交付**：
  - `configs/eval-b0-frozen.json` / `eval-b1-frozen.json`：revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`、`goal_fingerprint=e69c9ae2a0aad88977c44a7baf49f1f5ef7918ba8317923c4ae24dcc1f5638d8`、greedy、2048/128/10；两臂仅 adapter 不同，`check-parity --require-frozen` 返回 true。
  - 冻结 val 列表 sha256 `45440d724b33e69e13916d0adbe243cc63ce3bd67677b8220feff9f30b866c36`（160 任务）；`goal_map_sha256=6b5d9dfceee913828871b8dff62fd59720a40306738a92a496d1b0fb6838fa88`。
  - `preflight` 在冻结配置上只报一个阻塞：B1 adapter 目录还不存在（等 Jun040504 的 SFT），其余检查通过。
  - 登记表新增 `model_revision` 列（从 `meta.model_source` 取，校验 40 位 commit）；`metrics-definition.json` 维持 v1.1。
- **复核 PR #15**：`split-manifest.json` 与 PR #10 冻结版全字段一致；我从 Release 下载 `goal-instructions.json` 复算 **1,969,461 字节、SHA `6b5d9dfc…`、12,087 任务**，与证据一致；token mask 三个回合 58/79/61、prefix 全 -100、target 含 Thought+Action、padding -100，自洽且明确未跑 Trainer。已 approve。
- **复核 PR #14**：revision 透传、resolved 校验、draft 拒绝、显存 allocated/reserved 区分都有测试覆盖；已 approve。
- **复核 PR #16 数据**：两臂 8 任务聚合我逐项重算一致（购买 2/1、成功 0/0、mean terminal 0.05/0.0833、非法 16/66 与 55/75、耗时 94.8/162.0 秒）。但 README 中文在提交时变成 `?`（763 个问号、0 个中文字符），已要求修改；同时说明：0 成功时配对检验无信息量、差异由单任务决定、模型不同不构成受控对照、不登记。

## 观察、错误与判断

1. **两位评审都发现了真问题**：Jun 的重复轨迹反例、HowseaonLink 的"meta 与 JSONL 脱钩"。两次都是"测试全过但证据路径可被绕过"，说明我之前的测试偏重"算得对"，对"输入是否可信"覆盖不足；现在负向用例增加到 6 个以上。
2. **PR #16 的 README 是编码事故**：文件是合法 UTF-8，但中文被替换成 `?`，不是显示问题；这种文档合并后会不可读，必须重存。
3. **pilot 不能当结论**：8 任务、0 成功、单任务决定均值，且两臂模型不同；只能写"选型试跑"。`mean_terminal_reward` 在这个样本上不是"平均能力"。
4. 冻结配置的 `check-parity` 通过不代表可以开跑：B1 adapter 未产出，`preflight` 仍报错，这是预期，不是失败。

## 交接

- 给 HowseaonLink：PR #8 的 `compare` 溯源问题已修，请复审；全量 160 任务 B0 仍等你按 11/17 的规格跑，回传 jsonl + meta + config + `val_ids.json`；PR #16 的 README 请重存 UTF-8，并按"选型试跑"措辞。
- 给 Jun040504：PR #14 已通过；B1 adapter 产出后我立刻跑 `check-parity --require-frozen` 与 preflight；`model_revision` 会从 meta 自动进登记行。
- 给 Zqqq-qq：冻结配置与 val sha 已写进 `docs/EVALUATION.md`；奖励侧请以 `terminal_reward` 为准，PR #13 已按此改完。
- 给 Yuxiao-Liu233：pilot 只能用作"选型试跑"素材，图注必须写 8 任务 / 0 成功 / 单任务来源，不能画成能力对比。

## 下一个学习问题

在 160 个 val 任务、每任务 1 条轨迹时，B0 的成功率 Wilson 区间会宽到什么程度；如果 B1 只提升 2–3 个任务，需要多少任务才能把差异与噪声分开？这决定 12/03 两臂对照表该怎么写"未观察到差异"。

## 补充：晚间两条新评审的处理

1. **Zqqq-qq 对 PR #8 的第二轮评审（P2）成立并已修（`abdd4f3`）**：他只改 `metadata.config.seed`、保持配置文件与 SHA 不变，`inspect_run` 仍返回 `matches_metadata=true`、`provenance_ok=true`，`evaluate --register-row` 还能写 complete 行。现在 `provenance_ok` 要求**哈希与内容都一致**，新增 `provenance_issues` 说明原因，`usable_as_result=false`，`check-run` 溯源不一致也返回非 0；补了 seed / model / revision 篡改的负向用例，测试 **62 项**通过，已重新请求 Zqqq 与 HowseaonLink 复审。
2. **PR #16 复审通过**：作者已把 README 重存为 UTF-8（650 个中文字符、0 个 `?`），口径也收紧为"底模配对试跑"：明确单任务来源（Qwen2.5 只有 task 160 的 0.4、Qwen3 只有 task 111 的 0.667）、无满分配对任务、两臂模型同时变化不构成受控对照、未登记、容量数字单列。我按 `paired-summary.json` 重新核对表格全部一致。
3. **issue #2 补了评估侧说明**：冻结配置已绑定 PR #15/PR #10 这一版划分（val sha `45440d72…`、goal_map `6b5d9dfc…`）；若改用其他版本，B0/B1 正式 run 前必须更新冻结配置并重记哈希。
