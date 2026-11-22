# Terminal Rewards versus Rule-Based Potential Shaping for a Resource-Constrained WebShop Agent

**COMP7608A — Project Proposal**
**Team 32 — registration details to be checked by members**
**Working submission draft: 21 November 2026; not submitted**

| Registered name | Student ID |
|---|---|
| Liu Yuxiao | 3036841254 |
| Lin Haoxiang | 3036842454 |
| Zhang Qing | 3036843862 |
| Qiu Yaxuan | 3036842040 |
| Gu Junwei | 3036906664 |

Names and IDs follow the [course registration extract](team-registration.md). English and this document structure are working choices, not verified course formatting requirements. Outstanding requirements are listed in the [submission checklist](course-requirements-checklist.md).

## 1. Motivation and research question

WebShop evaluates agents that interpret a shopping request, navigate product pages, select options, and purchase an item [1]. In our text-based setup, a language-model policy observes page text and interaction history and produces search or click actions. Completing a purchase does not necessarily satisfy the request: a valid sequence may still select an unsuitable product or specification.

We investigate whether a fixed, interpretable intermediate reward changes the behavior of a small shopping agent under a limited training budget. Our main question is: **from the same supervised-finetuned policy, does rule-based potential shaping produce different held-out shopping outcomes from terminal-only rewards under a matched episode/update schedule?** We do not assume improvement.

SPA-RL motivates the broader study of stepwise feedback [2]. However, our preliminary learned additive progress estimator did not reliably outperform simple prediction baselines and exhibited credit-assignment and cycling concerns. We retain these negative findings and use a rule-based potential for the executable main comparison. This is an adaptation with a different reward mechanism, not a reproduction of neural SPA.

## 2. Data and starting policy

Our provided material contains 1,624 expert shopping dialogues. An audit identified task overlap in the original split. After grouping by task and checking instructions against the actual environment mapping, we use 1,451 training dialogues and 160 validation dialogues, exclude 13 records, and reserve 200 environment task IDs for final testing. Reserved IDs are not additional expert demonstrations. Training tasks alone supply SFT examples and RL updates; validation supports development, and test remains inaccessible to tuning.

Both reward arms start independently from the same original one-epoch SFT LoRA adapter of Qwen2.5-1.5B-Instruct, with the base revision and adapter/configuration hashes recorded. The SFT stage uses a parameter-efficient quantized setup related to QLoRA [3]; the proposed RL runs use unquantized FP32. Historical resume and synchronization modifications are disclosed in the retained SFT evidence. We will not relabel that run as a different, unexecuted configuration.

## 3. Reward design and policy learning

Let R be the environment score on purchase completion, or zero if the finite task ends without purchase. Page-match scores are diagnostic and cannot replace R. Terminal-only training gives R at the final action and zero earlier.

The shaping arm adds the transition difference:

    r_OP(t) = r_terminal(t) + Phi(s_next) - Phi(s_current).

Phi is a fixed lexical match between currently selected option values and the shopping instruction, bounded by 0.25. It uses visible selection state, not hidden target-product answers. No selected option gives zero potential. At purchase completion or the task action limit, successor potential is set to zero. With gamma=1 and zero initial/terminal potential, differences telescope: the episode total remains R, and returning to the same potential cannot earn net shaping reward. This follows the potential-difference construction studied in reward shaping [4]; it does not guarantee that our heuristic identifies correct options or improves finite-budget learning.

We use the repository's assistant-turn PPO engineering path, based on PPO's clipped objective [5]. Each sampled reply is one action; its joint log probability sums actual generated-token log probabilities. Replies are sampled anew from each arm's current policy. Observations/history/padding are excluded from direct action loss. A separate value head estimates the pre-reply state value; complete-episode GAE uses gamma=lambda=1. Old probabilities and calculated targets stay fixed during an update. The first experiment takes one optimizer step per batch, so it is not an evaluation of repeated PPO epochs or clipping's practical benefit.

Shared settings are FP32, AdamW learning rate 1e-5, clip 0.2, value-loss coefficient 0.5, gradient clipping 1.0, context length 2,048, and at most 128 generated tokens per reply. Both arms have zero grounding bonus and zero reference-KL coefficient. Sampling uses temperature=1, top-p=1, top-k=0 and disabled dropout. These choices and their limitations will be reported.

## 4. Budget and evaluation

Seeds 42 and 43 are specified before running. Each arm/seed receives four updates with four fixed training tasks per update: 16 episodes with an eight-action task limit. Across two arms and two seeds this is 16 optimizer updates and 64 episodes, capped at 512 policy decisions and 65,536 generated tokens. Equal schedules and caps do not force equal realized consumption; actual decisions, applied environment steps, tokens, and elapsed costs are recorded separately. We select the fourth-update checkpoint rather than choosing by validation performance. No extra updates are added to obtain a favorable result.

We evaluate the original SFT policy and all four final policies on the same 160 validation tasks, one greedy FP32 episode per task, with a ten-action evaluation limit. The historical 4-bit evaluation cannot substitute for this baseline. Full-reward success requires a completed purchase scoring at least 1−1e−6, with all valid attempted episodes, including normal failures, in the denominator. We also report purchase completion, mean terminal score, invalid-action rate, steps, loops, and resource cost. Technical interruptions are reported and handled by a predetermined rerun rule.

Each seed is reported separately, with task-paired differences, bootstrap intervals and single-policy Wilson intervals. Two training seeds are not treated as 320 independent tasks or as strong evidence of seed robustness. Final-test use requires separate approval of frozen identities, selection rules and evaluation configuration; the present execution plan does not authorize test runs.

## 5. Feasibility, progress and results

Environment, SFT handoff and data checks have retained artifacts: the WebShop bridge reports `alignment_checked=1624` with zero mismatches and a verified split, the B1 adapter identity is locked (`c39e2a78…`), and reviewers checked artifact integrity, probability/mask/reward relationships, checkpoint chains and fresh-process reload records for the earlier three-update engineering run. That was record-based verification, not independent GPU reproduction.

The planned PPO training ran for both seeds: four updates and 16 episodes per arm, totalling 16 optimizer updates, 64 episodes, 272 decisions, 220 applied `/step` calls and 16,128 generated tokens. Training explicitly selected FP32. Endpoint evaluation produced the following stored development results on the same **159 of 160** validation tasks; task 6700 was interrupted by an environment timeout.

**Precision correction (23 November).** The retained full-evaluation scripts called CLI rollout without the FP32 wrapper or an explicit dtype. In that source version, nonquantized loading defaults to BF16 on supported hardware and otherwise FP16. Historical batch metadata did not log actual parameter dtypes. These records must therefore be labelled **nonquantized, automatic-precision development evaluation**, with historical runtime dtype unverified, rather than completed FP32 validation. The earlier eight-task FP32 wrapper check is separate.

| stage | full successes | success rate | Wilson 95% | invalid actions |
|---|---:|---:|---|---:|
| B0 base | 17/159 | 10.69% | 6.78–16.45% | 72.96% |
| B1 SFT LoRA | 52/159 | 32.70% | 25.90–40.33% | 8.08% |
| TG, seed 42 | 52/159 | 32.70% | 25.90–40.33% | 10.94% |
| TG, seed 43 | 52/159 | 32.70% | 25.90–40.33% | 9.15% |
| OP, seed 42 | 50/159 | 31.45% | 24.74–39.03% | 10.44% |
| OP, seed 43 | 50/159 | 31.45% | 24.74–39.03% | 12.68% |

Task-paired bootstrap uses 10,000 resamples, Python `random.Random(42).choices`, frozen validation-ID order and linear quantiles. B1 − B0 is +35 tasks (+22.01 pp, 95% CI +15.09 to +29.56). TG − B1 is 0.00 pp [−5.66, +5.66] in each seed; OP − B1 is −1.26 pp [−6.29, +3.77]. OP − TG is −1.26 pp, with intervals [−6.92, +4.40] and [−6.29, +3.77] for seeds 42 and 43.

The retained records show a higher SFT success rate, and no observed OP success-rate improvement over TG at this budget and evaluation path. Equal TG/B1 success totals do **not** imply unchanged behaviour: each seed has ten tasks switching from success to failure and ten switching in the other direction. The intervals do not establish equivalence or general ineffectiveness of shaping.

Terminal task rewards count non-purchases as zero; the last page-match score is diagnostic only. Under that definition, OP − TG mean reward differences are −0.00933 [−0.03983, +0.02107] and −0.01426 [−0.04277, +0.01373]. Historical 4-bit results remain separate; differences across these runs cannot be attributed solely to precision.

Execution deviations remain explicit. Baselines have four batches of 40/40/39/40 completed tasks. PPO endpoints have five files of 40/40/20/40/19; the fifth supplies nineteen tasks not completed in the third, leaving only task 6700 missing. Twenty-eight actual per-batch metadata files have been recovered and match the raw run IDs and config/ID hashes. Old top-level metadata belongs to failed whole-run attempts and is not metadata for concatenated batches. Chunking, technical-task handling, actual historical precision and load-time identities still require protocol-owner acceptance.

The [23 November correction and evidence handoff](../results/eval-nov22-closeout-20261123/README.md) provides the reproducible audit, original batch metadata and retained execution source. Current file hashes and a post-fix load check are explicitly collection-time evidence, not historical attestation. The corrected loader supports explicit FP32 evaluation and records actual parameter dtypes. Formal registration and held-out test remain pending; the test split is untouched.


## 6. Schedule and responsibilities

The team targets proposal finalization on 22 November and submission on 23 November, necessary experimental work by the night of 24 November, project acceptance and presentation freeze on 25 November, presentation on 26 November, and report submission on 15 December. Course-specific submission times and formatting remain to be confirmed.

HowseaonLink executes environment/GPU jobs; Zqqq-qq coordinates data, rewards, RL and technical integration; Jun040504 checks SFT/model identity and restoration; QiuQiu-xuan independently evaluates results; Yuxiao-Liu233 integrates the presentation. Every member writes and reviews their own report contribution. Tutor assistance and actual student work will be distinguished.

## References

[1] Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. 2022. [WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents](https://arxiv.org/abs/2207.01206).

[2] Hanlin Wang, Chak Tou Leong, Jiashuo Wang, Jian Wang, and Wenjie Li. 2025. [SPA-RL: Reinforcing LLM Agents via Stepwise Progress Attribution](https://arxiv.org/abs/2505.20732).

[3] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. 2023. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314).

[4] Andrew Y. Ng, Daishi Harada, and Stuart Russell. 1999. [Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping](https://aima.cs.berkeley.edu/~russell/papers/icml99-shaping.pdf).

[5] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).
