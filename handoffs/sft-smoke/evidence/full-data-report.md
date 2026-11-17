# SFT data and label inspection — 2026-11-17

## Scope

Completed locally: private Release SFT retrieval, exact SHA-256 verification, repository audit, preliminary offline split, and tokenizer-based label/history inspection. No GitHub changes, GPU training, model weights, or PPO runs were performed.

Source: `Zqqq-qq/webshop-stepwise-rl`, private Release `data-v1`, asset `webshop_sft.json`.

- Bytes: 9,312,587.
- SHA-256: `721e46dba410cff900e187bccea91a456c38b105b6d14d068d7762afb7791de3` (matches the published value).
- Dataset: 1,624 dialogues, 1,624 unique task IDs, 1,599 normalized unique instructions; 8 `reward` / `true_reward` disagreements. These rewards are not used as SFT loss weights by the current implementation.
- Source checkout: `9df3063321ccda2a3a96e1a5d7d9d720bdd296e0`.
- Tokenizer: `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, matching the revision reported by the GPU executor. Only tokenizer/configuration files were downloaded.
- Tools: Python 3.11.14, Transformers 4.56.2, tokenizers 0.22.2, Jinja2 3.1.6.

## Preliminary split

Seed 42, validation fraction 0.1: 1,451 training dialogues, 160 validation dialogues, 200 reserved test task IDs, 13 excluded SFT dialogues. A reserved test ID is not a separately downloaded test conversation.

Train/validation/test task-ID intersections are zero. Train/validation normalized-instruction intersection is zero. `goal_map_sha256` is null: full environment task-instruction alignment and final split freezing are still pending. Test observations/rewards were not used to choose the context budget.

## Real-tokenizer inspection at max_length=2048

The implementation creates one supervised example for each assistant turn after removing the generic opening instruction/OK exchange. Thus dialogue counts and training-example counts differ.

| Measurement | Train | Validation |
|---|---:|---:|
| Dialogues | 1,451 | 160 |
| Candidate assistant-turn examples | 5,259 | 605 |
| Retained examples | 5,241 | 605 |
| Examples requiring older exchange removal | 188 | 27 |
| Skipped examples: latest exchange still exceeds budget | 18 (0.342%) | 0 |
| Raw length median | 975 | 1,011 |
| Raw length p95 | 1,934 | 2,022 |
| Raw length p99 | 2,988 | 2,682 |
| Raw length maximum | 15,624 | 6,450 |
| Fitted length p95 | 1,744 | 1,843 |
| Fitted length maximum | 2,048 | 2,048 |
| Supervised assistant suffix length median | 43 | 44 |
| Supervised assistant suffix length maximum | 192 | 103 |
| Label/history invariant failures among retained examples | 0 | 0 |

The raw length is the complete history prefix ending at the assistant turn, including the system prompt and chat-template tokens. The fitted length is after the existing `fit_history` removes old, complete user/assistant exchange pairs. No raw string/token slicing was applied to the latest exchange. Across the retained training examples, at most five older exchange pairs were removed. The maximum minimal system-plus-current-exchange length in train is 5,001 tokens, so simply removing history cannot retain every sample at 2,048.

### Label and padding checks

For every retained example, reconstructed the label logic in `src/webshop_lab/sft.py` with the actual tokenizer and independently tokenized the current assistant suffix. Verified:

- All preceding tokens, including system/user input and earlier assistant turns, have label `-100` and do not participate in this example's loss.
- The supervised labels exactly match the current assistant suffix, including template ending tokens, with nonempty supervision and a matching tokenized prefix.
- The system instruction and latest user/assistant pair are retained unchanged and total length is at most 2,048.
- For two differently sized real examples, reconstructed the current collator's right-padding contract: attention padding is 0 and label padding is `-100`.

This is an exhaustive token-level preprocessing check on retained train/validation examples plus a two-example padding-contract check. It does not execute PyTorch Trainer, gradients, GPU kernels, or model training and cannot establish downstream training success.

### Context recommendation

Keep 2,048 as a reasonable initial smoke-training budget: it retains 99.658% of training-turn examples under the current history policy, with zero label invariant failures. This is a preprocessing recommendation, not an optimal-context or task-success claim. Report 18 skipped train examples and the history-removal counts. Before formal experiments, review skipped/long examples with the training/data owners and decide whether to keep the budget, separately study 4,096, or adjust observations. Do not silently alter the shared config or claim every example is retained.

## Local artifacts and reproduction

All following paths are relative to the local checkout `webshop-stepwise-rl/`:

- Raw SFT: `data/raw/webshop_sft.json`.
- Audit: `artifacts/sft-audit.json`.
- Split files and manifest: `data/processed/`.
- Token/label report: `artifacts/sft-label-length-audit.json`.
- Inspection script: `artifacts/check_sft_labels.py`.
- Fixed tokenizer files: `artifacts/qwen-tokenizer/`; each SHA is in the inspection report.

```bash
PYTHONPATH=src .venv-tools/bin/python -m webshop_lab.cli audit --output artifacts/sft-audit.json
PYTHONPATH=src .venv-tools/bin/python -m webshop_lab.cli split
PYTHONPATH=src .venv-tools/bin/python artifacts/check_sft_labels.py
```

Raw data, generated splits, tokenizer assets and diagnostics remain local under ignored paths. No existing training source or configuration was modified. A final goal-map verified split, a two-optimizer-step GPU run, adapter saving/reloading, and real rollout evaluation remain separate tasks.
