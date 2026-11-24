# Issue #48: fixed-feature CPU head-only optimizer diagnosis

This is Yuxiao's addendum to Qiu's original [#49 evaluation audit](../../docs/PRM_EVALUATION_AUDIT.md), preserving Qiu's checker, independent original-memory audit, and attribution. The experiments here are a separate **re-extracted fixed B1 feature** analysis, not an extension of the historical neural/LoRA training.

## Evidence and reproduction

- Supplementary ZIP (How, publicly published Release): SHA256 `2d1c2763783759adb47bf8eeb056cc36b7134cd030a2f6b274ea32df68a86aa4`. The program verifies this whole ZIP and each listed member SHA before loading.
- Stored `F.npy` SHA256 `8efb5c98d3ce2ef43a948593715ff8c34ecd94d3ca1ea9288444d3d10b5592b6`. Actual shape **(8, 151936)**, dtype float64; these are only the summed logits. The supplementary `F.meta.json` incorrectly describes a stored extra `N` column. We verify ordered task IDs / labels / assistant-turn counts, then explicitly append `N=[3,3,10,10,3,3,3,3]` for **(8, 151937)** without modifying the original attachment.
- Historical `training-state.pt` SHA256 `c19461e3b22b822e5162cc3fb93e0825c5e398fddd26c55aa6a1326b87895c3c`. CPU PyTorch `weights_only=True` loading checks optimizer FP32 moments, finite nonnegative variance, 114 states, 512 counts, LR/WD/betas/epsilon; it does not restore a model or prove optimizer-ID/name linkage.
- Actual code: [`scripts/prm_cpu_head_diagnostic.py`](../../scripts/prm_cpu_head_diagnostic.py). Cloud workflow: [`.github/workflows/prm-cpu-head-diagnostic.yml`](../../.github/workflows/prm-cpu-head-diagnostic.yml). Its JSON artifact stores every 64 exposures' eight predictions, labels, MSE, maximum error, clip count, optimizer step count, input hashes, and provenance.

Download the exact supplementary ZIP from [the data-v1 release](https://github.com/Zqqq-qq/webshop-stepwise-rl/releases/download/data-v1/prm-evidence-supplement-20261124.zip), install pinned `numpy==2.1.3` and CPU `torch==2.8.0`, then run:

```sh
python -m unittest discover -s tests -p 'test_prm_cpu_head_diagnostic.py' -v
python scripts/prm_cpu_head_diagnostic.py \
  --archive /path/to/prm-evidence-supplement-20261124.zip \
  --output artifacts/prm-cpu-head/results.json
```

## Frozen diagnostic protocol

- Same eight rows in historical order, FP32 zero linear head, AdamW (`betas=.9/.999`, `eps=1e-8`, `wd=.01`), clip global norm at 1. No model forward, no LoRA updates, no GPU, no dev/test.
- Batch 1: LR `1e-7`, `3e-7`, `1e-6`; up to 512 optimizer steps; evaluate all eight after each 64 updates. Stop at strict MSE `<.001` **and** max absolute error `<.1`, or non-finite values.
- If none meets that fit gate: choose the LR with lowest **final eight-training-example MSE**, not a dev score, and compare batch 1 vs 8 at matched 512 sample exposures. Thus 512 vs 64 optimizer steps respectively. Evaluate every 64 exposures.
- The JSON includes both training curves and full prediction/label replay; independent reviewers can recompute each metric.

## Preliminary NumPy cross-check (not the cloud PyTorch signoff)

| 512 exposures | Optimizer steps | MSE | Max absolute error | Clip events |
|---|---:|---:|---:|---:|
| LR 1e-7, batch 1 | 512 | 0.13999064 | 0.64872158 | 512 |
| LR 3e-7, batch 1 | 512 | 0.18139606 | 0.60981709 | 512 |
| LR 1e-6, batch 1 | 512 | 0.36911613 | 0.91935408 | 512 |
| LR 1e-7, batch 8 | 64 | 0.12317892 | 0.54773194 | 64 |

The LR 1e-7 / clip 1 NumPy result reproduces the published reference `MSE .13999/max .64871` at the displayed precision; this is a useful independent arithmetic check, not a guarantee of PyTorch kernel bitwise identity. LR 3e-7 was nonmonotonic: MSE reached 0.09933 after 448 updates then rebounded to 0.18140 at 512. None reached the gate. Based solely on the *final fit* metric, LR 1e-7 is the conservative next diagnostic control, not an approved GPU setting.

## Evidence hierarchy and remaining work

1. **Fixed-representation expressivity:** the eight rows have numerical rank 8 with condition about 205, so a linear least-squares solution exists on those frozen re-extracted inputs. This does not imply gradient training can find it or that neural-state features are equivalent.
2. **Gradient memorization:** historical neural PRM at 512 updates remains MSE 0.13853125, max error 0.64872032, a failed eight-example memory gate. New *fixed-feature head-only* runs are optimization diagnostics, not historical neural PRM reproduction.
3. **Generalization / step attribution:** not evaluated, no test authorization. CPU PyTorch result and independent Zqq/Tutor replay must be checked before any equipment-side execution order. No automatic 1200-1500 step plan or GPU launch follows.

The original contemporaneous `F` was **not** saved; this ZIP contains How's explicitly **re-extracted** features. Do not describe it as the original same-run feature export. Original #49 full tokenizer/raw-step provenance checks require different inputs from this supplementary ZIP and are not silently declared completed by this run.
