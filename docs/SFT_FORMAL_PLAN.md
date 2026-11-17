# Formal SFT candidate and acceptance gates

Status: **draft, not frozen and not executed**. Owner: Jun040504. GPU executor:
HowseaonLink. This document is a plan, not evidence that training succeeded.

## Candidate configuration

`configs/sft-qwen15b-formal-draft.json` starts from the two-update handoff in
[PR #12](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/12).
The SFT command refuses configurations with `configuration_status: draft`.

| Item | Candidate |
|---|---|
| Base | Qwen/Qwen2.5-1.5B-Instruct |
| Hub revision | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Quantization | 4-bit NF4, double quantization, bf16 if supported else fp16 |
| Adapter | q_proj/v_proj, rank 8, alpha 16, dropout 0.05; base weights frozen by PEFT |
| Context | 2048 total tokens |
| Optimization | 1 epoch, batch 1, accumulation 8, LR 0.0002, seed 42 |
| Schedule | Existing runner: cosine, warmup ratio 0.03, gradient checkpointing |
| Logging | Every optimizer update (`logging_steps=1`) |
| Inputs | `data/processed/train.json`, `data/processed/val.json`, from the accepted goal-map-aligned split |
| Output | New directory per run; replace `RUN_ID`, never reuse a smoke output |

The 2048-token starting point comes from the preliminary tokenizer audit:
train 5,241 retained / 5,259 candidate examples, 188 history-trimmed, 18 skipped;
validation 605 retained, 27 history-trimmed, zero skipped. Recompute these counts
on the accepted formal files; agreement is expected only if input hashes and
tokenizer/formatting are unchanged. Loss supervises the current assistant suffix;
system, user, previous assistant turns and padding have labels -100.
These checks do not prove 2048 is optimal or that training improves shopping.

## Required before freezing a runnable copy

1. HowseaonLink returns PR #12's actual two-update logs, finite loss and gradient
   evidence, changed LoRA tensors, peak VRAM, duration, versions, and successful
   fresh-process adapter reload. Investigate failures before formal training.
2. Zqqq-qq confirms the formal split using the full environment goal map:
   zero alignment mismatches, verified split health, disjoint task IDs and
   normalized instructions, source/split/goal-map SHA-256 and goal fingerprint.
   PR #10 contains environment evidence, but its existence is not owner sign-off.
3. Jun040504 and HowseaonLink review measured capacity and freeze the settings.
   Check actual trainable names are LoRA-only and the padded assistant-only mask
   remains correct. Do not infer training capacity from inference memory.
4. Copy the draft to a run-specific JSON under `configs/runs/`, set
   `configuration_status` to `frozen`, choose a fresh output directory, and
   commit the reviewed config. The status field is an operator marker, not
   automatic proof that these gates were satisfied.

Raw data, model weights, adapters and checkpoints stay outside Git. Do not change
the formal input files to reproduce the preliminary smoke subset.

## Execution and saved artifacts

After acceptance, from the repository root in the validated GPU environment:

```bash
# Create artifacts first. Replace RUN_ID with the reviewed run identifier.
mkdir -p artifacts
set -o pipefail
PYTHONPATH=src python -m webshop_lab.cli sft \
  --config configs/runs/sft-qwen15b-RUN_ID.json \
  2>&1 | tee artifacts/sft-qwen15b-RUN_ID-console.log
```

Use the normal epoch budget; do not pass the smoke's `--max-steps 2` override.
The runner refuses an existing nonempty output directory. It evaluates and saves
at epoch boundaries, retains at most two Trainer checkpoints, and writes the final
adapter/tokenizer to `OUTPUT/adapter/`. It now explicitly saves
`OUTPUT/trainer_state.json` (including Trainer log history), `config.json` and
`run.json`. There is no automatic best-checkpoint selection: the delivered B1 is
the final adapter for this one-epoch candidate. Keep console logs even on failure;
run.json is written only after successful training, saving and evaluation.

The run record contains config/train/val hashes, requested/resolved model versions,
source commit, retained/skipped counts, GPU, torch version, peak allocated memory,
validation loss, and elapsed train/save/evaluation seconds (excluding model loading
and preprocessing). Preserve the full environment freeze, total wall time, split
manifest, goal-map hash and fingerprint alongside it; these are not all collected
automatically by the SFT runner. Hash the adapter files and record their shared
storage location. Never enter token loss as shopping reward in the result register.

## Version provenance and offline loading

The shared loader passes optional `revision` to both tokenizer and model. Qwen
SFT, B0, B1 and exploration configs use the same exact commit. Legacy configs
without a revision keep their prior behavior; formal comparisons should use a pin.
SFT `run.json` and rollout `.meta.json` include `model_source` with requested and
resolved revisions. A resolved exact-pin mismatch or model/tokenizer disagreement
raises an error. A local directory may have no resolved Hub commit; null means
unknown, not verified. In that case retain file hashes such as PR #12's verified
snapshot manifest. A revision argument cannot authenticate arbitrary local files.

Pre-download the pinned snapshot in the executor's normal Hugging Face cache,
then use the same model ID and revision with `HF_HUB_OFFLINE=1`. This avoids
relying on a mutable `refs/main`. The CPU loader mocks validate argument flow,
not a real cached-model load; the executor must verify the actual offline path.

PR #12 hashes the old production source and will intentionally fail its source
guard if these files are mixed into that handoff. Execute its original pinned
branch as documented, or explicitly regenerate/review a new handoff manifest;
do not silently bypass its hash checks. These production changes do not alter
the already delivered smoke files.

## Reload and B0/B1 acceptance

In a fresh process load the final adapter onto the same pinned base and compare
saved/reloaded LoRA tensor hashes, followed by a finite forward pass and generation.
Record commands, source/config/base/adapter hashes, environment and output. PR #12's
reload checks demonstrate the intended contract, but its smoke-specific source and
data hashes must not be assumed to validate the formal adapter unchanged.

QiuQiu-xuan supplies the accepted evaluator and fixed validation protocol;
HowseaonLink executes B0/B1 on the same tasks, prompt, context, decoding settings
and environment. Only the adapter differs. Repoint a run-specific B1 eval config
to the formal adapter, retain matching pinned model revisions, and record JSONL,
metadata, exact configs/IDs and evaluator commit. Confirm full task coverage and
parity before comparing terminal shopping reward and full-reward success rate.
PR #8's existing parity fields do not check `revision` or `model_source`; manually
compare the requested and resolved versions in both metadata files as an additional
gate until that evaluator check is extended. Null resolved values require the same
verified local-file hash manifest on both arms.
Keep the reserved test tasks untouched during tuning.

## Responsibility

As specified in [TEAM.md](TEAM.md), Jun040504 owns SFT implementation, labeling,
configuration, reload acceptance and interpretation. HowseaonLink owns the GPU
environment and training execution. Zqqq-qq owns data protocol/final split and
subsequent reward/RL; QiuQiu-xuan owns independent evaluation. SFT ownership
does not transfer all data preparation, GPU execution or evaluation to Jun.
