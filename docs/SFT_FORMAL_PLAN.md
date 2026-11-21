# Formal SFT candidate and acceptance gates

Status updated 2026-11-21: **formal configuration remains draft; target-device engineering acceptance and local candidate review are complete**. Owner: Jun040504. GPU executor: HowseaonLink. The SFT owner has now selected the original PR #18 adapter as B1 and frozen its identity below. Formal experiment registration and data/protocol gates remain separate.

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

1. Engineering evidence is accepted for PR #12's original snapshot and PR #14's
   pinned-loader long-input probe; see the evidence ledger below. This does not
   claim that later PR #12 script revisions were GPU-executed.
2. Zqqq-qq confirms the formal split using the full environment goal map:
   zero alignment mismatches, verified split health, disjoint task IDs and
   normalized instructions, source/split/goal-map SHA-256 and goal fingerprint.
   PR #10 contains environment evidence, but its existence is not owner sign-off.
3. The reviewed long-input probe supports the current batch/context starting
   settings. Preserve LoRA-only trainable names and assistant-only/padding mask
   checks. Resolve formal adoption and the execution/resume protocol below before
   calling any configuration frozen; capacity acceptance alone is insufficient.
4. Only if a new formal run is explicitly selected, copy the draft to a run-specific JSON under `configs/runs/`, set
   `configuration_status` to `frozen`, choose a fresh output directory, and
   commit the reviewed config. The status field is an operator marker, not
   automatic proof that these gates were satisfied.

Raw data, model weights, adapters and checkpoints stay outside Git. Do not change
the formal input files to reproduce the preliminary smoke subset.

## Execution and saved artifacts

For a separately approved new run after all gates pass, from the repository root
in the validated GPU environment (this command is not an instruction to rerun the
accepted local candidate now):

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
relying on a mutable `refs/main`. The CPU loader mocks validate argument flow. The separate target-device PR #14
probe verified actual pinned-cache offline loading; its scope and source identity
are recorded below. Future runs must retain their own file/version evidence.

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
PR #8 was merged at `21f3c83da9c9f599665cd89df479fbca7f512b88`; its final evaluator
checks revision/source provenance and supports frozen-protocol parity. Use that
accepted evaluator or a reviewed successor, rather than assuming this older PR
branch already contains it. PR #18's historical sampling metadata lacks the newer
`model_source` field: retain its pinned snapshot/config/weight-hash sidecars and
state this limitation instead of fabricating metadata. Null resolved values require
the same verified local-file hash manifest on both arms.
Keep the reserved test tasks untouched during tuning.

## Responsibility

As specified in [TEAM.md](TEAM.md), Jun040504 owns SFT implementation, labeling,
configuration, reload acceptance and interpretation. HowseaonLink owns the GPU
environment and training execution. Zqqq-qq owns data protocol/final split and
subsequent reward/RL; QiuQiu-xuan owns independent evaluation. SFT ownership
does not transfer all data preparation, GPU execution or evaluation to Jun.

## Capacity review follow-up (2026-11-19)

The runner records peak_allocated_gib, peak_reserved_gib and device_total_gib in GiB (2**30 bytes). Statistics reset before model loading and are sampled after final evaluation/saving, with CUDA synchronized. Legacy peak_allocated_gb remains an alias. Reserved memory is allocator reservation, not device-wide used memory or a guarantee of headroom; retain external monitoring and failure logs.

The previously reported old-loader figures are superseded for this acceptance decision by the pinned-loader long-input evidence. The reviewed PR #14 probe used 16 train examples of 2027–2048 tokens and four validation examples of 2043–2048 tokens, batch 1 / accumulation 8 / two updates. It reported 5.7945 GiB peak allocated and 10.3281 GiB peak reserved on a 15.9283 GiB device. Both step losses and gradient norms were finite. A separate fresh-process supplement matched the recorded LoRA digest and passed a finite forward/eight-token generation check. These checks establish the recorded short-run engineering path, not a guarantee against every future OOM or a shopping score.

## Evidence ledger and adopted B1 identity (2026-11-21)

The following are artifact audits of target-device records, not independent GPU execution on the review machine. Original weights and raw datasets remain outside Git.

| Record | Accepted scope | Immutable identity / review |
|---|---|---|
| PR #12 original two-update smoke | Finite step loss/gradients, actual LoRA changes, save and fresh-process reload/generation | Original handoff `bcaac8fe0bc3e1bd779f5d80fe8b8ae37168ac81`; [original bundle](https://github.com/Zqqq-qq/webshop-stepwise-rl/releases/download/data-v1/webshop-pr12-original-smoke-evidence.zip), SHA-256 `9521c84a6fe75c44c34aff02cd767778a02d7495fe8e62f4bd7a63302d34b109` |
| PR #14 pinned-loader capacity probe | Offline loading, near-2048-token training/evaluation and short capacity | Source `0fc56073740d7fbf8b9dc320451517fbb2cc7a40`; [capacity bundle](https://github.com/Zqqq-qq/webshop-stepwise-rl/releases/download/data-v1/webshop-pr14-0fc5607-capacity-acceptance.zip), SHA-256 `52f273ddbbb597dc440a72d609c55ecd4dd2e369a3d9d41ebccadde31126c078` |
| PR #14 reload supplement | Same recorded adapter identity, fresh process, finite forward and short generation | [Supplement](https://github.com/Zqqq-qq/webshop-stepwise-rl/releases/download/data-v1/webshop-pr14-0fc5607-reload-generation-supplement.zip), SHA-256 `3a6a7dc1daef57b3c5e43a78105831f0bd423bae75dee139a6bc7d43be0af4a9`; [acceptance](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/14#issuecomment-7008854291) |
| PR #18 local one-epoch candidate | Traceable retained execution records and recomputed sanitized validation metrics | Review head `ba768137780e993abf628077b239e1b1ffb49c1c`; [bundle](https://github.com/Zqqq-qq/webshop-stepwise-rl/releases/download/data-v1/webshop-pr18-qwen25-sft-and-validation-evidence.zip), SHA-256 `c62a1a97ac259a4a0f5ffb7d70e054ca2523c14d9866f53a40b56562c6aaae3b`; [Jun review](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/18#pullrequestreview-5987404594) |

SFT-owner decision (2026-11-21, explicitly authorized by the user): adopt PR #18's original SFT adapter as B1 and the common starting policy for later reward-arm comparisons. Its identity is frozen in [configs/identities/b1-pr18.json](../configs/identities/b1-pr18.json). This closes the SFT-owner model-selection decision. It does not close data-owner sign-off, formal evaluation registration or reward/budget protocol decisions. No retraining is requested.

The candidate used 5241 retained training examples / 605 validation examples, 656 recorded optimizer steps, seed 42, and validation token loss 0.400295227766037. Its executed logging interval was 5, whereas the unexecuted draft above uses 1. Keep the original config bytes and actual wrapper; do not relabel the run as an execution of this draft. Resumed steps and runtime changes are part of its identity.

Candidate adapter file SHA-256: `c39e2a78e7c5973328e9d2c23fb6320e566352e63cc089562488f3dd46ae5f9c`. Recorded trained/reloaded LoRA tensor digest: `93760afbdf4b2a24d4d89bfa2e986ec2db7576ce516f9fc439e485558f304640`. The reviewer checked agreement between records, not the actual tensor payload on the target machine.

The sanitized 160-task validation derivatives independently reproduce purchases 57/160 → 158/160, full-reward successes 19/160 → 50/160, mean terminal reward 0.2388541667 → 0.7059375, and invalid actions 585/1219 → 18/557. The paired reward difference is +0.4670833333, with a 10000-resample seed-42 interval [0.4034375, 0.5301041667]. Raw page/action trajectories were not independently replayed in this review. Both arms used the same optimized environment path; comprehensive equivalence to unchanged upstream scoring is not established here. These are local candidate val results, not final-test or RL results.

### Runtime synchronization patch policy

PR #18 resumed from checkpoints 100 and 600. Its final wrapper changed Trainer's last-batch predicate from `(step + 1) == steps_in_epoch` to `(step + 1 + steps_skipped) == steps_in_epoch`. The disclosed patch and retained wrapper agree byte-for-byte. An independent integer schedule check gives 441 remaining microbatches after step 600: 55 synchronization events with the old predicate, 56 with the patch, including the final one-sample remainder. This supports the reported failure explanation; it does not prove effective gradient scaling, sample order or restored optimizer/scheduler/RNG equivalence.

The first launch command is reconstructed, and microbatch/scaling/state-restoration comparisons were not captured. Preserve these limitations. The 29.31 GiB reserved-memory anomaly is excluded from physical-capacity conclusions.

This documentation update does not transplant the in-memory patch into production or add resume support to the shared SFT command. Keep it versioned with the candidate evidence. Before a future run relies on resume, review a version-specific implementation and verify partial-accumulation scaling, consumed batches and optimizer/scheduler/RNG restoration. A future fresh run uses a new directory and the original pinned base, not the two-update smoke adapter. Never bypass the draft gate or overwrite existing outputs to reproduce a historical run.

### Remaining formal experiment gates

1. Zqqq-qq explicitly confirms the mapped data version in [Issue #2](https://github.com/Zqqq-qq/webshop-stepwise-rl/issues/2): train 1451 / val 160 dialogues, 200 reserved test IDs, 13 exclusions; goal fingerprint `e69c9ae2a0aad88977c44a7baf49f1f5ef7918ba8317923c4ae24dcc1f5638d8`; goal-map SHA-256 `6b5d9dfceee913828871b8dff62fd59720a40306738a92a496d1b0fb6838fa88`.
2. The SFT owner has adopted this exact adapter under its disclosed historical execution protocol. How must resolve the local original adapter path and recheck the locked file hashes before use. Train SHA-256 is `00e9a068f90011ba85eb42e26dc3e048241f94a77820c4e64b8df999f54b045e`, val SHA-256 is `2ac4dcb4388735c031be4ffc46eb3454301340238f15c9bef01f616b2216b8b3`. The versioned identity lock records actual config/wrapper/adapter hashes and links the retained execution evidence. Record environment optimization and evaluation provenance in each run-specific record. Do not change historical configs to say frozen retroactively.
3. Qiu independently accepts evaluation provenance and decides what can enter the formal register. Preserve full raw trajectories/meta/configs and evaluator/source identities through an accessible private channel. Frozen val IDs SHA-256 is `45440d724b33e69e13916d0adbe243cc63ce3bd67677b8220feff9f30b866c36`; the reserved test set remains untouched.
4. If data or protocol changes invalidate adoption, scope and review the affected rerun first. No new training is launched by this update; the formal JSON retains `configuration_status: draft`.

## Applying the frozen B1 identity

The lock is an identity/adoption record, not a JSON accepted by the SFT or rollout command. Its status freezes the selected artifact, not all experiment settings. Keep the original formal-training draft guard unchanged because this decision adopts an existing artifact rather than launching that unexecuted draft.

Use the PR #18 original `output/adapter/` directory. On the target device, verify both `adapter_model.safetensors` and `adapter_config.json` against `configs/identities/b1-pr18.json`; the latter fixes LoRA scaling and module configuration as well as the weight file. Verify the same pinned base/tokenizer file identities. Stop on a mismatch and identify the different artifact rather than editing the lock to fit it.

For PPO execution configs, set `adapter` to that verified local directory, `adapter_sha256` to `c39e2a78e7c5973328e9d2c23fb6320e566352e63cc089562488f3dd46ae5f9c`, `revision` to `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, and `base_weights_sha256` to the lock's base-weight digest. Both initial arms must use this same artifact. A resumed round loads its own arm's PPO checkpoint; record that new checkpoint separately and preserve the B1 source.

For evaluation, a B1 config points to the same original SFT adapter; PPO-updated policies must be labeled separately. Each run retains its actual execution config hash, source, environment and task/decoding provenance. Selecting B1 does not by itself authorize formal result registration, final test execution, merging a PR, or changing main.
