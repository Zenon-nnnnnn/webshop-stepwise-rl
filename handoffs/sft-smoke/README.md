# Two-update Qwen SFT engineering smoke handoff

Prepared 2026-11-17. Owner: Jun040504; intended GPU executor: HowseaonLink.

## Status and boundary

Ready for GPU execution, **not a completed GPU training run**. CPU/tokenizer checks below were executed locally; no loss, VRAM, gradient, adapter or shopping-performance result is claimed. This folder is standalone and does not modify the repository's main branch or training implementation.

The current split has not passed complete-environment goal-map verification. These real-data subsets are explicitly authorized only as an engineering smoke: no environment actions, formal benchmark, final adapter selection, experiment-register row, or performance claim. The smoke adapter must not become the initialization for a formal model; start formal training from the original verified base after split freezing.

## Decisions for the executor

| Question | Decision and reason |
|---|---|
| Base model | Qwen/Qwen2.5-1.5B-Instruct at revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; use the already verified local snapshot. Weight and tokenizer hashes are checked before loading. |
| Quantization | Existing 4-bit NF4 + double quantization; compute dtype follows CUDA BF16 support, otherwise FP16, as in the repository loader. |
| LoRA modules | `q_proj`, `v_proj`, rank 8, alpha 16, dropout 0.05; bias none, causal LM. Preserve the initial narrow configuration to check the current pipeline before adding experimental variables. This is not an optimal-module claim. |
| Context | 2,048 total tokens including template, history and target; keep the existing whole-exchange removal policy. The full-data analysis retained 99.658% of train-turn examples. |
| Data | Deterministically sampled real-data subset: 16 preliminary train dialogues and 4 preliminary validation dialogues; seed 42 after sorting task IDs. No reward-based selection and no test task IDs. |
| Training budget | Exactly 2 optimizer updates, batch size 1, gradient accumulation 8, learning rate 2e-4, seed 42. Nominally 16 training microbatches. This is not two dialogues or two full epochs. |
| Logging | Every update; reject nonfinite loss/grad norm, require final global_step=2 and actual LoRA parameter changes. |
| Validation | One tiny validation pass after the two updates. Its loss is a pipeline check, not a generalization estimate. |
| Saving | Save adapter, tokenizer, config, Trainer state, steps and run evidence in a fresh `outputs/` directory. |
| Reload | Run a second command in a new Python process; check exact LoRA tensor hash, finite forward loss and short greedy generation with an explicit attention mask. |

The executor uses a standalone Trainer wrapper with the repository's loader, template and history functions. It keeps the initial learning settings, but disables epoch evaluation/checkpoints during the two updates, performs one explicit evaluation afterwards, and saves a final adapter. This avoids the misleading suggestion that it ran the full production entry point or an entire epoch.

## Data preparation and local evidence

`prepare.py` builds `smoke-config.json`, `data/train-smoke.json`, `data/val-smoke.json` and `data/subset-manifest.json`. The manifest records source/subset hashes, task IDs, the split manifest SHA and source implementation hashes. All generated data remain outside Git via this folder's `.gitignore`.

The prepared subset was checked with the actual tokenizer at the pinned model revision:

| Check | Train | Validation |
|---|---:|---:|
| Dialogues | 16 | 4 |
| Retained assistant-turn examples | 53 | 15 |
| Skipped examples | 0 | 0 |
| Examples needing older-exchange removal | 2 | 2 |
| Maximum fitted length | 1,934 | 1,995 |
| Label/history invariant failures | 0 | 0 |

Prompt/history tokens have labels -100; current assistant suffix tokens match the independently tokenized template suffix. System and current exchange remain intact. A two-example variable-length padding contract check passes. The GPU script additionally checks actual collated torch tensors.

CPU verification completed: preparation, real-tokenizer subset check, Python syntax compilation and `run_smoke.py --help`. GPU execution and fresh-process reload are pending. No model weights were downloaded on the preparation machine.

## Execution on the GPU host

### GitHub branch delivery (no raw data in Git)

This folder is published under `handoffs/sft-smoke/` on the PR branch. The raw subsets are deliberately omitted. Use the original SFT from the private `data-v1` Release, verify its SHA, and make it available at `data/raw/webshop_sft.json` in the repository checkout. Rebuild the exact smoke subsets locally; no separate ZIP transfer is required.

From the repository root on the PR branch:

```bash
conda activate webshop-gpu
PYTHONPATH=src python -m webshop_lab.cli audit --output artifacts/jun-sft-audit.json
PYTHONPATH=src python -m webshop_lab.cli split --output artifacts/jun-offline-split
cd handoffs/sft-smoke
python prepare.py --repo ../.. --split-dir ../../artifacts/jun-offline-split
python check_subset.py --repo ../.. --tokenizer /path/to/verified/qwen/snapshot
python run_smoke.py run --repo ../.. --model-path /path/to/verified/qwen/snapshot
python run_smoke.py reload --repo ../.. --model-path /path/to/verified/qwen/snapshot
```

Use the exact local snapshot path from the executor's earlier successful model check. The isolated offline split does not overwrite any existing formal `data/processed/` files. `prepare.py` compares its result against `evidence/subset-manifest.json` and stops if it differs. The handoff's English full-data report, audit summary, token/label evidence and expected subset manifest are in `evidence/`. Do not execute commands against main or merge the PR as part of this handoff.

### Alternative: privately transferred standalone folder

Transfer this complete folder privately, including its `data/` directory, beside the existing repository checkout. Do not put the private subsets into a Git commit. Activate the already tested `webshop-gpu` environment. The expected stack is Python 3.10, torch 2.8.0+cu128, transformers 4.56.2, peft 0.17.1, accelerate 1.10.1 and bitsandbytes 0.48.2. Capture actual `pip freeze` with the execution logs.

From this folder, replace the two paths with the actual checkout and local HF snapshot (not a remote model name):

```bash
conda activate webshop-gpu
python run_smoke.py run --repo /path/to/webshop-stepwise-rl --model-path /path/to/verified/qwen/snapshot
```

After it exits successfully, launch a **new process**:

```bash
python run_smoke.py reload --repo /path/to/webshop-stepwise-rl --model-path /path/to/verified/qwen/snapshot
```

No WebShop service, shopping rollout or external model API is needed. The scripts use the local model directory and check the official weight SHA plus both tokenizer hashes. If source hashes, data hashes or model hashes differ, stop and reconcile versions; do not remove the checks. If an output directory already exists, choose a new output path in the config and record the changed config SHA; do not overwrite evidence.

Optional regeneration and tokenizer preflight (only if the source data and preliminary split are present in that checkout):

```bash
python prepare.py --repo /path/to/webshop-stepwise-rl
python check_subset.py --repo /path/to/webshop-stepwise-rl --tokenizer /path/to/verified/qwen/snapshot
```

Regeneration must reproduce the subset hashes in the handed-off manifest. The runner checks the local subset bytes before training.

## Acceptance checklist

- [ ] Base weight, tokenizer, input subset and implementation hashes match.
- [ ] All trainable parameters are LoRA parameters; the trainable parameter count is recorded.
- [ ] Labels ignore prompt/history/padding and keep current assistant supervision; actual tensor padding check passes.
- [ ] Exactly 2 optimizer updates; both step losses and any logged gradient norms are finite. Loss need not decrease in only two steps.
- [ ] At least one LoRA tensor changes; max absolute change and changed tensor count are recorded.
- [ ] Tiny validation forward loss is finite; do not interpret it as shopping success.
- [ ] Adapter/config/tokenizer/state/logs are saved; load/preprocess/train elapsed time, total time, peak allocated/reserved VRAM and versions are recorded.
- [ ] New-process reload preserves the exact LoRA state hash and passes a forward and generation smoke.
- [ ] Artifacts are shared privately and independently checked; no formal benchmark/register claim is made.

Return `outputs/qwen15b-2step-seed42/` with `run-evidence.json`, `reload-evidence.json`, `steps.jsonl`, `trainer-state.json`, `config.json` and `adapter/`, plus console error/success logs, actual pip freeze, the source commit and manifest. On failure, return the error and partial artifacts; do not label the run passed or replace failures with synthetic numbers.

## Review follow-up: self-contained provenance and always-on checks

Updated 2026-11-18 in response to HowseaonLink's two non-blocking review items.
Both run and reload evidence now contain `runtime_provenance`: Python/platform,
the six main package versions, the installed package name/version inventory,
torch CUDA build and cuDNN version, observed repository Git HEAD, tracked-dirty
status, five source/config file hashes, and the hashes of the three handoff scripts
plus their config (four files total). Python's optimization level is recorded too.
The existing weight/subset checks remain mandatory.

For a checkout reconstructed through the API, optionally add
`--source-commit FULL_40_CHARACTER_COMMIT_SHA` to **both** run and reload commands.
It records an operator declaration separately from observed Git HEAD and reports
whether they match. An API copy can have a different local Git commit; never
present the declared commit as independently verified. Reviewers must compare the
recorded file hashes with the target GitHub commit. No credentials or environment
variables are dumped. Keep external console logs and pip freeze as well.

All validation in `prepare.py`, `check_subset.py` and `run_smoke.py` now uses
explicit `require` checks that raise ValueError, including under `python -O`.
This covers source/data/model hashes, masks, finite losses/gradients, two updates,
actual parameter changes, output isolation and exact reload identity. Checks in
unchanged production modules are outside this patch.

CPU follow-up validation: seven tests pass both normally and with `-O`; a tampered
source hash is rejected under optimization. Real-tokenizer preparation/checks
under `-O` reproduce the existing config/manifest/subset bytes and 53/15 examples,
zero skips, 2/2 history removals, with mask/padding checks passing. These tests do
not execute GPU training or verify the actual runtime_provenance GPU fields.

Preserve the reviewed execution contract: use the isolated preliminary split,
not a goal-map-aligned formal split, when reproducing this exact manifest. The
existing three-file manifest gate remains unchanged to keep the dataset/config
hashes stable; data.py and cli.py are now additionally recorded in runtime
provenance, but their hashes are observations rather than additional frozen gates.
prepare.py deterministically regenerates tracked smoke-config.json and should
leave its bytes unchanged. Invoke the scripts as files as documented; module-style
imports from a repository-root Python session are not part of this interface.

Do not replace scripts midway through a running GPU job. If the original pinned
commit was already executed, preserve its evidence unchanged and return its
source hashes, actual versions and console logs as a separate provenance sidecar.
Do not relabel it as a run of this update. A new execution must use a fresh output
directory and matching run/reload script bytes; never overwrite an older result.

Acceptance remains pending until the executor supplies finite per-step loss and
gradient evidence, changed LoRA tensors, allocated/reserved VRAM, saved outputs,
and fresh-process reload hash/forward/generation evidence. A review saying the
CPU preflight passed is not GPU training evidence.

## Earlier full-data work: summary

- Original SFT verified: 9,312,587 bytes, 1,624 dialogues, SHA-256 `721e46dba410cff900e187bccea91a456c38b105b6d14d068d7762afb7791de3`.
- 1,599 normalized unique shopping instructions, 8 reward/true_reward disagreements. These rewards are not SFT loss weights.
- Preliminary split: 1,451 train, 160 validation, 200 reserved test IDs, 13 excluded rows; task-ID intersections and train/validation instruction intersection are zero. Complete-environment goal-map verification remains pending.
- Full train-turn preprocessing: 5,259 candidates, 5,241 retained, 188 history-trimmed, 18 skipped; validation: 605 retained, 27 history-trimmed, zero skipped.
- Label/history checks passed for all 5,846 retained examples; this is preprocessing evidence, not a GPU/gradient test.

The independent full-data report and detailed JSON were produced locally before this handoff. This follow-up changes only the PR handoff files; it does not merge or change main.
