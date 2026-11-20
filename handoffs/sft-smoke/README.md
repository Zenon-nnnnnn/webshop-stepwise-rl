# Two-update Qwen SFT engineering smoke handoff

Prepared 2026-11-17. Owner: Jun040504; intended GPU executor: HowseaonLink.

## Status and boundary

Status updated 2026-11-21: **the original pinned handoff completed target-device two-update training and fresh-process reload/generation, and its evidence was independently accepted**. Later script revisions have CPU verification only and must not inherit the original GPU result. See the acceptance ledger below. This folder remains an engineering handoff, separate from formal training and shopping-performance evidence.

The exact smoke manifest intentionally retains the preliminary offline split. Later full goal-map alignment was verified separately in PR #10/#15; formal data adoption still requires the Issue #2 owner's confirmation. These real-data subsets are explicitly authorized only as an engineering smoke: no environment actions, formal benchmark, final adapter selection, experiment-register row, or performance claim. The smoke adapter must not become the initialization for a formal model; start formal training from the original verified base after split freezing.

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

CPU verification completed: preparation, real-tokenizer subset check, Python syntax compilation and `run_smoke.py --help`. The original snapshot's GPU execution and fresh-process reload were subsequently accepted; that evidence does not establish GPU execution of this newer script revision. No model weights were downloaded on the preparation machine.

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

The original snapshot's required artifacts were subsequently supplied and accepted;
see the ledger below. Any new execution of the revised scripts requires its own
finite-step, changed-tensor, memory, save and fresh-process reload records. A CPU
review alone does not establish GPU behavior of those revisions.

## Earlier full-data work: summary

- Original SFT verified: 9,312,587 bytes, 1,624 dialogues, SHA-256 `721e46dba410cff900e187bccea91a456c38b105b6d14d068d7762afb7791de3`.
- 1,599 normalized unique shopping instructions, 8 reward/true_reward disagreements. These rewards are not SFT loss weights.
- Preliminary split: 1,451 train, 160 validation, 200 reserved test IDs, 13 excluded rows; task-ID intersections and train/validation instruction intersection are zero. Later complete-environment goal-map checks passed separately; formal owner adoption remains pending in Issue #2.
- Full train-turn preprocessing: 5,259 candidates, 5,241 retained, 188 history-trimmed, 18 skipped; validation: 605 retained, 27 history-trimmed, zero skipped.
- Label/history checks passed for all 5,846 retained examples; this is preprocessing evidence, not a GPU/gradient test.

The independent full-data report and detailed JSON were produced locally before this handoff. This follow-up changes only the PR handoff files; it does not merge or change main.

## Evaluation-side follow-up (2026-11-19)

CPU preflight now checks the same three pinned tokenizer/config hashes as the GPU runner before loading. Both paths require exact statistics from evidence/subset-tokenizer-check.json: train examples/skips/trims/max-length = 53/0/2/1934; validation = 15/0/2/1995. Tokenizer or count drift fails explicitly.

contracts.py supplies the actual list collator shared by CPU preflight and the GPU tensor wrapper. The CPU check examines its output on real unequal-length examples: original inputs/labels preserved, correct pad IDs, zero attention and ignored padding labels. GPU tensor checks remain required; CPU checks do not establish torch/Trainer behavior. Provenance records contracts.py too (five handoff files including config).

Nine CPU tests pass normally and under -O, including corrupt tokenizer/count/padding cases. Real-tokenizer preflight reproduces the exact statistics. Hyperparameters, subset bytes and expected manifests are unchanged. This revision is not GPU-validated. The original run belongs to an older snapshot; its subsequently shared artifacts were independently accepted as documented below.

## Target-device acceptance ledger (2026-11-21)

The original handoff at `bcaac8fe0bc3e1bd779f5d80fe8b8ae37168ac81` completed two optimizer updates on the deterministic 53/15-example smoke subset. Retained records show finite step losses 0.7474 / 0.8753, finite gradient norms, 56 changed LoRA tensors and matching recorded saved/reloaded LoRA digest `caf1c2e68b9e7099697a5dcb12cfc1aaa56752bf553baa7ec6eac524b4cd3753`. The fresh-process forward/generation check passed. These are reviewed target-device records; weights were not independently rehashed on the review machine.

[Original private evidence ZIP](https://github.com/Zqqq-qq/webshop-stepwise-rl/releases/download/data-v1/webshop-pr12-original-smoke-evidence.zip), SHA-256 `9521c84a6fe75c44c34aff02cd767778a02d7495fe8e62f4bd7a63302d34b109`. [Executor handoff](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/12#issuecomment-7008040535). Preserve the original evidence: do not relabel it as a run of `37634bc` or `f10b399`, and do not mark the checklist above as proof of those later scripts.

Separate [PR #14 acceptance](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/14#issuecomment-7008854291) covers pinned-loader offline loading, near-2048-token short-training capacity and fresh-process generation. [PR #18 review](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/18#pullrequestreview-5987404594) accepts the accessible local one-epoch candidate records and independently recomputed sanitized 160-task validation metrics. These are distinct runs with distinct adapters and evidence bundles; neither makes the two-update smoke adapter a formal initialization.

The remaining work is formal mapped-data adoption by Zqqq-qq, a run-specific adoption/execution decision in PR #14, and independent formal evaluation provenance/registration by QiuQiu-xuan. GPU smoke execution is no longer waiting for Jun's initial configuration. No rerun or main modification is requested by this status update.
