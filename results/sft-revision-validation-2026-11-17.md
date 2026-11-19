# Pinned model revision and formal SFT draft: validation

Status: CPU validation completed; GPU/offline loading and formal training not executed.

## Change

- Forward optional revision to both Hugging Face loaders, with the existing
  no-revision behavior preserved. Pin the four Qwen SFT/evaluation/exploration
  configs to `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- Record requested/resolved base and tokenizer versions in SFT run.json and
  rollout metadata. Reject resolved commit conflicts; retain null for unknown
  local provenance rather than fabricating an actual revision.
- Explicitly save Trainer state and record train/save/evaluation elapsed time.
- Provide an executable-schema formal SFT draft, blocked while its status is
  draft, and an English acceptance/execution/reload plan.

## Local checks

Source base: main `eb76f4efe35f71842ae80a8020875148ff10cd1f`.
The local checkout started at `9df3063`; all six files changed through `eb76f4e`
were fetched at that commit before applying this patch. No production source
changed between those two base commits.

| Check | Result |
|---|---|
| `PYTHONPATH=src python3 -m unittest discover -s tests -v` | 15 passed: 8 existing, 7 new |
| Integration with PR #8 source at `ee85d06b278e2b4173d56bcfb4ba8cb12757ddac` | 61 passed: 54 evaluation-branch tests plus 7 new |
| `python3 -m compileall -q src scripts tests` | passed |
| `python3 scripts/check_repository.py` | passed |
| `git diff --check` | passed |
| `PYTHONPATH=src python3 -m webshop_lab.cli sft --help` | passed |
| Audit of the existing SHA-verified local SFT | passed; 12 original train/test ID overlaps reproduced |

The seven added tests exercise exact-pin propagation through the quantized
adapter-load path; legacy compatibility; mismatched resolved commit rejection
before training preparation; model/tokenizer disagreement; honest unknown local
provenance; requested branch vs resolved commit; and draft refusal before GPU
imports/output creation. Loaders are mocks, not actual weight downloads or CUDA.

The integration tree combined the reviewed PR #8 source and these source changes
locally; it is not a merged Git commit and does not change either remote branch.
The private audit output and full test logs remain local. No raw data or weights
are included in this PR.

## Outstanding acceptance

- Actual pinned-cache offline load and GPU training/save/reload, including loss,
  gradients, LoRA tensor changes, peak memory, duration and environment versions.
- PR #12's original source-hash guard must not be bypassed to mix this patch into
  its pinned smoke; use its documented branch or a separately reviewed new manifest.
- Formal split sign-off and final run-specific config freeze after smoke evidence.
- Real same-task B0/B1 evaluation. PR #8 does not automatically compare revision
  fields yet; manually compare the recorded source versions and any local manifests.

No training success, shopping improvement, or RL result is claimed.

## 2026-11-19 capacity-record follow-up

Added allocated/reserved allocator peaks and total device capacity with explicit GiB units, reset before loading and synchronization before recording. Legacy allocated field retained. The branch suite now has 16 passing CPU tests, including a mocked memory conversion test. These are not GPU measurements. The updated offline loader and representative long-sequence capacity still require executor validation; formal config remains a draft.
