# Data readiness and release-copy audit — 2026-11-17

## Current status: original data available

**The missing-SFT blocker is resolved.** The team's private [data-v1 Release](https://github.com/Zqqq-qq/webshop-stepwise-rl/releases/tag/data-v1) now provides the original SFT, product archive and split index archive. Member download, verification and import instructions are in [SHARED_DATA.md](https://github.com/Zqqq-qq/webshop-stepwise-rl/blob/main/docs/SHARED_DATA.md). Raw data and model weights must remain outside Git history.

The Release copy of `webshop_sft.json` was downloaded and independently checked locally:
- 9,312,587 bytes; 1,624 dialogues / 1,624 unique task IDs.
- SHA-256: `721e46dba410cff900e187bccea91a456c38b105b6d14d068d7762afb7791de3`.
- 1,599 normalized unique instructions, 969 records with reward 1.0, and 8 reward/true_reward disagreements.
- Preliminary split: 1,451 train / 160 validation dialogues, 200 reserved environment test task IDs, 13 excluded records. No cross-split task-ID intersection or train/validation normalized-instruction intersection was found.

Reserved test IDs identify environment tasks; they are not 200 downloaded expert demonstrations. The local split remains an offline candidate until checked against the complete environment goal map.

## Reproduced material audit using the Release copy

| Measurement | Result |
|---|---:|
| Upstream training task IDs | 1,624 |
| Upstream reserved test task IDs | 200 |
| Original train/test task-ID overlap | 12 |
| Bundled exploration trajectories | 78 across 26 tasks |
| Bundled full-reward exploration trajectories | 1 |
| Bundled progress train and validation data identical | Yes |
| Bundled PPO example trajectories | 41 |

Overlapping IDs: 818, 978, 2640, 2933, 4161, 5413, 5900, 7475, 7603, 7624, 7999, 8573.

The upstream counts are repository-material checks; the SFT counts above were rechecked against the downloaded Release file. None are trained-model benchmark results.

## Verification and source revision

These commands were run against source commit `9df3063321ccda2a3a96e1a5d7d9d720bdd296e0`:

```bash
PYTHONPATH=src .venv-tools/bin/python -m unittest discover -s tests -v
.venv-tools/bin/python -m compileall -q src scripts
.venv-tools/bin/python scripts/check_repository.py
PYTHONPATH=src .venv-tools/bin/python -m webshop_lab.cli audit --output artifacts/sft-audit.json
PYTHONPATH=src .venv-tools/bin/python -m webshop_lab.cli split
```

All 8 tests on that source revision passed; compilation and repository checks passed. The 8-test count is historical and must not be interpreted as the size of the later evaluation suite in PR #8. The explicit source path avoids a local installed-package import issue; that installation issue is not claimed fixed.

The subsequent tokenizer/label inspection and two-update execution handoff are recorded in [PR #12](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/12), including the full-data report and hash manifests. No GPU training or PPO result is claimed here.

## Environment evidence and remaining work

[PR #10](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/10) now contains the environment owner's import, health, goal-map and split evidence, including reported `alignment_mismatches=0` and `split_status.verified=true`. This supersedes the earlier claim that the team cannot start the environment because the SFT is missing. Formal data acceptance should review that evidence and agree the exact goal fingerprint/split manifest; the local offline report alone does not establish final freezing.

The original author's combined expert-trajectory archive was not downloaded in this local workflow. Its internal equivalence to the supplied SFT remains unverified. This is a provenance limitation, **not a blocker to using the team's verified Release copy**.

Earlier direct Drive connections timed out before an HTTP response; those failures never established that the files were missing or links invalid. The environment owner subsequently reported independent product/index downloads with hashes matching the Release. Members should use the private Release rather than wait for this local network route.

## Member actions

1. Obtain only the assets needed from `data-v1`; verify original and index-part hashes before use. The two index parts must be reassembled, not independently unzipped.
2. Reproduce audit/split locally or review the recorded evidence. Keep raw conversations, generated split datasets and weights out of Git.
3. Review PR #10's environment mapping/fingerprint and finalize the agreed split protocol before formal model comparisons.
4. Execute and report the engineering-only SFT/reload smoke from PR #12. Successful preprocessing or data sharing does not prove training, RL or shopping performance.
