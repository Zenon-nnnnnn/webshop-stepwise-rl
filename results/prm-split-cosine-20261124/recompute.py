"""Zero-dependency offline verifier and metric recomputation for PR #54 evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
ARMS_DIR = HERE / "rtx3090-fp32-three-arms"

EXPECTED_SAMPLES_SHA = "9c88beffaef277f519712ab4a07c45766212579770d6292572c7f189bca4d8fe"
EXPECTED_TRAIN_IDS_SHA = "8c85e56b3ac2e1f0a09b54560ad0f9bf77eedddb241d73617c7ff32416522e20"
EXPECTED_BASE_SHA = "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
EXPECTED_B1_SHA = "c39e2a78e7c5973328e9d2c23fb6320e566352e63cc089562488f3dd46ae5f9c"
EXPECTED_TRAIN_PY_SHA = "d71a30626cb4a747837a09a9e3814abc4bcccb2fbad2d2da6fe98febf4ab0ac8"

# Locked ground-truth labels and turn counts in canonical file order of samples.jsonl (9c88beff...)
CANONICAL_EIGHT = [
    (351, 3, 2.0 / 3.0),
    (95, 3, 0.0),
    (5478, 10, 0.0),
    (3542, 10, 0.0),
    (359, 3, 2.0 / 3.0),
    (5031, 3, 0.0),
    (0, 3, 1.0),
    (6225, 3, 1.0),
]
BINARY_PAYLOADS = {"adapter/adapter_model.safetensors", "head.pt", "training-state.pt"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cosine_mult(step_idx: int, total_steps: int) -> float:
    return 0.5 * (1.0 + math.cos(math.pi * step_idx / total_steps))


def verify_arm(arm: str, labels_by_task: list[tuple[int, int, float]]) -> dict:
    ckpt = ARMS_DIR / arm
    manifest = json.loads((ckpt / "prm-manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((ckpt / "summary.json").read_text(encoding="utf-8"))
    ref_preds = json.loads((ckpt / "reference-predictions.json").read_text(encoding="utf-8"))
    reload_report = json.loads((ARMS_DIR / f"{arm}-reload.json").read_text(encoding="utf-8"))
    attempt = json.loads((ARMS_DIR / f"{arm}.attempt.json").read_text(encoding="utf-8"))

    assert attempt["status"] == "complete" and attempt["test_authorization"] is False
    assert manifest["schema_version"] == 2
    assert manifest["architecture"]["representation"] == "final_hidden_state"
    assert manifest["architecture"]["input_dim"] == 1536

    ident = manifest["identity"]
    assert ident["base_sha256"] == EXPECTED_BASE_SHA
    assert ident["initialization"]["sft_adapter_files_sha256"]["adapter_model.safetensors"] == EXPECTED_B1_SHA
    assert summary["samples_sha256"] == EXPECTED_SAMPLES_SHA
    assert summary["train_ids_sha256"] == EXPECTED_TRAIN_IDS_SHA
    assert summary["retained_samples"] == 8 and summary["rejected"] == []

    # Verify every core source file SHA-256 recorded in manifest matches repo src/webshop_lab/prm/
    for src_name, src_digest in ident["source_sha256"].items():
        actual = sha256(REPO_ROOT / "src/webshop_lab/prm" / src_name)
        assert actual == src_digest, f"{arm}: source mismatch for {src_name}: {actual} != {src_digest}"

    # Verify SHA256SUMS & prm-manifest.json hashes for all tracked text artifacts
    sums = {}
    for line in (ckpt / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, rel = line.split("  ", 1)
            sums[rel] = digest
    assert sums["prm-manifest.json"] == sha256(ckpt / "prm-manifest.json")
    for rel, expected_digest in manifest["files_sha256"].items():
        assert sums[rel] == expected_digest
        local_file = ckpt / rel
        if rel not in BINARY_PAYLOADS:
            assert local_file.is_file(), f"Missing tracked evidence file: {local_file}"
            assert sha256(local_file) == expected_digest, f"SHA mismatch on {local_file}"

    # Verify fresh-process reload turn-by-turn and total-by-total against reference-predictions.json
    assert reload_report["reference_match"] is True
    reload_preds = reload_report["predictions"]
    assert len(ref_preds) == len(reload_preds) == len(labels_by_task) == 8

    max_reload_step_diff = 0.0
    max_reload_total_diff = 0.0
    per_sample = []
    sq_errors = []
    abs_errors = []

    for (expected_tid, expected_turns, label), ref_row, rel_row in zip(labels_by_task, ref_preds, reload_preds):
        assert ref_row["task_id"] == rel_row["task_id"] == expected_tid
        assert ref_row["turn_end_indices"] == rel_row["turn_end_indices"]
        assert len(ref_row["step_scores"]) == len(rel_row["step_scores"]) == expected_turns
        for a, b in zip(ref_row["step_scores"], rel_row["step_scores"]):
            d = abs(a - b)
            max_reload_step_diff = max(max_reload_step_diff, d)
            assert d <= 1e-5 + 1e-5 * abs(a)
        step_sum = math.fsum(ref_row["step_scores"])
        assert abs(step_sum - ref_row["total"]) <= 1e-5
        td = abs(ref_row["total"] - rel_row["total"])
        max_reload_total_diff = max(max_reload_total_diff, td)
        assert td <= 1e-5 + 1e-5 * abs(ref_row["total"])

        err = abs(ref_row["total"] - label)
        abs_errors.append(err)
        sq_errors.append((ref_row["total"] - label) ** 2)
        per_sample.append({
            "task_id": expected_tid,
            "turns": expected_turns,
            "label": label,
            "pred_total": ref_row["total"],
            "abs_error": err,
            "step_scores": ref_row["step_scores"],
        })

    # Verify optimization records and exact learning-rate schedule at every step
    opt = summary["optimization"]
    records = summary["records"]
    assert len(records) == summary["optimizer_steps"]
    assert summary["samples_consumed"] == summary["optimizer_steps"] * opt["gradient_accumulation_steps"]
    for idx, rec in enumerate(records):
        assert rec["optimizer_step"] == idx + 1
        mult = cosine_mult(idx, opt["cosine_total_steps"]) if opt["lr_scheduler"] == "cosine" else 1.0
        assert math.isclose(rec["learning_rate"], opt["learning_rate"] * mult, rel_tol=1e-12, abs_tol=0.0)
        assert math.isclose(rec["head_learning_rate"], opt["head_learning_rate"] * mult, rel_tol=1e-12, abs_tol=0.0)

    mse = math.fsum(sq_errors) / len(sq_errors)
    max_err = max(abs_errors)
    by_tid = {r["task_id"]: r["pred_total"] for r in per_sample}
    full_3turn_mean = 0.5 * (by_tid[0] + by_tid[6225])
    zero_3turn_mean = 0.5 * (by_tid[95] + by_tid[5031])

    return {
        "arm": arm,
        "optimizer_steps": summary["optimizer_steps"],
        "samples_consumed": summary["samples_consumed"],
        "optimization": opt,
        "clipped_updates": sum(1 for r in records if r["gradient_clipped"]),
        "elapsed_seconds": summary["elapsed_seconds"],
        "peak_allocated_gib": summary["peak_allocated_gib"],
        "peak_reserved_gib": summary["peak_reserved_gib"],
        "max_reload_step_diff": max_reload_step_diff,
        "max_reload_total_diff": max_reload_total_diff,
        "mse": mse,
        "max_abs_error": max_err,
        "memory_gate_passed": mse < 0.001 and max_err < 0.1,
        "three_turn_full_mean_0_6225": full_3turn_mean,
        "three_turn_zero_mean_95_5031": zero_3turn_mean,
        "three_turn_full_minus_zero": full_3turn_mean - zero_3turn_mean,
        "binary_payload_sha256": {k: manifest["files_sha256"][k] for k in sorted(BINARY_PAYLOADS)},
        "per_sample": per_sample,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, help="Optional local path to samples.jsonl (9c88beff...) to cross-check raw labels")
    args = parser.parse_args()

    actual_train_py = sha256(REPO_ROOT / "src/webshop_lab/prm/train.py")
    assert actual_train_py == EXPECTED_TRAIN_PY_SHA, f"train.py SHA mismatch: {actual_train_py}"
    assert sha256(REPO_ROOT / "results/env-2026-11-17/train_ids.json") == EXPECTED_TRAIN_IDS_SHA

    labels = list(CANONICAL_EIGHT)
    if args.samples is not None:
        assert sha256(args.samples) == EXPECTED_SAMPLES_SHA, "samples.jsonl SHA-256 mismatch"
        raw = [json.loads(line) for line in args.samples.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert [(r["task_id"], sum(1 for m in r["messages"] if m["role"] == "assistant"), float(r["terminal_reward"])) for r in raw] == [
            (tid, turns, lbl) for tid, turns, lbl in labels
        ]

    report = {
        "verified_train_py_sha256": actual_train_py,
        "samples_sha256": EXPECTED_SAMPLES_SHA,
        "train_ids_sha256": EXPECTED_TRAIN_IDS_SHA,
        "arms": [verify_arm(arm, labels) for arm in ("hidden-lr-low-acc8", "hidden-split-cos-acc8", "hidden-split-cos-acc1")],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
