"""CLI imports GPU/environment libraries only for commands that need them."""
import argparse
import importlib
from .common import ROOT


def main():
    p = argparse.ArgumentParser(description="WebShop team experiment tools (run from repository root)")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("import-assets", help="Safely unpack local data.zip/indexes.zip and import SFT JSON")
    s.add_argument("--source", required=True)
    s = sub.add_parser("audit", help="Audit original splits and sample trajectories; no GPU required")
    s.add_argument("--sft", default=str(ROOT / "data/raw/webshop_sft.json"))
    s.add_argument("--output", default="artifacts/audit.json")
    s = sub.add_parser("split", help="Create deterministic task/instruction-disjoint train/val partitions")
    s.add_argument("--sft", default="data/raw/webshop_sft.json")
    s.add_argument("--test-ids", default=str(ROOT / "upstream/eval_agent/data/webshop/test_indices.json"))
    s.add_argument("--output", default="data/processed")
    s.add_argument("--val-fraction", type=float, default=.1)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--goal-map", help="Environment-exported goal instructions, required before final training/evaluation")
    s = sub.add_parser("doctor", help="Report versions, GPU and local asset availability")
    s.add_argument("--output", default="artifacts/doctor.json")
    s = sub.add_parser("sft", help="Assistant-only LoRA SFT with optional 4-bit base")
    s.add_argument("--config", required=True)
    s.add_argument("--max-steps", type=int, default=-1)
    s = sub.add_parser("env-server", help="Local single-client WebShop environment HTTP bridge")
    s.add_argument("--port", type=int, default=8765)
    s = sub.add_parser("rollout", help="Collect real episodes against the environment bridge")
    s.add_argument("--config", required=True)
    s.add_argument("--ids", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--repeats", type=int, default=1)
    s = sub.add_parser("summarize", help="Summarize episode metrics with confidence interval")
    s.add_argument("--input", required=True)
    s.add_argument("--output", required=True)
    s = sub.add_parser("reward-export", help="Convert annotated episodes to step-reward training records")
    s.add_argument("--input", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--mode", choices=["terminal", "uniform", "spa"], required=True)
    s.add_argument("--alpha", type=float, default=1.)
    s.add_argument("--beta", type=float, default=.5)
    s = sub.add_parser("progress-train", help="Train low-cost additive linear baseline, not neural SPA")
    s.add_argument("--train", required=True)
    s.add_argument("--val", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--dimensions", type=int, default=4096)
    s.add_argument("--epochs", type=int, default=100)
    s.add_argument("--seed", type=int, default=42)
    s = sub.add_parser("progress-predict", help="Annotate episodes using the low-cost linear estimator")
    s.add_argument("--model", required=True)
    s.add_argument("--input", required=True)
    s.add_argument("--output", required=True)
    args = p.parse_args()
    mapping = {
        "import-assets": ("data", "import_assets"), "audit": ("data", "audit"), "split": ("data", "split"),
        "doctor": ("doctor", "run"), "sft": ("sft", "run"), "env-server": ("environment", "serve"),
        "rollout": ("rollout", "run"), "summarize": ("metrics", "run"), "reward-export": ("rewards", "run"),
        "progress-train": ("progress", "train"), "progress-predict": ("progress", "predict")}
    module, func = mapping[args.command]
    getattr(importlib.import_module(f"webshop_lab.{module}"), func)(args)


if __name__ == "__main__":
    main()
