"""Prepare and optionally run a paired, bounded PPO engineering comparison."""
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

from .common import file_hash, git_head, read_json, write_json
from .ppo import load_progress_model, training_tasks
from .rollout import request


def arm_configs(base, candidate, model, model_sha):
    reward_keys = {"reward_mode", "progress_model", "progress_model_sha256"}
    if reward_keys.intersection(base) or base.get("beta", 0.0) != 0.0:
        raise ValueError("Supply a shared config without arm-specific reward settings; beta must be zero")
    if candidate not in ("progress", "option_potential"):
        raise ValueError("Select progress or option_potential")
    if (candidate == "progress" and not (model and model_sha)
            or candidate == "option_potential" and (model or model_sha)):
        raise ValueError("A model path and hash are required only for the learned progress arm")
    terminal = dict(base, reward_mode="terminal")
    treatment = dict(base, reward_mode=candidate)
    if candidate == "progress":
        treatment.update(progress_model=str(Path(model).resolve()), progress_model_sha256=model_sha)
    return terminal, treatment


def common_budget(rounds, max_steps, max_new_tokens):
    if not rounds or any(not 1 <= len(ids) <= 8 for ids in rounds):
        raise ValueError("Each update requires 1–8 frozen train tasks")
    if type(max_steps) is not int or not 1 <= max_steps <= 10:
        raise ValueError("Invalid policy turn cap")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 128:
        raise ValueError("Invalid generation cap")
    episodes = sum(map(len, rounds))
    turns = episodes * max_steps
    return {"updates": len(rounds), "episodes": episodes, "policy_turn_cap": turns,
            "environment_call_cap": turns, "generated_token_cap": turns * max_new_tokens}


def run(args):
    base = read_json(args.config)
    shared_context = base.get("context_length")
    if type(shared_context) is not int or not base.get("max_new_tokens", 0) < shared_context <= 2048:
        raise ValueError("Invalid shared context budget")
    if any(base.get(k, v) != v for k, v in (("temperature", 1.0), ("top_p", 1.0), ("top_k", 0))):
        raise ValueError("Use the PPO replay-compatible sampling configuration")
    terminal, treatment = arm_configs(base, args.candidate, args.progress_model, args.progress_model_sha256)
    split_dir, goal_map = Path(args.split_dir).resolve(), Path(args.goal_map).resolve()
    output = Path(args.output).resolve()
    endpoint = base["environment_url"].rstrip("/")
    if not endpoint.startswith("http://127.0.0.1:"):
        raise ValueError("Use a local WebShop bridge")
    health = request(endpoint + "/health")
    if (not health["ready"] or not health["alignment_checked"] or health["alignment_mismatches"]
            or not health["split_status"]["verified"] or health["goal_fingerprint"] != base["goal_fingerprint"]):
        raise ValueError("WebShop health, split or fingerprint check failed")
    bridge_ready = args.candidate != "option_potential" or health.get("option_state") is True
    if args.execute and not bridge_ready:
        raise ValueError("Option potential execution requires env-server --option-state")
    if args.execute and subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Commit the execution source before starting a comparison")

    rounds = []
    for id_file in args.round_ids:
        ids, splits, _ = training_tasks(SimpleNamespace(split_dir=split_dir, goal_map=goal_map, ids=id_file), health)
        rounds.append(ids)
    budget = common_budget(rounds, base["max_steps"], base["max_new_tokens"])
    model_hash = file_hash(Path(base["model"]) / "model.safetensors")
    adapter_hash = file_hash(Path(base["adapter"]) / "adapter_model.safetensors")
    if (Path(base["model"]).name != base["revision"] or model_hash != base["base_weights_sha256"]
            or adapter_hash != base["adapter_sha256"]):
        raise ValueError("Shared model or SFT adapter identity check failed")
    if base.get("policy_dtype") != "float32" or base.get("load_in_4bit"):
        raise ValueError("Use the verified unquantized FP32 policy path")
    load_progress_model(treatment, split_dir)
    output.mkdir(parents=True, exist_ok=False)
    schedule = []
    for index, ids in enumerate(rounds, start=1):
        path = output / f"round-{index}-ids.json"
        write_json(path, ids)
        schedule.append({"path": str(path), "sha256": file_hash(path), "tasks": ids})
    arms = {}
    for label, config in (("terminal", terminal), ("candidate", treatment)):
        path = output / f"{label}.config.json"
        write_json(path, config)
        commands = []
        for index, ids in enumerate(schedule, start=1):
            cmd = [sys.executable, "-m", "webshop_lab.cli", "ppo-smoke", "--config", str(path),
                   "--ids", ids["path"], "--split-dir", str(split_dir), "--goal-map", str(goal_map),
                   "--output", str(output / label / f"round-{index}")]
            if index > 1:
                cmd += ["--resume", str(output / label / f"round-{index - 1}")]
            commands.append(cmd)
        arms[label] = {"config": str(path), "config_sha256": file_hash(path), "commands": commands}
    plan = {"source_commit": git_head(), "candidate": args.candidate, "health": health,
            "bridge_ready_for_candidate": bridge_ready, "mode": "execute" if args.execute else "preflight",
            "shared_config_sha256": file_hash(args.config), "model_sha256": model_hash,
            "initial_adapter_sha256": adapter_hash, "goal_map_sha256": file_hash(goal_map),
            "split_hashes": splits, "schedule": schedule, "arms": arms, "equal_budget_caps": budget,
            "counts_note": "Matched episodes and caps; actual decisions, calls and tokens can differ."}
    write_json(output / "plan.json", plan)
    print(f"Prepared {len(rounds)} updates per arm; bridge ready={bridge_ready}; {output / 'plan.json'}",
          flush=True)
    if not args.execute:
        return
    counters = {}
    for label in arms:
        parent = None
        totals = dict.fromkeys(("updates", "turns", "environment_calls", "generated_tokens"), 0)
        for index, cmd in enumerate(arms[label]["commands"], start=1):
            with (output / f"{label}-round-{index}.log").open("x") as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            summary = read_json(output / label / f"round-{index}" / "summary.json")
            if (summary["cumulative_updates"] != index or not summary["same_process_reload"]
                    or (parent and (summary["parent_run_id"] != parent["run_id"]
                                    or summary["actor_before"] != parent["actor_after"]
                                    or summary["critic_before"] != parent["critic_after"]))):
                raise RuntimeError(f"Checkpoint chain inconsistency in {label} round {index}")
            parent = summary
            for key in totals:
                totals[key] += summary[key]
        counters[label] = totals
        if (totals["updates"] != budget["updates"] or totals["turns"] > budget["policy_turn_cap"]
                or totals["environment_calls"] > budget["environment_call_cap"]
                or totals["generated_tokens"] > budget["generated_token_cap"]):
            raise RuntimeError(f"{label} exceeded the shared comparison budget")
    write_json(output / "training-summary.json", counters)
