"""Local asset import, upstream audit and leakage-aware SFT partitioning."""
import collections
import shutil
import zipfile
from pathlib import Path
from .common import ROOT, read_json, write_json, file_hash, goal_hash, task_partition, instruction_hash


def import_assets(args):
    source = Path(args.source).expanduser().resolve()
    env = ROOT / "upstream/envs/webshop"
    manifest = {"assets": [], "note": "Local assets; do not commit datasets or model weights."}
    for name in ["data.zip", "indexes.zip", "webshop_sft.json"]:
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; see docs/DATA.md")
    for name in ["data.zip", "indexes.zip"]:
        path = source / name
        dest = env if name == "data.zip" else env / "search_index"
        with zipfile.ZipFile(path) as z:
            members = [i for i in z.infolist() if not i.is_dir()]
            # Validate the entire archive before extracting any file.
            for i in members:
                target = (dest / i.filename).resolve()
                if not target.is_relative_to(dest.resolve()) or i.filename.startswith("/"):
                    raise ValueError(f"Unsafe archive path: {i.filename}")
                if i.filename.split("/", 1)[0] != ("data" if name == "data.zip" else "indexes"):
                    raise ValueError(f"Unexpected archive layout in {name}: {i.filename}")
            missing = sum(i.file_size for i in members if not (dest / i.filename).exists())
            if shutil.disk_usage(ROOT).free < missing + 2_000_000_000:
                raise RuntimeError("Insufficient free disk space for extraction plus 2GB reserve")
            for i in members:
                target = dest / i.filename
                if target.exists():
                    if target.stat().st_size != i.file_size:
                        raise ValueError(f"Existing file has wrong size: {target}; move it away and retry")
                    # Check CRC against archive contents without decompressing the archive again.
                    import zlib
                    crc = 0
                    with target.open('rb') as f:
                        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                            crc = zlib.crc32(chunk, crc)
                    if crc & 0xffffffff != i.CRC:
                        raise ValueError(f"Existing file CRC mismatch: {target}; move it away and retry")
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    tmp = target.with_name(target.name + ".partial")
                    with z.open(i) as src, tmp.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    tmp.replace(target)
            manifest["assets"].append({"name": name, "sha256": file_hash(path), "bytes": path.stat().st_size})
    target = ROOT / "data/raw/webshop_sft.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if source / "webshop_sft.json" != target.resolve():
        shutil.copy2(source / "webshop_sft.json", target)
    manifest["assets"].append({"name": "webshop_sft.json", "sha256": file_hash(target), "bytes": target.stat().st_size})
    write_json(ROOT / "data/asset-manifest.json", manifest)
    print("Imported assets. Next: webshop-lab audit; webshop-lab split")


def audit(args):
    base = ROOT / "upstream"
    train = set(read_json(base / "eval_agent/data/webshop/train_indices.json"))
    test = set(read_json(base / "eval_agent/data/webshop/test_indices.json"))
    explore = read_json(base / "exploration/webshop/exploration_outputs/exploration.json")
    tiny = read_json(base / "exploration/webshop/exploration_outputs/exploration_tiny.json")
    report = {
        "upstream_train_tasks": len(train), "upstream_test_tasks": len(test),
        "upstream_train_test_overlap": sorted(train & test),
        "upstream_exploration_trajectories": len(explore),
        "upstream_exploration_unique_tasks": len({r['id'] for r in explore}),
        "upstream_exploration_reward_one": sum(float(r['agent_final_reward']) >= 1 - 1e-6 for r in explore),
        "upstream_progress_train_equals_validation": explore == tiny,
        "upstream_ppo_trajectories": len(read_json(base / "prm/sampled_data_rl_training_webshop_flatten.json")),
        "warning": "These are audit counts, NOT trained-agent benchmark results. Task IDs depend on the exact environment goal ordering."
    }
    path = Path(args.sft)
    if path.exists():
        rows = read_json(path)
        report["sft"] = {"rows": len(rows), "unique_ids": len({r['id'] for r in rows}),
                         "unique_instructions": len({goal_hash(r) for r in rows}),
                         "reward_distribution": dict(collections.Counter(str(r.get('reward')) for r in rows)),
                         "reward_true_reward_disagreements": sum(r.get('reward') != r.get('true_reward') for r in rows),
                         "sha256": file_hash(path)}
    write_json(args.output, report)
    print(f"Audit saved to {args.output}; original task-ID overlap: {len(train & test)}")


def split(args):
    rows = read_json(args.sft)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty SFT JSON list")
    for r in rows:
        if not isinstance(r.get('id'), int) or not r.get('conversations'):
            raise ValueError("Every SFT row needs integer id and conversations")
        for j, t in enumerate(r['conversations']):
            if t.get('from') != ('human' if j % 2 == 0 else 'gpt') or not isinstance(t.get('value'), str):
                raise ValueError(f"Malformed alternating conversation at task {r['id']}")
    test_ids = sorted(set(read_json(args.test_ids)))
    reserved = set()
    if args.goal_map:
        goals = read_json(args.goal_map)
        reserved = {instruction_hash(r['instruction']) for r in goals if int(r['task_id']) in test_ids}
        if not set(test_ids) <= {int(r['task_id']) for r in goals}:
            raise ValueError('Goal map does not cover every reserved test task')
    train, val, excluded = task_partition(rows, test_ids, args.val_fraction, args.seed, reserved)
    out = Path(args.output)
    write_json(out / "train.json", train)
    write_json(out / "val.json", val)
    train_ids = sorted({int(r['id']) for r in train})
    val_ids = sorted({int(r['id']) for r in val})
    for name, ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
        write_json(out / f"{name}_ids.json", ids)
    manifest = {"seed": args.seed, "source_sha256": file_hash(args.sft),
                "test_ids_sha256": file_hash(args.test_ids), "excluded_rows": excluded,
                "train_rows": len(train), "val_rows": len(val), "test_tasks": len(test_ids),
                "goal_map_sha256": file_hash(args.goal_map) if args.goal_map else None,
                "train_goal_hashes": sorted({goal_hash(r) for r in train}),
                "val_goal_hashes": sorted({goal_hash(r) for r in val}),
                "notes": ["Reserved default upstream test IDs; removed matching SFT IDs and known instruction duplicates.",
                          "Test instructions absent from SFT must also be checked against the instantiated environment before final evaluation.",
                          "Use val IDs for development; never use test rewards for tuning."]}
    write_json(out / "split-manifest.json", manifest)
    print(f"train={len(train)}, val={len(val)}, test={len(test_ids)}, excluded={excluded}")
