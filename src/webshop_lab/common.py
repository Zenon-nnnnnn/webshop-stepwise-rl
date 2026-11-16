"""Small, dependency-free utilities shared by data and evaluation commands."""
import hashlib
import json
import math
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def goal_text(row):
    """Extract the actual shopping instruction, not the shared system preamble."""
    for turn in row["conversations"]:
        value = turn["value"]
        if turn["from"] == "human" and "Instruction:" in value:
            value = value.split("Instruction:", 1)[1]
            value = value.removeprefix(" [SEP]").strip()
            return " ".join(value.split("[SEP]", 1)[0].lower().split())
    raise ValueError(f"Cannot find shopping instruction for task {row.get('id')}")


def goal_hash(row):
    return instruction_hash(goal_text(row))


def normalize_instruction(value):
    value = ' '.join(value.lower().split()).rstrip('.')
    return re.sub(r',?\s*and price lower than [\d.]+ dollars\.?$', '', value).strip().rstrip('.')


def instruction_hash(value):
    # Group budget variants of the same underlying shopping request together.
    return hashlib.sha256(normalize_instruction(value).encode()).hexdigest()


ACTION = re.compile(r"^\s*Action:\s*(search|click)\[([^\r\n\]]+)\]\s*$", re.MULTILINE | re.IGNORECASE)


def parse_action(text):
    """Require one unambiguous action line; never silently execute a second action."""
    matches = list(ACTION.finditer(text))
    if len(matches) != 1:
        raise ValueError("Expected exactly one Action: search[...] or Action: click[...] line")
    m = matches[0]
    return f"{m[1].lower()}[{m[2].strip().lower()}]"


def is_executable(action, available):
    kind, value = action.split("[", 1)
    value = value[:-1]
    if kind == "search":
        return bool(available.get("has_search_bar")) and bool(value)
    return value in {str(x).lower() for x in available.get("clickables", [])}


def wilson(successes, total):
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    den = 1 + z * z / total
    mid = (p + z * z / (2 * total)) / den
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return [max(0., mid - radius), min(1., mid + radius)]


def task_partition(rows, test_ids, val_fraction=.1, seed=42, reserved_goal_hashes=()):
    """Reserve upstream test IDs; group all repeated normalized instructions together."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    test_ids = set(test_ids)
    # If an SFT instruction is seen under a reserved test ID, exclude all copies.
    blocked_goals = {goal_hash(r) for r in rows if int(r["id"]) in test_ids} | set(reserved_goal_hashes)
    candidates = [r for r in rows if int(r["id"]) not in test_ids and goal_hash(r) not in blocked_goals]
    # Connected components prevent BOTH same-ID and same-instruction leakage.
    parents = list(range(len(candidates)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    seen = {}
    for i, r in enumerate(candidates):
        for key in [("id", int(r["id"])), ("goal", goal_hash(r))]:
            if key in seen:
                parents[find(i)] = find(seen[key])
            seen[key] = i
    groups = {}
    for i, row in enumerate(candidates):
        groups.setdefault(find(i), []).append(row)
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError("Need at least two independent task groups")
    random.Random(seed).shuffle(keys)
    n = min(len(keys) - 1, max(1, round(len(keys) * val_fraction)))
    val_keys = set(keys[:n])
    train = [r for k in keys if k not in val_keys for r in groups[k]]
    val = [r for k in keys if k in val_keys for r in groups[k]]
    assert not ({int(r['id']) for r in train} & {int(r['id']) for r in val})
    assert not ({goal_hash(r) for r in train} & {goal_hash(r) for r in val})
    return train, val, len(rows) - len(candidates)


def combine_rewards(progress, grounding, final_reward, mode="spa", alpha=1., beta=.5):
    if not progress or len(progress) != len(grounding):
        raise ValueError("Need equal nonempty progress and grounding vectors")
    if not all(math.isfinite(float(v)) for v in [*progress, *grounding, final_reward, alpha, beta]):
        raise ValueError("Non-finite reward")
    if mode == "terminal":
        return [0.] * (len(progress) - 1) + [float(final_reward)]
    if mode == "uniform":
        return [float(final_reward) / len(progress)] * len(progress)
    if mode == "spa":
        return [alpha * float(p) + beta * float(g) for p, g in zip(progress, grounding)]
    raise ValueError(f"Unknown reward mode: {mode}")
