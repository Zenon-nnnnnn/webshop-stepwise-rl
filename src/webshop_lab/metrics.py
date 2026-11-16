import collections
import json
import math
from pathlib import Path
from .common import write_json, wilson, file_hash


def load_episodes(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        raise ValueError('No completed episode records')
    seen = set()
    for row in rows:
        key = (row['task_id'], row.get('repeat', 0))
        if key in seen:
            raise ValueError(f'Duplicate task/repeat pair: {key}; do not pool duplicate runs')
        seen.add(key)
        reward = float(row['final_reward'])
        if not math.isfinite(reward) or not 0 <= reward <= 1:
            raise ValueError('Invalid final reward')
        if not row['steps']:
            raise ValueError('Episode must have at least one step')
    if len({row.get('goal_fingerprint') for row in rows}) != 1:
        raise ValueError('Episodes from different environment goal mappings cannot be pooled')
    return rows


def run(args):
    rows = load_episodes(args.input)
    n = len(rows)
    successes = sum(bool(r['done']) and float(r['final_reward']) >= 1 - 1e-6 for r in rows)
    step_count = sum(len(r['steps']) for r in rows)
    invalid = sum(not bool(s['grounding']) for r in rows for s in r['steps'])
    repeats = collections.Counter(r['task_id'] for r in rows)
    result = {'episodes': n, 'unique_tasks': len(repeats), 'successes': successes,
              'success_rate': successes / n,
              'success_rate_wilson_95': wilson(successes, n) if max(repeats.values()) == 1 else None,
              'mean_final_reward': sum(float(r['final_reward']) for r in rows) / n,
              'mean_steps': step_count / n, 'invalid_action_rate': invalid / step_count,
              'grounding_accuracy': 1 - invalid / step_count,
              'termination_counts': dict(collections.Counter(r['termination'] for r in rows)),
              'input_sha256': file_hash(args.input), 'goal_fingerprint': rows[0].get('goal_fingerprint'),
              'notes': ['Success requires both done=true and final_reward approximately 1; purchase alone is not success.',
                        'Independent-task Wilson interval omitted for repeated trajectories; use task-cluster bootstrap for repeated-task inference.',
                        'Check metadata selected_ids against completed IDs before calling a run complete.']}
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
