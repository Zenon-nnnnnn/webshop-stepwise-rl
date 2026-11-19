"""Independent recomputation of the headline metrics from a raw episode JSONL.

Deliberately standalone: this file does not import `webshop_lab`. A second, deliberately dumb
implementation is the only way to catch a mistake in the main one (a wrong denominator, a filter
that silently drops episodes, a rate computed over successes instead of attempts).

Usage:
    python scripts/verify_metrics.py --jsonl artifacts/b0-val.jsonl
    python scripts/verify_metrics.py --jsonl artifacts/b0-val.jsonl --evaluation artifacts/b0-val-evaluation.json

Exit code 0 means every compared field agrees within 1e-9; 1 means a mismatch or unreadable input.
"""
import argparse
import collections
import json
import sys


def recompute(path):
    rows = [json.loads(line) for line in open(path, encoding='utf-8').read().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f'no episode records in {path}')
    episodes = len(rows)
    purchases = sum(1 for row in rows if row['done'] is True)
    successes = sum(1 for row in rows
                    if row['done'] is True and float(row['final_reward']) >= 1 - 1e-6)
    steps = sum(len(row['steps']) for row in rows)
    invalid = sum(1 for row in rows for step in row['steps'] if not step.get('grounding'))
    per_task = collections.Counter(row['task_id'] for row in rows)
    return {
        'episodes': episodes,
        'purchase_terminated': purchases,
        'full_reward_successes': successes,
        'purchase_without_full_reward': purchases - successes,
        'max_steps_terminated': episodes - purchases,
        'success_rate': successes / episodes,
        'mean_final_reward': sum(float(row['final_reward']) for row in rows) / episodes,
        'mean_steps': steps / episodes,
        'invalid_action_rate': invalid / steps if steps else 0.0,
        'unique_tasks': len(per_task),
        'tasks_with_repeats': sorted(task for task, count in per_task.items() if count > 1),
        'reward_not_in_unit_interval': [row['task_id'] for row in rows
                                        if not 0 <= float(row['final_reward']) <= 1],
    }


FIELDS = ('episodes', 'purchase_terminated', 'full_reward_successes', 'purchase_without_full_reward',
          'max_steps_terminated', 'success_rate', 'mean_final_reward', 'mean_steps',
          'invalid_action_rate', 'unique_tasks')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jsonl', required=True)
    parser.add_argument('--evaluation', help='evaluate output JSON to compare against')
    parser.add_argument('--tolerance', type=float, default=1e-9)
    args = parser.parse_args()
    mine = recompute(args.jsonl)
    print(json.dumps(mine, ensure_ascii=False, indent=2))
    if mine['reward_not_in_unit_interval']:
        print(f'FAIL reward outside [0, 1] for tasks {mine["reward_not_in_unit_interval"][:10]}')
        return 1
    if not args.evaluation:
        return 0
    recorded = json.load(open(args.evaluation, encoding='utf-8'))['metrics']
    failures = []
    print(f'{"field":32} {"recomputed":>18} {"recorded":>18}  verdict')
    for field in FIELDS:
        left, right = mine[field], recorded.get(field)
        if isinstance(left, float) or isinstance(right, float):
            ok = isinstance(right, (int, float)) and abs(float(left) - float(right)) <= args.tolerance
        else:
            ok = left == right
        failures.append(field) if not ok else None
        print(f'{field:32} {left!r:>18} {right!r:>18}  {"PASS" if ok else "FAIL"}')
    if failures:
        print(f'MISMATCH in {len(failures)} field(s): {", ".join(failures)}')
        return 1
    print(f'All {len(FIELDS)} fields agree within {args.tolerance}.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
