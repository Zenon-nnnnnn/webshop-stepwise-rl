"""Paired comparison of two runs evaluated on the same tasks.

Success is a per-task binary outcome, so the two arms are compared as paired data: only the tasks
where the arms disagree carry information (McNemar), and the risk difference gets a paired
bootstrap interval that resamples whole tasks. Treating every trajectory as independent would
overstate confidence whenever a task has repeated samples.
"""
import collections
import math
import random

from .evaluation import cluster_bootstrap_ci, is_success, percentile, summarize_episodes


def _by_task(rows):
    return {(row['task_id'], int(row.get('repeat', 0))): row for row in rows}


def mcnemar_exact(a_only, b_only):
    """Two-sided exact McNemar p-value for paired binary outcomes (binomial, p=0.5)."""
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(0, min(a_only, b_only) + 1))
    return min(1.0, 2 * tail / 2 ** discordant)


def paired_bootstrap_difference(rows_a, rows_b, resamples=2000, seed=42):
    """Percentile interval for success_rate(b) - success_rate(a), resampling task clusters."""
    a = {key: 1 if is_success(row) else 0 for key, row in _by_task(rows_a).items()}
    b = {key: 1 if is_success(row) else 0 for key, row in _by_task(rows_b).items()}
    keys = sorted(a)
    rng = random.Random(seed)
    differences = []
    for _ in range(resamples):
        total = hits_a = hits_b = 0
        for _ in keys:
            key = keys[rng.randrange(len(keys))]
            total += 1
            hits_a += a[key]
            hits_b += b[key]
        differences.append(hits_b / total - hits_a / total)
    return [round(percentile(differences, .025), 6), round(percentile(differences, .975), 6)]


def compare_runs(rows_a, rows_b, labels=('a', 'b')):
    by_a, by_b = _by_task(rows_a), _by_task(rows_b)
    if set(by_a) != set(by_b):
        only_a = sorted(set(by_a) - set(by_b))[:5]
        only_b = sorted(set(by_b) - set(by_a))[:5]
        raise ValueError('The two runs were not evaluated on the same (task, repeat) set; '
                         f'only in {labels[0]}: {only_a}; only in {labels[1]}: {only_b}')
    if len(by_a) < 2:
        raise ValueError('Need at least two paired tasks to compare')
    metrics_a = summarize_episodes(rows_a)
    metrics_b = summarize_episodes(rows_b)
    keys = sorted(by_a)
    a_only = sum(1 for key in keys if is_success(by_a[key]) and not is_success(by_b[key]))
    b_only = sum(1 for key in keys if is_success(by_b[key]) and not is_success(by_a[key]))
    difference = metrics_b['success_rate'] - metrics_a['success_rate']
    interval = paired_bootstrap_difference(rows_a, rows_b)
    p_value = mcnemar_exact(a_only, b_only)
    notes = [
        'Paired by (task, repeat); only discordant tasks inform the McNemar test.',
        'The interval is a paired task-cluster bootstrap (2000 resamples, seed 42).',
        'A difference whose interval contains 0 is not evidence of an improvement.',
        'This compares two runs; it says nothing about other seeds or about the test split.',
    ]
    if len(keys) < 50:
        notes.append(f'Only {len(keys)} paired tasks: intervals are wide and one task moves the rate '
                     f'by {100 / len(keys):.1f} percentage points.')
    return {
        'tasks': len(keys),
        labels[0]: {
            'episodes': metrics_a['episodes'],
            'purchase_terminated': metrics_a['purchase_terminated'],
            'full_reward_successes': metrics_a['full_reward_successes'],
            'success_rate': metrics_a['success_rate'],
            'success_rate_ci95': metrics_a['success_rate_ci95'],
            'invalid_action_rate': metrics_a['invalid_action_rate'],
            'mean_final_reward': metrics_a['mean_final_reward'],
        },
        labels[1]: {
            'episodes': metrics_b['episodes'],
            'purchase_terminated': metrics_b['purchase_terminated'],
            'full_reward_successes': metrics_b['full_reward_successes'],
            'success_rate': metrics_b['success_rate'],
            'success_rate_ci95': metrics_b['success_rate_ci95'],
            'invalid_action_rate': metrics_b['invalid_action_rate'],
            'mean_final_reward': metrics_b['mean_final_reward'],
        },
        'paired': {
            f'only_{labels[0]}': a_only,
            f'only_{labels[1]}': b_only,
            'both_success': sum(1 for key in keys if is_success(by_a[key]) and is_success(by_b[key])),
            'neither_success': sum(1 for key in keys if not is_success(by_a[key]) and not is_success(by_b[key])),
            'mcnemar_exact_p': round(p_value, 6),
        },
        'risk_difference': round(difference, 6),
        'risk_difference_ci95': interval,
        'ci_method': 'paired task-cluster bootstrap, 2000 resamples, seed 42',
        'notes': notes,
    }


def run(args):
    from .common import write_json
    from .evaluation import check_parity, load_episodes
    import json
    labels = (args.a_label or 'a', args.b_label or 'b')
    rows_a = load_episodes(args.a_jsonl)
    rows_b = load_episodes(args.b_jsonl)
    parity = None
    if args.a_meta and args.b_meta:
        parity = check_parity(a_meta=args.a_meta, b_meta=args.b_meta)
    result = compare_runs(rows_a, rows_b, labels)
    result['labels'] = labels
    result['fairness'] = parity
    if parity and not parity['parity']:
        result.setdefault('notes', []).append('Both arms are not run under the same settings; '
                                             'see fairness.blocking_issues.')
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if parity and not parity['parity']:
        raise SystemExit(1)
