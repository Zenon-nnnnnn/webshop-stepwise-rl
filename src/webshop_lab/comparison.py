"""Paired comparison of two runs evaluated on the same tasks.

Success is a per-task outcome, so the two arms are compared as paired data **by task**, not by
trajectory: repeated trajectories of one task are correlated, and counting them as independent
samples inflates significance. With one trajectory per task the task-level outcome is binary and an
exact McNemar test applies; with repeats the test is replaced by a paired task-cluster bootstrap of
the difference in per-task success rates, and no McNemar p-value is reported.
"""
import collections
import math
import random

from .evaluation import is_success, percentile, summarize_episodes


def group_by_task(rows):
    groups = collections.defaultdict(dict)
    for row in rows:
        groups[row['task_id']][int(row.get('repeat', 0))] = row
    return groups


def _task_success_means(groups):
    return {task: sum(1 for row in repeats.values() if is_success(row)) / len(repeats)
            for task, repeats in groups.items()}


def mcnemar_exact(a_only, b_only):
    """Two-sided exact McNemar p-value for paired binary outcomes (binomial, p=0.5)."""
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(0, min(a_only, b_only) + 1))
    return min(1.0, 2 * tail / 2 ** discordant)


def paired_task_bootstrap(means_a, means_b, resamples=2000, seed=42):
    """Percentile interval for the task-averaged difference, resampling whole tasks."""
    keys = sorted(means_a)
    rng = random.Random(seed)
    differences = []
    for _ in range(resamples):
        total = 0.0
        for _ in keys:
            task = keys[rng.randrange(len(keys))]
            total += means_b[task] - means_a[task]
        differences.append(total / len(keys))
    return [round(percentile(differences, .025), 6), round(percentile(differences, .975), 6)]


def compare_runs(rows_a, rows_b, labels=('a', 'b')):
    groups_a, groups_b = group_by_task(rows_a), group_by_task(rows_b)
    if set(groups_a) != set(groups_b):
        only_a = sorted(set(groups_a) - set(groups_b))[:5]
        only_b = sorted(set(groups_b) - set(groups_a))[:5]
        raise ValueError('The two runs were not evaluated on the same tasks; '
                         f'only in {labels[0]}: {only_a}; only in {labels[1]}: {only_b}')
    for task in groups_a:
        if set(groups_a[task]) != set(groups_b[task]):
            raise ValueError(f'Task {task} has different repeat indices in the two runs; '
                             'align the repeats before comparing')
    tasks = len(groups_a)
    if tasks < 2:
        raise ValueError('Need at least two paired tasks to compare')
    means_a, means_b = _task_success_means(groups_a), _task_success_means(groups_b)
    repeats = {len(repeats) for repeats in groups_a.values()} | {len(repeats) for repeats in groups_b.values()}
    single = repeats == {1}
    metrics_a, metrics_b = summarize_episodes(rows_a), summarize_episodes(rows_b)
    difference = sum(means_b[task] - means_a[task] for task in groups_a) / tasks
    interval = paired_task_bootstrap(means_a, means_b)
    paired = None
    p_value = None
    if single:
        a_only = sum(1 for task in groups_a if means_a[task] == 1 and means_b[task] == 0)
        b_only = sum(1 for task in groups_a if means_b[task] == 1 and means_a[task] == 0)
        p_value = mcnemar_exact(a_only, b_only)
        paired = {f'only_{labels[0]}': a_only, f'only_{labels[1]}': b_only,
                  'both_success': sum(1 for task in groups_a if means_a[task] == 1 and means_b[task] == 1),
                  'neither_success': sum(1 for task in groups_a if means_a[task] == 0 and means_b[task] == 0)}
    notes = [
        'Paired by task; repeated trajectories of one task are one cluster, not independent samples.',
        'The interval is a paired task-cluster bootstrap of the task-averaged success-rate difference '
        '(2000 resamples, seed 42).',
        'A difference whose interval contains 0 is not evidence of an improvement.',
        'This compares two runs; it says nothing about other seeds or about the test split.',
    ]
    if single:
        notes.append('One trajectory per task, so an exact McNemar test on discordant tasks is reported.')
    else:
        notes.append('Repeated trajectories detected: McNemar is not reported (its independence '
                     'assumption fails); read the task-cluster interval instead.')
    if len(repeats) > 1:
        notes.append(f'Repeats per task differ between tasks: {sorted(repeats)}; the risk difference is '
                     'the task-averaged paired difference.')
    if tasks < 50:
        notes.append(f'Only {tasks} paired tasks: intervals are wide and one task moves the rate '
                     f'by {100 / tasks:.1f} percentage points.')
    return {
        'tasks': tasks,
        'repeats_per_task': sorted(repeats),
        'episodes': {labels[0]: metrics_a['episodes'], labels[1]: metrics_b['episodes']},
        labels[0]: {
            'episodes': metrics_a['episodes'],
            'purchase_terminated': metrics_a['purchase_terminated'],
            'full_reward_successes': metrics_a['full_reward_successes'],
            'success_rate': metrics_a['success_rate'],
            'mean_terminal_reward': metrics_a['mean_terminal_reward'],
            'success_rate_ci95': metrics_a['success_rate_ci95'],
            'invalid_action_rate': metrics_a['invalid_action_rate'],
        },
        labels[1]: {
            'episodes': metrics_b['episodes'],
            'purchase_terminated': metrics_b['purchase_terminated'],
            'full_reward_successes': metrics_b['full_reward_successes'],
            'success_rate': metrics_b['success_rate'],
            'mean_terminal_reward': metrics_b['mean_terminal_reward'],
            'success_rate_ci95': metrics_b['success_rate_ci95'],
            'invalid_action_rate': metrics_b['invalid_action_rate'],
        },
        'paired': paired,
        'mcnemar_applicable': single,
        'mcnemar_exact_p': p_value,
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
