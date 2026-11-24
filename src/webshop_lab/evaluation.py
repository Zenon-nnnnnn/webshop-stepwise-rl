"""Auditable evaluation: run completeness, fairness parity, metrics and failure triage.

Rules this module enforces (they mirror docs/EXPERIMENTS.md and docs/EVALUATION.md):

- Purchase is not success. WebShop sets ``done=True`` as soon as an item is bought, while the
  final reward can stay below 1. Purchase terminations and full-reward successes are counted
  separately, and a run is never summarized from a partial JSONL file.
- A run is only comparable with another one when the task list, model, prompt, context budget
  and sampling settings are identical; only the adapter (the treatment) may differ.
- Failure classes are a triage aid, not ground truth. Each case keeps the raw steps so a human
  can confirm or reject the label before it enters the report.

Standard library only: these commands run on the CPU-only machines used for log review.
"""
import collections
import datetime
import json
import math
import random
import re
from pathlib import Path

from .common import file_hash, file_hash_lf, git_head, normalize_instruction, read_json, wilson, write_json

SUCCESS_EPS = 1e-6
FAILURE_CLASSES = ('format', 'budget', 'specification', 'loop', 'retrieval', 'other', 'engineering')
FAILURE_LABELS = {
    'format': 'format（动作格式/不可执行）',
    'budget': 'budget（超出预算）',
    'specification': 'specification（规格不符）',
    'loop': 'loop（无效循环/步数耗尽）',
    'retrieval': 'retrieval（检索不匹配）',
    'other': 'other（未归类，需人工查看）',
    'engineering': 'engineering（工程故障，单独统计）',
}
# Words that carry no retrieval signal; the budget suffix is stripped before tokenizing.
STOPWORDS = frozenset("""
a an the and or of for with to in on at is are was were be been being this that these those
it its as by from i i'm need want looking for price lower than dollars dollar
""".split())
INFRASTRUCTURE_ERROR = re.compile(r'connection|timeout|timed out|server|traceback|not ready|name or service', re.I)
PRICE = re.compile(r'Price:\s*\$?([0-9][0-9,]*(?:\.[0-9]+)?)', re.I)
BUDGET = re.compile(r'price\s+lower\s+than\s+\$?([0-9]+(?:\.[0-9]+)?)', re.I)


def load_jsonl(path):
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f'{path}:{number} is not valid JSON: {error}') from error
    return rows


def load_episodes(path, allow_duplicates=False):
    """Load episode records and reject files that cannot be pooled as one run.

    ``allow_duplicates`` exists for the completeness audit: it must be able to *report* a
    duplicated (task, repeat) pair instead of failing before it can say what is wrong.
    """
    rows = load_jsonl(path)
    if not rows:
        raise ValueError('No completed episode records')
    seen = set()
    for row in rows:
        key = (row['task_id'], row.get('repeat', 0))
        if key in seen and not allow_duplicates:
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


def is_success(row):
    """A task counts as solved only when the episode ended and the final reward is full."""
    return bool(row['done']) and float(row['final_reward']) >= 1 - SUCCESS_EPS


def terminal_reward(row):
    """Task-level reward: the recorded score if the episode ended with a purchase, else 0.

    The environment returns a page-match score at every step (upstream
    ``web_agent_text_env.py`` sets ``self.reward = search_max`` on search pages and ``maxr`` on
    item pages, and only ``done()`` returns the purchase score). So ``final_reward`` alone is not
    a task reward: a run that never buys can still end on a page scoring 0.9. Task reward must be
    0 for anything that did not purchase.
    """
    return float(row['final_reward']) if bool(row['done']) else 0.0


def percentile(values, quantile):
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def cluster_bootstrap_ci(rows, resamples=2000, seed=42):
    """Percentile interval that resamples whole tasks, for repeated-trajectory designs.

    Trajectories of the same task are correlated, so they are resampled as a cluster instead of
    being treated as independent samples. Fixed seed keeps the numbers reproducible.
    """
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row['task_id']].append(1 if is_success(row) else 0)
    keys = sorted(groups)
    if not keys:
        return None
    rng = random.Random(seed)
    rates = []
    for _ in range(resamples):
        total = hits = 0
        for _ in keys:
            sample = groups[keys[rng.randrange(len(keys))]]
            total += len(sample)
            hits += sum(sample)
        rates.append(hits / total)
    return [round(percentile(rates, .025), 6), round(percentile(rates, .975), 6)]


def summarize_episodes(rows):
    """Metrics for one run; purchase terminations and full-reward successes stay separate."""
    n = len(rows)
    if not n:
        raise ValueError('No episodes to summarize')
    purchase_terminated = sum(1 for row in rows if bool(row['done']))
    successes = sum(1 for row in rows if is_success(row))
    step_count = sum(len(row['steps']) for row in rows)
    invalid = sum(1 for row in rows for step in row['steps'] if not bool(step['grounding']))
    repeats = collections.Counter(row['task_id'] for row in rows)
    independent = max(repeats.values()) == 1
    interval = wilson(successes, n) if independent else cluster_bootstrap_ci(rows)
    return {
        'episodes': n, 'unique_tasks': len(repeats),
        'purchase_terminated': purchase_terminated,
        'full_reward_successes': successes, 'successes': successes,
        'purchase_without_full_reward': purchase_terminated - successes,
        'max_steps_terminated': n - purchase_terminated,
        'success_rate': successes / n,
        'success_rate_ci95': interval,
        'ci_method': 'wilson (one episode per task)' if independent
        else 'cluster bootstrap over tasks, 2000 resamples, seed 42',
        'conditional_success_given_purchase': (successes / purchase_terminated) if purchase_terminated else None,
        'mean_terminal_reward': sum(terminal_reward(row) for row in rows) / n,
        'mean_final_reward': sum(float(row['final_reward']) for row in rows) / n,
        'mean_steps': step_count / n,
        'invalid_action_rate': invalid / step_count,
        'grounding_accuracy': 1 - invalid / step_count,
        'episodes_with_invalid_actions': sum(1 for row in rows if any(not s['grounding'] for s in row['steps'])),
        'termination_counts': dict(collections.Counter(row['termination'] for row in rows)),
        'goal_fingerprint': rows[0].get('goal_fingerprint'),
        'notes': [
            'Success requires done=true AND final_reward approximately 1; a purchase alone is not success.',
            'mean_terminal_reward is the task reward: the recorded score only when the episode purchased, else 0.',
            'mean_final_reward is the last recorded environment score; for max_steps episodes it is a page-match score, not task completion.',
            'success_rate uses every attempted episode as the denominator, including max_steps terminations.',
            'Independent-task Wilson interval omitted for repeated trajectories; a task-cluster bootstrap is reported instead.',
            'Check metadata selected_ids against completed IDs (webshop-lab check-run) before quoting a run.',
        ],
    }


def inspect_run(jsonl_path, meta_path, config_path=None):
    """Compare the completed episode list with the metadata that was written before collection."""
    meta = read_json(meta_path)
    missing_keys = sorted({'run_id', 'config', 'selected_ids', 'repeats', 'health', 'config_sha256'} - set(meta))
    if missing_keys:
        raise ValueError(f'{meta_path} lacks required metadata keys: {missing_keys}')
    rows = load_episodes(jsonl_path, allow_duplicates=True)
    selected = [int(task) for task in meta['selected_ids']]
    repeats = int(meta.get('repeats', 1))
    if repeats < 1:
        raise ValueError('Metadata repeats must be positive')
    expected = [(task, repeat) for task in selected for repeat in range(repeats)]
    completed = [(int(row['task_id']), int(row.get('repeat', 0))) for row in rows]
    expected_set, completed_set = set(expected), set(completed)
    missing_pairs = sorted(expected_set - completed_set)
    unexpected_pairs = sorted(completed_set - expected_set)
    duplicate_pairs = sorted(pair for pair, count in collections.Counter(completed).items() if count > 1)
    expected_fingerprint = (meta.get('health') or {}).get('goal_fingerprint')
    run_id = meta['run_id']
    run_id_mismatches = sorted({int(row['task_id']) for row in rows if row.get('run_id') != run_id})
    fingerprint_mismatches = sorted({int(row['task_id']) for row in rows
                                     if expected_fingerprint and row.get('goal_fingerprint') != expected_fingerprint})
    # Identity failures are as blocking as a wrong task set: a merged file from another run would
    # otherwise look complete just because it happens to cover the same task IDs.
    if unexpected_pairs or duplicate_pairs or run_id_mismatches or fingerprint_mismatches:
        status = 'mismatched'
    elif missing_pairs:
        status = 'partial'
    else:
        status = 'complete'
    config_file = None
    if config_path:
        candidate = Path(config_path)
        if not candidate.exists():
            raise FileNotFoundError(f'Config not found: {candidate}')
        raw, normalized = file_hash(candidate), file_hash_lf(candidate)
        expected_sha = meta.get('config_sha256')
        canonical = expected_sha if expected_sha in {raw, normalized} else raw
        config_file = {'path': str(candidate), 'sha256': canonical,
                       'raw_sha256': raw, 'lf_sha256': normalized,
                       'newline_normalized': canonical != raw,
                       'matches_metadata': canonical == expected_sha,
                       'matches_embedded': read_json(candidate) == meta.get('config')}
    provenance_issues = []
    if run_id_mismatches:
        provenance_issues.append(f'{len(run_id_mismatches)} episode(s) carry a different run_id')
    if fingerprint_mismatches:
        provenance_issues.append(f'{len(fingerprint_mismatches)} episode(s) carry a different goal_fingerprint')
    if config_file:
        if not config_file['matches_metadata']:
            provenance_issues.append('config file sha256 does not match metadata.config_sha256')
        if not config_file['matches_embedded']:
            provenance_issues.append('config file content differs from metadata.config')
    provenance_ok = not provenance_issues
    summary = (f'{len(completed)} of {len(expected)} expected episodes '
               f'({len(selected)} tasks x {repeats} repeats)')
    if status == 'partial':
        summary += f'; missing {len(missing_pairs)} episode(s)'
    elif status == 'mismatched':
        summary += (f'; unexpected {len(unexpected_pairs)}, duplicate {len(duplicate_pairs)}, '
                    f'wrong run_id {len(run_id_mismatches)}, wrong fingerprint '
                    f'{len(fingerprint_mismatches)}')
    return {
        'status': status, 'summary': summary,
        'complete': status == 'complete', 'partial': status == 'partial',
        'provenance_ok': provenance_ok, 'provenance_issues': provenance_issues,
        'meta_path': str(meta_path), 'jsonl_path': str(jsonl_path),
        'run_id': run_id,
        'selected_ids': selected,
        'selected_tasks': len(selected), 'repeats': repeats,
        'expected_episodes': len(expected), 'completed_episodes': len(completed),
        'completed_unique_tasks': len({task for task, _ in completed}),
        'missing_pairs': [list(pair) for pair in missing_pairs],
        'unexpected_pairs': [list(pair) for pair in unexpected_pairs],
        'duplicate_pairs': [list(pair) for pair in duplicate_pairs],
        'order_ok': completed == expected,
        'run_id_mismatch_tasks': run_id_mismatches,
        'goal_fingerprint_mismatch_tasks': fingerprint_mismatches,
        'goal_fingerprint': expected_fingerprint,
        'model_source': meta.get('model_source'),
        'config_file': config_file,
        'notes': [
            'A partial or mismatched file must not be reported as the frozen task set.',
            'run_id or goal_fingerprint mismatches are identity failures, so status becomes mismatched.',
            'provenance_issues lists why provenance_ok is false; both the config file hash and its content must agree with the metadata.',
            'order_ok is informational: episodes are appended sequentially by rollout.',
        ],
    }


def _tokens(text):
    words = re.findall(r"[a-z0-9]+(?:['.][a-z0-9]+)*", str(text).lower())
    return {word for word in words if len(word) > 2 and word not in STOPWORDS}


def _observation_signature(text):
    return ' '.join(str(text).split())[:400]


def _prices(text):
    found = []
    for raw in PRICE.findall(str(text)):
        try:
            found.append(float(raw.replace(',', '')))
        except ValueError:
            continue
    return found


def _budget(instruction):
    match = BUDGET.search(str(instruction))
    return float(match.group(1)) if match else None


def _format_reason(steps):
    parse_failures = [step for step in steps if not step.get('grounding') and step.get('action') is None]
    if parse_failures:
        return (f'{len(parse_failures)} step(s) had no single parseable Action line; '
                f'first error: {parse_failures[0].get("error")}')
    unavailable = [step for step in steps if not step.get('grounding')]
    if unavailable:
        return (f'{len(unavailable)} parsed action(s) were not executable on the current page; '
                f'first: {unavailable[0].get("action")!r} ({unavailable[0].get("error")})')
    return None


def _purchase_outcome(row, steps):
    """Classify a completed purchase. WebShop can return reward 0 even when an item was bought."""
    if not row.get('done'):
        return None, None
    reward = float(row['final_reward'])
    if reward >= 1 - SUCCESS_EPS:
        return None, None
    budget = _budget(row.get('instruction', ''))
    purchases = [step for step in steps if str(step.get('action', '')).startswith('click[buy now]')]
    bought = purchases[-1] if purchases else steps[-1]
    if budget is not None and any(price > budget + 1e-9 for price in _prices(bought.get('observation', ''))):
        return 'budget', f'purchased an item priced above the {budget:.2f} budget'
    if reward <= 0:
        return ('specification',
                'purchased, but the reward is 0: no required attribute matched '
                '(wrong product type or wrong specification)')
    return ('specification',
            'purchased with partial reward: some attributes match, at least one is wrong')


def _loop_reason(steps):
    actions = [str(step.get('action')) for step in steps if step.get('action')]
    pairs = collections.Counter((str(step.get('action')), _observation_signature(step.get('observation', '')))
                                for step in steps if step.get('action'))
    if actions and (max(collections.Counter(actions).values()) >= 3 or max(pairs.values()) >= 2):
        return 'the same action was repeated without a new observation'
    return None


def _retrieval_reason(row, steps):
    searches = [str(step.get('action')) for step in steps if str(step.get('action', '')).startswith('search[')]
    queries = [action[action.index('[') + 1:-1] for action in searches]
    goal_tokens = _tokens(normalize_instruction(row.get('instruction', '')))
    if queries and goal_tokens and all(not (_tokens(query) & goal_tokens) for query in queries):
        return 'no search query shared a content word with the shopping instruction'
    return None


def classify_episode(row):
    """Heuristic triage of one failed episode.

    Returns ``{'class', 'reason', 'secondary'}``. One episode can carry several signals (a run can
    emit unparsable actions and still end with a wrong purchase), so the purchase outcome is the
    primary label and the remaining signals are kept in ``secondary`` instead of being dropped.
    """
    steps = row.get('steps') or []
    if not steps:
        return {'class': 'engineering', 'reason': 'episode recorded no steps', 'secondary': []}
    if row.get('termination') not in {'purchase', 'max_steps'}:
        return {'class': 'engineering', 'reason': f"unexpected termination {row.get('termination')!r}",
                'secondary': []}
    errors = [str(step.get('error')) for step in steps if step.get('error')]
    if any(INFRASTRUCTURE_ERROR.search(error) for error in errors):
        return {'class': 'engineering', 'reason': 'a step log contains an infrastructure error',
                'secondary': []}
    signals = {}
    format_reason = _format_reason(steps)
    if format_reason:
        signals['format'] = format_reason
    purchase_class, purchase_reason = _purchase_outcome(row, steps)
    if purchase_class:
        signals[purchase_class] = purchase_reason
    if not row.get('done'):
        loop_reason = _loop_reason(steps)
        if loop_reason:
            signals['loop'] = loop_reason
    retrieval_reason = _retrieval_reason(row, steps)
    if retrieval_reason:
        signals['retrieval'] = retrieval_reason
    for name in ('budget', 'specification', 'format', 'loop', 'retrieval'):
        if name in signals:
            return {'class': name, 'reason': signals[name],
                    'secondary': sorted(other for other in signals if other != name)}
    return {'class': 'other', 'reason': 'no rule matched; inspect this log manually', 'secondary': []}


def failure_breakdown(rows, per_class=1):
    """Count failed episodes per class and keep representative cases for the report.

    ``counts`` uses one primary label per episode; ``any_counts`` counts every episode that shows
    the signal, so a purchase failure that also produced unparsable actions is visible in both.
    """
    failures = [row for row in rows if not is_success(row)]
    counts = {name: 0 for name in FAILURE_CLASSES}
    any_counts = {name: 0 for name in FAILURE_CLASSES}
    labelled = []
    for row in failures:
        verdict = classify_episode(row)
        labelled.append((row, verdict))
        counts[verdict['class']] += 1
        for name in {verdict['class'], *verdict.get('secondary', [])}:
            any_counts[name] += 1
    cases = collections.defaultdict(list)
    ordered = sorted(labelled, key=lambda pair: (int(pair[0]['task_id']), int(pair[0].get('repeat', 0))))
    for row, verdict in ordered:
        if len(cases[verdict['class']]) < per_class:
            cases[verdict['class']].append(_case_record(row, verdict))
    denominator = len(failures) or 1
    return {
        'episodes': len(rows), 'failed_episodes': len(failures),
        'counts': counts,
        'any_counts': any_counts,
        'rates_of_failures': {name: round(count / denominator, 6) for name, count in counts.items()},
        'unclassified_tasks': sorted(int(row['task_id']) for row, verdict in labelled
                                     if verdict['class'] == 'other'),
        'cases': [case for name in FAILURE_CLASSES for case in cases[name]],
        'notes': [
            'counts assigns one primary label per episode; any_counts also counts secondary signals.',
            'Labels are heuristic triage from recorded steps; confirm each case manually before quoting it.',
            'An "other" class means no rule matched, not that the model behaved well.',
            'engineering episodes are infrastructure problems and must be excluded from method comparison.',
        ],
    }


def _case_record(row, verdict):
    steps = []
    for step in row['steps']:
        steps.append({
            'step': step.get('step'),
            'action': step.get('action'),
            'grounding': int(bool(step.get('grounding'))),
            'environment_reward': step.get('environment_reward'),
            'error': step.get('error'),
            'observation': _observation_signature(step.get('observation', '')),
        })
    return {
        'class': verdict['class'], 'reason': verdict['reason'],
        'secondary': verdict.get('secondary', []),
        'task_id': int(row['task_id']), 'repeat': int(row.get('repeat', 0)),
        'instruction': row.get('instruction', ''),
        'termination': row.get('termination'), 'done': bool(row['done']),
        'final_reward': float(row['final_reward']),
        'steps': steps,
    }


def write_case_report(cases, path, source):
    """Markdown case sheet that can be pasted into the report after human verification."""
    lines = ['# 失败案例工作表（自动分诊，引用前必须人工复核）', '',
             f'来源：`{source}`。分类规则见 `docs/EVALUATION.md`；下面的标签只是初筛，不是最终结论。', '']
    for name in FAILURE_CLASSES:
        group = [case for case in cases if case['class'] == name]
        if not group:
            continue
        lines += [f'## {FAILURE_LABELS[name]}（{len(group)} 例）', '']
        for case in group:
            lines += [f"### task {case['task_id']}（repeat {case['repeat']}）— "
                      f"termination={case['termination']}, done={case['done']}, "
                      f"final_reward={case['final_reward']:.3f}", '',
                      f"指令：{case['instruction']}", '',
                      f"分诊理由：{case['reason']}",
                      f"其他信号：{', '.join(case.get('secondary') or []) or '无'}", '',
                      '| step | action | grounding | env reward | error | observation (truncated) |',
                      '|---|---|---|---|---|---|']
            for step in case['steps']:
                observation = str(step['observation']).replace('|', '\\|')
                error = str(step['error'] or '').replace('|', '\\|')
                lines.append(f"| {step['step']} | `{step['action']}` | {step['grounding']} | "
                             f"{step['environment_reward']} | {error} | {observation} |")
            lines.append('')
    if not any(case for case in cases):
        lines += ['没有任何失败案例：要么全部成功，要么输入为空。', '']
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


PARITY_FIELDS = ('model', 'revision', 'load_in_4bit', 'seed', 'temperature', 'top_p', 'top_k',
                 'context_length', 'max_new_tokens', 'max_steps', 'environment_url')


def _parity_source(meta_path, config_path):
    if not meta_path and not config_path:
        raise ValueError('Provide --a-meta/--b-meta or --a-config/--b-config')
    if meta_path:
        meta = read_json(meta_path)
        return {'kind': 'meta', 'path': str(meta_path), 'config': meta['config'],
                'selected_ids': [int(task) for task in meta['selected_ids']],
                'repeats': int(meta.get('repeats', 1)),
                'goal_fingerprint': (meta.get('health') or {}).get('goal_fingerprint')}
    config = read_json(config_path)
    return {'kind': 'config', 'path': str(config_path), 'config': config,
            'selected_ids': None, 'repeats': None,
            'goal_fingerprint': config.get('goal_fingerprint')}


def check_parity(a_meta=None, b_meta=None, a_config=None, b_config=None, require_frozen=False):
    """Compare the two arms that are supposed to differ only by the adapter."""
    a = _parity_source(a_meta, a_config)
    b = _parity_source(b_meta, b_config)
    fields = []
    blocking = []
    for name in PARITY_FIELDS:
        left, right = a['config'].get(name), b['config'].get(name)
        ok = left == right
        fields.append({'field': name, 'a': left, 'b': right, 'equal': ok})
        if not ok:
            blocking.append(f'{name} differs: {left!r} vs {right!r}')
    fingerprint_a = a['goal_fingerprint']
    fingerprint_b = b['goal_fingerprint']
    fingerprints_equal = bool(fingerprint_a) and fingerprint_a == fingerprint_b
    fields.append({'field': 'goal_fingerprint', 'a': fingerprint_a, 'b': fingerprint_b,
                   'equal': fingerprints_equal})
    if not fingerprint_a or not fingerprint_b:
        blocking.append('goal_fingerprint is empty: fix it from /health before final evaluation')
        if require_frozen:
            blocking.append('--require-frozen was set, so an unfrozen goal fingerprint is an error')
    elif fingerprint_a != fingerprint_b:
        blocking.append('the two arms used different environment goal mappings')
    same_tasks = same_order = None
    if a['selected_ids'] is not None and b['selected_ids'] is not None:
        same_tasks = set(a['selected_ids']) == set(b['selected_ids'])
        same_order = a['selected_ids'] == b['selected_ids']
        if not same_tasks:
            blocking.append('the two arms were run on different task sets')
        if not same_order:
            blocking.append('the two arms used the same tasks in a different order; --ids files differ')
        if a['repeats'] != b['repeats']:
            blocking.append(f"repeat counts differ: {a['repeats']} vs {b['repeats']}")
    adapters = (a['config'].get('adapter'), b['config'].get('adapter'))
    warnings = []
    if adapters[0] == adapters[1]:
        warnings.append('both arms use the same adapter: this pair cannot show a training effect')
    warnings.append('Prompt and system message live in the code commit; compare commits as well.')
    return {
        'a': {'path': a['path'], 'kind': a['kind']},
        'b': {'path': b['path'], 'kind': b['kind']},
        'fields': fields,
        'same_task_set': same_tasks, 'same_task_order': same_order,
        'goal_fingerprint': {'a': fingerprint_a, 'b': fingerprint_b, 'equal': fingerprints_equal},
        'adapters': {'a': adapters[0], 'b': adapters[1]},
        'parity': not blocking,
        'blocking_issues': blocking, 'warnings': warnings,
        'notes': ['Parity covers configuration values only; the code commit and the instruction split '
                  'hash must be compared separately (see check-register).'],
    }


def evaluate(args):
    audit = inspect_run(args.jsonl, args.meta, getattr(args, 'config', None))
    if audit['status'] == 'mismatched':
        raise SystemExit(f'Run is mismatched: {audit["summary"]}. The task set or run identity does '
                         'not match the metadata; rerun instead of summarizing this file.')
    if audit['status'] == 'partial' and not getattr(args, 'allow_partial', False):
        raise SystemExit(f'Run is {audit["status"]}: {audit["summary"]}. '
                         'Refusing to summarize an incomplete task set; rerun or pass --allow-partial '
                         'and label the numbers as partial.')
    if getattr(args, 'register_row', None):
        if not audit['config_file']:
            raise SystemExit('--config is required with --register-row: the row must name and hash '
                             'the configuration that produced the run')
        if not audit['provenance_ok']:
            raise SystemExit('Refusing to register: ' + '; '.join(audit['provenance_issues'])
                             + '. The config file and the metadata do not describe the same run.')
    rows = load_episodes(args.jsonl)
    metrics = summarize_episodes(rows)
    failures = failure_breakdown(rows, max(1, int(getattr(args, 'cases_per_class', 1))))
    result = {
        'schema_version': 1,
        'generated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        'run': {
            'run_id': audit['run_id'],
            'method': getattr(args, 'method', None),
            'meta_path': str(args.meta), 'jsonl_path': str(args.jsonl),
            'jsonl_sha256': file_hash(args.jsonl),
            'config_path': str(args.config) if getattr(args, 'config', None) else None,
            'config_sha256': audit['config_file']['sha256'] if audit['config_file'] else None,
            'commit': git_head(),
        },
        'completeness': audit,
        'metrics': metrics,
        'failures': failures,
        'usable_as_result': bool(audit['complete'] and audit['provenance_ok']),
        'warnings': [],
    }
    if audit['status'] == 'partial':
        result['warnings'].append('Partial run: rates use the completed subset, not the frozen task set.')
    if audit['run_id_mismatch_tasks']:
        result['warnings'].append('Some episodes carry a different run_id than the metadata.')
    if audit['goal_fingerprint_mismatch_tasks']:
        result['warnings'].append('Some episodes carry a different goal_fingerprint than the metadata.')
    if audit['config_file'] and not audit['config_file']['matches_metadata']:
        result['warnings'].append('The config file hash does not match config_sha256 in the metadata: '
                                  'the config was edited after the run started.')
    write_json(args.output, result)
    if getattr(args, 'cases', None):
        write_case_report(failures['cases'], args.cases, args.jsonl)
    if getattr(args, 'figures_dir', None):
        from .figures import write_run_figures
        write_run_figures(metrics, failures['counts'], args.figures_dir,
                          label=getattr(args, 'method', None) or audit['run_id'][:8])
    if getattr(args, 'register_row', None):
        from .register import row_from_run, write_register_row
        snapshot = audit['config_file'] or audit_row_config(args)
        row = row_from_run(args, audit, {'path': snapshot['path'], 'sha256': snapshot['sha256']}, metrics)
        write_register_row(args.register_row, row)
        print(f'Register row written to {args.register_row}')
    print(json.dumps({'completeness': {'status': audit['status'], 'summary': audit['summary']},
                      'metrics': {key: metrics[key] for key in
                                  ('episodes', 'purchase_terminated', 'full_reward_successes',
                                   'purchase_without_full_reward', 'success_rate',
                                   'success_rate_ci95', 'mean_terminal_reward',
                                   'mean_final_reward', 'mean_steps')},
                      'failures': {'counts': failures['counts'],
                                   'any_counts': failures['any_counts']}}, ensure_ascii=False, indent=2))


def audit_row_config(args):
    """Config snapshot used for the register row (the file that was actually hashed)."""
    if not getattr(args, 'config', None):
        raise SystemExit('--config is required when writing a register row: the row must name the '
                         'configuration that produced the run')
    return {'path': str(args.config), 'sha256': file_hash(args.config)}


def check_run(args):
    audit = inspect_run(args.jsonl, args.meta, getattr(args, 'config', None))
    if getattr(args, 'output', None):
        write_json(args.output, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if audit['status'] != 'complete' or not audit['provenance_ok']:
        raise SystemExit(1)


def parity(args):
    result = check_parity(args.a_meta, args.b_meta, args.a_config, args.b_config,
                          getattr(args, 'require_frozen', False))
    if getattr(args, 'output', None):
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result['parity']:
        raise SystemExit(1)
