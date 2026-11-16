"""Experiment register: one CSV row per real run, validated against its evidence columns.

The register is the only place a number may enter the report from. A row is acceptable only when
it names the run, the frozen configuration (path and hash), the code commit, the instruction-split
hash and the environment goal fingerprint, and when its counts agree with its own rate.
"""
import csv
import json
import math
import re
import subprocess
from pathlib import Path

from .common import ROOT, file_hash, git_head, read_json, write_json

COLUMNS = (
    'run_id', 'status', 'method', 'model', 'seed', 'split_sha256', 'goal_fingerprint', 'git_commit',
    'config_path', 'config_sha256', 'episodes', 'successes', 'purchase_terminated', 'success_rate',
    'success_rate_ci95', 'mean_reward', 'mean_steps', 'invalid_action_rate', 'peak_vram_gb',
    'evidence_path', 'notes',
)
STATUSES = ('planned', 'running', 'complete', 'failed')
# R-TG / R-PG are the current reward arms (terminal+G and stepwise+G, same grounding weight).
# R0 / R1 stay accepted only for exports produced before the 11/16 plan; new runs use R-TG/R-PG.
METHODS = ('B0', 'B1', 'P0', 'R-TG', 'R-PG', 'R0', 'R1')
LEGACY_METHODS = ('R0', 'R1')
HEX64 = re.compile(r'[0-9a-f]{64}')
HEX_COMMIT = re.compile(r'[0-9a-f]{7,40}')
CI95 = re.compile(r'\[\s*(?:0(?:\.\d+)?|1(?:\.0+)?)\s*,\s*(?:0(?:\.\d+)?|1(?:\.0+)?)\s*\]')


def read_register(path):
    with Path(path).open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def write_register_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS), lineterminator='\n')
        writer.writeheader()
        writer.writerow({column: row.get(column, '') for column in COLUMNS})


def _relative(path):
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def row_from_run(args, audit, config_snapshot, metrics):
    """Build a register row from one verified evaluation run."""
    method = getattr(args, 'method', None)
    if method not in METHODS:
        raise SystemExit(f'--method is required and must be one of {METHODS} when writing a register row')
    split_ids = getattr(args, 'split_ids', None)
    if not split_ids:
        raise SystemExit('--split-ids is required: the row must record which frozen ID list was used')
    if not audit['complete']:
        raise SystemExit('Refusing to register an incomplete run: ' + audit['summary'])
    commit = git_head()
    if not commit:
        raise SystemExit('Cannot determine the git commit; run this from the repository checkout')
    config = read_json(config_snapshot['path'])
    interval = metrics['success_rate_ci95']
    if not interval:
        raise SystemExit('Run has no success interval; cannot register a result without uncertainty')
    peak = getattr(args, 'peak_vram_gb', None)
    return {
        'run_id': audit['run_id'],
        'status': 'complete',
        'method': method,
        'model': config.get('model', ''),
        'seed': str(config.get('seed', '')),
        'split_sha256': file_hash(split_ids),
        'goal_fingerprint': audit['goal_fingerprint'] or '',
        'git_commit': commit,
        'config_path': _relative(config_snapshot['path']),
        'config_sha256': config_snapshot['sha256'],
        'episodes': str(metrics['episodes']),
        'successes': str(metrics['full_reward_successes']),
        'purchase_terminated': str(metrics['purchase_terminated']),
        'success_rate': f"{metrics['success_rate']:.6f}",
        'success_rate_ci95': f'[{interval[0]:.6f}, {interval[1]:.6f}]',
        'mean_reward': f"{metrics['mean_final_reward']:.6f}",
        'mean_steps': f"{metrics['mean_steps']:.2f}",
        'invalid_action_rate': f"{metrics['invalid_action_rate']:.6f}",
        'peak_vram_gb': '' if peak is None else f'{float(peak):.2f}',
        'evidence_path': _relative(args.jsonl),
        'notes': f'{method} on the frozen val list; purchase terminations and full-reward successes split',
    }


def _git_has_commit(root, commit):
    try:
        out = subprocess.run(['git', '-C', str(root), 'cat-file', '-e', commit + '^{commit}'],
                             capture_output=True, text=True, check=False)
    except OSError:
        return None
    return out.returncode == 0


def validate(fieldnames, rows, root=ROOT, check_git=True):
    """Validate the register header and every data row; returns a JSON-serialisable report."""
    errors, warnings = [], []
    fields = list(fieldnames or [])
    if fields != list(COLUMNS):
        errors.append('Header must be exactly: ' + ','.join(COLUMNS)
                      + ' (found: ' + ','.join(fields) + ')')
        return {'rows': len(rows), 'errors': errors, 'warnings': warnings, 'status': 'invalid'}
    if not (Path(root) / '.git').exists():
        if check_git:
            warnings.append('Not a git checkout: commit existence was not verified')
    seen = set()
    for number, row in enumerate(rows, start=2):
        label = f'line {number}'

        def problem(message):
            errors.append(f'{label}: {message}')

        run_id = (row.get('run_id') or '').strip()
        if not run_id:
            problem('run_id is empty')
        elif run_id in seen:
            problem('duplicate run_id')
        else:
            seen.add(run_id)
        status = (row.get('status') or '').strip()
        if status not in STATUSES:
            problem(f'status must be one of {STATUSES} (found {status!r})')
            continue
        method = (row.get('method') or '').strip()
        if method not in METHODS:
            problem(f'method must be one of {METHODS} (found {method!r})')
        elif method in LEGACY_METHODS:
            warnings.append(f'{label}: {method} is a legacy label; current arms are R-TG / R-PG')
        if not (row.get('model') or '').strip():
            problem('model is empty')
        seed = _int_field(row, 'seed', problem, required=status == 'complete', minimum=0)
        commit = (row.get('git_commit') or '').strip()
        if commit:
            if not HEX_COMMIT.fullmatch(commit):
                problem('git_commit must be 7-40 lowercase hex characters')
            elif check_git:
                exists = _git_has_commit(root, commit)
                if exists is False:
                    problem(f'git_commit {commit} does not exist in this repository')
                elif exists is None:
                    warnings.append(f'{label}: git commit existence could not be checked')
        elif status == 'complete':
            problem('git_commit is empty for a complete run')
        split = (row.get('split_sha256') or '').strip()
        if split and not HEX64.fullmatch(split):
            problem('split_sha256 must be 64 lowercase hex characters (sha256 of the frozen ID list)')
        elif not split and status == 'complete':
            problem('split_sha256 is empty for a complete run')
        fingerprint = (row.get('goal_fingerprint') or '').strip()
        if fingerprint and not HEX64.fullmatch(fingerprint):
            problem('goal_fingerprint must be 64 lowercase hex characters')
        elif not fingerprint and status == 'complete':
            problem('goal_fingerprint is empty for a complete run')
        config_path = (row.get('config_path') or '').strip()
        config_hash = (row.get('config_sha256') or '').strip()
        if config_path:
            candidate = Path(config_path)
            candidate = candidate if candidate.is_absolute() else Path(root) / candidate
            if not candidate.is_file():
                warnings.append(f'{label}: config_path {config_path} is not present in this checkout')
            elif config_hash:
                actual = file_hash(candidate)
                if actual != config_hash:
                    problem(f'config_sha256 does not match {config_path}')
        elif status == 'complete':
            problem('config_path is empty for a complete run')
        if config_hash and not HEX64.fullmatch(config_hash):
            problem('config_sha256 must be 64 lowercase hex characters')
        elif not config_hash and status == 'complete':
            problem('config_sha256 is empty for a complete run')
        episodes = _int_field(row, 'episodes', problem, required=status == 'complete', minimum=0)
        successes = _int_field(row, 'successes', problem, required=status == 'complete', minimum=0)
        purchases = _int_field(row, 'purchase_terminated', problem, required=status == 'complete', minimum=0)
        rate = _float_field(row, 'success_rate', problem, required=status == 'complete', minimum=0, maximum=1)
        mean_reward = _float_field(row, 'mean_reward', problem, required=status == 'complete',
                                   minimum=0, maximum=1)
        mean_steps = _float_field(row, 'mean_steps', problem, required=status == 'complete', minimum=0)
        invalid = _float_field(row, 'invalid_action_rate', problem, required=status == 'complete',
                               minimum=0, maximum=1)
        _float_field(row, 'peak_vram_gb', problem, required=False, minimum=0)
        if episodes is not None and episodes < 1 and status == 'complete':
            problem('episodes must be at least 1 for a complete run')
        if None not in (episodes, successes) and successes > episodes:
            problem('successes cannot exceed episodes')
        if None not in (purchases, episodes) and purchases > episodes:
            problem('purchase_terminated cannot exceed episodes')
        if None not in (successes, purchases) and successes > purchases:
            problem('successes cannot exceed purchase_terminated: success requires a purchase')
        if None not in (episodes, successes, rate) and episodes:
            if abs(rate - successes / episodes) > 1e-6:
                problem(f'success_rate {rate} does not match successes/episodes '
                        f'{successes / episodes:.6f}')
        interval_raw = (row.get('success_rate_ci95') or '').strip()
        if status == 'complete':
            if not interval_raw:
                problem('success_rate_ci95 is empty for a complete run')
            elif not CI95.fullmatch(interval_raw):
                problem('success_rate_ci95 must look like [0.123456, 0.456789]')
            elif rate is not None:
                low, high = [float(part) for part in interval_raw.strip('[]').split(',')]
                if not low <= rate <= high:
                    problem(f'success_rate {rate} is outside its own interval {interval_raw}')
        if status == 'complete' and mean_steps is not None and mean_steps <= 0:
            problem('mean_steps must be positive for a complete run')
        evidence = (row.get('evidence_path') or '').strip()
        if not evidence:
            if status == 'complete':
                problem('evidence_path is empty for a complete run')
        else:
            candidate = Path(evidence)
            candidate = candidate if candidate.is_absolute() else Path(root) / candidate
            if not candidate.exists():
                warnings.append(f'{label}: evidence_path {evidence} is not present in this checkout')
        if status == 'complete' and not (row.get('notes') or '').strip():
            warnings.append(f'{label}: notes is empty')
    return {
        'rows': len(rows), 'errors': errors, 'warnings': warnings,
        'status': 'invalid' if errors else 'ok',
        'head': git_head(Path(root)),
        'notes': ['A header-only register is valid while no real run exists.',
                  'Complete rows need run_id, method, model, seed, split_sha256, goal_fingerprint, '
                  'git_commit, config_path, config_sha256, counts, rate, interval and evidence_path.',
                  'Numbers that cannot be traced to these columns must not enter the report.'],
    }


def _int_field(row, key, problem, required, minimum=None):
    raw = (row.get(key) or '').strip()
    if not raw:
        if required:
            problem(f'{key} is empty')
        return None
    try:
        value = int(raw)
    except ValueError:
        problem(f'{key} must be an integer (found {raw!r})')
        return None
    if minimum is not None and value < minimum:
        problem(f'{key} must be >= {minimum}')
    return value


def _float_field(row, key, problem, required, minimum=None, maximum=None):
    raw = (row.get(key) or '').strip()
    if not raw:
        if required:
            problem(f'{key} is empty')
        return None
    try:
        value = float(raw)
    except ValueError:
        problem(f'{key} must be a number (found {raw!r})')
        return None
    if not math.isfinite(value):
        problem(f'{key} must be finite')
        return None
    if minimum is not None and value < minimum:
        problem(f'{key} must be >= {minimum}')
    if maximum is not None and value > maximum:
        problem(f'{key} must be <= {maximum}')
    return value


def run(args):
    register = Path(args.register)
    if not register.is_file():
        raise SystemExit(f'Register not found: {register}')
    fieldnames, rows = read_register(register)
    result = validate(fieldnames, rows, Path(args.root), not args.no_git_check)
    result['register'] = str(register)
    if getattr(args, 'output', None):
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result['errors']:
        raise SystemExit(1)


def results_table(fieldnames, rows, commit=None, source='results/experiment-register.csv'):
    """Markdown result table. Only ``status=complete`` rows are quoted; nothing is recomputed."""
    if list(fieldnames or []) != list(COLUMNS):
        raise ValueError('Register header does not match register.COLUMNS; run check-register first')
    complete = [row for row in rows if (row.get('status') or '').strip() == 'complete']
    order = {name: index for index, name in enumerate(METHODS)}
    complete.sort(key=lambda row: (order.get(str(row.get('method')).strip(), len(order)),
                                   str(row.get('run_id', ''))))
    lines = [f'生成自 `{source}`（commit `{commit or "unknown"}`）。只含 `status=complete` 行；'
             '数值来自登记表，未重新计算，区间含义见 docs/EVALUATION.md。', '',
             '| method | model | episodes | purchases | successes | success_rate | 95% CI | invalid action | seed | config | notes |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for row in complete:
        lines.append('| {method} | {model} | {episodes} | {purchases} | {successes} | {rate} | {ci} | '
                     '{invalid} | {seed} | `{config}` | {notes} |'.format(
                         method=row.get('method', ''), model=row.get('model', ''),
                         episodes=row.get('episodes', ''), purchases=row.get('purchase_terminated', ''),
                         successes=row.get('successes', ''), rate=row.get('success_rate', ''),
                         ci=row.get('success_rate_ci95', ''), invalid=row.get('invalid_action_rate', ''),
                         seed=row.get('seed', ''), config=row.get('config_path', ''),
                         notes=(row.get('notes') or '').replace('|', '\\|')))
    if not complete:
        lines.append('| — | — | — | — | — | — | — | — | — | — | 暂无 complete 行；真实结果出来后再填 |')
    return '\n'.join(lines) + '\n'


def table(args):
    fieldnames, rows = read_register(args.register)
    text = results_table(fieldnames, rows, git_head(), source=str(args.register))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding='utf-8')
    print(f'Wrote {output} from {len(rows)} register rows')
