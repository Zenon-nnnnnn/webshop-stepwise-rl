"""Pre-run gates: spend GPU time only on inputs whose results can be audited afterwards.

Run this before a B0/B1 (or R-TG/R-PG) evaluation. It checks the things that are cheap on CPU and
expensive to discover after hours of rollout: a config that drifted, an adapter path that does not
exist, an ID list with duplicates, an unfrozen goal fingerprint, two arms that are not comparable,
or a register whose header no longer matches the tooling.
"""
from pathlib import Path

from .common import ROOT, file_hash, read_json, write_json
from .evaluation import check_parity
from .register import read_register, validate

REQUIRED_CONFIG = ('model', 'context_length', 'max_new_tokens', 'max_steps', 'temperature', 'seed')


def check_config(path, role='a', require_frozen=False):
    problems, warnings = [], []
    config = read_json(path)
    missing = [key for key in REQUIRED_CONFIG if key not in config]
    if missing:
        problems.append(f'missing required keys: {missing}')
    else:
        for key in ('context_length', 'max_new_tokens', 'max_steps'):
            if int(config[key]) < 1:
                problems.append(f'{key} must be positive')
        budget = int(config['context_length']) - int(config['max_new_tokens'])
        if budget < 512:
            warnings.append(f'context {config["context_length"]} minus generation '
                            f'{config["max_new_tokens"]} leaves only {budget} tokens for history')
        if float(config.get('temperature', 0)) > 0:
            warnings.append('temperature > 0: sampling settings must match across arms and be '
                            'recorded; greedy is easier to compare')
        if not config.get('goal_fingerprint'):
            message = 'goal_fingerprint is empty: fill it from /health before a final evaluation'
            (problems if require_frozen else warnings).append(message)
    adapter = config.get('adapter')
    if adapter:
        if not Path(adapter).is_dir():
            problems.append(f'adapter directory does not exist: {adapter}')
        elif not any((Path(adapter) / name).exists() for name in
                     ('adapter_config.json', 'adapter_model.safetensors', 'adapter_model.bin')):
            problems.append(f'adapter directory has no adapter_config/model file: {adapter}')
    return {
        'role': role, 'path': str(path), 'sha256': file_hash(path),
        'model': config.get('model'), 'adapter': adapter,
        'goal_fingerprint': config.get('goal_fingerprint'),
        'problems': problems, 'warnings': warnings,
    }


def check_ids(path):
    problems, warnings = [], []
    ids = read_json(path)
    if not isinstance(ids, list):
        problems.append('ID file must be a JSON list')
        ids = []
    elif not ids:
        problems.append('ID file is empty')
    else:
        if any(type(task) is not int for task in ids):
            problems.append('every task ID must be an integer (strings or floats are rejected)')
        duplicates = sorted({task for task in ids if ids.count(task) > 1})
        if duplicates:
            problems.append(f'{len(duplicates)} duplicated task ID(s), e.g. {duplicates[:5]}')
        if any(task < 0 for task in ids):
            problems.append('negative task ID')
        if len(ids) < 20:
            warnings.append(f'only {len(ids)} tasks: intervals will be wide and comparisons weak')
    return {
        'path': str(path), 'sha256': file_hash(path),
        'count': len(ids), 'unique': len(set(ids)),
        'problems': problems, 'warnings': warnings,
    }


def run(args):
    configs = []
    if args.a_config:
        configs.append(check_config(args.a_config, 'a', args.require_frozen))
    if args.b_config:
        configs.append(check_config(args.b_config, 'b', args.require_frozen))
    ids = check_ids(args.ids) if args.ids else None
    parity = None
    if args.a_config and args.b_config:
        parity = check_parity(a_config=args.a_config, b_config=args.b_config,
                              require_frozen=args.require_frozen)
    register = None
    if args.register:
        fieldnames, rows = read_register(args.register)
        register = validate(fieldnames, rows, Path(args.root), check_git=not args.no_git_check)
    problems = [f'config {item["role"]}: {message}' for item in configs for message in item['problems']]
    problems += [f'ids: {message}' for message in (ids or {}).get('problems', [])]
    warnings = [f'config {item["role"]}: {message}' for item in configs for message in item['warnings']]
    warnings += [f'ids: {message}' for message in (ids or {}).get('warnings', [])]
    # An unfrozen fingerprint blocks a final evaluation, not a smoke run; --require-frozen decides.
    for message in (parity or {}).get('blocking_issues', []):
        if 'goal_fingerprint' in message and not args.require_frozen:
            warnings.append(f'parity: {message}')
        else:
            problems.append(f'parity: {message}')
    warnings += [f'parity: {message}' for message in (parity or {}).get('warnings', [])
                 if 'goal_fingerprint' not in message]
    problems += [f'register: {message}' for message in (register or {}).get('errors', [])]
    warnings += [f'register: {message}' for message in (register or {}).get('warnings', [])]
    result = {
        'schema_version': 1,
        'ready': not problems,
        'configs': configs,
        'ids': ids,
        'parity': parity,
        'register': register,
        'problems': problems, 'warnings': warnings,
        'notes': [
            'ready=true only means the inputs are internally consistent; it does not prove the '
            'environment mapping, the model load or the adapter behaviour.',
            'Keep this file with the run: it records the hashes that the register row must repeat.',
        ],
    }
    if args.output:
        write_json(args.output, result)
    print(f'ready={result["ready"]}  configs={len(configs)}  '
          f'ids={ids["count"] if ids else "-"}  parity={"checked" if parity else "-"}')
    for message in problems:
        print(f'PROBLEM {message}')
    for message in warnings:
        print(f'warning {message}')
    if not result['ready']:
        raise SystemExit(1)
