"""Recompute the Nov 22 development runs from a privately extracted evidence pack.

This script checks stored records. It does not load a policy or run a benchmark.
"""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values, p):
    values = sorted(values)
    at = (len(values) - 1) * p
    lo = math.floor(at)
    return values[lo] + (values[math.ceil(at)] - values[lo]) * (at - lo)


def paired(left, right, ids, field):
    differences = [right[i][field] - left[i][field] for i in ids]
    rng = random.Random(42)
    samples = [sum(rng.choices(differences, k=len(ids))) / len(ids)
               for _ in range(10000)]
    return {'difference': sum(differences) / len(ids),
            'ci95': [quantile(samples, .025), quantile(samples, .975)]}


def audit(root):
    frozen = read(root / 'val_ids.json')
    assert len(frozen) == len(set(frozen)) == 160
    expected = [i for i in frozen if i != 6700]
    assert len(expected) == 159
    result = {'scope': 'stored development records; historical FP32 label unverified',
              'scheduled_tasks': 160, 'covered_tasks': 159, 'missing_tasks': [6700],
              'bootstrap': {'resamples': 10000, 'seed': 42,
                            'implementation': 'Python random.Random.choices; linear quantiles',
                            'task_order': 'frozen val_ids.json'},
              'runs': {}, 'comparisons': {}, 'test_authorization': False}
    aligned = {}
    shared = None
    for directory in sorted((root / 'runs').iterdir()):
        config = read(directory / 'config.json')
        comparable = {k: v for k, v in config.items() if k not in ('seed', 'adapter')}
        if shared is None:
            shared = comparable
        assert shared == comparable, directory.name
        assert not config['load_in_4bit']
        records = []
        batches = []
        for path in sorted((directory / 'chunks').glob('rollout-*.jsonl')):
            meta_path = path.with_suffix('.meta.json')
            meta = read(meta_path)
            ids_path = path.with_name(path.stem.replace('rollout-', 'ids-') + '.json')
            ids = read(ids_path)
            rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            assert meta['config'] == config, path
            assert meta['config_sha256'] == digest(directory / 'config.json'), path
            assert meta['ids_sha256'] == digest(ids_path), path
            assert meta['selected_ids'] == ids, path
            assert [r['task_id'] for r in rows] == ids[:len(rows)], path
            assert rows and all(r['run_id'] == meta['run_id'] for r in rows), path
            assert meta['repeats'] == 1 and all(r['repeat'] == 0 for r in rows), path
            health = meta['health']
            assert health['ready'] and health['split_status']['verified'], path
            assert health['alignment_mismatches'] == 0 and health['option_state'], path
            assert all(r['goal_fingerprint'] == health['goal_fingerprint'] ==
                       config['goal_fingerprint'] for r in rows), path
            batches.append({'file': str(path.relative_to(root)), 'run_id': meta['run_id'],
                            'rows': len(rows), 'planned': len(ids),
                            'jsonl_sha256': digest(path), 'meta_sha256': digest(meta_path)})
            records.extend(rows)
        assert len(records) == len({r['task_id'] for r in records}) == 159, directory
        assert {r['task_id'] for r in records} == set(expected), directory
        data = {r['task_id']: {'success': int(r['done'] and r['final_reward'] >= 1 - 1e-6),
                               'terminal_reward': r['final_reward'] if r['done'] else 0.}
                for r in records}
        aligned[directory.name] = data
        steps = [s for r in records for s in r['steps']]
        result['runs'][directory.name] = {
            'episodes': len(records), 'purchases': sum(bool(r['done']) for r in records),
            'successes': sum(v['success'] for v in data.values()),
            'mean_terminal_reward': sum(v['terminal_reward'] for v in data.values()) / 159,
            'decisions': len(steps), 'invalid_actions': sum(not s['grounding'] for s in steps),
            'batches': batches, 'historical_runtime_dtype_recorded': False}
    assert set(aligned) == {'b0', 'b1', 'tg42', 'op42', 'tg43', 'op43'}
    for left, right in [('b0', 'b1'), ('b1', 'tg42'), ('b1', 'tg43'),
                        ('b1', 'op42'), ('b1', 'op43'), ('tg42', 'op42'), ('tg43', 'op43')]:
        a, b = aligned[left], aligned[right]
        result['comparisons'][right + '_minus_' + left] = {
            'success': paired(a, b, expected, 'success'),
            'terminal_reward': paired(a, b, expected, 'terminal_reward'),
            'only_left_success': sum(a[i]['success'] and not b[i]['success'] for i in expected),
            'only_right_success': sum(b[i]['success'] and not a[i]['success'] for i in expected)}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('Verified 28 batches, six runs of 159 tasks. Historical dtype remains unverified.')
