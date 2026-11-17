"""Build a deterministic real-data engineering smoke subset, never a benchmark."""
import argparse
import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
SFT_SHA = '721e46dba410cff900e187bccea91a456c38b105b6d14d068d7762afb7791de3'

def require(condition, message):
    """Always-on validation, including under python -O."""
    if not condition:
        raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--split-dir', type=Path, help='Use an isolated offline split, leaving formal split files untouched')
    args = parser.parse_args()
    repo = args.repo.resolve()
    split_dir = args.split_dir.resolve() if args.split_dir else repo/'data/processed'
    require(sha(repo/'data/raw/webshop_sft.json') == SFT_SHA, 'Wrong original SFT version')
    split = json.loads((split_dir/'split-manifest.json').read_text())
    require(split['source_sha256'] == SFT_SHA and split['seed'] == 42, 'prepare.py: validation failed at original line 31')
    test_ids = set(json.loads((split_dir/'test_ids.json').read_text()))
    manifest = {'purpose': 'two-update engineering smoke; not formal training/evaluation',
                'source_sft_sha256': SFT_SHA, 'split_manifest_sha256': sha(split_dir/'split-manifest.json'),
                'environment_goal_map_verified': bool(split.get('goal_map_sha256')), 'seed': 42,
                'selection': 'sort each split by task ID; independently shuffle with seed 42; take first 16/4 dialogues',
                'datasets': {}}
    manifest['source_files_sha256'] = {name: sha(repo/name) for name in [
        'src/webshop_lab/model.py', 'src/webshop_lab/common.py', 'configs/sft-qwen15b.json']}
    selections = {}
    for name, count in [('train', 16), ('val', 4)]:
        source = split_dir/f'{name}.json'
        rows = sorted(json.loads(source.read_text()), key=lambda row: row['id'])
        random.Random(42).shuffle(rows)
        selected = rows[:count]
        require(len(selected) == count, 'prepare.py: validation failed at original line 46')
        require(not ({r['id'] for r in selected} & test_ids), 'prepare.py: validation failed at original line 47')
        selections[name] = selected
        dest = HERE/f'data/{name}-smoke.json'
        write(dest, selected)
        manifest['datasets'][name] = {'source_sha256': sha(source), 'subset_sha256': sha(dest),
            'dialogues': count, 'task_ids': [r['id'] for r in selected]}
    require(not ({r['id'] for r in selections['train']} & {r['id'] for r in selections['val']}), 'prepare.py: validation failed at original line 53')
    config = json.loads((repo/'configs/sft-qwen15b.json').read_text())
    config.update({'train_file': 'data/train-smoke.json', 'val_file': 'data/val-smoke.json',
                   'output_dir': 'outputs/qwen15b-2step-seed42', 'logging_steps': 1,
                   'model_revision': '989aa7980e4cf806f80c7fef2b1adb7bc71aa306',
                   'max_steps': 2, 'purpose': manifest['purpose']})
    write(HERE/'smoke-config.json', config)
    write(HERE/'data/subset-manifest.json', manifest)
    expected = HERE/'evidence/subset-manifest.json'
    if expected.exists():
        require(manifest == json.loads(expected.read_text()), 'Subset does not match the handed-off manifest; stop and reconcile inputs')
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    main()
