"""Dependency-free linear progress baseline, NOT the paper's neural SPA estimator.

Learn w such that sum_t dot(w, phi(observation_t, response_t)) approximates
the purchase terminal reward (zero for an episode without a purchase). Hashed bag-of-words features are fixed and use no test data.
The decomposition is not uniquely identified by terminal supervision.
"""
import hashlib
import json
import math
import random
import re
from pathlib import Path
from .common import read_json, write_json, file_hash, instruction_hash
from .evaluation import load_episodes, terminal_reward


def features(step, dimensions):
    tokens = re.findall(r'[a-z0-9]+', (step['observation'] + '\n' + step['response']).lower())
    result = {0: 1.}  # Per-step bias is part of the additive estimator.
    for token in tokens:
        slot = 1 + int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], 'little') % (dimensions - 1)
        result[slot] = result.get(slot, 0.) + 1. / max(1, len(tokens))
    return result


def total_features(episode, dimensions):
    result = {}
    for step in episode['steps']:
        for key, value in features(step, dimensions).items():
            result[key] = result.get(key, 0.) + value
    return result


def dot(weights, vector):
    return sum(weights[i] * v for i, v in vector.items())


def train(args):
    train_rows, val_rows = load_episodes(args.train), load_episodes(args.val)
    if {r['task_id'] for r in train_rows} & {r['task_id'] for r in val_rows}:
        raise ValueError('Progress train/val task overlap')
    if {instruction_hash(r['instruction']) for r in train_rows} & {instruction_hash(r['instruction']) for r in val_rows}:
        raise ValueError('Progress train/val instruction overlap')
    if {r.get('goal_fingerprint') for r in train_rows} != {r.get('goal_fingerprint') for r in val_rows}:
        raise ValueError('Progress train/val environment mismatch')
    if args.dimensions < 2 or args.epochs < 1:
        raise ValueError('dimensions>=2 and epochs>=1 required')
    train_targets = [terminal_reward(row) for row in train_rows]
    if len(set(train_targets)) < 2:
        raise ValueError('Training rewards have no variation; collect more diverse trajectories')
    w = [0.] * args.dimensions
    data = [(total_features(row, args.dimensions), target)
            for row, target in zip(train_rows, train_targets)]
    validation = [(total_features(row, args.dimensions), terminal_reward(row)) for row in val_rows]
    best_loss, best_w, best_epoch = math.inf, None, None
    rng = random.Random(args.seed)
    for epoch in range(args.epochs):
        rng.shuffle(data)
        for vector, y in data:
            error = dot(w, vector) - y
            # Normalized SGD avoids instability from variable trajectory lengths.
            lr = .1 / (1. + sum(v * v for v in vector.values()))
            for i, value in vector.items():
                w[i] -= lr * (2 * error * value + 1e-4 * w[i])
        loss = sum((dot(w, x) - y)**2 for x, y in validation) / len(validation)
        if loss < best_loss:
            best_loss, best_w, best_epoch = loss, list(w), epoch + 1
    write_json(args.output, {'kind': 'hashed-linear-additive-v1', 'dimensions': args.dimensions,
        'weights': best_w, 'target': 'terminal_reward', 'best_epoch': best_epoch, 'validation_mse': best_loss, 'seed': args.seed,
        'train_sha256': file_hash(args.train), 'val_sha256': file_hash(args.val),
        'goal_fingerprint': train_rows[0].get('goal_fingerprint'),
        'train_task_ids': sorted({r['task_id'] for r in train_rows}),
        'val_task_ids': sorted({r['task_id'] for r in val_rows}),
        'note': 'Low-cost linear baseline; not a reproduction of neural SPA and not evidence of causal step attribution.'})
    print(f'Linear baseline saved; validation MSE={best_loss:.6f}, epoch={best_epoch}')


def predict(args):
    model = read_json(args.model)
    if model.get('kind') != 'hashed-linear-additive-v1':
        raise ValueError('Unsupported estimator format')
    rows = load_episodes(args.input)
    output = Path(args.output)
    if output.exists():
        raise ValueError('Refusing to overwrite annotated episodes')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as f:
        for row in rows:
            if row.get('goal_fingerprint') != model['goal_fingerprint']:
                raise ValueError('Estimator/environment mismatch')
            for step in row['steps']:
                step['progress'] = dot(model['weights'], features(step, model['dimensions']))
            row['progress_model_sha256'] = file_hash(args.model)
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'Annotated {len(rows)} trajectories; predictions are not policy improvements.')
