"""Rollout JSONL → PRM 样本，并在任务/指令层面隔离 fit 与 dev。

只做数据转换与切分，不加载模型；tokenizer 由调用方传入。
"""
import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from ..model import SYSTEM
from .tokenize import TrajectoryTooLong, tokenize_trajectory

SCHEMA_VERSION = 'prm-data-v1'
REWARD_KEYS = {'task_id', 'instruction', 'steps', 'done', 'final_reward', 'success', 'termination'}


def normalize_instruction(text):
    if not isinstance(text, str):
        raise ValueError('instruction 必须是字符串')
    return ' '.join(text.lower().split())


def terminal_reward(row):
    """任务级奖励：只有完成购买才取记录分，其余为 0（页面匹配分不是任务奖励）。"""
    if not isinstance(row, dict):
        raise ValueError('episode 必须是对象')
    done = row.get('done')
    if type(done) is not bool:
        raise ValueError('episode.done 必须是布尔值，不能把缺失当成失败')
    value = float(row.get('final_reward', 0.0))
    reward = value if done else 0.0
    if not (0.0 <= reward <= 1.0):
        raise ValueError('terminal reward 必须位于 [0,1]')
    return reward


def episode_messages(row):
    """把一条 rollout 还原成 system + (user, assistant)* 的完整对话。"""
    if not isinstance(row, dict):
        raise ValueError('episode 必须是对象')
    missing = REWARD_KEYS - set(row)
    if missing:
        raise ValueError('episode 缺少字段: %s' % sorted(missing))
    instruction = row['instruction']
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('instruction 不能为空')
    steps = row['steps']
    if not isinstance(steps, list) or not steps:
        raise ValueError('episode 至少需要一个 step')
    messages = [{'role': 'system',
                 'content': SYSTEM + '\nShopping instruction: ' + instruction.strip()}]
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError('step 必须是对象')
        observation, response = step.get('observation'), step.get('response')
        if not isinstance(observation, str) or not isinstance(response, str):
            raise ValueError('step 缺少 observation/response 文本')
        if not response.strip():
            raise ValueError('空的 assistant 回复不能作为评分位置')
        messages.append({'role': 'user', 'content': observation})
        messages.append({'role': 'assistant', 'content': response})
    return messages


def load_episodes(paths):
    """读取一个或多个 rollout JSONL，按 (task_id, repeat) 去重并保留来源。"""
    episodes, seen, sources = [], set(), []
    for path in paths:
        p = Path(path)
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        sources.append({'path': str(p), 'sha256': digest})
        with p.open(encoding='utf-8') as stream:
            for line_no, line in enumerate(stream, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError('%s:%d 不是合法 JSON' % (p, line_no)) from exc
                key = (int(row['task_id']), int(row.get('repeat', 0)))
                if key in seen:
                    raise ValueError('重复的 task_id/repeat: %s' % (key,))
                seen.add(key)
                row['_source'] = sources[-1]
                episodes.append(row)
    if not episodes:
        raise ValueError('没有可用轨迹')
    return episodes, sources


def split_by_task(episodes, fit_fraction=0.8, seed=42):
    """按任务与归一化指令双重隔离切分，返回 (fit, dev) 与切分摘要。"""
    if not (0.0 < fit_fraction < 1.0):
        raise ValueError('fit_fraction 必须在 (0,1)')
    tasks = {}
    for row in episodes:
        task = int(row['task_id'])
        tasks.setdefault(task, []).append(row)
    ordered = sorted(tasks)
    random.Random(seed).shuffle(ordered)
    target_tasks = max(1, int(round(len(ordered) * fit_fraction)))
    fit_tasks = set(ordered[:target_tasks])
    fit = [r for r in episodes if int(r['task_id']) in fit_tasks]
    dev = [r for r in episodes if int(r['task_id']) not in fit_tasks]
    if not dev:
        raise ValueError('dev 为空，请增加轨迹或降低 fit_fraction')
    fit_instructions = {normalize_instruction(r['instruction']) for r in fit}
    dev_instructions = {normalize_instruction(r['instruction']) for r in dev}
    overlap = fit_instructions & dev_instructions
    if overlap:
        raise ValueError('fit/dev 指令重叠 %d 条，需重新切分' % len(overlap))
    summary = {'seed': seed, 'fit_fraction': fit_fraction,
               'fit_tasks': len(fit_tasks), 'dev_tasks': len(ordered) - len(fit_tasks),
               'fit_episodes': len(fit), 'dev_episodes': len(dev),
               'instruction_overlap': 0}
    return fit, dev, summary


def reward_distribution(episodes):
    values = [terminal_reward(r) for r in episodes]
    buckets = Counter()
    for v in values:
        if v <= 0.0:
            buckets['zero'] += 1
        elif v >= 1.0:
            buckets['full'] += 1
        else:
            buckets['partial'] += 1
    return {'count': len(values), 'mean': sum(values) / len(values) if values else 0.0,
            'buckets': dict(buckets)}


def build_samples(episodes, tokenizer, max_length):
    """把 episode 编码成 PRM 样本；超长样本显式计数，绝不截断。"""
    samples, rejected = [], Counter()
    for row in episodes:
        try:
            messages = episode_messages(row)
            encoded = tokenize_trajectory(tokenizer, messages, max_length)
        except TrajectoryTooLong as exc:
            rejected['too_long'] += 1
            continue
        except ValueError as exc:
            reason = str(exc)
            if '空的 assistant' in reason:
                rejected['empty_action'] += 1
            elif 'reserved template token' in reason or 'Reserved template token' in reason:
                rejected['reserved_token'] += 1
            else:
                raise
            continue
        samples.append({
            'task_id': int(row['task_id']),
            'repeat': int(row.get('repeat', 0)),
            'instruction_sha256': hashlib.sha256(
                normalize_instruction(row['instruction']).encode()).hexdigest(),
            'terminal_reward': terminal_reward(row),
            'num_turns': len(encoded['turn_end_indices']),
            'num_tokens': len(encoded['input_ids']),
            'template_sha256': encoded['template_sha256'],
            'encoded': encoded,
        })
    return samples, dict(rejected)


def write_dataset(out_dir, config_fingerprint, sources, split_summary, fit_samples, dev_samples,
                  rejections, tokenizer_identity):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    payload = {}
    for name, rows in (('fit', fit_samples), ('dev', dev_samples)):
        path = out / (name + '.jsonl')
        with path.open('w', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        with path.open('rb') as stream:
            digest = hashlib.sha256(stream.read()).hexdigest()
        payload[name] = {'path': str(path), 'sha256': digest, 'samples': len(rows),
                         'turns': sum(r['num_turns'] for r in rows)}
    manifest = {
        'schema_version': SCHEMA_VERSION,
        'config_fingerprint': config_fingerprint,
        'tokenizer_identity': tokenizer_identity,
        'sources': sources,
        'split': split_summary,
        'files': payload,
        'rejections': rejections,
        'reward_fit': reward_distribution([
            {'done': True, 'final_reward': r['terminal_reward']} for r in fit_samples]),
        'reward_dev': reward_distribution([
            {'done': True, 'final_reward': r['terminal_reward']} for r in dev_samples]),
        'score_position': 'assistant_im_end',
        'truncation': 'forbidden: overlong trajectories are rejected and counted',
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                                       encoding='utf-8')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description='Build PRM fit/dev data from rollout JSONL')
    parser.add_argument('--rollout', action='append', required=True)
    parser.add_argument('--config', required=True, help='PRMConfig JSON (for max_length/fingerprint)')
    parser.add_argument('--output', required=True)
    parser.add_argument('--fit-fraction', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)

    from .config import load_config
    config = load_config(args.config)
    if config.load_in_4bit:
        raise SystemExit('PRM 只按非量化路径构建样本')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer, revision=config.tokenizer_revision,
                                              trust_remote_code=False)
    tokenizer_identity = {'path': config.tokenizer, 'revision': config.tokenizer_revision,
                          'template_sha256': hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()}

    episodes, sources = load_episodes(args.rollout)
    fit, dev, summary = split_by_task(episodes, args.fit_fraction, args.seed)
    fit_samples, fit_rejected = build_samples(fit, tokenizer, config.max_length)
    dev_samples, dev_rejected = build_samples(dev, tokenizer, config.max_length)
    rejections = {'fit': fit_rejected, 'dev': dev_rejected}
    manifest = write_dataset(args.output, config.fingerprint(), sources, summary,
                             fit_samples, dev_samples, rejections, tokenizer_identity)
    print(json.dumps({'fit': manifest['files']['fit']['samples'],
                      'dev': manifest['files']['dev']['samples'],
                      'rejections': rejections}, ensure_ascii=False))


if __name__ == '__main__':
    main()
