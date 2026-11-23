"""Rollout JSONL → PRM 样本，并在任务/指令层面隔离 fit 与 dev。

只做数据转换与切分，不加载模型；tokenizer 由调用方传入。
强制要求输入来自冻结 train 池，val/test 一律拒绝。
"""
import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

from ..model import SYSTEM
from .tokenize import TrajectoryTooLong, tokenize_trajectory

SCHEMA_VERSION = 'prm-data-v1'
LABEL_KEYS = ('done', 'final_reward', 'success', 'termination')
ALLOWED_TERMINATIONS = ('purchase', 'max_steps')
SUCCESS_EPS = 1e-6


class TechnicalInterruption(ValueError):
    """技术中断：单列计数，不能当成普通失败或标签 0 写进训练集。"""


class SourceNotAllowed(ValueError):
    """输入轨迹属于冻结 val/test，不能进入 PRM 训练集。"""


def normalize_instruction(text):
    if not isinstance(text, str):
        raise ValueError('instruction 必须是字符串')
    return ' '.join(text.lower().split())


def raw_reward(row):
    """先校验原始分数字段，再决定是否归零；布尔、NaN、Inf 与非数值一律报错。"""
    if not isinstance(row, dict):
        raise ValueError('episode 必须是对象')
    if 'final_reward' not in row:
        raise ValueError('episode 缺少 final_reward，不能默认成 0')
    value = row['final_reward']
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('final_reward 必须是数值，不能是布尔或其他类型')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('final_reward 必须有限，NaN/Inf 不能被当成 0')
    if not (0.0 <= value <= 1.0):
        raise ValueError('final_reward 必须位于 [0,1]')
    return value


def terminal_reward(row):
    """任务级奖励：只有完成购买才取记录分，其余为 0（页面匹配分不是任务奖励）。"""
    value = raw_reward(row)
    done = row.get('done')
    if type(done) is not bool:
        raise ValueError('episode.done 必须是布尔值，不能把缺失当成失败')
    termination = row.get('termination')
    if termination not in ALLOWED_TERMINATIONS:
        raise TechnicalInterruption('termination=%r 属于技术中断，需单列' % (termination,))
    if done != (termination == 'purchase'):
        raise ValueError('done 与 termination 不一致: %r/%r' % (done, termination))
    success = row.get('success')
    if type(success) is not bool:
        raise ValueError('episode.success 必须是布尔值')
    if success != (done and value >= 1.0 - SUCCESS_EPS):
        raise ValueError('success 与 purchase 分数不一致')
    return value if done else 0.0


def episode_messages(row, require_labels=True):
    """把一条 rollout 还原成 system + (user, assistant)* 的完整对话。

    require_labels=False 用于推理：不要求 done/final_reward/success/termination。
    """
    if not isinstance(row, dict):
        raise ValueError('episode 必须是对象')
    if require_labels:
        missing = [k for k in LABEL_KEYS if k not in row]
        if missing:
            raise ValueError('episode 缺少字段: %s' % missing)
        terminal_reward(row)
    instruction = row.get('instruction')
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('instruction 不能为空')
    steps = row.get('steps')
    if not isinstance(steps, list) or not steps:
        raise ValueError('episode 至少需要一个 step')
    messages = [{'role': 'system',
                 'content': SYSTEM + '\nShopping instruction: ' + instruction.strip()}]
    for step in steps:
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


def load_split(split_dir):
    """读取冻结划分；返回 task id 集合与指令哈希集合。"""
    split_dir = Path(split_dir)
    result = {}
    for name in ('train', 'val', 'test'):
        path = split_dir / (name + '_ids.json')
        if not path.exists():
            raise ValueError('缺少划分文件: %s' % path)
        ids = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(ids, list) or not ids:
            raise ValueError('%s 必须是非空列表' % path.name)
        result[name] = {int(v) for v in ids}
    if result['train'] & result['val'] or result['train'] & result['test']:
        raise ValueError('冻结划分之间存在任务交叉')
    return result


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


def gate_train_pool(episodes, splits, label_instructions):
    """强制检查输入属于冻结 train 池，且指令不与 val/test 重合。"""
    train = splits['train']
    forbidden = splits['val'] | splits['test']
    outside = sorted({int(r['task_id']) for r in episodes} - train)
    if outside:
        raise SourceNotAllowed(
            '有 %d 个任务不在冻结 train 池内（例如 %s），val/test 轨迹不能训练 PRM'
            % (len(outside), outside[:5]))
    leaked = [r for r in episodes
              if normalize_instruction(r['instruction']) in label_instructions]
    if leaked:
        raise SourceNotAllowed('有 %d 条轨迹的指令与 val/test 重合，需移除' % len(leaked))
    return {'train_tasks': len(train), 'forbidden_tasks': len(forbidden),
            'instruction_overlap_with_eval': 0}


def split_by_task(episodes, fit_fraction=0.8, seed=42):
    """按任务与归一化指令双重隔离切分，返回 (fit, dev) 与切分摘要。"""
    if not (0.0 < fit_fraction < 1.0):
        raise ValueError('fit_fraction 必须在 (0,1)')
    tasks = {}
    for row in episodes:
        tasks.setdefault(int(row['task_id']), []).append(row)
    ordered = sorted(tasks)
    random.Random(seed).shuffle(ordered)
    target = max(1, int(round(len(ordered) * fit_fraction)))
    if target >= len(ordered):
        raise ValueError('dev 为空，请增加轨迹或降低 fit_fraction')
    fit_tasks = set(ordered[:target])
    fit = [r for r in episodes if int(r['task_id']) in fit_tasks]
    dev = [r for r in episodes if int(r['task_id']) not in fit_tasks]
    overlap = ({normalize_instruction(r['instruction']) for r in fit}
               & {normalize_instruction(r['instruction']) for r in dev})
    if overlap:
        raise ValueError('fit/dev 指令重叠 %d 条，需重新切分' % len(overlap))
    return fit, dev, {'seed': seed, 'fit_fraction': fit_fraction,
                      'fit_tasks': len(fit_tasks), 'dev_tasks': len(ordered) - len(fit_tasks),
                      'fit_episodes': len(fit), 'dev_episodes': len(dev),
                      'instruction_overlap': 0}


def reward_distribution(episodes):
    values = [terminal_reward(r) for r in episodes]
    buckets = Counter()
    for v in values:
        buckets['zero' if v <= 0.0 else ('full' if v >= 1.0 else 'partial')] += 1
    return {'count': len(values), 'mean': sum(values) / len(values) if values else 0.0,
            'buckets': dict(buckets)}


def build_samples(episodes, tokenizer, max_length):
    """把 episode 编码成 PRM 样本；超长/技术中断/空动作显式计数，绝不截断。"""
    samples, rejected = [], Counter()
    for row in episodes:
        try:
            messages = episode_messages(row)
            encoded = tokenize_trajectory(tokenizer, messages, max_length)
        except TrajectoryTooLong:
            rejected['too_long'] += 1
            continue
        except TechnicalInterruption:
            rejected['technical_interruption'] += 1
            continue
        except ValueError as exc:
            reason = str(exc)
            if '空的 assistant' in reason:
                rejected['empty_action'] += 1
            elif 'template token' in reason.lower():
                rejected['reserved_token'] += 1
            else:
                rejected['invalid_label_or_schema'] += 1
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
            'source': row.get('_source', {}),
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
                         'turns': sum(r['num_turns'] for r in rows),
                         'label_mean': (sum(r['terminal_reward'] for r in rows) / len(rows))
                         if rows else 0.0}
    manifest = {
        'schema_version': SCHEMA_VERSION,
        'config_fingerprint': config_fingerprint,
        'tokenizer_identity': tokenizer_identity,
        'sources': sources,
        'split': split_summary,
        'files': payload,
        'rejections': rejections,
        'score_position': 'assistant_im_end',
        'truncation': 'forbidden: overlong trajectories are rejected and counted',
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                                       encoding='utf-8')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description='Build PRM fit/dev data from rollout JSONL')
    parser.add_argument('--rollout', action='append', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--split-dir', required=True,
                        help='frozen data/processed directory with train/val/test ids')
    parser.add_argument('--output', required=True)
    parser.add_argument('--fit-fraction', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)

    from .config import load_config
    config = load_config(args.config)
    if config.load_in_4bit:
        raise SystemExit('PRM 只按非量化路径构建样本')

    splits = load_split(args.split_dir)
    episodes, sources = load_episodes(args.rollout)
    eval_instructions = set()
    for name in ('val', 'test'):
        path = Path(args.split_dir) / (name + '.json')
        if path.exists():
            for row in json.loads(path.read_text(encoding='utf-8')):
                if isinstance(row, dict) and 'instruction' in row:
                    eval_instructions.add(normalize_instruction(row['instruction']))
    gate = gate_train_pool(episodes, splits, eval_instructions)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer, revision=config.tokenizer_revision,
                                              trust_remote_code=False)
    tokenizer_identity = {'path': config.tokenizer, 'revision': config.tokenizer_revision,
                          'template_sha256': hashlib.sha256(
                              tokenizer.chat_template.encode()).hexdigest()}

    fit, dev, summary = split_by_task(episodes, args.fit_fraction, args.seed)
    fit_samples, fit_rejected = build_samples(fit, tokenizer, config.max_length)
    dev_samples, dev_rejected = build_samples(dev, tokenizer, config.max_length)
    rejections = {'fit': fit_rejected, 'dev': dev_rejected}
    summary['train_pool_gate'] = gate
    manifest = write_dataset(args.output, config.fingerprint(), sources, summary,
                             fit_samples, dev_samples, rejections, tokenizer_identity)
    print(json.dumps({'fit': manifest['files']['fit']['samples'],
                      'dev': manifest['files']['dev']['samples'],
                      'rejections': rejections}, ensure_ascii=False))


if __name__ == '__main__':
    main()
