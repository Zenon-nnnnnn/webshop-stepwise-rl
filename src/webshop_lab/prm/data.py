"""Rollout JSONL → PRM 样本，并在任务/指令层面隔离 fit 与 dev。

只做数据转换与切分，不加载模型；tokenizer 由调用方传入。
强制要求输入来自冻结 train 池，并用已锁定 goal-map 校验指令身份；val/test 一律拒绝。
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
    """输入轨迹不属于冻结 train 池，或指令身份不匹配，不能进入 PRM 训练集。"""


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


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_split(split_dir):
    """读取冻结划分 ID；返回 id 集合与文件 SHA。"""
    split_dir = Path(split_dir)
    result, digests = {}, {}
    for name in ('train', 'val', 'test'):
        path = split_dir / (name + '_ids.json')
        if not path.exists():
            raise ValueError('缺少划分文件: %s' % path)
        ids = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(ids, list) or not ids:
            raise ValueError('%s 必须是非空列表' % path.name)
        result[name] = {int(v) for v in ids}
        digests[name + '_ids'] = _sha256(path)
    if result['train'] & result['val'] or result['train'] & result['test']:
        raise ValueError('冻结划分之间存在任务交叉')
    return result, digests


def load_goal_map(path):
    """读取已锁定 goal-map：task_id → 官方指令。缺失或不可解析直接报错。"""
    path = Path(path)
    if not path.exists():
        raise ValueError('缺少 goal-map: %s' % path)
    rows = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        raise ValueError('goal-map 结构不符')
    mapping = {}
    for row in rows:
        if not isinstance(row, dict) or 'task_id' not in row or 'instruction' not in row:
            raise ValueError('goal-map 条目缺少 task_id/instruction')
        task = int(row['task_id'])
        if task in mapping:
            raise ValueError('goal-map 中 task_id 重复: %d' % task)
        mapping[task] = row['instruction']
    return mapping, _sha256(path)


def gate_train_pool(episodes, splits, goal_map):
    """强制检查：任务属于冻结 train、指令与 goal-map 一致、且不与 val/test 指令重合。"""
    train = splits['train']
    forbidden_ids = splits['val'] | splits['test']
    outside = sorted({int(r['task_id']) for r in episodes} - train)
    if outside:
        raise SourceNotAllowed(
            '有 %d 个任务不在冻结 train 池内（例如 %s），val/test 轨迹不能训练 PRM'
            % (len(outside), outside[:5]))
    missing = sorted({int(r['task_id']) for r in episodes} - set(goal_map))
    if missing:
        raise SourceNotAllowed('goal-map 缺少任务指令: %s' % missing[:5])
    mismatched, leaked = [], []
    forbidden_instructions = {normalize_instruction(goal_map[t])
                              for t in forbidden_ids if t in goal_map}
    for row in episodes:
        task = int(row['task_id'])
        actual = normalize_instruction(row['instruction'])
        if actual != normalize_instruction(goal_map[task]):
            mismatched.append(task)
        if actual in forbidden_instructions:
            leaked.append(task)
    if mismatched:
        raise SourceNotAllowed('任务指令与 goal-map 不一致: %s' % sorted(set(mismatched))[:5])
    if leaked:
        raise SourceNotAllowed('指令与 val/test 重合: %s' % sorted(set(leaked))[:5])
    return {'train_tasks': len(train), 'forbidden_tasks': len(forbidden_ids),
            'goal_map_entries': len(goal_map), 'forbidden_instructions': len(forbidden_instructions),
            'instruction_overlap_with_eval': 0}


def load_episodes(paths):
    """读取一个或多个 rollout JSONL，按 (task_id, repeat) 去重并保留来源。"""
    episodes, seen, sources = [], set(), []
    for path in paths:
        p = Path(path)
        sources.append({'path': str(p), 'sha256': _sha256(p)})
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
    """编码样本；超长/技术中断/空动作显式记录被剔除的 task 与指令哈希，绝不截断。"""
    samples, rejected = [], Counter()
    dropped = []
    for row in episodes:
        task = int(row['task_id'])
        sha = hashlib.sha256(normalize_instruction(row['instruction']).encode()).hexdigest()
        reason = None
        try:
            messages = episode_messages(row)
            encoded = tokenize_trajectory(tokenizer, messages, max_length)
        except TrajectoryTooLong:
            reason = 'too_long'
        except TechnicalInterruption:
            reason = 'technical_interruption'
        except ValueError as exc:
            text = str(exc)
            if '空的 assistant' in text:
                reason = 'empty_action'
            elif 'template token' in text.lower():
                reason = 'reserved_token'
            else:
                reason = 'invalid_label_or_schema'
        if reason:
            rejected[reason] += 1
            dropped.append({'task_id': task, 'instruction_sha256': sha, 'reason': reason})
            continue
        samples.append({
            'task_id': task,
            'repeat': int(row.get('repeat', 0)),
            'instruction_sha256': sha,
            'terminal_reward': terminal_reward(row),
            'num_turns': len(encoded['turn_end_indices']),
            'num_tokens': len(encoded['input_ids']),
            'template_sha256': encoded['template_sha256'],
            'source': row.get('_source', {}),
            'encoded': encoded,
        })
    return samples, dict(rejected), dropped


def write_dataset(out_dir, config_fingerprint, sources, split_summary, fit_samples, dev_samples,
                  rejections, dropped, tokenizer_identity, goal_map_sha256, split_sha256):
    if not fit_samples or not dev_samples:
        raise ValueError('过滤后 fit 或 dev 为空，应停止正式训练流程而不是继续')
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    payload = {}
    for name, rows in (('fit', fit_samples), ('dev', dev_samples)):
        path = out / (name + '.jsonl')
        with path.open('w', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        payload[name] = {'path': str(path), 'sha256': _sha256(path), 'samples': len(rows),
                         'tasks': len({r['task_id'] for r in rows}),
                         'turns': sum(r['num_turns'] for r in rows),
                         'label_mean': sum(r['terminal_reward'] for r in rows) / len(rows),
                         'label_buckets': dict(Counter(
                             'zero' if r['terminal_reward'] <= 0 else
                             ('full' if r['terminal_reward'] >= 1 else 'partial') for r in rows))}
    manifest = {
        'schema_version': SCHEMA_VERSION,
        'config_fingerprint': config_fingerprint,
        'tokenizer_identity': tokenizer_identity,
        'goal_map_sha256': goal_map_sha256,
        'split_sha256': split_sha256,
        'sources': sources,
        'split': split_summary,
        'files': payload,
        'rejections': rejections,
        'dropped': dropped,
        'dropped_tasks': sorted({d['task_id'] for d in dropped}),
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
    parser.add_argument('--split-dir', required=True)
    parser.add_argument('--goal-map', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--fit-fraction', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)

    from .config import load_config
    config = load_config(args.config)
    if config.load_in_4bit:
        raise SystemExit('PRM 只按非量化路径构建样本')

    splits, split_sha = load_split(args.split_dir)
    goal_map, goal_sha = load_goal_map(args.goal_map)
    episodes, sources = load_episodes(args.rollout)
    gate = gate_train_pool(episodes, splits, goal_map)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer, revision=config.tokenizer_revision,
                                              trust_remote_code=False)
    tokenizer_identity = {'path': config.tokenizer, 'revision': config.tokenizer_revision,
                          'template_sha256': hashlib.sha256(
                              tokenizer.chat_template.encode()).hexdigest()}

    fit, dev, summary = split_by_task(episodes, args.fit_fraction, args.seed)
    fit_samples, fit_rejected, fit_dropped = build_samples(fit, tokenizer, config.max_length)
    dev_samples, dev_rejected, dev_dropped = build_samples(dev, tokenizer, config.max_length)
    rejections = {'fit': fit_rejected, 'dev': dev_rejected}
    summary['train_pool_gate'] = gate
    manifest = write_dataset(args.output, config.fingerprint(), sources, summary,
                             fit_samples, dev_samples, rejections, fit_dropped + dev_dropped,
                             tokenizer_identity, goal_sha, split_sha)
    print(json.dumps({'fit': manifest['files']['fit']['samples'],
                      'dev': manifest['files']['dev']['samples'],
                      'rejections': rejections,
                      'fit_label_buckets': manifest['files']['fit']['label_buckets']},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
