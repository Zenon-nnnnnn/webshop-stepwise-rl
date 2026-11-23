"""加载与运行 PRM：独立进程、只依赖已保存的 adapter + head + manifest。"""
import argparse
import hashlib
import json
from pathlib import Path

from .data import episode_messages, terminal_reward
from .tokenize import collate_trajectories, tokenize_trajectory

REQUIRED_IDENTITY = ('base_revision', 'base_sha256', 'tokenizer_sha256',
                     'initialization', 'config_fingerprint',
                     'template_sha256', 'source_commit')


def build_backbone(config, device='cuda'):
    """FP32 非量化；sft_b1 初始化时把 B1 合并进底模，再挂 PRM 专用 adapter。"""
    import torch
    from transformers import AutoModelForCausalLM

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')

    base = AutoModelForCausalLM.from_pretrained(
        config.model, revision=config.revision, torch_dtype=torch.float32,
        device_map={'': 0} if device == 'cuda' else None, trust_remote_code=False,
        attn_implementation='sdpa')
    if config.initialization == 'sft_b1':
        from peft import PeftModel
        base = PeftModel.from_pretrained(base, config.sft_adapter).merge_and_unload()
    return base


def identity_from(config, template_sha256, source_commit):
    """身份必须绑定实际初始化文件，不能只写 initialization 字符串。"""
    base_sha = hashlib.sha256(Path(config.model, 'model.safetensors').read_bytes()).hexdigest()
    identity = {
        'base_revision': config.revision,
        'base_sha256': base_sha,
        'tokenizer_sha256': config.tokenizer_files_sha256.get('tokenizer.json', ''),
        'initialization': config.initialization,
        'init_files_sha256': dict(sorted(config.sft_adapter_files_sha256.items())),
        'config_fingerprint': config.fingerprint(),
        'template_sha256': template_sha256,
        'source_commit': source_commit,
    }
    missing = [k for k in REQUIRED_IDENTITY if not identity.get(k)]
    if missing:
        raise ValueError('身份字段缺失: %s' % missing)
    if config.initialization == 'sft_b1':
        if set(identity['init_files_sha256']) != {'adapter_model.safetensors',
                                                  'adapter_config.json'}:
            raise ValueError('sft_b1 身份必须包含原 B1 的 adapter 两个文件 SHA')
    elif identity['init_files_sha256']:
        raise ValueError('base 初始化不应携带 SFT adapter 身份')
    return identity


def load_prm(config, checkpoint, expected_identity, device='cuda'):
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer
    from .model_core import NeuralPRM, restore_head, verify_prm

    manifest = verify_prm(checkpoint, expected_identity)
    base = build_backbone(config, device=device)
    backbone = PeftModel.from_pretrained(base, str(Path(checkpoint) / 'adapter'), is_trainable=False)
    model = NeuralPRM(backbone).to(base.dtype)
    restore_head(model, checkpoint, expected_identity)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer, revision=config.tokenizer_revision,
                                              trust_remote_code=False)
    return model, tokenizer, manifest


def score_episode(model, tokenizer, row, max_length, device='cuda'):
    """对一条轨迹逐回合输出 p_t 与总和；推理路径不要求也不读取终局标签。"""
    import torch

    messages = episode_messages(row, require_labels=False)
    encoded = tokenize_trajectory(tokenizer, messages, max_length)
    batch = collate_trajectories([encoded], tokenizer.pad_token_id)
    ids = torch.tensor(batch['input_ids'], dtype=torch.long, device=device)
    attention = torch.tensor(batch['attention_mask'], dtype=torch.long, device=device)
    ends = torch.tensor(batch['turn_end_indices'], dtype=torch.long, device=device)
    turns = torch.tensor(batch['turn_mask'], dtype=torch.bool, device=device)
    with torch.inference_mode():
        out = model(input_ids=ids, attention_mask=attention, turn_end_indices=ends, turn_mask=turns)
    valid = turns[0]
    scores = out['step_scores'][0][valid].float().cpu().tolist()
    result = {'task_id': int(row['task_id']), 'turn_scores': scores,
              'total': float(out['totals'][0]), 'turns': len(scores),
              'tokens': len(encoded['input_ids']),
              'template_sha256': encoded['template_sha256'],
              'score_position': encoded['score_position']}
    if 'done' in row:
        try:
            result['terminal_reward_for_diagnosis'] = terminal_reward(row)
        except ValueError:
            result['terminal_reward_for_diagnosis'] = None
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description='Score one rollout trajectory with a saved PRM')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--rollout', required=True)
    parser.add_argument('--task-id', type=int, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--output')
    args = parser.parse_args(argv)

    from .config import load_config, verify_input_files
    config = load_config(args.config)
    verify_input_files(config)

    row = None
    with Path(args.rollout).open(encoding='utf-8') as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            candidate = json.loads(line)
            if int(candidate['task_id']) == args.task_id:
                row = candidate
                break
    if row is None:
        raise SystemExit('轨迹中找不到 task_id=%d' % args.task_id)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer, revision=config.tokenizer_revision,
                                              trust_remote_code=False)
    template_sha256 = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
    identity = identity_from(config, template_sha256, args.source_commit)

    model, tokenizer, manifest = load_prm(config, args.checkpoint, identity, device=args.device)
    result = score_episode(model, tokenizer, row, config.max_length, device=args.device)
    result['checkpoint_identity'] = identity
    result['vocab_size'] = manifest['vocab_size']
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        Path(args.output).write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
