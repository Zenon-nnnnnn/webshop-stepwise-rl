"""GPU-only two-update SFT smoke and separate-process adapter reload check.

Run from this folder. No WebShop environment is required. Never register this as
a shopping benchmark or use its adapter for the formal B0/B1 comparison.
"""
import argparse
import hashlib
import json
import math
import importlib.metadata
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from prepare import HERE, sha, write, require

WEIGHT_SHA = 'dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee'
TOKENIZER_SHAS = {
    'config.json': '98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670',
    'tokenizer.json': 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539',
    'tokenizer_config.json': '5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583'}

def validate_sources(repo, manifest):
    for filename, expected in manifest['source_files_sha256'].items():
        require(sha(repo/filename) == expected, f'Implementation/config changed: {filename}')

def runtime_provenance(repo, torch, declared_commit=None):
    """Capture observations separately from an API-snapshot operator's declared commit."""
    def git(*args):
        try:
            result = subprocess.run(['git', '-C', str(repo), *args], capture_output=True,
                                    text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result if result.returncode in (0, 1) else None
    head_result = git('rev-parse', 'HEAD')
    head = head_result.stdout.strip() if head_result and head_result.returncode == 0 else None
    diff = git('diff', '--quiet', 'HEAD', '--')
    installed = {dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions()
                 if dist.metadata.get('Name')}
    versions = {}
    for name in ['torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes', 'tokenizers']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    source_names = ['src/webshop_lab/model.py', 'src/webshop_lab/common.py',
                    'src/webshop_lab/data.py', 'src/webshop_lab/cli.py', 'configs/sft-qwen15b.json']
    handoff_names = ['run_smoke.py', 'prepare.py', 'check_subset.py', 'smoke-config.json']
    return {'python_version': platform.python_version(), 'python_optimization_level': sys.flags.optimize,
            'platform': platform.platform(),
            'package_versions': versions, 'installed_packages': dict(sorted(installed.items())),
            'torch_cuda_build': torch.version.cuda, 'cudnn_version': torch.backends.cudnn.version(),
            'source': {'repo_git_head': head,
                'repo_tracked_dirty': None if diff is None else diff.returncode != 0,
                'declared_commit': declared_commit,
                'declared_matches_git_head': None if not head or not declared_commit else head == declared_commit,
                'source_files_sha256': {name: sha(repo/name) for name in source_names},
                'handoff_files_sha256': {name: sha(HERE/name) for name in handoff_names},
                'note': 'Git HEAD and an operator-declared API snapshot are distinct. File hashes identify executed bytes; a declaration alone does not verify a commit.'}}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['run', 'reload'])
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--model-path', type=Path, required=True, help='Verified local HF snapshot directory')
    p.add_argument('--source-commit', help='Optional full commit SHA for an API-downloaded snapshot; recorded as a declaration, not proof')
    args = p.parse_args()
    require(args.source_commit is None or re.fullmatch(r'[0-9a-f]{40}', args.source_commit),
            '--source-commit must be a full lowercase Git commit SHA')
    sys.path.insert(0, str(args.repo.resolve()/'src'))
    import torch
    from transformers import Trainer, TrainingArguments, TrainerCallback, set_seed
    from peft import LoraConfig, get_peft_model
    from webshop_lab.model import load, training_messages, fit_history
    require(torch.cuda.is_available(), 'Execute on the CUDA host, not the preparation computer')
    cfg = json.loads((HERE/'smoke-config.json').read_text())
    manifest = json.loads((HERE/'data/subset-manifest.json').read_text())
    validate_sources(args.repo.resolve(), manifest)
    for name in ['train', 'val']:
        require(sha(HERE/cfg[f'{name}_file']) == manifest['datasets'][name]['subset_sha256'], 'run_smoke.py: validation failed at original line 38')
    model_path = args.model_path.resolve()
    require(sha(model_path/'model.safetensors') == WEIGHT_SHA, 'Base model weight SHA mismatch')
    for filename, expected in TOKENIZER_SHAS.items():
        require(sha(model_path/filename) == expected, f'Tokenizer mismatch: {filename}')
    provenance = runtime_provenance(args.repo.resolve(), torch, args.source_commit)
    out = HERE/cfg['output_dir']
    set_seed(cfg['seed'])
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    def lora_hash(model):
        h = hashlib.sha256()
        for name, param in sorted(model.named_parameters()):
            if 'lora_' in name:
                h.update(name.encode())
                h.update(param.detach().float().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def examples(path, tok):
        items, trimmed, skipped = [], 0, 0
        for row in json.loads(path.read_text()):
            messages = training_messages(row)
            for i, msg in enumerate(messages):
                if msg['role'] != 'assistant':
                    continue
                history = messages[:i+1]
                try:
                    kept, ids = fit_history(tok, history, cfg['max_length'])
                except ValueError:
                    skipped += 1
                    continue
                prefix = tok.apply_chat_template(kept[:-1], tokenize=True, add_generation_prompt=True)
                require(ids[:len(prefix)] == prefix and len(prefix) < len(ids), 'run_smoke.py: validation failed at original line 70')
                text = tok.apply_chat_template(kept, tokenize=False, add_generation_prompt=False)
                prompt = tok.apply_chat_template(kept[:-1], tokenize=False, add_generation_prompt=True)
                require(text.startswith(prompt), 'run_smoke.py: validation failed at original line 73')
                expected = tok.encode(text[len(prompt):], add_special_tokens=False)
                labels = [-100]*len(prefix)+ids[len(prefix):]
                require(labels[len(prefix):] == expected, 'run_smoke.py: validation failed at original line 76')
                require(all(v == -100 for v in labels[:len(prefix)]), 'run_smoke.py: validation failed at original line 77')
                require(kept[0] == history[0] and kept[-2:] == history[-2:], 'run_smoke.py: validation failed at original line 78')
                trimmed += len(kept) < len(history)
                items.append({'input_ids': ids, 'attention_mask': [1]*len(ids), 'labels': labels})
        return items, {'examples': len(items), 'skipped': skipped, 'history_trimmed': trimmed,
                       'label_checks_passed': True}

    def collate(items):
        width = max(len(x['input_ids']) for x in items)
        return {key: torch.tensor([item[key]+[pad]*(width-len(item[key])) for item in items])
                for key, pad in [('input_ids', tok.pad_token_id), ('attention_mask', 0), ('labels', -100)]}

    if args.mode == 'reload':
        evidence = json.loads((out/'run-evidence.json').read_text())
        require(evidence['passed'] and sha(HERE/'smoke-config.json') == evidence['config_sha256'], 'run_smoke.py: validation failed at original line 91')
        runtime = {**cfg, 'model': str(model_path), 'adapter': str(out/'adapter')}
        model, tok, dtype = load(runtime, training=False)
        require(lora_hash(model) == evidence['lora_state_sha256'], 'Reloaded LoRA parameters differ')
        val, _ = examples(HERE/cfg['val_file'], tok)
        model.eval()
        probe = collate([val[0]])
        with torch.no_grad():
            output = model(**{k: v.to('cuda') for k,v in probe.items()})
        loss = float(output.loss)
        require(math.isfinite(loss), 'run_smoke.py: validation failed at original line 101')
        # Greedy generation uses a synthetic instruction, not an environment task.
        inputs = tok('Reply with one short shopping action.', return_tensors='pt').to('cuda')
        with torch.no_grad():
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=8,
                                       pad_token_id=tok.pad_token_id)
        require(generated.shape[1] > inputs['input_ids'].shape[1], 'run_smoke.py: validation failed at original line 107')
        torch.cuda.synchronize()
        write(out/'reload-evidence.json', {'passed': True, 'finite_probe_loss': loss,
            'runtime_provenance': provenance,
            'lora_state_sha256': lora_hash(model), 'matches_saved_training_state': True,
            'adapter_files_sha256': {f.name: sha(f) for f in (out/'adapter').iterdir() if f.is_file()},
            'elapsed_seconds': time.perf_counter()-started,
            'peak_allocated_gb': torch.cuda.max_memory_allocated()/1024**3,
            'warning': 'Fresh-process load, forward and generation only; not shopping performance'})
        print('Fresh-process adapter reload passed')
        return

    require(not out.exists(), 'Use a fresh output directory; never overwrite a previous run')
    out.mkdir(parents=True)
    runtime = {**cfg, 'model': str(model_path)}
    model, tok, dtype = load(runtime, training=True)
    model = get_peft_model(model, LoraConfig(r=cfg['lora_r'], lora_alpha=cfg['lora_alpha'],
        target_modules=cfg['target_modules'], lora_dropout=.05, bias='none', task_type='CAUSAL_LM'))
    model.config.use_cache = False
    model.enable_input_require_grads()
    trainable = {name: param for name,param in model.named_parameters() if param.requires_grad}
    require(trainable and all('lora_' in name for name in trainable), 'run_smoke.py: validation failed at original line 127')
    before = {name: param.detach().float().cpu().clone() for name,param in trainable.items()}
    train, train_stats = examples(HERE/cfg['train_file'], tok)
    val, val_stats = examples(HERE/cfg['val_file'], tok)
    require(len(train) >= 16 and val, 'run_smoke.py: validation failed at original line 131')
    pair = next(([train[0], item] for item in train[1:] if len(item['input_ids']) != len(train[0]['input_ids'])), None)
    require(pair is not None, 'run_smoke.py: validation failed at original line 133')
    padded = collate(pair)
    require((padded['attention_mask'] == 0).any(), 'run_smoke.py: validation failed at original line 135')
    require(torch.all(padded['labels'][padded['attention_mask'] == 0] == -100), 'run_smoke.py: validation failed at original line 136')
    logs = []
    class Capture(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            entry = {'step': state.global_step, **(logs or {})}
            for key in ['loss', 'eval_loss', 'grad_norm']:
                if key in entry:
                    require(math.isfinite(float(entry[key])), f'Nonfinite {key}')
            with (out/'steps.jsonl').open('a') as f:
                f.write(json.dumps(entry)+'\n')
            log_entries.append(entry)
    log_entries = logs
    ta = TrainingArguments(output_dir=str(out), max_steps=2, num_train_epochs=1,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
        learning_rate=cfg['learning_rate'], warmup_ratio=.03, lr_scheduler_type='cosine',
        bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        eval_strategy='no', save_strategy='no', logging_steps=1, logging_first_step=True,
        logging_nan_inf_filter=False, report_to='none', seed=cfg['seed'])
    trainer = Trainer(model=model, args=ta, train_dataset=train, eval_dataset=val,
                      data_collator=collate, callbacks=[Capture()])
    result = trainer.train()
    torch.cuda.synchronize()
    training_seconds = time.perf_counter()-started
    require(trainer.state.global_step == 2, 'run_smoke.py: validation failed at original line 161')
    loss_entries = [entry for entry in logs if 'loss' in entry]
    require({entry['step'] for entry in loss_entries} == {1,2}, 'run_smoke.py: validation failed at original line 163')
    changes = {name: float((param.detach().float().cpu()-before[name]).abs().max()) for name,param in trainable.items()}
    require(all(math.isfinite(v) for v in changes.values()) and any(v>0 for v in changes.values()), 'run_smoke.py: validation failed at original line 165')
    metrics = trainer.evaluate()
    require(math.isfinite(float(metrics['eval_loss'])), 'run_smoke.py: validation failed at original line 167')
    trainer.save_model(str(out/'adapter'))
    tok.save_pretrained(out/'adapter')
    trainer.state.save_to_json(str(out/'trainer-state.json'))
    write(out/'config.json', cfg)
    write(out/'run-evidence.json', {'passed': True, 'global_step': trainer.state.global_step,
        'runtime_provenance': provenance,
        'purpose': cfg['purpose'], 'config_sha256': sha(HERE/'smoke-config.json'),
        'subset_manifest_sha256': sha(HERE/'data/subset-manifest.json'),
        'model_revision': cfg['model_revision'], 'base_weights_sha256': WEIGHT_SHA,
        'train': train_stats, 'val': val_stats, 'padding_check_passed': True,
        'trainable_parameters': sum(p.numel() for p in trainable.values()),
        'all_trainable_parameters_are_lora': True,
        'changed_lora_tensors': sum(v>0 for v in changes.values()),
        'max_lora_absolute_change': max(changes.values()), 'training_metrics': result.metrics,
        'lora_state_sha256': lora_hash(model),
        'validation_metrics': metrics, 'load_preprocess_and_train_seconds': training_seconds,
        'total_seconds': time.perf_counter()-started,
        'peak_allocated_gb': torch.cuda.max_memory_allocated()/1024**3,
        'peak_reserved_gb': torch.cuda.max_memory_reserved()/1024**3,
        'gpu': torch.cuda.get_device_name(0), 'torch_version': torch.__version__,
        'warning': 'Engineering-only smoke on preliminary split; not a benchmark or final model'})
    print('Two optimizer steps passed. Run reload in a NEW process next.')

if __name__ == '__main__':
    main()
