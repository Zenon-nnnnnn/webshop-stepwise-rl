"""GPU-only two-update SFT smoke and separate-process adapter reload check.

Run from this folder. No WebShop environment is required. Never register this as
a shopping benchmark or use its adapter for the formal B0/B1 comparison.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from prepare import HERE, sha, write

WEIGHT_SHA = 'dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee'
TOKENIZER_SHAS = {
    'config.json': '98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670',
    'tokenizer.json': 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539',
    'tokenizer_config.json': '5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583'}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['run', 'reload'])
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--model-path', type=Path, required=True, help='Verified local HF snapshot directory')
    args = p.parse_args()
    sys.path.insert(0, str(args.repo.resolve()/'src'))
    import torch
    from transformers import Trainer, TrainingArguments, TrainerCallback, set_seed
    from peft import LoraConfig, get_peft_model
    from webshop_lab.model import load, training_messages, fit_history
    assert torch.cuda.is_available(), 'Execute on the CUDA host, not the preparation computer'
    cfg = json.loads((HERE/'smoke-config.json').read_text())
    manifest = json.loads((HERE/'data/subset-manifest.json').read_text())
    for filename, expected in manifest['source_files_sha256'].items():
        assert sha(args.repo.resolve()/filename) == expected, f'Implementation/config changed: {filename}'
    for name in ['train', 'val']:
        assert sha(HERE/cfg[f'{name}_file']) == manifest['datasets'][name]['subset_sha256']
    model_path = args.model_path.resolve()
    assert sha(model_path/'model.safetensors') == WEIGHT_SHA, 'Base model weight SHA mismatch'
    for filename, expected in TOKENIZER_SHAS.items():
        assert sha(model_path/filename) == expected, f'Tokenizer mismatch: {filename}'
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
                assert ids[:len(prefix)] == prefix and len(prefix) < len(ids)
                text = tok.apply_chat_template(kept, tokenize=False, add_generation_prompt=False)
                prompt = tok.apply_chat_template(kept[:-1], tokenize=False, add_generation_prompt=True)
                assert text.startswith(prompt)
                expected = tok.encode(text[len(prompt):], add_special_tokens=False)
                labels = [-100]*len(prefix)+ids[len(prefix):]
                assert labels[len(prefix):] == expected
                assert all(v == -100 for v in labels[:len(prefix)])
                assert kept[0] == history[0] and kept[-2:] == history[-2:]
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
        assert evidence['passed'] and sha(HERE/'smoke-config.json') == evidence['config_sha256']
        runtime = {**cfg, 'model': str(model_path), 'adapter': str(out/'adapter')}
        model, tok, dtype = load(runtime, training=False)
        assert lora_hash(model) == evidence['lora_state_sha256'], 'Reloaded LoRA parameters differ'
        val, _ = examples(HERE/cfg['val_file'], tok)
        model.eval()
        probe = collate([val[0]])
        with torch.no_grad():
            output = model(**{k: v.to('cuda') for k,v in probe.items()})
        loss = float(output.loss)
        assert math.isfinite(loss)
        # Greedy generation uses a synthetic instruction, not an environment task.
        inputs = tok('Reply with one short shopping action.', return_tensors='pt').to('cuda')
        with torch.no_grad():
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=8,
                                       pad_token_id=tok.pad_token_id)
        assert generated.shape[1] > inputs['input_ids'].shape[1]
        torch.cuda.synchronize()
        write(out/'reload-evidence.json', {'passed': True, 'finite_probe_loss': loss,
            'lora_state_sha256': lora_hash(model), 'matches_saved_training_state': True,
            'adapter_files_sha256': {f.name: sha(f) for f in (out/'adapter').iterdir() if f.is_file()},
            'elapsed_seconds': time.perf_counter()-started,
            'peak_allocated_gb': torch.cuda.max_memory_allocated()/1024**3,
            'warning': 'Fresh-process load, forward and generation only; not shopping performance'})
        print('Fresh-process adapter reload passed')
        return

    assert not out.exists(), 'Use a fresh output directory; never overwrite a previous run'
    out.mkdir(parents=True)
    runtime = {**cfg, 'model': str(model_path)}
    model, tok, dtype = load(runtime, training=True)
    model = get_peft_model(model, LoraConfig(r=cfg['lora_r'], lora_alpha=cfg['lora_alpha'],
        target_modules=cfg['target_modules'], lora_dropout=.05, bias='none', task_type='CAUSAL_LM'))
    model.config.use_cache = False
    model.enable_input_require_grads()
    trainable = {name: param for name,param in model.named_parameters() if param.requires_grad}
    assert trainable and all('lora_' in name for name in trainable)
    before = {name: param.detach().float().cpu().clone() for name,param in trainable.items()}
    train, train_stats = examples(HERE/cfg['train_file'], tok)
    val, val_stats = examples(HERE/cfg['val_file'], tok)
    assert len(train) >= 16 and val
    pair = next(([train[0], item] for item in train[1:] if len(item['input_ids']) != len(train[0]['input_ids'])), None)
    assert pair is not None
    padded = collate(pair)
    assert (padded['attention_mask'] == 0).any()
    assert torch.all(padded['labels'][padded['attention_mask'] == 0] == -100)
    logs = []
    class Capture(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            entry = {'step': state.global_step, **(logs or {})}
            for key in ['loss', 'eval_loss', 'grad_norm']:
                if key in entry:
                    assert math.isfinite(float(entry[key])), f'Nonfinite {key}'
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
    assert trainer.state.global_step == 2
    loss_entries = [entry for entry in logs if 'loss' in entry]
    assert {entry['step'] for entry in loss_entries} == {1,2}
    changes = {name: float((param.detach().float().cpu()-before[name]).abs().max()) for name,param in trainable.items()}
    assert all(math.isfinite(v) for v in changes.values()) and any(v>0 for v in changes.values())
    metrics = trainer.evaluate()
    assert math.isfinite(float(metrics['eval_loss']))
    trainer.save_model(str(out/'adapter'))
    tok.save_pretrained(out/'adapter')
    trainer.state.save_to_json(str(out/'trainer-state.json'))
    write(out/'config.json', cfg)
    write(out/'run-evidence.json', {'passed': True, 'global_step': trainer.state.global_step,
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
