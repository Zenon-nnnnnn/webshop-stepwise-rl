"""Small-batch LoRA training. Saves adapters, tokenizer, config and run metadata."""
import json
import subprocess
import time
from pathlib import Path
from .common import ROOT, read_json, write_json, file_hash
from .model import load, model_source, training_messages, fit_history


def run(args):
    cfg = read_json(args.config)
    if cfg.get('configuration_status') == 'draft':
        raise ValueError('Draft SFT configuration: complete the acceptance gates in docs/SFT_FORMAL_PLAN.md and freeze a separate run config before execution.')
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments, set_seed
    set_seed(cfg.get('seed', 42))
    out = Path(cfg['output_dir'])
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Output already exists: {out}. Use a new run directory.')
    model, tok, dtype = load(cfg, training=True)
    source = model_source(cfg, model, tok)
    model = get_peft_model(model, LoraConfig(r=cfg.get('lora_r', 8), lora_alpha=cfg.get('lora_alpha', 16),
        target_modules=cfg.get('target_modules', ['q_proj', 'v_proj']), lora_dropout=.05,
        bias='none', task_type='CAUSAL_LM'))
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    class Examples(torch.utils.data.Dataset):
        def __init__(self, path):
            self.items = []
            self.skipped = 0
            for row in read_json(path):
                messages = training_messages(row)
                for i, msg in enumerate(messages):
                    if msg['role'] != 'assistant':
                        continue
                    try:
                        kept, ids = fit_history(tok, messages[:i + 1], cfg.get('max_length', 2048))
                    except ValueError:
                        self.skipped += 1
                        continue
                    prefix = tok.apply_chat_template(kept[:-1], tokenize=True, add_generation_prompt=True)
                    if ids[:len(prefix)] != prefix or len(prefix) >= len(ids):
                        raise ValueError('Chat-template prefix mismatch; inspect tokenizer before training')
                    labels = [-100] * len(prefix) + ids[len(prefix):]
                    self.items.append({'input_ids': ids, 'attention_mask': [1] * len(ids), 'labels': labels})
            if not self.items:
                raise ValueError('No trainable assistant examples after preprocessing')
        def __len__(self):
            return len(self.items)
        def __getitem__(self, index):
            return self.items[index]
    train = Examples(cfg['train_file'])
    val = Examples(cfg['val_file'])
    def collate(examples):
        size = max(len(x['input_ids']) for x in examples)
        return {k: torch.tensor([x[k] + [pad] * (size - len(x[k])) for x in examples])
                for k, pad in [('input_ids', tok.pad_token_id), ('attention_mask', 0), ('labels', -100)]}
    ta = TrainingArguments(output_dir=str(out), num_train_epochs=cfg.get('epochs', 1),
        max_steps=args.max_steps, per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg.get('gradient_accumulation_steps', 8),
        learning_rate=cfg.get('learning_rate', 2e-4), warmup_ratio=.03, lr_scheduler_type='cosine',
        bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        eval_strategy='epoch', save_strategy='epoch', save_total_limit=2,
        logging_steps=cfg.get('logging_steps', 5), report_to='none', seed=cfg.get('seed', 42))
    trainer = Trainer(model=model, args=ta, train_dataset=train, eval_dataset=val, data_collator=collate)
    start = time.monotonic()
    trainer.train()
    trainer.save_state()
    adapter = out / 'adapter'
    trainer.save_model(str(adapter))
    tok.save_pretrained(adapter)
    metrics = trainer.evaluate()
    write_json(out / 'config.json', cfg)
    write_json(out / 'run.json', {'config_sha256': file_hash(args.config), 'train_sha256': file_hash(cfg['train_file']),
        'model_source': source, 'elapsed_train_save_eval_seconds': time.monotonic() - start,
        'val_sha256': file_hash(cfg['val_file']), 'train_examples': len(train), 'val_examples': len(val),
        'skipped_train_examples': train.skipped, 'skipped_val_examples': val.skipped,
        'max_steps_override': args.max_steps, 'gpu': torch.cuda.get_device_name(0),
        'peak_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3,
        'torch_version': torch.__version__, 'metrics': metrics,
        'git_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        'warning': 'Validation token loss is not shopping success rate. Run real episodes separately.'})
