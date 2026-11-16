"""Collect episodes sequentially. No simulator answer/reward enters the policy prompt."""
import datetime
import json
import math
import time
import urllib.request
import uuid
from pathlib import Path
from .common import read_json, write_json, file_hash, parse_action, is_executable
from .model import load, fit_history, SYSTEM


def request(url, body=None):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.load(response)


def run(args):
    import torch
    from transformers import set_seed
    cfg = read_json(args.config)
    base = cfg.get('environment_url', 'http://127.0.0.1:8765').rstrip('/')
    if not base.startswith('http://127.0.0.1:'):
        raise ValueError('Use the local environment bridge at http://127.0.0.1:PORT')
    health = request(base + '/health')
    if not health.get('ready'):
        raise RuntimeError('Environment is not ready')
    if not health.get('alignment_checked') or health.get('alignment_mismatches'):
        raise RuntimeError('SFT/environment task alignment is unverified or mismatched. See docs/TROUBLESHOOTING.md; do not report invalid benchmark results.')
    if not health.get('split_status', {}).get('verified'):
        raise RuntimeError('Environment instruction split check failed. Re-run split --goal-map artifacts/goal-instructions.json.')
    if cfg.get('goal_fingerprint') and cfg['goal_fingerprint'] != health['goal_fingerprint']:
        raise ValueError('Environment goal fingerprint differs from the frozen experiment configuration')
    out = Path(args.output)
    if out.exists():
        raise ValueError(f'Refusing to overwrite {out}; choose a new run path')
    if args.repeats < 1:
        raise ValueError('repeats must be positive')
    ids = read_json(args.ids)
    if not isinstance(ids, list) or any(type(i) is not int for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Expected a JSON list of unique integer task IDs')
    if args.limit:
        if args.limit < 1:
            raise ValueError('limit must be positive')
        ids = ids[:args.limit]
    if not ids:
        raise ValueError('Empty task list')
    set_seed(cfg.get('seed', 42))
    model, tok, _ = load(cfg)
    model.eval()
    max_steps = int(cfg.get('max_steps', 10))
    if max_steps < 1:
        raise ValueError('max_steps must be positive')
    max_new = int(cfg.get('max_new_tokens', 128))
    max_context = int(cfg.get('context_length', 2048)) - max_new
    out.parent.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())
    write_json(out.with_suffix('.meta.json'), {'run_id': run_id, 'config': cfg,
        'config_sha256': file_hash(args.config), 'ids_sha256': file_hash(args.ids), 'selected_ids': ids,
        'repeats': args.repeats, 'health': health, 'torch_version': torch.__version__,
        'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
    with out.open('x', encoding='utf-8') as f:
        for task in ids:
            for repeat in range(args.repeats):
                start = time.monotonic()
                state = request(base + '/reset', {'task_id': task})
                messages = [{'role': 'system', 'content': SYSTEM + '\nShopping instruction: ' + state['instruction']},
                            {'role': 'user', 'content': state['observation']}]
                steps = []
                done, reward = False, 0.
                for index in range(max_steps):
                    kept, tokens = fit_history(tok, messages, max_context, generation=True)
                    input_ids = torch.tensor([tokens], device=model.device)
                    temp = float(cfg.get('temperature', 0.))
                    generation_kwargs = {'do_sample': temp > 0, 'max_new_tokens': max_new,
                                         'pad_token_id': tok.pad_token_id, 'eos_token_id': model.generation_config.eos_token_id}
                    if temp > 0:
                        generation_kwargs.update(temperature=temp, top_p=cfg.get('top_p', .9))
                    with torch.inference_mode():
                        generated = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), **generation_kwargs)
                    text = tok.decode(generated[0, len(tokens):], skip_special_tokens=True).strip()
                    valid, action, error = False, None, None
                    before = state['observation']
                    available = state['available']
                    try:
                        action = parse_action(text)
                        if not is_executable(action, available):
                            raise ValueError('Action is not in available actions')
                        state = request(base + '/step', {'action': action})
                        valid = True
                        reward = float(state['reward'])
                        done = bool(state['done'])
                        feedback = state['observation']
                    except ValueError as e:
                        error = str(e)
                        feedback = 'Observation: Invalid action or format. Reply with one valid Action: line.'
                    # Network/server failures propagate and stop the run; never count them as model failures.
                    steps.append({'step': index, 'observation': before, 'available': available,
                                  'response': text, 'action': action, 'grounding': int(valid),
                                  'error': error, 'environment_reward': reward if valid else 0.})
                    messages.extend([{'role': 'assistant', 'content': text}, {'role': 'user', 'content': feedback}])
                    if done:
                        break
                record = {'schema_version': 1, 'run_id': run_id, 'task_id': task, 'repeat': repeat,
                          'instruction': state.get('instruction', messages[0]['content'].split('Shopping instruction: ', 1)[1]),
                          'goal_fingerprint': health['goal_fingerprint'], 'final_reward': reward,
                          'done': done, 'success': done and reward >= 1 - 1e-6,
                          'termination': 'purchase' if done else 'max_steps',
                          'steps': steps, 'seconds': time.monotonic() - start}
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                f.flush()
                print(f'task={task} repeat={repeat} reward={reward:.3f} done={done} steps={len(steps)}')
