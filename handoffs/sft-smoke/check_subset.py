"""CPU tokenizer check for the frozen smoke subset; does not import torch."""
import argparse
import json
import sys
from pathlib import Path
from prepare import HERE, sha, write, require

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.repo.resolve()/'src'))
    from transformers import AutoTokenizer
    from webshop_lab.model import fit_history, training_messages
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    cfg = json.loads((HERE/'smoke-config.json').read_text())
    manifest = json.loads((HERE/'data/subset-manifest.json').read_text())
    result = {'context': cfg['max_length'], 'datasets': {}, 'gpu_training_executed': False}
    samples = []
    for name in ['train', 'val']:
        path = HERE/cfg[f'{name}_file']
        require(sha(path) == manifest['datasets'][name]['subset_sha256'], 'check_subset.py: validation failed at original line 23')
        kept_count, skipped, trimmed, lengths = 0, 0, 0, []
        for row in json.loads(path.read_text()):
            messages = training_messages(row)
            for i, message in enumerate(messages):
                if message['role'] != 'assistant':
                    continue
                try:
                    kept, ids = fit_history(tok, messages[:i+1], cfg['max_length'])
                except ValueError:
                    skipped += 1
                    continue
                prefix = tok.apply_chat_template(kept[:-1], tokenize=True, add_generation_prompt=True)
                require(ids[:len(prefix)] == prefix and len(ids)>len(prefix), 'check_subset.py: validation failed at original line 36')
                text = tok.apply_chat_template(kept, tokenize=False, add_generation_prompt=False)
                prompt = tok.apply_chat_template(kept[:-1], tokenize=False, add_generation_prompt=True)
                require(text.startswith(prompt), 'check_subset.py: validation failed at original line 39')
                labels = [-100]*len(prefix)+ids[len(prefix):]
                require(labels[len(prefix):] == tok.encode(text[len(prompt):], add_special_tokens=False), 'check_subset.py: validation failed at original line 41')
                require(kept[0] == messages[0] and kept[-2:] == messages[i-1:i+1], 'check_subset.py: validation failed at original line 42')
                kept_count += 1
                trimmed += len(kept)<i+1
                lengths.append(len(ids))
                samples.append((ids, labels))
        result['datasets'][name] = {'examples': kept_count, 'skipped': skipped, 'trimmed': trimmed,
                                    'max_length': max(lengths), 'label_invariants_passed': True}
    pair = next(( [samples[0], item] for item in samples[1:] if len(item[0]) != len(samples[0][0]) ), None)
    require(pair, 'check_subset.py: validation failed at original line 50')
    width = max(len(ids) for ids,labels in pair)
    for ids,labels in pair:
        n = width-len(ids)
        attention = [1]*len(ids)+[0]*n
        padded = labels+[-100]*n
        require(all(label == -100 for mask,label in zip(attention,padded) if mask==0), 'check_subset.py: validation failed at original line 56')
    result['padding_contract_passed'] = True
    require(result['datasets']['train']['examples'] >= 16, 'check_subset.py: validation failed at original line 58')
    write(HERE/'data/subset-tokenizer-check.json', result)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
