"""Shared CPU/GPU smoke input and collation contracts (no torch import)."""
from prepare import require, sha

TOKENIZER_SHAS = {
    'config.json': '98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670',
    'tokenizer.json': 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539',
    'tokenizer_config.json': '5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583'}

def verify_tokenizer(path):
    for filename, expected in TOKENIZER_SHAS.items():
        require(sha(path/filename) == expected, f'Tokenizer mismatch: {filename}')

def verify_stats(actual, expected):
    for key in ['examples', 'skipped', 'trimmed', 'max_length']:
        value = actual.get('history_trimmed') if key == 'trimmed' and 'history_trimmed' in actual else actual.get(key)
        require(value == expected[key], f'Subset statistic mismatch for {key}: {value} != {expected[key]}')

def collate_lists(items, pad_id):
    width = max(len(item['input_ids']) for item in items)
    return {key: [item[key] + [pad] * (width-len(item[key])) for item in items]
            for key, pad in [('input_ids', pad_id), ('attention_mask', 0), ('labels', -100)]}

def verify_collated(batch, items, pad_id):
    width = max(len(item['input_ids']) for item in items)
    for index, item in enumerate(items):
        length = len(item['input_ids'])
        for key in ['input_ids', 'attention_mask', 'labels']:
            row = batch[key][index]
            require(len(row) == width and row[:length] == item[key], f'Collator changed original {key}')
        require(batch['input_ids'][index][length:] == [pad_id]*(width-length), 'Invalid input padding')
        require(batch['attention_mask'][index][length:] == [0]*(width-length), 'Invalid padding attention mask')
        require(batch['labels'][index][length:] == [-100]*(width-length), 'Invalid padding labels')
