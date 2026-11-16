"""One-run summary command. Metric definitions live in evaluation.py so every tool agrees."""
import json

from .common import file_hash, write_json
from .evaluation import load_episodes, summarize_episodes

__all__ = ['load_episodes', 'run']


def run(args):
    rows = load_episodes(args.input)
    result = summarize_episodes(rows)
    result['input_sha256'] = file_hash(args.input)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
