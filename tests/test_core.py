import argparse
import json
import tempfile
import unittest
import zipfile
from unittest.mock import patch
from pathlib import Path
from webshop_lab.common import (combine_rewards, goal_hash, instruction_hash, is_executable,
                                parse_action, read_json, task_partition, wilson)
from webshop_lab.metrics import load_episodes, run as summarize
from webshop_lab.progress import train, predict
from webshop_lab.rewards import run as export
from webshop_lab.model import fit_history
from webshop_lab.data import import_assets


def row(task, instruction):
    return {'id': task, 'conversations': [
        {'from': 'human', 'value': 'You are web shopping.'}, {'from': 'gpt', 'value': 'OK'},
        {'from': 'human', 'value': f'WebShop [SEP] Instruction: [SEP] {instruction} [SEP] Search'},
        {'from': 'gpt', 'value': 'Thought: search\nAction: search[shoes]'}]}


def episode(task, reward, instruction):
    return {'task_id': task, 'repeat': 0, 'instruction': instruction, 'goal_fingerprint': 'fixed-goals',
            'done': True, 'final_reward': reward, 'termination': 'purchase', 'steps': [
                {'observation': f'Instruction: {instruction}', 'response': 'Action: search[shoes]', 'grounding': 1},
                {'observation': 'Buy Now', 'response': 'Action: click[buy now]', 'grounding': 1}]}


class CoreTests(unittest.TestCase):
    def test_asset_import_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            source, root = d / 'source', d / 'repo'
            source.mkdir()
            root.mkdir()
            (source / 'webshop_sft.json').write_text('[]')
            with zipfile.ZipFile(source / 'data.zip', 'w') as z:
                z.writestr('../../../escaped.txt', 'should never be extracted')
            with zipfile.ZipFile(source / 'indexes.zip', 'w') as z:
                z.writestr('indexes/segments_1', 'dummy')
            with patch('webshop_lab.data.ROOT', root):
                with self.assertRaises(ValueError):
                    import_assets(argparse.Namespace(source=str(source)))
            self.assertFalse((d / 'escaped.txt').exists())
    def test_action_parser_and_availability(self):
        self.assertEqual(parse_action('Thought: find\nAction: click[Buy Now]'), 'click[buy now]')
        for invalid in ['click[buy now]', '', 'Action: click[x]\nAction: click[y]', 'Action: click[]']:
            with self.assertRaises(ValueError):
                parse_action(invalid)
        self.assertFalse(is_executable('search[shoes]', {'has_search_bar': False}))
        self.assertTrue(is_executable('click[red]', {'clickables': ['red']}))

    def test_partition_groups_ids_instructions_and_budget_variants(self):
        rows = [row(1, 'red shoes, and price lower than 20.00 dollars'),
                row(2, 'red shoes, and price lower than 30.00 dollars'),
                row(3, 'blue coat'), row(4, 'green hat'), row(5, 'white socks'),
                row(6, 'yellow bag'), row(7, 'purple tie'), row(8, 'pink belt')]
        self.assertEqual(goal_hash(rows[0]), goal_hash(rows[1]))
        train_rows, val_rows, excluded = task_partition(rows, [1], seed=42)
        self.assertEqual(excluded, 2)
        self.assertFalse({r['id'] for r in train_rows + val_rows} & {1, 2})
        self.assertFalse({goal_hash(r) for r in train_rows} & {goal_hash(r) for r in val_rows})
        self.assertEqual((train_rows, val_rows, excluded), task_partition(rows, [1], seed=42))
        other = task_partition(rows, [], reserved_goal_hashes=[instruction_hash('white socks')])
        self.assertNotIn(5, [r['id'] for r in other[0] + other[1]])

    def test_same_id_cannot_cross_splits(self):
        rows = [row(1, 'red shoes'), row(1, 'red boots'), row(2, 'blue hat'), row(3, 'green bag')]
        a, b, _ = task_partition(rows, [], val_fraction=.5)
        self.assertFalse({r['id'] for r in a} & {r['id'] for r in b})

    def test_reward_modes_and_nonfinite_rejection(self):
        self.assertEqual(combine_rewards([.1, .4], [1, 0], .5, 'spa'), [.6, .4])
        self.assertEqual(combine_rewards([.1, .4], [1, 0], .5, 'terminal'), [0, .5])
        self.assertEqual(combine_rewards([.1, .4], [1, 0], .5, 'uniform'), [.25, .25])
        with self.assertRaises(ValueError):
            combine_rewards([float('nan')], [1], 0)
        self.assertIsNone(wilson(0, 0))
        lo, hi = wilson(1, 2)
        self.assertLess(lo, .5)
        self.assertGreater(hi, .5)

    def test_history_preserves_system_and_current_exchange(self):
        class Tokenizer:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return list(''.join(m['role'] + ':' + m['content'] for m in messages) +
                            ('assistant:' if add_generation_prompt else ''))
        msgs = [{'role': 'system', 'content': 'task'}, {'role': 'user', 'content': 'old'},
                {'role': 'assistant', 'content': 'old reply'}, {'role': 'user', 'content': 'current'}]
        kept, ids = fit_history(Tokenizer(), msgs, 45, generation=True)
        self.assertEqual(kept, [msgs[0], msgs[-1]])
        self.assertEqual(msgs[1]['content'], 'old')
        with self.assertRaises(ValueError):
            fit_history(Tokenizer(), [msgs[0], msgs[-1]], 5, generation=True)

    def test_metrics_purchase_is_not_success(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            path = d / 'episodes.jsonl'
            rows = [episode(1, .5, 'red shoes'), episode(2, 1., 'blue shoes')]
            path.write_text('\n'.join(json.dumps(r) for r in rows))
            summarize(argparse.Namespace(input=str(path), output=str(d / 'summary.json')))
            self.assertEqual(read_json(d / 'summary.json')['success_rate'], .5)
            path.write_text('\n'.join(json.dumps(r) for r in [rows[0], rows[0]]))
            with self.assertRaises(ValueError):
                load_episodes(path)

    def test_linear_progress_and_reward_export_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            def put(name, rows):
                p = d / name
                p.write_text('\n'.join(json.dumps(r) for r in rows))
                return str(p)
            training = put('train.jsonl', [episode(1, 0., 'red shoes'), episode(2, 1., 'blue coat')])
            val = put('val.jsonl', [episode(3, .5, 'green hat')])
            model = str(d / 'model.json')
            train(argparse.Namespace(train=training, val=val, output=model, dimensions=32, epochs=10, seed=42))
            annotated = str(d / 'annotated.jsonl')
            predict(argparse.Namespace(model=model, input=training, output=annotated))
            export(argparse.Namespace(input=annotated, output=str(d / 'rl.json'), mode='spa', alpha=1., beta=.5))
            self.assertEqual(len(read_json(d / 'rl.json')), 2)
            with self.assertRaises(ValueError):
                export(argparse.Namespace(input=training, output=str(d / 'bad.json'), mode='spa', alpha=1., beta=.5))
            with self.assertRaises(ValueError):
                train(argparse.Namespace(train=training, val=training, output=model, dimensions=32, epochs=2, seed=42))


if __name__ == '__main__':
    unittest.main()
