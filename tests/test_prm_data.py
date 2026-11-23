"""CPU tests for PRM data conversion, split gating and fit/dev isolation."""
import json
import math
import tempfile
import unittest
from pathlib import Path

from webshop_lab.prm import data


def episode(task, instruction, observations=('o',), done=True, final=1.0):
    steps = [{'observation': obs, 'response': 'Thought: t\nAction: search[x]'}
             for obs in observations]
    return {'task_id': task, 'repeat': 0, 'instruction': instruction, 'steps': steps,
            'done': done, 'final_reward': final,
            'success': bool(done and final >= 1.0 - 1e-6),
            'termination': 'purchase' if done else 'max_steps'}


class RewardTests(unittest.TestCase):
    def test_purchase_keeps_recorded_score(self):
        self.assertAlmostEqual(data.terminal_reward(episode(1, 'buy socks', final=0.75)), 0.75)

    def test_non_purchase_is_zero_even_if_page_score_is_high(self):
        self.assertEqual(data.terminal_reward(episode(1, 'x', done=False, final=0.9)), 0.0)

    def test_missing_done_is_rejected(self):
        row = episode(1, 'x')
        row.pop('done')
        with self.assertRaises(ValueError):
            data.terminal_reward(row)

    def test_nan_and_inf_are_rejected_not_zeroed(self):
        for value in (float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                data.terminal_reward(episode(1, 'x', done=False, final=value))

    def test_boolean_score_is_rejected(self):
        with self.assertRaises(ValueError):
            data.terminal_reward(episode(1, 'x', final=True))

    def test_environment_error_is_a_technical_interruption(self):
        row = episode(1, 'x', done=False, final=0.0)
        row['termination'] = 'environment_error'
        with self.assertRaises(data.TechnicalInterruption):
            data.terminal_reward(row)

    def test_done_and_termination_must_agree(self):
        row = episode(1, 'x', done=False, final=0.0)
        row['termination'] = 'purchase'
        with self.assertRaises(ValueError):
            data.terminal_reward(row)

    def test_success_must_match_purchase(self):
        row = episode(1, 'x', done=True, final=0.5)
        row['success'] = True
        with self.assertRaises(ValueError):
            data.terminal_reward(row)


class MessageTests(unittest.TestCase):
    def test_messages_alternate_and_start_with_system(self):
        messages = data.episode_messages(episode(1, 'buy socks', ('o1', 'o2')))
        self.assertEqual([m['role'] for m in messages],
                         ['system', 'user', 'assistant', 'user', 'assistant'])
        self.assertIn('Shopping instruction: buy socks', messages[0]['content'])

    def test_inference_messages_do_not_require_labels(self):
        row = {'task_id': 1, 'instruction': 'buy socks',
               'steps': [{'observation': 'o', 'response': 'Thought: t\nAction: search[x]'}]}
        messages = data.episode_messages(row, require_labels=False)
        self.assertEqual(messages[-1]['role'], 'assistant')

    def test_training_messages_require_labels(self):
        row = {'task_id': 1, 'instruction': 'buy socks',
               'steps': [{'observation': 'o', 'response': 'Thought: t\nAction: search[x]'}]}
        with self.assertRaises(ValueError):
            data.episode_messages(row)

    def test_empty_assistant_action_is_rejected(self):
        row = episode(1, 'x')
        row['steps'][0]['response'] = '   '
        with self.assertRaises(ValueError):
            data.episode_messages(row)


class SplitsGateTests(unittest.TestCase):
    def make_split_dir(self, tmp, train, val, test):
        d = Path(tmp)
        for name, ids in (('train', train), ('val', val), ('test', test)):
            (d / (name + '_ids.json')).write_text(json.dumps(ids), encoding='utf-8')
            (d / (name + '.json')).write_text(json.dumps([]), encoding='utf-8')
        return d

    def test_val_tasks_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self.make_split_dir(tmp, [1, 2, 3], [4, 5], [6])
            splits = data.load_split(d)
            rows = [episode(4, 'val task', final=0.5)]
            with self.assertRaises(data.SourceNotAllowed):
                data.gate_train_pool(rows, splits, set())

    def test_train_tasks_pass_the_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self.make_split_dir(tmp, [1, 2, 3], [4], [5])
            splits = data.load_split(d)
            rows = [episode(1, 'train task', final=0.5)]
            info = data.gate_train_pool(rows, splits, set())
            self.assertEqual(info['instruction_overlap_with_eval'], 0)

    def test_instruction_leak_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self.make_split_dir(tmp, [1, 2], [4], [5])
            splits = data.load_split(d)
            rows = [episode(1, 'Buy  Socks', final=0.5)]
            with self.assertRaises(data.SourceNotAllowed):
                data.gate_train_pool(rows, splits, {data.normalize_instruction('buy socks')})

    def test_overlapping_splits_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self.make_split_dir(tmp, [1, 2], [2], [3])
            with self.assertRaises(ValueError):
                data.load_split(d)


class SplitTests(unittest.TestCase):
    def test_split_is_deterministic_and_disjoint(self):
        rows = [episode(i, 'instruction %d' % i, final=0.5) for i in range(10)]
        fit_a, dev_a, summary_a = data.split_by_task(rows, 0.7, seed=7)
        fit_b, dev_b, _ = data.split_by_task(rows, 0.7, seed=7)
        self.assertEqual([r['task_id'] for r in fit_a], [r['task_id'] for r in fit_b])
        self.assertEqual(set(r['task_id'] for r in fit_a) & set(r['task_id'] for r in dev_a), set())
        self.assertEqual(summary_a['instruction_overlap'], 0)

    def test_duplicate_normalized_instructions_across_split_are_rejected(self):
        rows = [episode(1, 'Buy  Socks', final=0.5),
                episode(2, 'buy socks', final=0.5),
                episode(3, 'other', final=0.5)]
        with self.assertRaises(ValueError):
            data.split_by_task(rows, 0.34, seed=1)


class BuildTests(unittest.TestCase):
    def test_overlong_trajectory_is_counted_not_truncated(self):
        rows = [episode(1, 'a')]
        original = data.tokenize_trajectory

        def boom(*args, **kwargs):
            raise data.TrajectoryTooLong('too long')

        data.tokenize_trajectory = boom
        try:
            samples, rejected = data.build_samples(rows, object(), 8)
        finally:
            data.tokenize_trajectory = original
        self.assertEqual(samples, [])
        self.assertEqual(rejected['too_long'], 1)

    def test_technical_interruption_is_counted_separately(self):
        rows = [episode(1, 'a', done=False, final=0.0)]
        rows[0]['termination'] = 'environment_error'
        samples, rejected = data.build_samples(rows, object(), 2048)
        self.assertEqual(samples, [])
        self.assertEqual(rejected['technical_interruption'], 1)

    def test_empty_action_is_counted(self):
        rows = [episode(1, 'a')]
        rows[0]['steps'][0]['response'] = '  '
        samples, rejected = data.build_samples(rows, object(), 2048)
        self.assertEqual(samples, [])
        self.assertEqual(rejected.get('empty_action'), 1)


class ManifestTests(unittest.TestCase):
    def test_manifest_records_sources_split_and_rejections(self):
        samples = [{'task_id': 1, 'repeat': 0, 'instruction_sha256': 'a' * 64,
                    'terminal_reward': 1.0, 'num_turns': 2, 'num_tokens': 10,
                    'template_sha256': 'b' * 64, 'source': {}, 'encoded': {}}]
        with tempfile.TemporaryDirectory() as tmp:
            manifest = data.write_dataset(
                Path(tmp) / 'ds', 'f' * 64, [{'path': 'r.jsonl', 'sha256': 'c' * 64}],
                {'seed': 42, 'fit_fraction': 0.8, 'fit_tasks': 1, 'dev_tasks': 1,
                 'fit_episodes': 1, 'dev_episodes': 1, 'instruction_overlap': 0},
                samples, [], {'fit': {}, 'dev': {}},
                {'path': 'tok', 'revision': 'd' * 40, 'template_sha256': 'b' * 64})
            self.assertEqual(manifest['score_position'], 'assistant_im_end')
            self.assertEqual(manifest['split']['instruction_overlap'], 0)
            self.assertIn('truncation', manifest)
            on_disk = json.loads((Path(tmp) / 'ds' / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(on_disk['files']['fit']['samples'], 1)


if __name__ == '__main__':
    unittest.main()
