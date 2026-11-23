"""CPU tests for PRM data conversion and fit/dev isolation."""
import json
import tempfile
import unittest
from pathlib import Path

from webshop_lab.prm import data


def episode(task, instruction, observations, done=True, final=1.0):
    steps = [{'observation': obs, 'response': 'Thought: t\nAction: search[x]'}
             for obs in observations]
    return {'task_id': task, 'repeat': 0, 'instruction': instruction, 'steps': steps,
            'done': done, 'final_reward': final, 'success': bool(done and final >= 1.0),
            'termination': 'purchase' if done else 'max_steps'}


class RewardTests(unittest.TestCase):
    def test_purchase_keeps_recorded_score(self):
        self.assertAlmostEqual(data.terminal_reward(episode(1, 'buy socks', ['o'], True, 0.75)), 0.75)

    def test_non_purchase_is_zero_even_if_page_score_is_high(self):
        self.assertEqual(data.terminal_reward(episode(1, 'buy socks', ['o'], False, 0.9)), 0.0)

    def test_missing_done_is_rejected(self):
        row = episode(1, 'x', ['o'])
        row.pop('done')
        with self.assertRaises(ValueError):
            data.terminal_reward(row)

    def test_out_of_range_reward_is_rejected(self):
        with self.assertRaises(ValueError):
            data.terminal_reward(episode(1, 'x', ['o'], True, 1.5))


class MessageTests(unittest.TestCase):
    def test_messages_alternate_and_start_with_system(self):
        messages = data.episode_messages(episode(1, 'buy socks', ['o1', 'o2']))
        self.assertEqual([m['role'] for m in messages],
                         ['system', 'user', 'assistant', 'user', 'assistant'])
        self.assertIn('Shopping instruction: buy socks', messages[0]['content'])

    def test_empty_assistant_action_is_rejected(self):
        row = episode(1, 'x', ['o'])
        row['steps'][0]['response'] = '   '
        with self.assertRaises(ValueError):
            data.episode_messages(row)

    def test_missing_schema_field_is_rejected(self):
        row = episode(1, 'x', ['o'])
        row.pop('instruction')
        with self.assertRaises(ValueError):
            data.episode_messages(row)


class SplitTests(unittest.TestCase):
    def test_split_is_deterministic_and_disjoint(self):
        rows = [episode(i, 'instruction %d' % i, ['o'], final=0.5) for i in range(10)]
        fit_a, dev_a, summary_a = data.split_by_task(rows, 0.7, seed=7)
        fit_b, dev_b, _ = data.split_by_task(rows, 0.7, seed=7)
        self.assertEqual([r['task_id'] for r in fit_a], [r['task_id'] for r in fit_b])
        self.assertEqual(set(r['task_id'] for r in fit_a) & set(r['task_id'] for r in dev_a), set())
        self.assertEqual(summary_a['instruction_overlap'], 0)
        self.assertEqual(summary_a['fit_episodes'] + summary_a['dev_episodes'], len(rows))

    def test_duplicate_normalized_instructions_across_split_are_rejected(self):
        rows = [episode(1, 'Buy  Socks', ['o'], final=0.5),
                episode(2, 'buy socks', ['o'], final=0.5),
                episode(3, 'other', ['o'], final=0.5)]
        with self.assertRaises(ValueError):
            data.split_by_task(rows, 0.34, seed=1)


class LoaderTests(unittest.TestCase):
    def test_duplicates_and_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'r.jsonl'
            path.write_text(json.dumps(episode(1, 'a', ['o'])) + '\n', encoding='utf-8')
            _, sources = data.load_episodes([path])
            self.assertEqual(len(sources), 1)
            self.assertEqual(len(sources[0]['sha256']), 64)
            with self.assertRaises(ValueError):
                data.load_episodes([path, path])


class BuildTests(unittest.TestCase):
    def test_overlong_trajectory_is_counted_not_truncated(self):
        rows = [episode(1, 'a', ['o'])]
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

    def test_empty_action_is_counted(self):
        rows = [episode(1, 'a', ['o'])]
        rows[0]['steps'][0]['response'] = '  '
        samples, rejected = data.build_samples(rows, object(), 2048)
        self.assertEqual(samples, [])
        self.assertEqual(rejected.get('empty_action'), 1)


class ManifestTests(unittest.TestCase):
    def test_manifest_records_sources_split_and_rejections(self):
        samples = [{'task_id': 1, 'repeat': 0, 'instruction_sha256': 'a' * 64,
                    'terminal_reward': 1.0, 'num_turns': 2, 'num_tokens': 10,
                    'template_sha256': 'b' * 64, 'encoded': {}}]
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
            self.assertEqual(len(on_disk['files']['fit']['sha256']), 64)


if __name__ == '__main__':
    unittest.main()
