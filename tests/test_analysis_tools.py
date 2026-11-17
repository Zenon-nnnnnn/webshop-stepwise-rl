"""Tests for the pre-run gates, the paired comparison, the results table and the independent check.

Fixtures are synthetic: they exercise tooling, they are not model results.
"""
import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from webshop_lab import comparison, evaluation, preflight, register
from webshop_lab.common import file_hash

REPO = Path(__file__).resolve().parents[1]
FINGERPRINT = 'a' * 64
BASE_CONFIG = {'model': 'Qwen/Qwen2.5-1.5B-Instruct', 'adapter': None, 'load_in_4bit': True, 'seed': 42,
               'temperature': 0.0, 'top_p': 0.9, 'context_length': 2048, 'max_new_tokens': 128,
               'max_steps': 10, 'environment_url': 'http://127.0.0.1:8765',
               'goal_fingerprint': FINGERPRINT}


def step(number, observation, action, grounding=1, error=None, reward=0.0):
    return {'step': number, 'observation': observation, 'available': {'has_search_bar': True},
            'response': f'Thought: t\nAction: {action}', 'action': action,
            'grounding': grounding, 'error': error, 'environment_reward': reward}


def episode(task, reward, instruction='red shoes, and price lower than 30.00 dollars', steps=None,
            done=True, repeat=0, run_id='run-1'):
    if steps is None:
        steps = [step(0, 'Search', 'search[red shoes]'),
                 step(1, 'red shoes [SEP] Price: $20.00 [SEP] Buy Now', 'click[buy now]', reward=reward)]
    return {'schema_version': 1, 'run_id': run_id, 'task_id': task, 'repeat': repeat,
            'instruction': instruction, 'goal_fingerprint': FINGERPRINT,
            'final_reward': reward, 'done': done,
            'success': bool(done) and reward >= 1 - 1e-6,
            'termination': 'purchase' if done else 'max_steps', 'steps': steps, 'seconds': 1.0}


def write_json(path, value):
    Path(path).write_text(json.dumps(value), encoding='utf-8')
    return path


def write_jsonl(path, rows):
    Path(path).write_text('\n'.join(json.dumps(row) for row in rows) + '\n', encoding='utf-8')
    return path


def write_adapter(directory):
    (Path(directory)).mkdir(parents=True, exist_ok=True)
    (Path(directory) / 'adapter_config.json').write_text('{}', encoding='utf-8')
    (Path(directory) / 'adapter_model.safetensors').write_text('', encoding='utf-8')
    return str(directory)


class PreflightTests(unittest.TestCase):
    def _args(self, directory, **overrides):
        values = {'a_config': None, 'b_config': None, 'ids': None, 'register': None,
                  'root': str(directory), 'require_frozen': False, 'no_git_check': True,
                  'output': str(Path(directory) / 'preflight.json')}
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_ready_inputs_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = write_adapter(root / 'adapter')
            config = dict(BASE_CONFIG, adapter=adapter)
            left = write_json(root / 'b0.json', dict(config, adapter=None))
            right = write_json(root / 'b1.json', config)
            ids = write_json(root / 'val_ids.json', list(range(40)))
            preflight.run(self._args(directory, a_config=str(left), b_config=str(right), ids=str(ids)))
            result = json.loads((root / 'preflight.json').read_text(encoding='utf-8'))
            self.assertTrue(result['ready'], msg=result['problems'])
            self.assertEqual(result['ids']['count'], 40)
            self.assertTrue(result['parity']['parity'])

    def test_missing_adapter_and_duplicate_ids_are_blocking(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty = root / 'empty-adapter'
            empty.mkdir()
            config = write_json(root / 'b1.json', dict(BASE_CONFIG, adapter=str(empty)))
            ids = write_json(root / 'val_ids.json', [1, 2, 2, 3])
            with self.assertRaises(SystemExit):
                preflight.run(self._args(directory, ids=str(ids), a_config=str(config)))
            result = json.loads((root / 'preflight.json').read_text(encoding='utf-8'))
            self.assertFalse(result['ready'])
            joined = ' '.join(result['problems'])
            self.assertIn('adapter', joined)
            self.assertIn('duplicated', joined)

    def test_parity_drift_blocks_and_frozen_fingerprint_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = write_json(root / 'b0.json', BASE_CONFIG)
            right = write_json(root / 'b1.json', dict(BASE_CONFIG, context_length=1024))
            ids = write_json(root / 'val_ids.json', list(range(40)))
            with self.assertRaises(SystemExit):
                preflight.run(self._args(directory, a_config=str(left), b_config=str(right), ids=str(ids)))
            result = json.loads((root / 'preflight.json').read_text(encoding='utf-8'))
            self.assertTrue(any('context_length' in problem for problem in result['problems']))
            unfrozen = write_json(root / 'unfrozen.json', dict(BASE_CONFIG, goal_fingerprint=None))
            preflight.run(self._args(directory, a_config=str(unfrozen), ids=str(ids)))
            warned = json.loads((root / 'preflight.json').read_text(encoding='utf-8'))
            self.assertTrue(warned['ready'])
            with self.assertRaises(SystemExit):
                preflight.run(self._args(directory, a_config=str(unfrozen), ids=str(ids),
                                         require_frozen=True))


class ComparisonTests(unittest.TestCase):
    def test_identical_runs_show_no_difference(self):
        rows = [episode(task, 1. if task % 2 == 0 else 0.) for task in range(10)]
        result = comparison.compare_runs(rows, rows, ('B0', 'B1'))
        self.assertEqual(result['risk_difference'], 0.0)
        self.assertEqual(result['risk_difference_ci95'], [0.0, 0.0])
        self.assertEqual(result['paired']['mcnemar_exact_p'], 1.0)
        self.assertEqual(result['tasks'], 10)

    def test_paired_counts_and_exact_test(self):
        rows_a = [episode(task, 0.) for task in range(10)]
        rows_b = [episode(task, 1. if task < 3 else 0.) for task in range(10)]
        result = comparison.compare_runs(rows_a, rows_b, ('B0', 'B1'))
        self.assertEqual(result['paired']['only_B1'], 3)
        self.assertEqual(result['paired']['only_B0'], 0)
        self.assertAlmostEqual(result['risk_difference'], 0.3)
        self.assertAlmostEqual(result['paired']['mcnemar_exact_p'], 0.25)
        low, high = result['risk_difference_ci95']
        self.assertLessEqual(low, high)
        self.assertLess(low, 0.3)

    def test_different_task_sets_are_rejected(self):
        with self.assertRaises(ValueError):
            comparison.compare_runs([episode(1, 1.), episode(2, 0.)], [episode(1, 1.), episode(3, 0.)])
        with self.assertRaises(ValueError):
            comparison.compare_runs([episode(1, 1.)], [episode(1, 1.)])


class ResultsTableTests(unittest.TestCase):
    def test_only_complete_rows_are_rendered(self):
        rows = [
            {'run_id': 'r1', 'status': 'complete', 'method': 'B0', 'model': 'm', 'seed': '42',
             'episodes': '10', 'purchase_terminated': '5', 'successes': '2', 'success_rate': '0.2',
             'success_rate_ci95': '[0.05, 0.5]', 'invalid_action_rate': '0.1',
             'config_path': 'configs/eval-base.json', 'notes': 'b0'},
            {'run_id': 'r2', 'status': 'planned', 'method': 'B1', 'model': 'm'},
        ]
        text = register.results_table(list(register.COLUMNS), rows, commit='abc123')
        self.assertIn('| B0 |', text)
        self.assertNotIn('| B1 |', text)
        self.assertIn('abc123', text)
        empty = register.results_table(list(register.COLUMNS), [], commit='abc123')
        self.assertIn('暂无 complete 行', empty)

    def test_header_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            register.results_table(['run_id', 'status'], [])


class MetricsDefinitionTests(unittest.TestCase):
    def test_definition_matches_code(self):
        definition = json.loads((REPO / 'results/metrics-definition.json').read_text(encoding='utf-8'))
        names = [metric['name'] for metric in definition['metrics']]
        summary = evaluation.summarize_episodes([episode(1, 1.), episode(2, 0., done=False)])
        missing = sorted(set(names) - set(summary))
        self.assertEqual(missing, [], msg='metrics-definition.json documents fields the code does not return')
        self.assertEqual(set(definition['failure_classes']), set(evaluation.FAILURE_CLASSES))


class TerminalRewardTests(unittest.TestCase):
    def test_page_score_is_not_task_reward(self):
        rows = [episode(1, 1.), episode(2, 0.9, done=False)]
        metrics = evaluation.summarize_episodes(rows)
        self.assertAlmostEqual(metrics['mean_final_reward'], 0.95)
        self.assertAlmostEqual(metrics['mean_terminal_reward'], 0.5)
        self.assertEqual(evaluation.terminal_reward(rows[1]), 0.0)

    def test_register_row_records_the_terminal_reward(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = write_json(root / 'config.json', BASE_CONFIG)
            split = write_json(root / 'val_ids.json', [1, 2])
            jsonl = write_jsonl(root / 'run.jsonl', [episode(1, 1.), episode(2, 0.9, done=False)])
            args = argparse.Namespace(method='B0', split_ids=str(split), jsonl=str(jsonl),
                                      peak_vram_gb=None)
            audit = {'complete': True, 'run_id': 'run-1', 'goal_fingerprint': FINGERPRINT,
                     'summary': '2 of 2'}
            metrics = {'episodes': 2, 'full_reward_successes': 1, 'purchase_terminated': 1,
                       'success_rate': 0.5, 'success_rate_ci95': [0.09, 0.9],
                       'mean_terminal_reward': 0.5, 'mean_final_reward': 0.95,
                       'mean_steps': 2.0, 'invalid_action_rate': 0.0}
            row = register.row_from_run(args, audit, {'path': config, 'sha256': 'a' * 64}, metrics)
            self.assertEqual(row['mean_reward'], '0.500000')
            self.assertIn('task reward', row['notes'])

    def test_config_hash_accepts_a_windows_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'configs/eval-base.json'
            config.parent.mkdir(parents=True)
            text = json.dumps(BASE_CONFIG, indent=2) + '\n'
            config.write_text(text, encoding='utf-8', newline='\n')
            lf_hash = file_hash(config)
            config.write_bytes(text.encode('utf-8').replace(b'\n', b'\r\n'))
            row = {'run_id': 'r', 'status': 'complete', 'method': 'B0',
                   'model': BASE_CONFIG['model'], 'seed': '42', 'split_sha256': 'b' * 64,
                   'goal_fingerprint': FINGERPRINT, 'git_commit': 'a' * 40,
                   'config_path': 'configs/eval-base.json', 'config_sha256': lf_hash,
                   'episodes': '4', 'successes': '1', 'purchase_terminated': '2',
                   'success_rate': '0.25', 'success_rate_ci95': '[0.05, 0.6]',
                   'mean_reward': '0.4', 'mean_steps': '2.25', 'invalid_action_rate': '0.1',
                   'peak_vram_gb': '', 'evidence_path': 'artifacts/run.jsonl', 'notes': 'n'}
            result = register.validate(list(register.COLUMNS), [row], root, check_git=False)
            self.assertEqual(result['status'], 'ok', msg=result['errors'])
            self.assertTrue(any('CRLF' in warning for warning in result['warnings']))


class IndependentMetricCheckTests(unittest.TestCase):
    def test_script_agrees_with_evaluate_and_flags_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = write_json(root / 'config.json', BASE_CONFIG)
            jsonl = write_jsonl(root / 'run.jsonl', [episode(1, 1.), episode(2, .6), episode(3, 0., done=False)])
            meta = write_json(root / 'run.meta.json', {
                'run_id': 'run-1', 'config': BASE_CONFIG, 'config_sha256': 'x' * 64,
                'ids_sha256': 'y' * 64, 'selected_ids': [1, 2, 3], 'repeats': 1,
                'health': {'goal_fingerprint': FINGERPRINT},
                'started_utc': '2026-11-16T00:00:00+00:00'})
            evaluation.evaluate(argparse.Namespace(
                meta=str(meta), jsonl=str(jsonl), config=str(config),
                output=str(root / 'evaluation.json'), cases=None, figures_dir=None,
                allow_partial=False, cases_per_class=1, method='B0', split_ids=None,
                register_row=None, peak_vram_gb=None))
            command = [sys.executable, str(REPO / 'scripts/verify_metrics.py'),
                       '--jsonl', str(jsonl), '--evaluation', str(root / 'evaluation.json')]
            passed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(passed.returncode, 0, msg=passed.stdout + passed.stderr)
            recorded = json.loads((root / 'evaluation.json').read_text(encoding='utf-8'))
            recorded['metrics']['success_rate'] = 0.5
            (root / 'tampered.json').write_text(json.dumps(recorded), encoding='utf-8')
            failed = subprocess.run([sys.executable, str(REPO / 'scripts/verify_metrics.py'),
                                     '--jsonl', str(jsonl), '--evaluation', str(root / 'tampered.json')],
                                    capture_output=True, text=True)
            self.assertEqual(failed.returncode, 1)
            self.assertIn('success_rate', failed.stdout)


if __name__ == '__main__':
    unittest.main()
