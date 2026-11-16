"""Tests for the evaluation audit, failure triage, parity and register checks.

These fixtures are synthetic: they exercise the tooling, they are not model results.
"""
import argparse
import json
import tempfile
import unittest
from pathlib import Path

from webshop_lab import evaluation, figures, register
from webshop_lab.common import file_hash
from webshop_lab.metrics import run as summarize

FINGERPRINT = 'a' * 64
REPO = Path(__file__).resolve().parents[1]
CONFIG_SHA = file_hash(REPO / 'configs/eval-base.json')
CONFIG = {'model': 'Qwen/Qwen2.5-1.5B-Instruct', 'adapter': None, 'load_in_4bit': True, 'seed': 42,
          'temperature': 0.0, 'top_p': 0.9, 'context_length': 2048, 'max_new_tokens': 128,
          'max_steps': 10, 'environment_url': 'http://127.0.0.1:8765', 'goal_fingerprint': FINGERPRINT}


def step(number, observation, action, grounding=1, error=None, reward=0.0):
    return {'step': number, 'observation': observation, 'available': {'has_search_bar': True},
            'response': f'Thought: t\nAction: {action}', 'action': action,
            'grounding': grounding, 'error': error, 'environment_reward': reward}


def episode(task, reward, instruction='red shoes, and price lower than 30.00 dollars', steps=None,
            done=True, termination=None, repeat=0, run_id='run-1', fingerprint=FINGERPRINT):
    if steps is None:
        steps = [step(0, 'Search', 'search[red shoes]'),
                 step(1, 'red shoes [SEP] Price: $20.00 [SEP] Buy Now',
                      'click[buy now]', reward=reward)]
    return {'schema_version': 1, 'run_id': run_id, 'task_id': task, 'repeat': repeat,
            'instruction': instruction, 'goal_fingerprint': fingerprint,
            'final_reward': reward, 'done': done,
            'success': bool(done) and reward >= 1 - 1e-6,
            'termination': termination or ('purchase' if done else 'max_steps'),
            'steps': steps, 'seconds': 1.0}


def meta(ids, repeats=1, run_id='run-1', fingerprint=FINGERPRINT, config=None, config_sha='0' * 64):
    return {'run_id': run_id, 'config': config or dict(CONFIG),
            'config_sha256': config_sha, 'ids_sha256': 'b' * 64,
            'selected_ids': ids, 'repeats': repeats,
            'health': {'goal_fingerprint': fingerprint},
            'started_utc': '2026-11-16T00:00:00+00:00'}


def write_jsonl(path, rows):
    Path(path).write_text('\n'.join(json.dumps(row) for row in rows) + '\n', encoding='utf-8')
    return path


def write_json(path, value):
    Path(path).write_text(json.dumps(value), encoding='utf-8')
    return path


class CompletionAuditTests(unittest.TestCase):
    def test_exact_set_is_complete_and_missing_episode_is_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [episode(1, 1.), episode(2, .5)]
            meta_path = write_json(root / 'run.meta.json', meta([1, 2]))
            jsonl = write_jsonl(root / 'run.jsonl', rows)
            audit = evaluation.inspect_run(jsonl, meta_path)
            self.assertEqual(audit['status'], 'complete')
            self.assertTrue(audit['order_ok'])
            write_jsonl(jsonl, rows[:1])
            partial = evaluation.inspect_run(jsonl, meta_path)
            self.assertEqual(partial['status'], 'partial')
            self.assertFalse(partial['complete'])
            self.assertEqual(partial['missing_pairs'], [[2, 0]])

    def test_unexpected_and_duplicate_episodes_are_mismatched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta_path = write_json(root / 'run.meta.json', meta([1]))
            jsonl = write_jsonl(root / 'run.jsonl', [episode(1, 1.), episode(2, 1.)])
            self.assertEqual(evaluation.inspect_run(jsonl, meta_path)['status'], 'mismatched')
            write_jsonl(jsonl, [episode(1, 1.), episode(1, 1.)])
            duplicate = evaluation.inspect_run(jsonl, meta_path)
            self.assertEqual(duplicate['status'], 'mismatched')
            self.assertEqual(duplicate['duplicate_pairs'], [[1, 0]])
            write_jsonl(jsonl, [episode(1, 1., repeat=0), episode(1, 1., repeat=1)])
            self.assertEqual(evaluation.inspect_run(jsonl, meta_path)['status'], 'mismatched')

    def test_repeats_must_cover_every_repeat_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta_path = write_json(root / 'run.meta.json', meta([1], repeats=2))
            jsonl = write_jsonl(root / 'run.jsonl', [episode(1, 1., repeat=0)])
            audit = evaluation.inspect_run(jsonl, meta_path)
            self.assertEqual(audit['status'], 'partial')
            self.assertEqual(len(audit['missing_pairs']), 1)

    def test_run_id_and_fingerprint_mismatches_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta_path = write_json(root / 'run.meta.json', meta([1]))
            jsonl = write_jsonl(root / 'run.jsonl', [episode(1, 1., run_id='other', fingerprint='c' * 64)])
            audit = evaluation.inspect_run(jsonl, meta_path)
            self.assertEqual(audit['run_id_mismatch_tasks'], [1])
            self.assertEqual(audit['goal_fingerprint_mismatch_tasks'], [1])

    def test_config_edit_after_run_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = write_json(root / 'config.json', CONFIG)
            meta_path = write_json(root / 'run.meta.json',
                                   meta([1], config_sha=file_hash(config_path)))
            jsonl = write_jsonl(root / 'run.jsonl', [episode(1, 1.)])
            audit = evaluation.inspect_run(jsonl, meta_path, config_path)
            self.assertTrue(audit['config_file']['matches_metadata'])
            edit = dict(CONFIG, max_steps=12)
            write_json(config_path, edit)
            audit = evaluation.inspect_run(jsonl, meta_path, config_path)
            self.assertFalse(audit['config_file']['matches_metadata'])


class MetricTests(unittest.TestCase):
    def test_purchase_is_counted_but_not_a_success(self):
        rows = [episode(1, 1.), episode(2, .5), episode(3, 0., done=False)]
        metrics = evaluation.summarize_episodes(rows)
        self.assertEqual(metrics['episodes'], 3)
        self.assertEqual(metrics['purchase_terminated'], 2)
        self.assertEqual(metrics['full_reward_successes'], 1)
        self.assertEqual(metrics['purchase_without_full_reward'], 1)
        self.assertEqual(metrics['max_steps_terminated'], 1)
        self.assertAlmostEqual(metrics['success_rate'], 1 / 3)
        self.assertEqual(metrics['conditional_success_given_purchase'], .5)
        self.assertIsNotNone(metrics['success_rate_ci95'])

    def test_repeated_tasks_use_cluster_bootstrap(self):
        rows = [episode(1, 1., repeat=0), episode(1, 0., repeat=1), episode(1, 0., repeat=2)]
        first = evaluation.summarize_episodes(rows)
        second = evaluation.summarize_episodes(rows)
        self.assertIn('cluster bootstrap', first['ci_method'])
        self.assertEqual(first['success_rate_ci95'], second['success_rate_ci95'])
        low, high = first['success_rate_ci95']
        self.assertLessEqual(low, high)

    def test_summarize_command_keeps_purchase_and_success_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_jsonl(root / 'episodes.jsonl', [episode(1, .5)])
            summarize(argparse.Namespace(input=str(path), output=str(root / 'summary.json')))
            summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['success_rate'], 0.0)
            self.assertEqual(summary['purchase_terminated'], 1)
            self.assertEqual(summary['purchase_without_full_reward'], 1)


class FailureTriageTests(unittest.TestCase):
    def test_each_failure_class_is_detected(self):
        format_row = episode(1, 0., done=False, steps=[
            step(0, 'Search', 'search[shoes]'),
            step(1, 'Search', None, grounding=0,
                 error='Expected exactly one Action: search[...] or Action: click[...] line')])
        unavailable = episode(2, 0., done=False, steps=[
            step(0, 'Search', 'click[buy now]', grounding=0,
                 error='Action is not in available actions')])
        retrieval = episode(3, 0., done=False, instruction='red shoes, and price lower than 30.00 dollars',
                            steps=[step(0, 'Search', 'search[garden hose]'),
                                   step(1, 'Search', 'search[lawn mower]')])
        specification = episode(4, .6, steps=[
            step(0, 'Search', 'search[red shoes]'),
            step(1, 'red boots [SEP] Price: $20.00 [SEP] Buy Now', 'click[buy now]', reward=.6)])
        budget = episode(5, .6, steps=[
            step(0, 'Search', 'search[red shoes]'),
            step(1, 'red shoes [SEP] Price: $45.00 [SEP] Buy Now', 'click[buy now]', reward=.6)])
        # WebShop gives reward 0 when a purchase matches no required attribute; it is still a purchase.
        zero_budget = episode(8, 0., steps=[
            step(0, 'Search', 'search[red shoes]'),
            step(1, 'red shoes [SEP] Price: $45.00 [SEP] Buy Now', 'click[buy now]')])
        zero_spec = episode(9, 0., steps=[
            step(0, 'Search', 'search[red shoes]'),
            step(1, 'red boots [SEP] Price: $20.00 [SEP] Buy Now', 'click[buy now]')])
        loop = episode(6, 0., done=False, steps=[step(0, 'Search', 'search[red shoes]'),
                                                 step(1, 'No results', 'search[red shoes]'),
                                                 step(2, 'No results', 'search[red shoes]')])
        engineering = episode(7, 0., done=False, steps=[step(0, 'Search', 'search[red shoes]'),
                                                        step(1, 'Search', None, grounding=0,
                                                             error='connection refused by server')])
        expected = {1: 'format', 2: 'format', 3: 'retrieval', 4: 'specification',
                    5: 'budget', 6: 'loop', 7: 'engineering', 8: 'budget', 9: 'specification'}
        for row in (format_row, unavailable, retrieval, specification, budget, loop, engineering,
                    zero_budget, zero_spec):
            self.assertEqual(evaluation.classify_episode(row)['class'], expected[row['task_id']],
                             msg=f'task {row["task_id"]}')

    def test_purchase_outcome_outranks_format_noise(self):
        mess = episode(10, 0., steps=[
            step(0, 'Search', None, grounding=0,
                 error='Expected exactly one Action: search[...] or Action: click[...] line'),
            step(1, 'red boots [SEP] Price: $20.00 [SEP] Buy Now', 'click[buy now]')])
        verdict = evaluation.classify_episode(mess)
        self.assertEqual(verdict['class'], 'specification')
        self.assertIn('format', verdict['secondary'])
        breakdown = evaluation.failure_breakdown([mess, episode(11, 0., done=False, steps=[
            step(0, 'Search', 'search[red shoes]'),
            step(1, 'Search', None, grounding=0, error='Expected exactly one Action line')])])
        self.assertEqual(breakdown['counts']['specification'], 1)
        self.assertEqual(breakdown['counts']['format'], 1)
        self.assertEqual(breakdown['any_counts']['format'], 2)
        self.assertEqual(breakdown['any_counts']['specification'], 1)

    def test_successful_episode_is_not_classified(self):
        breakdown = evaluation.failure_breakdown([episode(1, 1.)])
        self.assertEqual(breakdown['failed_episodes'], 0)
        self.assertEqual(sum(breakdown['counts'].values()), 0)

    def test_case_sheet_lists_steps_for_each_class(self):
        rows = [episode(4, .6, steps=[step(0, 'Search', 'search[red shoes]'),
                                      step(1, 'red boots [SEP] Price: $20.00 [SEP] Buy Now',
                                           'click[buy now]', reward=.6)]),
                episode(5, .6, steps=[step(0, 'Search', 'search[red shoes]'),
                                      step(1, 'red shoes [SEP] Price: $45.00 [SEP] Buy Now',
                                           'click[buy now]', reward=.6)])]
        breakdown = evaluation.failure_breakdown(rows, per_class=1)
        self.assertEqual({case['class'] for case in breakdown['cases']},
                         {'specification', 'budget'})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cases.md'
            evaluation.write_case_report(breakdown['cases'], path, 'fixture.jsonl')
            text = path.read_text(encoding='utf-8')
            self.assertIn('人工复核', text)
            self.assertIn('task 4', text)
            self.assertIn('task 5', text)


class ParityTests(unittest.TestCase):
    def test_identical_arms_only_differing_by_adapter_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = dict(CONFIG, adapter=None)
            b = dict(CONFIG, adapter='artifacts/sft/adapter')
            left = write_json(root / 'b0.meta.json', meta([1, 2], config=a))
            right = write_json(root / 'b1.meta.json', meta([1, 2], config=b))
            result = evaluation.check_parity(a_meta=str(left), b_meta=str(right))
            self.assertTrue(result['parity'], msg=result['blocking_issues'])
            self.assertTrue(result['same_task_set'])
            self.assertNotEqual(result['adapters']['a'], result['adapters']['b'])

    def test_context_or_task_drift_breaks_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = write_json(root / 'b0.meta.json', meta([1, 2], config=dict(CONFIG, context_length=1024)))
            right = write_json(root / 'b1.meta.json', meta([1, 2]))
            result = evaluation.check_parity(a_meta=str(left), b_meta=str(right))
            self.assertFalse(result['parity'])
            self.assertTrue(any('context_length' in issue for issue in result['blocking_issues']))
            other = write_json(root / 'b1-other.meta.json', meta([1, 3]))
            result = evaluation.check_parity(a_meta=str(left), b_meta=str(other))
            self.assertFalse(result['same_task_set'])

    def test_unfrozen_fingerprint_blocks_final_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = write_json(root / 'b0.meta.json', meta([1], fingerprint=''))
            right = write_json(root / 'b1.meta.json', meta([1], fingerprint=''))
            result = evaluation.check_parity(a_meta=str(left), b_meta=str(right), require_frozen=True)
            self.assertFalse(result['parity'])
            self.assertTrue(any('goal_fingerprint' in issue for issue in result['blocking_issues']))


class RegisterTests(unittest.TestCase):
    def _row(self, **overrides):
        row = {
            'run_id': 'run-1', 'status': 'complete', 'method': 'B0', 'model': CONFIG['model'],
            'seed': '42', 'split_sha256': 'c' * 64, 'goal_fingerprint': FINGERPRINT,
            'git_commit': '0123456789abcdef0123456789abcdef01234567',
            'config_path': 'configs/eval-base.json', 'config_sha256': CONFIG_SHA,
            'episodes': '10', 'successes': '3', 'purchase_terminated': '5', 'success_rate': '0.300000',
            'success_rate_ci95': '[0.100000, 0.600000]', 'mean_reward': '0.400000', 'mean_steps': '6.50',
            'invalid_action_rate': '0.200000', 'peak_vram_gb': '7.50',
            'evidence_path': 'artifacts/b0-val.jsonl', 'notes': 'B0 run',
        }
        row.update(overrides)
        return row

    def test_header_only_register_is_valid(self):
        result = register.validate(list(register.COLUMNS), [], check_git=False)
        self.assertEqual(result['status'], 'ok')

    def test_header_mismatch_is_rejected(self):
        result = register.validate(['run_id', 'status'], [], check_git=False)
        self.assertEqual(result['status'], 'invalid')
        self.assertTrue(any('Header must be' in error for error in result['errors']))

    def test_complete_row_needs_consistent_counts(self):
        result = register.validate(list(register.COLUMNS), [self._row()], check_git=False)
        self.assertEqual(result['status'], 'ok', msg=result['errors'])
        bad_rate = register.validate(list(register.COLUMNS),
                                     [self._row(success_rate='0.500000', success_rate_ci95='[0.4, 0.6]')],
                                     check_git=False)
        self.assertEqual(bad_rate['status'], 'invalid')
        self.assertTrue(any('success_rate' in error for error in bad_rate['errors']))
        too_many = register.validate(list(register.COLUMNS),
                                     [self._row(successes='6')], check_git=False)
        self.assertTrue(any('purchase_terminated' in error for error in too_many['errors']))

    def test_complete_row_needs_evidence_columns(self):
        result = register.validate(list(register.COLUMNS),
                                   [self._row(split_sha256='', goal_fingerprint='',
                                              config_sha256='', evidence_path='', git_commit='')],
                                   check_git=False)
        self.assertEqual(result['status'], 'invalid')
        joined = ' '.join(result['errors'])
        for key in ('split_sha256', 'goal_fingerprint', 'config_sha256', 'evidence_path', 'git_commit'):
            self.assertIn(key, joined)

    def test_config_hash_is_verified_when_file_exists(self):
        config_path = Path(__file__).resolve().parents[1] / 'configs/eval-base.json'
        row = self._row(config_sha256=file_hash(config_path))
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(register.validate(list(register.COLUMNS), [row], root, check_git=False)['status'], 'ok')
        row['config_sha256'] = 'e' * 64
        result = register.validate(list(register.COLUMNS), [row], root, check_git=False)
        self.assertEqual(result['status'], 'invalid')
        self.assertTrue(any('config_sha256 does not match' in error for error in result['errors']))

    def test_planned_rows_may_omit_results(self):
        row = {'run_id': 'later', 'status': 'planned', 'method': 'B1', 'model': CONFIG['model'],
               'seed': '42', 'split_sha256': '', 'goal_fingerprint': '', 'git_commit': '',
               'config_path': '', 'config_sha256': '', 'episodes': '', 'successes': '',
               'purchase_terminated': '', 'success_rate': '', 'success_rate_ci95': '',
               'mean_reward': '', 'mean_steps': '', 'invalid_action_rate': '',
               'peak_vram_gb': '', 'evidence_path': '', 'notes': 'queued'}
        result = register.validate(list(register.COLUMNS), [row], check_git=False)
        self.assertEqual(result['status'], 'ok', msg=result['errors'])


class EndToEndEvaluationTests(unittest.TestCase):
    def _run(self, directory, rows, **overrides):
        root = Path(directory)
        config_path = write_json(root / 'eval-config.json', CONFIG)
        jsonl = write_jsonl(root / 'b0-val.jsonl', rows)
        meta_path = write_json(root / 'b0-val.meta.json',
                               meta([row['task_id'] for row in rows],
                                    config_sha=file_hash(config_path)))
        split_ids = write_json(root / 'val_ids.json', [row['task_id'] for row in rows])
        args = argparse.Namespace(
            meta=str(meta_path), jsonl=str(jsonl), config=str(config_path),
            output=str(root / 'evaluation.json'), cases=str(root / 'cases.md'),
            figures_dir=str(root / 'figures'), allow_partial=False, cases_per_class=1,
            method='B0', split_ids=str(split_ids), register_row=str(root / 'row.csv'),
            peak_vram_gb=7.5)
        for key, value in overrides.items():
            setattr(args, key, value)
        return args, root

    def test_evaluate_writes_metrics_cases_figures_and_register_row(self):
        with tempfile.TemporaryDirectory() as directory:
            args, root = self._run(directory, [episode(1, 1.), episode(2, .6), episode(3, 0., done=False)])
            evaluation.evaluate(args)
            result = json.loads((root / 'evaluation.json').read_text(encoding='utf-8'))
            self.assertEqual(result['completeness']['status'], 'complete')
            self.assertEqual(result['metrics']['purchase_terminated'], 2)
            self.assertEqual(result['metrics']['full_reward_successes'], 1)
            self.assertTrue((root / 'cases.md').is_file())
            self.assertTrue((root / 'figures' / 'B0-outcomes.svg').is_file())
            fieldnames, rows = register.read_register(root / 'row.csv')
            self.assertEqual(fieldnames, list(register.COLUMNS))
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row['method'], 'B0')
            self.assertEqual(row['purchase_terminated'], '2')
            self.assertEqual(row['successes'], '1')
            checked = register.validate(fieldnames, rows, root, check_git=False)
            self.assertEqual(checked['status'], 'ok', msg=checked['errors'])

    def test_evaluate_refuses_incomplete_run_without_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = write_json(root / 'eval-config.json', CONFIG)
            jsonl = write_jsonl(root / 'b0-val.jsonl', [episode(1, 1.)])
            meta_path = write_json(root / 'b0-val.meta.json',
                                   meta([1, 2], config_sha=file_hash(config_path)))
            args = argparse.Namespace(meta=str(meta_path), jsonl=str(jsonl), config=str(config_path),
                                      output=str(root / 'evaluation.json'), cases=None, figures_dir=None,
                                      allow_partial=False, cases_per_class=1, method='B0',
                                      split_ids=None, register_row=None, peak_vram_gb=None)
            with self.assertRaises(SystemExit):
                evaluation.evaluate(args)
            args.allow_partial = True
            evaluation.evaluate(args)
            result = json.loads((root / 'evaluation.json').read_text(encoding='utf-8'))
            self.assertEqual(result['completeness']['status'], 'partial')
            self.assertTrue(any('Partial run' in warning for warning in result['warnings']))
            with self.assertRaises(SystemExit):
                evaluation.evaluate(argparse.Namespace(**{**vars(args), 'allow_partial': True,
                                                          'register_row': str(root / 'row.csv')}))
            # An extra task is a mismatched run, which --allow-partial must never accept.
            write_jsonl(jsonl, [episode(1, 1.), episode(3, 1.)])
            with self.assertRaises(SystemExit):
                evaluation.evaluate(argparse.Namespace(**{**vars(args), 'allow_partial': True,
                                                          'cases': None, 'figures_dir': None}))


class FigureTests(unittest.TestCase):
    def test_bar_chart_is_svg_with_labels_and_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'chart.svg'
            figures.bar_chart([('B0', .2), ('B1', .35)], path, 'title', 'rate',
                              ymax=1.0, intervals=[(.1, .3), (.2, .5)])
            text = path.read_text(encoding='utf-8')
            self.assertTrue(text.startswith('<svg'))
            self.assertIn('B0', text)
            self.assertIn('0.350', text)
            self.assertIn('line', text)

    def test_run_figures_and_method_figure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = evaluation.summarize_episodes([episode(1, 1.), episode(2, .6)])
            counts = evaluation.failure_breakdown([episode(1, 1.), episode(2, .6)])['counts']
            paths = figures.write_run_figures(metrics, counts, root / 'figures', label='B1')
            self.assertEqual(len(paths), 2)
            self.assertTrue(all(path.is_file() for path in paths))
            rows = [{'status': 'complete', 'method': 'B0', 'success_rate': '0.25',
                     'success_rate_ci95': '[0.1, 0.4]', 'episodes': '8'},
                    {'status': 'complete', 'method': 'B1', 'success_rate': '0.50',
                     'success_rate_ci95': '[0.3, 0.7]', 'episodes': '8'}]
            path = figures.method_figure(rows, root / 'methods.svg')
            text = path.read_text(encoding='utf-8')
            self.assertIn('B0', text)
            self.assertIn('B1', text)
            with self.assertRaises(ValueError):
                figures.method_figure([{'status': 'planned', 'method': 'B0'}], root / 'empty.svg')


if __name__ == '__main__':
    unittest.main()
