"""CPU negative checks; no model weights or GPU runtime are required."""
import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from prepare import HERE, require, sha
from run_smoke import runtime_provenance, validate_sources


class HandoffChecks(unittest.TestCase):
    def run_python(self, program, optimized=False):
        return subprocess.run([sys.executable, *(['-O'] if optimized else []), '-c', program],
                              cwd=HERE, capture_output=True, text=True)

    def test_no_assert_statements_remain_in_execution_scripts(self):
        for name in ['prepare.py', 'check_subset.py', 'run_smoke.py']:
            tree = ast.parse((HERE/name).read_text())
            self.assertFalse(any(isinstance(node, ast.Assert) for node in ast.walk(tree)), name)

    def test_require_rejects_in_both_interpreter_modes(self):
        for optimized in [False, True]:
            failed = self.run_python("from prepare import require; require(False, 'deliberate failure')", optimized)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn('deliberate failure', failed.stderr)
            passed = self.run_python("from prepare import require; require(True, 'unused')", optimized)
            self.assertEqual(passed.returncode, 0, passed.stderr)

    def test_source_hash_gate_rejects_tampering_under_optimization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p = root/'source.py'
            p.write_text('original')
            expected = sha(p)
            validate_sources(root, {'source_files_sha256': {'source.py': expected}})
            p.write_text('tampered')
            program = ("from pathlib import Path; from run_smoke import validate_sources; "
                       f"validate_sources(Path({str(root)!r}), {{'source_files_sha256': {{'source.py': {expected!r}}}}})")
            result = self.run_python(program, optimized=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Implementation/config changed', result.stderr)

    def test_cli_help_does_not_import_gpu_dependencies(self):
        for optimized in [False, True]:
            result = subprocess.run([sys.executable, *(['-O'] if optimized else []),
                str(HERE/'run_smoke.py'), '--help'], cwd=HERE.parent, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--source-commit', result.stdout)

    def test_invalid_declared_commit_is_rejected_before_gpu_import(self):
        result = subprocess.run([sys.executable, '-O', str(HERE/'run_smoke.py'), 'run',
            '--repo', '.', '--model-path', '.', '--source-commit', 'main'],
            cwd=HERE, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('full lowercase Git commit SHA', result.stderr)

    def test_provenance_separates_declared_commit_from_observed_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['src/webshop_lab/model.py', 'src/webshop_lab/common.py',
                         'src/webshop_lab/data.py', 'src/webshop_lab/cli.py', 'configs/sft-qwen15b.json']:
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('test fixture')
            torch = SimpleNamespace(version=SimpleNamespace(cuda='test-cuda'),
                backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 123)))
            declared = 'a'*40
            def git_result(command, **kwargs):
                if 'rev-parse' in command:
                    return subprocess.CompletedProcess(command, 0, 'b'*40+'\n', '')
                return subprocess.CompletedProcess(command, 1, '', '')
            with patch('run_smoke.subprocess.run', side_effect=git_result):
                info = runtime_provenance(root, torch, declared)
            self.assertEqual(info['source']['repo_git_head'], 'b'*40)
            self.assertEqual(info['source']['declared_commit'], declared)
            self.assertFalse(info['source']['declared_matches_git_head'])
            self.assertTrue(info['source']['repo_tracked_dirty'])
            self.assertEqual(len(info['source']['source_files_sha256']), 5)
            self.assertEqual(len(info['source']['handoff_files_sha256']), 4)
            self.assertEqual(set(info['package_versions']),
                             {'torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes', 'tokenizers'})
            json.dumps(info)

    def test_missing_git_metadata_is_unknown_not_an_inferred_commit(self):
        with patch('run_smoke.subprocess.run', side_effect=FileNotFoundError), \
             patch('run_smoke.sha', return_value='f'*64):
            torch = SimpleNamespace(version=SimpleNamespace(cuda=None),
                backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: None)))
            source = runtime_provenance(Path('.'), torch, 'a'*40)['source']
        self.assertIsNone(source['repo_git_head'])
        self.assertIsNone(source['repo_tracked_dirty'])
        self.assertIsNone(source['declared_matches_git_head'])


if __name__ == '__main__':
    unittest.main()
