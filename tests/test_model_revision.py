"""CPU checks for version propagation and provenance; these do not execute CUDA."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from webshop_lab import model, sft

REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'


class ModelRevisionTests(unittest.TestCase):
    def fake_runtime(self, resolved=REVISION):
        tok = SimpleNamespace(pad_token_id=0, init_kwargs={'_commit_hash': resolved})
        base = SimpleNamespace(config=SimpleNamespace(_commit_hash=resolved))
        tokenizer_loader = Mock(return_value=tok)
        model_loader = Mock(return_value=base)
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True,
            is_bf16_supported=lambda: True), bfloat16='bf16', float16='fp16')
        transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=model_loader), BitsAndBytesConfig=Mock())
        peft = SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=Mock(return_value=base)),
                              prepare_model_for_kbit_training=Mock(return_value=base))
        return tok, base, tokenizer_loader, model_loader, {'torch': torch, 'transformers': transformers, 'peft': peft}

    def test_exact_revision_reaches_both_loaders_and_quantized_adapter_path(self):
        tok, base, tok_load, base_load, modules = self.fake_runtime()
        with patch.dict('sys.modules', modules):
            model.load({'model': 'Qwen/model', 'revision': REVISION,
                        'load_in_4bit': True, 'adapter': 'saved/adapter'})
        self.assertEqual(tok_load.call_args.kwargs['revision'], REVISION)
        self.assertEqual(base_load.call_args.kwargs['revision'], REVISION)
        self.assertIn('quantization_config', base_load.call_args.kwargs)
        modules['peft'].PeftModel.from_pretrained.assert_called_once_with(base, 'saved/adapter')

    def test_legacy_config_does_not_force_a_revision(self):
        _, _, tok_load, base_load, modules = self.fake_runtime()
        with patch.dict('sys.modules', modules):
            model.load({'model': 'legacy/model'})
        self.assertNotIn('revision', tok_load.call_args.kwargs)
        self.assertNotIn('revision', base_load.call_args.kwargs)

    def test_wrong_resolved_commit_is_rejected_before_training(self):
        _, _, _, _, modules = self.fake_runtime('f'*40)
        with patch.dict('sys.modules', modules), self.assertRaises(ValueError):
            model.load({'model': 'Qwen/model', 'revision': REVISION}, training=True)
        modules['peft'].prepare_model_for_kbit_training.assert_not_called()

    def test_tokenizer_model_commit_disagreement_is_rejected(self):
        tok, base, _, _, _ = self.fake_runtime()
        tok.init_kwargs['_commit_hash'] = 'a'*40
        with self.assertRaises(ValueError):
            model.model_source({'model': 'Qwen/model'}, base, tok)

    def test_local_source_does_not_claim_requested_commit_was_resolved(self):
        tok, base, _, _, _ = self.fake_runtime(None)
        info = model.model_source({'model': '/local/model', 'revision': REVISION}, base, tok)
        self.assertEqual(info['requested_revision'], REVISION)
        self.assertIsNone(info['resolved_model_revision'])
        self.assertIsNone(info['resolved_tokenizer_revision'])
        json.dumps(info)

    def test_branch_name_is_recorded_separately_from_resolved_commit(self):
        tok, base, _, _, _ = self.fake_runtime()
        info = model.model_source({'model': 'Qwen/model', 'revision': 'main'}, base, tok)
        self.assertEqual(info['requested_revision'], 'main')
        self.assertEqual(info['resolved_model_revision'], REVISION)

    def test_formal_draft_refuses_execution_before_gpu_import_or_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'draft.json'
            out = Path(directory)/'output'
            path.write_text(json.dumps({'configuration_status': 'draft', 'output_dir': str(out)}))
            with self.assertRaisesRegex(ValueError, 'Draft SFT configuration'):
                sft.run(argparse.Namespace(config=str(path), max_steps=-1))
            self.assertFalse(out.exists())
