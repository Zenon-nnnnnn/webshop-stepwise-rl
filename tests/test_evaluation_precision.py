"""CPU regressions for explicit evaluation precision, without downloading a model."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import test_model_revision
from webshop_lab import model


class EvaluationPrecisionTests(unittest.TestCase):
    def runtime(self):
        result = test_model_revision.ModelRevisionTests().fake_runtime()
        result[-1]["torch"].float32 = "fp32"
        return result

    def test_nonquantized_float32_config_reaches_model_loader(self):
        _, _, _, loader, modules = self.runtime()
        with patch.dict("sys.modules", modules):
            _, _, dtype = model.load({"model": "Qwen/model", "load_in_4bit": False,
                                      "policy_dtype": "float32"})
        self.assertEqual(loader.call_args.kwargs["torch_dtype"], "fp32")
        self.assertEqual(dtype, "fp32")

    def test_invalid_precision_fails_before_loading(self):
        _, _, _, loader, modules = self.runtime()
        with patch.dict("sys.modules", modules), self.assertRaises(ValueError):
            model.load({"model": "Qwen/model", "policy_dtype": "float3"})
        loader.assert_not_called()

    def test_explicit_float32_conflict_fails_before_loading(self):
        _, _, _, loader, modules = self.runtime()
        with patch.dict("sys.modules", modules), self.assertRaises(ValueError):
            model.load({"model": "Qwen/model", "policy_dtype": "float32"}, dtype="bf16")
        loader.assert_not_called()

    def test_quantization_does_not_satisfy_float32_contract(self):
        _, _, _, loader, modules = self.runtime()
        with patch.dict("sys.modules", modules), self.assertRaises(ValueError):
            model.load({"model": "Qwen/model", "policy_dtype": "float32",
                        "load_in_4bit": True})
        loader.assert_not_called()

    def test_legacy_nonquantized_config_remains_auto(self):
        _, _, _, loader, modules = self.runtime()
        with patch.dict("sys.modules", modules):
            model.load({"model": "Qwen/model", "load_in_4bit": False})
        self.assertEqual(loader.call_args.kwargs["torch_dtype"], "bf16")

    def test_precision_record_preserves_mixed_base_and_adapter_dtypes(self):
        base = SimpleNamespace(dtype="torch.bfloat16", numel=lambda: 100,
                               requires_grad=False)
        adapter = SimpleNamespace(dtype="torch.float32", numel=lambda: 4,
                                  requires_grad=True)
        fake = SimpleNamespace(named_parameters=lambda: iter([("base", base),
                                                              ("lora", adapter)]))
        result = model.precision_record(fake, "torch.bfloat16")
        self.assertEqual(result["selected_load_dtype"], "torch.bfloat16")
        self.assertEqual(result["parameter_numel_by_dtype"],
                         {"torch.bfloat16": 100, "torch.float32": 4})
        self.assertEqual(result["trainable_numel_by_dtype"], {"torch.float32": 4})
