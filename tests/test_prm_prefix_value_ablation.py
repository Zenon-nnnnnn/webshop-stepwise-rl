"""Unit tests for Prefix-Value Telescoping PRM helper logic and archived 5-arm recompute verification."""
import importlib.util
from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PrefixValuePRMAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recompute = _load_module(
            "prm_pv_recompute",
            REPO_ROOT / "results/prm-prefix-value-20261126/recompute.py",
        )
        try:
            import torch  # noqa: F401
            cls.ablation = _load_module(
                "run_prefix_value_prm_ablation",
                REPO_ROOT / "scripts/run_prefix_value_prm_ablation.py",
            )
        except ImportError:
            cls.ablation = None

    def test_committed_five_arm_evidence_recomputes_cleanly(self):
        report = self.recompute.verify_all(REPO_ROOT, bundle_path=None)
        arms = report["arms"]
        self.assertEqual(arms["arm1_pv_qv_fit501"]["dev32_correct"], "22/32")
        self.assertEqual(arms["arm2_pv_alllinear_fit501"]["dev32_correct"], "22/32")
        self.assertEqual(arms["arm2_pv_alllinear_fit501"]["by_visibility"]["V+"], "19/21")
        self.assertGreater(arms["arm2_pv_alllinear_fit501"]["r7544_click_gap"], 0.02)
        self.assertEqual(arms["arm4_pv_asin_context_fit501"]["dev32_correct"], "26/32")
        self.assertEqual(arms["arm4_pv_asin_context_fit501"]["by_visibility"]["V+"], "20/21")
        self.assertGreater(arms["arm4_pv_asin_context_fit501"]["r7544_click_gap"], 0.11)
        self.assertLess(arms["arm4_pv_asin_context_fit501"]["dev160_mse"], 0.082)

    def test_public_asin_lookup_uses_only_immediately_preceding_user_page(self):
        if self.ablation is None:
            self.skipTest("Requires torch environment")
        messages = [
            {"role": "system", "content": "Shop carefully."},
            {
                "role": "user",
                "content": "Instruction: [SEP] running shoes [SEP] B07YNGTFK3 [SEP] Women Running Shoes [SEP] $49.99 [SEP] B08CLOGS00 [SEP] Nordways Clogs [SEP] $19.99",
            },
            {"role": "assistant", "content": "Action: click[b07yngtfk3]"},
            {"role": "user", "content": "Product page [SEP] B09FUTURE1 [SEP] Secret Future Item [SEP] $9.99"},
            {"role": "assistant", "content": "Action: click[buy now]"},
        ]
        resolved = self.ablation.resolve_public_asin_from_prev_page(messages, 2)
        self.assertIn("B07YNGTFK3", resolved)
        self.assertIn("Women Running Shoes", resolved)
        self.assertIn("$49.99", resolved)
        self.assertNotIn("Nordways Clogs", resolved)
        # Non-ASIN action returns None
        self.assertIsNone(self.ablation.resolve_public_asin_from_prev_page(messages, 4))
        # ASIN appearing ONLY in a future message (index 3) cannot be resolved at index 2
        messages_future_only = [
            messages[0],
            messages[1],
            {"role": "assistant", "content": "Action: click[b09future1]"},
            messages[3],
        ]
        self.assertIsNone(self.ablation.resolve_public_asin_from_prev_page(messages_future_only, 2))


if __name__ == "__main__":
    unittest.main()
