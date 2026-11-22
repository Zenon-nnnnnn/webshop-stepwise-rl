import unittest

from webshop_lab.ppo_comparison import arm_configs, common_budget


class PPOComparisonTests(unittest.TestCase):
    def test_pair_differs_only_in_reward_inputs(self):
        base = {"adapter": "local", "seed": 42, "max_steps": 8, "max_new_tokens": 64}
        terminal, candidate = arm_configs(base, "option_potential", None, None)
        self.assertEqual(terminal, dict(base, reward_mode="terminal"))
        self.assertEqual(candidate, dict(base, reward_mode="option_potential"))
        with self.assertRaisesRegex(ValueError, "reward"):
            arm_configs(dict(base, beta=0.1), "option_potential", None, None)

    def test_limits_count_invalid_decisions_and_reset_per_round(self):
        budget = common_budget([[1, 2], [3]], max_steps=8, max_new_tokens=64)
        self.assertEqual(budget, {"updates": 2, "episodes": 3, "policy_turn_cap": 24,
                                  "environment_call_cap": 24, "generated_token_cap": 1536})
