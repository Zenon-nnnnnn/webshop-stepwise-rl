"""The hand-check worksheet, asserted against the code.

results/hand-check-metrics.md shows the arithmetic for four synthetic episodes; this test fails if
the metric definitions drift away from that worksheet.
"""
import unittest
from pathlib import Path

from webshop_lab import evaluation
from webshop_lab.common import wilson

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / 'tests/fixtures/hand-check-episodes.jsonl'


class HandCheckTests(unittest.TestCase):
    def setUp(self):
        self.rows = evaluation.load_episodes(FIXTURE)
        self.metrics = evaluation.summarize_episodes(self.rows)

    def test_hand_arithmetic(self):
        metrics = self.metrics
        self.assertEqual(metrics['episodes'], 4)
        self.assertEqual(metrics['purchase_terminated'], 2)
        self.assertEqual(metrics['full_reward_successes'], 1)
        self.assertEqual(metrics['purchase_without_full_reward'], 1)
        self.assertEqual(metrics['max_steps_terminated'], 2)
        self.assertEqual(metrics['success_rate'], 0.25)                      # 1 / 4
        self.assertEqual(metrics['mean_final_reward'], 0.4)                  # 1.6 / 4
        self.assertEqual(metrics['mean_steps'], 2.25)                        # 9 / 4
        self.assertAlmostEqual(metrics['invalid_action_rate'], 2 / 9)        # 2 invalid of 9 steps
        self.assertAlmostEqual(metrics['grounding_accuracy'], 7 / 9)
        self.assertEqual(metrics['episodes_with_invalid_actions'], 2)
        self.assertEqual(metrics['conditional_success_given_purchase'], 0.5)  # 1 / 2
        self.assertEqual(metrics['termination_counts'], {'purchase': 2, 'max_steps': 2})
        self.assertEqual([round(value, 6) for value in metrics['success_rate_ci95']],
                         [round(value, 6) for value in wilson(1, 4)])

    def test_hand_checked_failure_labels(self):
        breakdown = evaluation.failure_breakdown(self.rows)
        self.assertEqual(breakdown['counts']['specification'], 1)   # task 102: right item, wrong size
        self.assertEqual(breakdown['counts']['format'], 2)          # task 103 unparsable, 104 non-executable
        self.assertEqual(breakdown['failed_episodes'], 3)
        self.assertEqual(breakdown['any_counts']['loop'], 1)        # task 103 repeats search[red running shoes]


if __name__ == '__main__':
    unittest.main()
