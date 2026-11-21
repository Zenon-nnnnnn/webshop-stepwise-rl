import math
import unittest

from webshop_lab.ppo import episode_targets, replay_ratio


class PPOMathTests(unittest.TestCase):
    def test_terminal_return_does_not_include_page_scores(self):
        self.assertEqual(episode_targets([0.125, 0.25], False, 0.75),
                         (0.0, [-0.125, -0.25], [0.0, 0.0]))
        self.assertEqual(episode_targets([0.125, 0.25], True, 0.75),
                         (0.75, [0.625, 0.5], [0.75, 0.75]))

    def test_replay_rejects_accumulating_token_error(self):
        self.assertAlmostEqual(replay_ratio([-1.0], [-0.99], 0.05), math.exp(0.01))
        with self.assertRaisesRegex(ValueError, "joint"):
            replay_ratio([-1.0] * 128, [-0.999] * 128, 0.05)
        for invalid in ([], [float("nan")], [float("inf")]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                replay_ratio([-1.0], invalid, 0.05)
