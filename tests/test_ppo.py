import math
import unittest

from webshop_lab.ppo import assign_episode_rewards, episode_targets, replay_ratio


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


    def test_progress_rewards_are_paid_once_per_turn_and_use_suffix_returns(self):
        turns = [
            {"observation": "home", "response_text_parsed": "search", "grounding": 1, "old_value": 0.125,
             "response_token_ids": [1, 2, 3]},
            {"observation": "item", "response_text_parsed": "invalid", "grounding": 0, "old_value": 0.25,
             "response_token_ids": [7]},
        ]
        model = {"dimensions": 2, "weights": [0.25, 0.0]}
        self.assertEqual(assign_episode_rewards(turns, False, 0.9, model), 0.0)
        self.assertEqual([t["reward"] for t in turns], [0.25, 0.25])
        self.assertEqual([t["return_target"] for t in turns], [0.5, 0.25])
        self.assertEqual([t["advantage"] for t in turns], [0.375, 0.0])
        self.assertEqual([t["continuation_mask"] for t in turns], [1, 0])

    def test_terminal_mode_preserves_purchase_reward(self):
        turns = [{"old_value": 0.125}, {"old_value": 0.25}]
        self.assertEqual(assign_episode_rewards(turns, True, 0.75), 0.75)
        self.assertEqual([t["reward"] for t in turns], [0.0, 0.75])
        self.assertEqual([t["return_target"] for t in turns], [0.75, 0.75])
