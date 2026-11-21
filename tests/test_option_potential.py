import unittest

from webshop_lab.ppo import assign_episode_rewards, option_potential


class OptionPotentialTests(unittest.TestCase):
    def test_goal_match_and_unselected_options(self):
        goal = "Find a stone grey pack of 2, and price lower than 40 dollars"
        grey = option_potential(goal, {"color": "stone grey pack of 2"})
        red = option_potential(goal, {"color": "red pack of 2"})
        self.assertGreater(grey, red)
        self.assertEqual(option_potential(goal, {}), 0)
        self.assertLessEqual(grey, 0.25)

    def test_loop_and_end_do_not_mint_reward(self):
        turns = [{"old_value": 0.0, "potential_before": a, "potential_after": b}
                 for a, b in ((0.0, 0.25), (0.25, 0.0), (0.0, 0.25), (0.25, 0.25))]
        self.assertEqual(assign_episode_rewards(turns, True, 0.75, shaping=True), 0.75)
        self.assertEqual([t["reward"] for t in turns], [0.25, -0.25, 0.25, 0.5])
        self.assertEqual([t["return_target"] for t in turns], [0.75, 0.5, 0.75, 0.5])
