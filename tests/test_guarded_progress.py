import unittest

from webshop_lab.progress import trajectory_features


class GuardedProgressTests(unittest.TestCase):
    def test_invalid_repeated_search_and_reverse_pagination_get_no_features(self):
        def step(action, grounding=1):
            return {"action": action, "grounding": grounding, "observation": "page", "response": action}

        steps = [step("search[a]", 0), step("search[red shoes]"), step("search[RED  SHOES]"),
                 step("click[next >]"), step("click[< prev]"), step("click[item]")]
        vectors = trajectory_features(steps, 8, "hashed-linear-guarded-v2")
        self.assertEqual([bool(v) for v in vectors], [False, True, False, True, False, True])
        self.assertEqual([v.get(0, 0.0) for v in vectors], [0, 1, 0, 0, 0, 0])

    def test_legacy_features_retain_per_step_bias(self):
        step = {"observation": "page", "response": "search", "grounding": 0}
        self.assertEqual(trajectory_features([step], 8)[0][0], 1.0)
