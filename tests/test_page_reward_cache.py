import unittest
from types import SimpleNamespace
from webshop_lab import environment


class PageRewardCacheTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        def score(product, target):
            self.calls.append((product['name'], target['name']))
            return {'r_type': len(product['name']) / 10}
        self.goal = SimpleNamespace(get_type_reward=score)
        self.original = score
        self.product = {'name': 'blue shirt'}
        self.target = {'name': 'requested shirt'}

    def wrap(self, page):
        return getattr(environment, 'cached_type_score_page', lambda p, g: p)(page, self.goal)

    def test_repeated_combinations_reuse_type_score_without_changing_results(self):
        def page(_):
            return [self.goal.get_type_reward(self.product, self.target) for i in range(7)]
        result = self.wrap(page)(None)
        self.assertEqual(result, [{'r_type': 1.0}] * 7)
        self.assertEqual(len(self.calls), 1)
        self.assertIs(self.goal.get_type_reward, self.original)

    def test_cache_does_not_survive_page_render(self):
        def page(_):
            for i in range(3):
                self.goal.get_type_reward(self.product, self.target)
        wrapped = self.wrap(page)
        wrapped(None)
        wrapped(None)
        self.assertEqual(len(self.calls), 2)

    def test_cache_distinguishes_product_and_target_identity(self):
        other_product = dict(self.product)
        other_target = dict(self.target)
        def page(_):
            for i in range(3):
                self.goal.get_type_reward(self.product, self.target)
                self.goal.get_type_reward(other_product, self.target)
                self.goal.get_type_reward(self.product, other_target)
        self.wrap(page)(None)
        self.assertEqual(len(self.calls), 3)

    def test_mutating_a_return_value_does_not_change_later_scores(self):
        def page(_):
            value = self.goal.get_type_reward(self.product, self.target)
            value['r_type'] = -100
            return self.goal.get_type_reward(self.product, self.target)
        self.assertEqual(self.wrap(page)(None), {'r_type': 1.0})

    def test_failure_restores_original_scorer(self):
        def page(_):
            self.goal.get_type_reward(self.product, self.target)
            raise RuntimeError('page failed')
        with self.assertRaisesRegex(RuntimeError, 'page failed'):
            self.wrap(page)(None)
        self.assertIs(self.goal.get_type_reward, self.original)
