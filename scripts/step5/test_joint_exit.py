"""CPU tests for Step5B joint policies and matched-budget controls."""
import unittest
import numpy as np
from joint_exit import depths_from_mask, policy_masks, random_subset


class JointExitTests(unittest.TestCase):
    def test_only_loop3_can_exit(self):
        depths = depths_from_mask(np.array([True, False, True]))
        np.testing.assert_array_equal(depths, [3, 4, 3])

    def test_joint_conditions_are_conjunctive(self):
        features = ({'raw': {'confidence': np.array([.9, .9, .7]), 'margin': np.zeros(3)}},
                    np.array([True, False, True]), np.array([.001, .03, .001]), np.zeros(3))
        np.testing.assert_array_equal(policy_masks(features, .8), [True, True, False])
        np.testing.assert_array_equal(policy_masks(features, .8, True), [True, False, False])
        np.testing.assert_array_equal(policy_masks(features, .8, js_threshold=.01), [True, False, False])

    def test_random_matches_budget_and_is_seeded(self):
        indices = np.arange(10)
        first = random_subset(indices, 3, 17)
        second = random_subset(indices, 3, 17)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual(len(set(first)), 3)


if __name__ == '__main__':
    unittest.main()
