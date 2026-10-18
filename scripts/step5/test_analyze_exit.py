"""CPU semantic tests for Step5B regret and causal signal diagnostics."""
import unittest
import numpy as np
from analyze_exit import quantities, causal_signals, oracle, sorted_members


class DiagnosticTests(unittest.TestCase):
    def test_regret_signs_and_soft_labels(self):
        rows = [{'target': [1., 0.]}, {'target': [1., 0.]}, {'target': [.5, .5]}]
        logits = [[np.array([0., 1.]), np.array([1., 0.]), np.array([0., 1.])]] * 3
        logits = logits + [[np.array([1., 0.]), np.array([0., 1.]), np.array([0., 0.])]]
        values = quantities(logits, rows, [1.] * 4)
        np.testing.assert_array_equal(values['hard'], [True, True, False])
        np.testing.assert_array_equal(values['accuracy_regret'], [1., -1., 0.])
        self.assertGreater(values['nll_regret'][0], 0.)
        self.assertLess(values['nll_regret'][1], 0.)
        self.assertGreater(values['nll_regret'][2], 0.)

    def test_future_independence_and_candidate_permutation(self):
        logits = [[np.array([loop, 2. * loop, -loop])] for loop in [1., 2., 3., 4.]]
        gates = np.arange(12).reshape(3, 4) / 10.
        original = causal_signals(logits, [1.] * 4, gates, [0, 3])
        changed = [[scores[[2, 0, 1]] for scores in loop] for loop in logits]
        changed[3] = [np.array([100., -100., 0.])]
        new_gates = gates[[2, 0, 1]].copy()
        new_gates[:, 3] = -100.
        actual = causal_signals(changed, [1.] * 4, new_gates, [0, 3])
        for name in original:
            np.testing.assert_allclose(actual[name], original[name], atol=1e-12)

    def test_oracle_point_constraints_and_soft_denominator(self):
        raw = {'hard': np.array([True, True, False]),
               'accuracy_regret': np.array([1., -1., 0.]),
               'nll_regret': np.array([.1, -.1, .01]),
               'correct': np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=bool)}
        result = oracle(np.arange(3), raw, raw)
        self.assertEqual(result['point_estimate_budget_upper_bound']['exit_rows'], 3)
        self.assertEqual(result['strict_per_row_nonworsening']['exit_rows'], 1)
        self.assertAlmostEqual(result['point_estimate_budget_upper_bound']['nll_regret_raw'], .01 / 3.)

    def test_ranking_excludes_unstable_and_resolves_ties(self):
        ordered = sorted_members(np.arange(3), np.array([.9, -np.inf, .9]), ['b', 'c', 'a'])
        np.testing.assert_array_equal(ordered, [2, 0])


if __name__ == '__main__':
    unittest.main()
