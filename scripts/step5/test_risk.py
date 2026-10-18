"""Step5C semantic tests: causal features, isolation, targets and risk policies."""
import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from risk_core import (bootstrap_weights, features_and_targets, fit_predictors,
    policy_mask, policy_report, predict)
from risk_data import NearDuplicateIndex, identities, shingles
from risk_fit import lock_models, final_check


def example_data():
    generator = np.random.default_rng(17)
    features = generator.normal(size=(120, 6))
    features[:, 0] = 1. / (1. + np.exp(-features[:, 0]))
    nll = np.column_stack([features[:, 0] - .5, .5 * features[:, 0] - .25])
    events = np.tile([0, 1, 2], 40)
    return {'x': features, 'nll': nll, 'events': events, 'hard': np.ones(120, dtype=bool),
            'accuracy_regret': (events == 1).astype(float) - (events == 2),
            'base_nll': np.ones((120, 2)), 'base_accuracy': np.ones(120)}


class RiskTests(unittest.TestCase):
    def test_no_future_feature_or_target_leakage(self):
        row = [{'target': [1., 0.]}]
        logits = [np.array([[0., 1.], [0., 1.], [0., 1.], [1., 0.]])]
        gates = [np.zeros((2, 4))]
        original = features_and_targets(logits, gates, row, [1.] * 4)
        changed = [logits[0].copy()]
        changed[0][3] = [100., -100.]
        gates[0][:, 3] = 100.
        later = features_and_targets(changed, gates, [{'target': [0., 1.]}], [1.] * 4)
        np.testing.assert_allclose(original['x'], later['x'])
        self.assertNotEqual(original['nll'][0, 0], later['nll'][0, 0])

    def test_permutation_invariance_and_single_choice_rejection(self):
        row = [{'target': [1., 0., 0.]}]
        logits = [np.arange(12, dtype=float).reshape(4, 3)]
        gates = [np.arange(12, dtype=float).reshape(3, 4)]
        original = features_and_targets(logits, gates, row, [1.] * 4)
        permutation = [2, 0, 1]
        permuted = features_and_targets([logits[0][:, permutation]], [gates[0][permutation]],
                                        [{'target': [0., 1., 0.]}], [1.] * 4)
        np.testing.assert_allclose(original['x'], permuted['x'])
        np.testing.assert_allclose(original['nll'], permuted['nll'])
        with self.assertRaises(ValueError):
            features_and_targets([np.zeros((4, 1))], [np.zeros((1, 4))], [{'target': [1.]}], [1.] * 4)

    def test_risk_direction_and_soft_label_denominator(self):
        data = {'hard': np.array([True, False, True]), 'accuracy_regret': np.array([1., 0., -1.]),
                'nll': np.array([[.1, .1], [.6, .6], [-.1, -.1]]),
                'base_nll': np.ones((3, 2)), 'base_accuracy': np.array([1., 0., 0.])}
        report = policy_report(data, np.array([True, True, False]), bootstrap_weights(3, 100))
        self.assertAlmostEqual(report['accuracy_difference'], -.5)
        self.assertAlmostEqual(report['nll_difference_raw_scaled'][0], .7 / 3.)
        self.assertEqual(report['tail_exited_raw_scaled'], [1, 1])
        self.assertEqual(report['exit_rows'], 2)

    def test_standardization_only_train_and_constant_baseline(self):
        train = example_data()
        validation = {key: value.copy() for key, value in train.items()}
        validation['x'] += 5.
        model = fit_predictors(train, validation, list(range(6)))
        np.testing.assert_allclose(model['scaler'].mean_, train['x'].mean(axis=0))
        constant = fit_predictors(train, validation, None)
        output = predict(constant, validation)
        np.testing.assert_allclose(output['nll'][0], train['nll'].mean(axis=0))
        np.testing.assert_allclose(output['events'].sum(axis=1), 1.)

    def test_multiple_constraints_and_fallback(self):
        prediction = {'nll': np.array([[0., .1], [.2, 0.], [0., 0.]]),
                      'accuracy_regret': np.array([0., 0., .1])}
        np.testing.assert_array_equal(policy_mask(prediction, [.05, .05, .05]), [False, False, False])
        self.assertFalse(policy_mask(prediction, None).any())
        np.testing.assert_array_equal(policy_mask(prediction, [None, None, None]), [True, True, True])

    def test_exact_and_near_duplicate_normalization(self):
        first = {'state_json': '{"x": 1}', 'question': 'Hello  World', 'options': ['A', 'B']}
        second = {**first, 'question': 'hello world', 'options': ['B', 'A']}
        self.assertEqual(identities(first), identities(second))
        near = NearDuplicateIndex()
        tokens = shingles('one two three four five six seven eight nine ten')
        near.add_tokens(tokens, 'train:group')
        self.assertEqual(near.matches(tokens), ['train:group'])

    def test_synthetic_fit_lock_then_final_check_once(self):
        partitions = {}
        for name in ['risk_train', 'risk_val', 'policy_fit', 'final_check']:
            data = example_data()
            data['groups'] = np.array([f'{name}:{index}' for index in range(len(data['x']))])
            partitions[name] = data
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'results').mkdir()
            (root / 'results/protocol.json').write_text('{}')
            with patch('risk_fit.verify_sources', return_value={}), patch('risk_fit.load_partition', side_effect=lambda root, name: partitions[name]):
                lock_models(root)
                self.assertTrue((root / 'results/locked.json').exists())
                final_check(root)
                result = json.loads((root / 'results/final_metrics.json').read_text())
                self.assertIn('nll_predictability', result['go'])
                with self.assertRaises(FileExistsError):
                    final_check(root)


if __name__ == '__main__':
    unittest.main()
