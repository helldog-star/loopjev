"""Fit, lock, and independently assess the predeclared Step5C pilot."""
import argparse
import itertools
import json
import time
from pathlib import Path

import joblib
import numpy as np

from risk_core import (ACC_THRESHOLDS, NLL_THRESHOLDS, RANDOM_SEEDS, bootstrap_weights,
    fit_predictors, paired_mean_interval, policy_mask, policy_report, predict, prediction_report)
from risk_data import verify_sources
from step1_common import fingerprint
from step2_evaluate import write_json


def load_partition(root, name):
    spec = json.loads((root / 'results/protocol.json').read_text())
    path = root / f'cache/{name}.npz'
    report = json.loads((root / f'results/extraction_{name}.json').read_text())
    if fingerprint(path) != report['cache_sha256'] or report['manifest_sha256'] != spec['manifest_sha256'][name]:
        raise ValueError('Compact cache identity mismatch')
    with np.load(path, allow_pickle=False) as stream:
        data = {name: stream[name] for name in stream.files}
    if len(data['x']) != spec['counts'][name] or len(set(data['groups'])) != len(data['x']):
        raise ValueError('Unexpected group count')
    return data


def choose_policy(data, prediction, weights):
    best_threshold = None
    best = policy_report(data, policy_mask(prediction, None), weights)
    candidates = []
    for threshold in itertools.product(NLL_THRESHOLDS, NLL_THRESHOLDS, ACC_THRESHOLDS):
        mask = policy_mask(prediction, threshold)
        report = policy_report(data, mask, weights)
        candidates.append({'threshold': list(threshold), **report})
        if report['noninferior'] and (report['exit_rows'], -report['nll_difference_raw_scaled'][1]) > (
                best['exit_rows'], -best['nll_difference_raw_scaled'][1]):
            best_threshold, best = list(threshold), report
    return {'threshold': best_threshold, 'policy_fit': best}, candidates


def confidence_mask(data, threshold):
    return np.zeros(len(data['x']), dtype=bool) if threshold is None else data['x'][:, 0] >= threshold


def lock_models(root):
    verify_sources(root)
    destination = root / 'results'
    if (destination / 'models.joblib').exists() or (destination / 'locked.json').exists():
        raise FileExistsError('Model/lock exists; do not refit after check')
    train, validation, policy = [load_partition(root, name) for name in ['risk_train', 'risk_val', 'policy_fit']]
    if any(set(left['groups']) & set(right['groups']) for left, right in itertools.combinations([train, validation, policy], 2)):
        raise ValueError('Group leakage')
    models = {}
    validation_reports, fit_reports, choices, selections = {}, {}, {}, {}
    weights = bootstrap_weights(len(policy['x']), 1000)
    for name, columns in [('C0_mean', None), ('C1_confidence', [0]), ('C2_full', list(range(6)))]:
        model = fit_predictors(train, validation, columns)
        models[name] = model
        selections[name] = model['selection']
        validation_reports[name] = prediction_report(validation, predict(model, validation))
        fitted = predict(model, policy)
        fit_reports[name] = prediction_report(policy, fitted)
        choices[name], candidates = choose_policy(policy, fitted, weights)
        write_json(destination / f'policy_search_{name}.json', candidates)
        print(f'{name}: selected regularization and locked fit threshold; exits={choices[name]["policy_fit"]["exit_rows"]}', flush=True)
    confidence_best = {'threshold': None, 'policy_fit': policy_report(policy, confidence_mask(policy, None), weights)}
    confidence_trials = []
    for threshold in [.8, .85, .9, .92, .94, .96, .98, .99]:
        report = policy_report(policy, confidence_mask(policy, threshold), weights)
        confidence_trials.append({'threshold': threshold, **report})
        if report['noninferior'] and report['exit_rows'] > confidence_best['policy_fit']['exit_rows']:
            confidence_best = {'threshold': threshold, 'policy_fit': report}
    choices['confidence_threshold'] = confidence_best
    write_json(destination / 'development_metrics.json', {'risk_val': validation_reports, 'policy_fit': fit_reports,
        'regularization': selections, 'policies': choices, 'confidence_trials': confidence_trials,
        'status': 'Model and policy selection only; final-check unopened'})
    joblib.dump(models, destination / 'models.joblib')
    write_json(destination / 'locked.json', {'version': 'step5c-lock-v1', 'locked_at_unix': time.time(),
        'protocol_sha256': fingerprint(destination / 'protocol.json'),
        'models_sha256': fingerprint(destination / 'models.joblib'),
        'development_sha256': fingerprint(destination / 'development_metrics.json'),
        'policies': choices, 'final_check_access': 'Only after this lock; no refitting/rethresholding',
        'primary_model': 'C2_full', 'bootstrap_final': 2000, 'bootstrap_seed': 17})
    print('Models and policies locked; final-check may now be extracted', flush=True)


def prediction_comparison(data, predictions, weights):
    report = {}
    for baseline in ['C0_mean', 'C1_confidence']:
        current = {}
        for target, scale in enumerate(['raw', 'scaled']):
            error_full = (predictions['C2_full']['nll'][:, target] - data['nll'][:, target]) ** 2
            error_reference = (predictions[baseline]['nll'][:, target] - data['nll'][:, target]) ** 2
            rmse_full, rmse_reference = np.sqrt(error_full.mean()), np.sqrt(error_reference.mean())
            current[scale] = {'paired_squared_error': paired_mean_interval(error_full - error_reference, weights),
                'relative_rmse_improvement': float(1. - rmse_full / rmse_reference) if rmse_reference > 0 else None,
                'meets_1percent_and_ci': bool(rmse_reference > 0 and 1. - rmse_full / rmse_reference >= .01 and
                                              np.quantile(weights @ (error_full - error_reference) / len(error_full), .95) < 0)}
        report[baseline] = current
    return report


def budget_comparison(data, chosen, weights):
    count, rows = int(chosen.sum()), len(chosen)
    hard = data['hard']
    identifiers = data['groups']
    ordered = np.array(sorted(range(rows), key=lambda index: (-data['x'][index, 0], identifiers[index])))
    confidence = np.zeros(rows, dtype=bool)
    confidence[ordered[:count]] = True
    random_masks, random_points = [], []
    for seed in RANDOM_SEEDS:
        mask = np.zeros(rows, dtype=bool)
        mask[np.random.default_rng(seed).choice(rows, size=count, replace=False)] = True
        random_masks.append(mask)
        random_points.append({'accuracy_difference': float(-(mask * data['accuracy_regret']).sum() / hard.sum()),
                              'nll_difference_raw_scaled': (mask[:, None] * data['nll']).mean(axis=0).tolist()})
    reference_masks = {'matched_confidence_ranking': confidence.astype(float),
                       'expected_uniform_random': np.full(rows, count / rows)}
    comparisons = {}
    for name, reference in reference_masks.items():
        multiplier = chosen.astype(float) - reference
        paired = {scale: paired_mean_interval(multiplier * data['nll'][:, target], weights)
                  for target, scale in enumerate(['raw', 'scaled'])}
        accuracy_vector = -(multiplier * data['accuracy_regret'])
        valid = weights @ hard.astype(float) > 0
        draws = (weights @ accuracy_vector)[valid] / (weights @ hard.astype(float))[valid]
        difference = float(accuracy_vector.sum() / hard.sum())
        lower = float(np.quantile(draws, .05))
        comparisons[name] = {'policy_minus_reference_nll': paired,
            'accuracy_difference': difference, 'accuracy_one_sided_95_lower': lower,
            'meets_practical_gain': bool(all(paired[scale]['mean_difference'] <= -.005 and
                                            paired[scale]['one_sided_95_upper'] < 0 for scale in ['raw', 'scaled'])
                                         and lower > -.01)}
    return {'exit_rows_each': count, 'seeds': RANDOM_SEEDS, 'random_seed_points': random_points,
        'comparisons': comparisons,
        'confidence_reference': policy_report(data, confidence, weights),
        'limits': ['exact exit counts only, not token/FLOP-matched',
                   'confidence top-count ranking is a transductive budget diagnostic, not an online threshold',
                   'random mean comparison uses exact uniform expectation r*d, finite seeds describe subset variation']}


def final_check(root):
    verify_sources(root)
    destination = root / 'results'
    if (destination / 'final_metrics.json').exists() or (destination / 'final_started.json').exists():
        raise FileExistsError('Final-check already accessed; do not repeatedly retest')
    locked = json.loads((destination / 'locked.json').read_text())
    for file, key in [('protocol.json', 'protocol_sha256'), ('models.joblib', 'models_sha256'),
                      ('development_metrics.json', 'development_sha256')]:
        if fingerprint(destination / file) != locked[key]:
            raise ValueError(f'Lock changed: {file}')
    write_json(destination / 'final_started.json', {'started_at_unix': time.time(),
               'lock_sha256': fingerprint(destination / 'locked.json')})
    data = load_partition(root, 'final_check')
    models = joblib.load(destination / 'models.joblib')
    weights = bootstrap_weights(len(data['x']), locked['bootstrap_final'], locked['bootstrap_seed'])
    predictions = {name: predict(model, data) for name, model in models.items()}
    reports = {name: prediction_report(data, prediction) for name, prediction in predictions.items()}
    policies = {name: policy_report(data, policy_mask(predictions[name], locked['policies'][name]['threshold']), weights)
                for name in predictions}
    policies['confidence_threshold'] = policy_report(data, confidence_mask(data,
        locked['policies']['confidence_threshold']['threshold']), weights)
    comparison = prediction_comparison(data, predictions, weights)
    selected = policy_mask(predictions['C2_full'], locked['policies']['C2_full']['threshold'])
    budget = budget_comparison(data, selected, weights)
    fixed_fractions = {}
    for fraction in [.1, .2, .4]:
        count = int(np.floor(len(data['x']) * fraction))
        threshold = locked['policies']['C2_full']['threshold']
        risk = predictions['C2_full']['nll'][:, 0]
        ordered = np.array(sorted(range(len(risk)), key=lambda index: (risk[index], data['groups'][index])))
        mask = np.zeros(len(risk), dtype=bool)
        mask[ordered[:count]] = True
        fixed_fractions[str(fraction)] = {'raw_ridge_ranking': policy_report(data, mask, weights),
                                          'budget': budget_comparison(data, mask, weights)}
    predictive_go = all(item[scale]['meets_1percent_and_ci'] for item in comparison.values() for scale in ['raw', 'scaled'])
    policy_go = policies['C2_full']['noninferior'] and policies['C2_full']['theoretical_loop_saving'] >= .1
    budget_go = all(item['meets_practical_gain'] for item in budget['comparisons'].values())
    write_json(destination / 'final_metrics.json', {'status': 'completed fresh-group pilot; not official test',
        'prediction': reports, 'prediction_comparisons': comparison, 'locked_policy_reports': policies,
        'matched_budget': budget, 'descriptive_fixed_exit_fractions': fixed_fractions,
        'go': {'nll_predictability': predictive_go, 'quality_and_10percent_loops': bool(policy_go),
               'matched_budget_gain': budget_go, 'dynamic_inference_investment': bool(predictive_go and policy_go and budget_go)},
        'limitations': ['single backbone seed', 'fresh train-heldout groups with shared task sources/templates',
                       'individual exploratory intervals, no familywise multiplicity guarantee',
                       'rank coverage points are descriptive, not policies tuned on final-check',
                       'offline loop savings only; no measured acceleration']})
    print(f'Final-check complete. Predictability={predictive_go}, quality+cost={policy_go}, budget={budget_go}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['fit', 'check'])
    parser.add_argument('--root', default='artifacts/step5/risk_uniform4_v1')
    args = parser.parse_args()
    if args.action == 'fit':
        lock_models(Path(args.root))
    else:
        final_check(Path(args.root))


if __name__ == '__main__':
    main()
