"""Low-dimensional causal features and statistical helpers for Step5C."""
import numpy as np
from scipy.special import expit, logsumexp, softmax
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


FEATURES = ['confidence_raw', 'margin_raw', 'normalized_entropy_raw', 'js_raw_2_3',
            'argmax_stable_2_3', 'cumulative_gate_mean_3']
NLL_THRESHOLDS = [-.05, 0., .02, .05, .1, .2, None]
ACC_THRESHOLDS = [0., .01, .03, .05, None]
ALPHAS = [.1, 1., 10., 100.]
LOGISTIC_C = [.1, 1., 10.]
RANDOM_SEEDS = [17, 29, 43, 71, 101, 127, 151, 179, 211, 239,
                271, 307, 331, 367, 401, 431, 463, 503, 541, 577, 613]
TAIL_DELTA = .5


def features_and_targets(logits, gates, records, temperatures):
    features, nll, hard, events, base_nll, base_accuracy = [], [], [], [], [], []
    for index, row in enumerate(records):
        if len(row['target']) < 2:
            raise ValueError('Step5C choice requires K >= 2')
        previous, current = softmax(logits[index][1]), softmax(logits[index][2])
        middle = (previous + current) / 2.
        js = .5 * np.sum(previous * np.log(np.maximum(previous, 1e-300) / middle))
        js += .5 * np.sum(current * np.log(np.maximum(current, 1e-300) / middle))
        ordered = np.sort(current)
        candidate_gates = np.asarray(gates[index])[:, :3]
        cumulative = 1. - np.prod(1. - expit(candidate_gates), axis=1)
        features.append([current.max(), ordered[-1] - ordered[-2],
                         -np.sum(current * np.log(np.maximum(current, 1e-300))) / np.log(len(current)),
                         js, float(previous.argmax() == current.argmax()), cumulative.mean()])
        target = np.asarray(row['target'], dtype=float)
        is_hard = np.count_nonzero(target) == 1 and np.isclose(target.max(), 1.)
        hard.append(is_hard)
        correct3 = np.argmax(logits[index][2]) == target.argmax()
        correct4 = np.argmax(logits[index][3]) == target.argmax()
        events.append(1 if not correct3 and correct4 else 2 if correct3 and not correct4 else 0)
        regrets, baselines = [], []
        for ts in [[1.] * 4, temperatures]:
            shallow = np.asarray(logits[index][2]) / ts[2]
            deep = np.asarray(logits[index][3]) / ts[3]
            shallow_loss = logsumexp(shallow) - target @ shallow
            deep_loss = logsumexp(deep) - target @ deep
            regrets.append(shallow_loss - deep_loss)
            baselines.append(deep_loss)
        nll.append(regrets)
        base_nll.append(baselines)
        base_accuracy.append(float(correct4))
    result = {'x': np.asarray(features), 'nll': np.asarray(nll), 'hard': np.asarray(hard),
              'events': np.asarray(events), 'base_nll': np.asarray(base_nll),
              'base_accuracy': np.asarray(base_accuracy)}
    result['accuracy_regret'] = (result['events'] == 1).astype(float) - (result['events'] == 2)
    result['accuracy_regret'][~result['hard']] = 0.
    if not all(np.isfinite(result[name]).all() for name in ['x', 'nll', 'base_nll']):
        raise ValueError('Nonfinite features or targets')
    return result


def event_probabilities(model, features):
    if isinstance(model, np.ndarray):
        return np.tile(model, (len(features), 1))
    probabilities = np.zeros((len(features), 3))
    probabilities[:, model.classes_.astype(int)] = model.predict_proba(features)
    return probabilities


def fit_predictors(train, validation, columns):
    if columns is None:
        means = train['nll'].mean(axis=0)
        rates = np.bincount(train['events'][train['hard']], minlength=3) / train['hard'].sum()
        return {'columns': None, 'nll_mean': means, 'event_model': rates, 'selection': {'kind': 'train means'}}
    scaler = StandardScaler().fit(train['x'][:, columns])
    train_features = scaler.transform(train['x'][:, columns])
    validation_features = scaler.transform(validation['x'][:, columns])
    models, selection = [], []
    for target in range(2):
        candidates = []
        for alpha in ALPHAS:
            model = Ridge(alpha=alpha).fit(train_features, train['nll'][:, target])
            loss = np.mean((model.predict(validation_features) - validation['nll'][:, target]) ** 2)
            candidates.append((float(loss), alpha, model))
        loss, alpha, model = min(candidates, key=lambda item: (item[0], -item[1]))
        models.append(model)
        selection.append({'alpha': alpha, 'validation_mse': loss})
    labels = train['events'][train['hard']]
    if len(np.unique(labels)) < 2:
        event_model = np.bincount(labels, minlength=3) / len(labels)
        event_selection = {'kind': 'constant; one observed event class'}
    else:
        candidates = []
        for regularization in LOGISTIC_C:
            model = LogisticRegression(C=regularization, max_iter=2000, random_state=17).fit(
                train_features[train['hard']], labels)
            if model.n_iter_.max() >= 2000:
                raise RuntimeError('Logistic regression did not converge')
            probability = event_probabilities(model, validation_features[validation['hard']])
            selected = probability[np.arange(validation['hard'].sum()), validation['events'][validation['hard']]]
            loss = -np.log(np.maximum(selected, 1e-12)).mean()
            candidates.append((float(loss), regularization, model))
        loss, regularization, event_model = min(candidates, key=lambda item: (item[0], item[1]))
        event_selection = {'C': regularization, 'validation_log_loss': loss}
    return {'columns': columns, 'scaler': scaler, 'nll_models': models, 'event_model': event_model,
            'selection': {'nll': selection, 'events': event_selection}}


def predict(model, data):
    if model['columns'] is None:
        nll = np.tile(model['nll_mean'], (len(data['x']), 1))
        probability = event_probabilities(model['event_model'], data['x'])
    else:
        features = model['scaler'].transform(data['x'][:, model['columns']])
        nll = np.column_stack([regressor.predict(features) for regressor in model['nll_models']])
        probability = event_probabilities(model['event_model'], features)
    return {'nll': nll, 'events': probability, 'accuracy_regret': probability[:, 1] - probability[:, 2]}


def bootstrap_weights(count, repeats, seed=17):
    return np.random.default_rng(seed).multinomial(count, np.full(count, 1. / count), size=repeats)


def paired_mean_interval(values, weights, minimum_effect=0.):
    draws = weights @ values / len(values)
    upper = float(np.quantile(draws, .95))
    return {'mean_difference': float(values.mean()), 'two_sided_95': np.quantile(draws, [.025, .975]).tolist(),
            'one_sided_95_upper': upper, 'supports_improvement': upper < -minimum_effect}


def policy_mask(prediction, threshold):
    if threshold is None:
        return np.zeros(len(prediction['nll']), dtype=bool)
    raw, scaled, accuracy = [np.inf if value is None else value for value in threshold]
    return ((prediction['nll'][:, 0] <= raw) & (prediction['nll'][:, 1] <= scaled) &
            (prediction['accuracy_regret'] <= accuracy))


def policy_report(data, mask, weights):
    hard = data['hard']
    denominator = weights @ hard.astype(float)
    valid = denominator > 0
    acc_vector = -(mask * data['accuracy_regret'])
    accuracy = float(acc_vector.sum() / hard.sum())
    acc_draws = (weights @ acc_vector)[valid] / denominator[valid]
    nll_vector = mask[:, None] * data['nll']
    nll_draws = weights @ nll_vector / len(mask)
    lower = float(np.quantile(acc_draws, .05))
    upper = np.quantile(nll_draws, .95, axis=0)
    return {'rows': len(mask), 'hard_rows': int(hard.sum()), 'exit_rows': int(mask.sum()),
            'exit_fraction': float(mask.mean()), 'mean_loops': float(4. - mask.mean()),
            'theoretical_loop_saving': float(mask.mean() / 4.), 'accuracy_difference': accuracy,
            'accuracy_one_sided_95_lower': lower,
            'nll_difference_raw_scaled': nll_vector.mean(axis=0).tolist(),
            'nll_one_sided_95_upper_raw_scaled': upper.tolist(),
            'noninferior': bool(lower > -.01 and np.all(upper < .02)),
            'absolute_accuracy': float(data['base_accuracy'][hard].mean() + accuracy),
            'absolute_nll_raw_scaled': (data['base_nll'].mean(axis=0) + nll_vector.mean(axis=0)).tolist(),
            'tail_exited_raw_scaled': ((data['nll'][mask] > TAIL_DELTA).sum(axis=0)).tolist(),
            'max_exited_regret_raw_scaled': data['nll'][mask].max(axis=0).tolist() if mask.any() else None}


def prediction_report(data, prediction):
    report = {'nll': {}, 'events': {}}
    for target, name in enumerate(['raw', 'scaled']):
        truth, estimate = data['nll'][:, target], prediction['nll'][:, target]
        ordered = np.argsort(estimate, kind='stable')
        tail = truth > TAIL_DELTA
        high = np.argsort(-estimate, kind='stable')[:max(1, int(np.ceil(.2 * len(truth))))]
        report['nll'][name] = {'mae': float(np.mean(np.abs(truth - estimate))),
            'rmse': float(np.sqrt(np.mean((truth - estimate) ** 2))), 'tail_events': int(tail.sum()),
            'tail_recall_top20': float(tail[high].sum() / tail.sum()) if tail.any() else None,
            'tail_recall_predicted_gt_delta': float((tail & (estimate > TAIL_DELTA)).sum() / tail.sum()) if tail.any() else None,
            'deciles_low_to_high': [{'rows': len(members), 'predicted_mean': float(estimate[members].mean()),
                                    'observed_mean': float(truth[members].mean()),
                                    'tail_fraction': float(tail[members].mean())}
                                   for members in np.array_split(ordered, min(10, len(ordered)))]}
    hard = data['hard']
    for event, name in [(1, 'rescued_by_loop4'), (2, 'degraded_by_loop4')]:
        truth = data['events'][hard] == event
        probability = prediction['events'][hard, event]
        high = np.argsort(-probability, kind='stable')[:max(1, int(np.ceil(.2 * len(truth))))]
        bins = np.minimum((probability * 5).astype(int), 4)
        report['events'][name] = {'events': int(truth.sum()), 'prevalence': float(truth.mean()),
            'brier': float(np.mean((probability - truth) ** 2)),
            'auroc': float(roc_auc_score(truth, probability)) if 0 < truth.sum() < len(truth) else None,
            'average_precision': float(average_precision_score(truth, probability)) if truth.any() else None,
            'recall_top20': float(truth[high].sum() / truth.sum()) if truth.any() else None,
            'calibration_bins': [{'rows': int((bins == value).sum()),
                'predicted_mean': float(probability[bins == value].mean()),
                'observed_fraction': float(truth[bins == value].mean())}
                for value in range(5) if np.any(bins == value)]}
    return report
