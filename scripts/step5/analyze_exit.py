"""Step5B cached diagnostics of third-loop versus fourth-loop decisions."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.special import logsumexp, softmax

from offline_exit import confidence, evaluate
from step1_common import fingerprint
from step1_finish import selection_group
from step2_evaluate import write_json


def quantities(logits, records, temperatures):
    hard = np.array([np.count_nonzero(row['target']) == 1 and
                     np.isclose(max(row['target']), 1.) for row in records])
    losses, correct, probabilities = [], [], []
    for loop in range(4):
        current = [np.asarray(scores, dtype=float) / temperatures[loop] for scores in logits[loop]]
        probabilities.append([softmax(scores) for scores in current])
        losses.append([logsumexp(scores) - np.asarray(row['target']) @ scores
                       for scores, row in zip(current, records, strict=True)])
        correct.append([scores.argmax() == np.argmax(row['target'])
                        for scores, row in zip(current, records, strict=True)])
    losses, correct = np.array(losses).T, np.array(correct).T
    return {'hard': hard, 'losses': losses, 'correct': correct,
            'nll_regret': losses[:, 2] - losses[:, 3],
            'accuracy_regret': (correct[:, 3].astype(float) - correct[:, 2]) * hard,
            'probabilities': probabilities}


def causal_signals(logits, temperatures, gates, offsets):
    existing = confidence(logits, temperatures, gates, offsets)
    signals = {name: values[:, 2] for name, values in existing.items()}
    for scale, ts in [('raw', [1.] * 4), ('scaled', temperatures)]:
        margins, entropy, stability, js_values = [], [], [], []
        for previous, current in zip(logits[1], logits[2], strict=True):
            previous, current = softmax(previous / ts[1]), softmax(current / ts[2])
            ordered = np.sort(current)
            middle = (previous + current) / 2.
            js_value = .5 * np.sum(previous * np.log(np.maximum(previous, 1e-300) / middle))
            js_value += .5 * np.sum(current * np.log(np.maximum(current, 1e-300) / middle))
            margins.append(ordered[-1] - ordered[-2])
            entropy.append(-np.sum(current * np.log(np.maximum(current, 1e-300))) / np.log(len(current)))
            stability.append(previous.argmax() == current.argmax())
            js_values.append(js_value)
        signals[f'margin_{scale}'] = np.array(margins)
        signals[f'negative_normalized_entropy_{scale}'] = -np.array(entropy)
        signals[f'negative_js_{scale}'] = -np.array(js_values)
        signals[f'stable_confidence_{scale}'] = np.where(stability, signals[f'maxprob_{scale}'], -np.inf)
    return signals


def subset_summary(indices, raw, scaled):
    indices = np.asarray(indices, dtype=int)
    if not len(indices):
        return {'rows': 0}
    hard_indices = indices[raw['hard'][indices]]
    result = {'rows': len(indices), 'hard_rows': len(hard_indices),
              'nll_regret_raw': float(raw['nll_regret'][indices].mean()),
              'nll_regret_scaled': float(scaled['nll_regret'][indices].mean())}
    if len(hard_indices):
        correct = raw['correct'][hard_indices]
        result.update(accuracy_loop3=float(correct[:, 2].mean()),
                      accuracy_loop4=float(correct[:, 3].mean()),
                      accuracy_regret=float(raw['accuracy_regret'][hard_indices].mean()),
                      rescue_rows=int((~correct[:, 2] & correct[:, 3]).sum()),
                      degradation_rows=int((correct[:, 2] & ~correct[:, 3]).sum()))
    return result


def transitions(indices, raw, scaled, signals):
    hard_indices = indices[raw['hard'][indices]]
    correct = raw['correct']
    result = {}
    for name, mask in {
        'both_correct': correct[:, 2] & correct[:, 3],
        'rescued_by_loop4': ~correct[:, 2] & correct[:, 3],
        'degraded_by_loop4': correct[:, 2] & ~correct[:, 3],
        'both_wrong': ~correct[:, 2] & ~correct[:, 3],
    }.items():
        members = hard_indices[mask[hard_indices]]
        item = subset_summary(members, raw, scaled)
        if len(members):
            item['signals'] = {key: {'median': float(np.median(values[members][np.isfinite(values[members])])),
                                     'quantiles_10_90': np.quantile(values[members][np.isfinite(values[members])], [.1, .9]).tolist()}
                               for key, values in signals.items() if np.isfinite(values[members]).any()}
        result[name] = item
    result['prediction_agreement_fraction'] = float(np.mean([
        raw['probabilities'][2][index].argmax() == raw['probabilities'][3][index].argmax()
        for index in indices]))
    result['all_rows'] = subset_summary(indices, raw, scaled)
    result['nll_regret_quantiles'] = {scale: np.quantile(data['nll_regret'][indices], [0, .1, .5, .9, .95, .99, 1]).tolist()
                                     for scale, data in [('raw', raw), ('scaled', scaled)]}
    return result


def sorted_members(indices, signal, group_ids):
    valid = indices[np.isfinite(signal[indices])]
    return np.array(sorted(valid, key=lambda index: (-signal[index], group_ids[index])), dtype=int)


def signal_analysis(indices, signals, raw, scaled, group_ids):
    result = {}
    for name, values in signals.items():
        ordered = sorted_members(indices, values, group_ids)
        bins = []
        for members in np.array_split(ordered, min(10, len(ordered))):
            item = subset_summary(members, raw, scaled)
            item['mean_signal'] = float(values[members].mean())
            if name in ('maxprob_raw', 'maxprob_scaled'):
                hard_members = members[raw['hard'][members]]
                item['hard_mean_confidence'] = float(values[hard_members].mean()) if len(hard_members) else None
            bins.append(item)
        curves = []
        for fraction in [.01, .05, .1, .2, .25, .4, .6, .8, 1.]:
            count = min(len(ordered), int(np.floor(len(indices) * fraction)))
            members = ordered[:count]
            if not count:
                continue
            item = subset_summary(members, raw, scaled)
            item.update(exit_fraction=count / len(indices), theoretical_loop_saving=count / len(indices) / 4.,
                        overall_accuracy_regret=float(raw['accuracy_regret'][members].sum() / raw['hard'][indices].sum()),
                        overall_nll_regret_raw=float(raw['nll_regret'][members].sum() / len(indices)),
                        overall_nll_regret_scaled=float(scaled['nll_regret'][members].sum() / len(indices)))
            curves.append(item)
        result[name] = {'eligible_rows': len(ordered), 'descending_deciles': bins, 'coverage_curve': curves}
    return result


def oracle(indices, raw, scaled):
    hard = raw['hard'][indices]
    accuracy = raw['accuracy_regret'][indices]
    raw_loss, scaled_loss = raw['nll_regret'][indices], scaled['nll_regret'][indices]
    safe = (accuracy <= 0) & (raw_loss <= 0) & (scaled_loss <= 0)
    matrix = np.stack([accuracy, raw_loss, scaled_loss])
    upper = np.array([.01 * hard.sum(), .02 * len(indices), .02 * len(indices)]) - 1e-9
    solution = milp(-np.ones(len(indices)), integrality=np.ones(len(indices)),
                    bounds=Bounds(0., 1.), constraints=LinearConstraint(matrix, -np.inf, upper),
                    options={'time_limit': 60., 'mip_rel_gap': 0.})
    if not solution.success:
        raise RuntimeError(f'Oracle solver did not certify optimum: {solution.message}')
    mask = solution.x > .5
    if np.any(matrix @ mask.astype(float) > upper + 1e-7):
        raise ValueError('Invalid oracle constraint solution')
    return {'strict_per_row_nonworsening': {'exit_rows': int(safe.sum()),
                'theoretical_loop_saving': float(safe.mean() / 4.),
                'subset': subset_summary(indices[safe], raw, scaled)},
            'point_estimate_budget_upper_bound': {'exit_rows': int(mask.sum()),
                'theoretical_loop_saving': float(mask.mean() / 4.),
                'accuracy_regret': float(accuracy[mask].sum() / hard.sum()),
                'nll_regret_raw': float(raw_loss[mask].sum() / len(indices)),
                'nll_regret_scaled': float(scaled_loss[mask].sum() / len(indices)),
                'solver': solution.message},
            'limits': 'Uses labels and loop4, cannot deploy. Point-constraint optimum is not a confidence-bound guarantee; safe subset is a feasible construction, not maximum.'}


def plot_report(metrics, destination):
    current = metrics['partitions']['policy_check']
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    names = ['both_correct', 'rescued_by_loop4', 'degraded_by_loop4', 'both_wrong']
    axes[0, 0].bar(['Both correct', '4 rescues', '4 degrades', 'Both wrong'],
                   [current['transitions'][name]['rows'] for name in names], color='#52687a')
    axes[0, 0].set(title='Loop3/4 hard-label transitions', ylabel='Decision rows', xlabel='Correctness transition')
    for name, label in [('maxprob_raw', 'Raw confidence'), ('maxprob_scaled', 'Scaled confidence')]:
        bins = [item for item in current['signals'][name]['descending_deciles'] if item['hard_rows']]
        axes[0, 1].plot([item['hard_mean_confidence'] for item in bins],
                        [item['accuracy_loop3'] for item in bins], marker='.', label=label)
    axes[0, 1].plot([0, 1], [0, 1], color='gray', linestyle='--', label='Perfect calibration')
    axes[0, 1].set(title='Loop3 reliability, confidence quantiles', xlabel='Mean confidence on hard rows', ylabel='Loop3 accuracy')
    axes[0, 1].legend(fontsize=8)
    for name in ['maxprob_raw', 'stable_confidence_raw', 'negative_js_raw', 'gate_cumulative_min']:
        points = current['signals'][name]['coverage_curve']
        axes[1, 0].plot([100 * item['exit_fraction'] for item in points],
                        [100 * item['overall_accuracy_regret'] for item in points], label=name)
        axes[1, 1].plot([100 * item['exit_fraction'] for item in points],
                        [item['overall_nll_regret_raw'] for item in points], label=name)
    axes[1, 0].axhline(1., color='gray', linestyle='--')
    axes[1, 1].axhline(.02, color='gray', linestyle='--')
    axes[1, 0].set(title='Accuracy regret of ranked loop3 exits', xlabel='Rows exited (%)', ylabel='Full-policy accuracy loss (pp)')
    axes[1, 1].set(title='Raw NLL regret of ranked loop3 exits', xlabel='Rows exited (%)', ylabel='Full-policy NLL increase (nats/row)')
    axes[1, 0].legend(fontsize=7)
    axes[1, 1].legend(fontsize=7)
    fig.suptitle('Step5B development diagnostics: E2 Uniform-4, policy-check (previously viewed)')
    fig.savefig(destination / 'diagnostics.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='artifacts/step3/uniform4')
    parser.add_argument('--out-dir', default='artifacts/step5/diagnostics_uniform4_v1')
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    output, run = Path(args.out_dir), Path(args.run)
    if output.exists():
        raise FileExistsError(output)
    if args.bootstrap < 100:
        raise ValueError('At least 100 bootstrap draws required')
    source = run / 'results/validation_predictions.pt'
    cache_path = run / 'cache/validation.pt'
    protocol_path = run / 'results/protocol.json'
    protocol = json.loads(protocol_path.read_text())
    data = torch.load(source, map_location='cpu', weights_only=False)['multi_loop_head']
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    records, logits = data['records'], data['logits']
    temperatures = [item['temperature'] for item in data['temperatures']]
    if protocol['version'] != 'step3-multi-loop-v1' or protocol['args']['smoke'] or protocol['args']['loops'] != 4:
        raise ValueError('Expected formal four-loop E2')
    if len(records) != 2048 or len(logits) != 4 or records != cache['rows']:
        raise ValueError('Unpaired records')
    if len({row['group_id'] for row in records}) != len(records) or any(row['kind'] != 'choice' for row in records):
        raise ValueError('Expected unique choice groups')
    adapters = {path.name: fingerprint(path) for path in (run / 'results/adapter').iterdir() if path.is_file()}
    if adapters != cache['config']['adapter_sha256'] or cache['config']['step3_scripts_sha256'] != protocol['scripts_sha256']:
        raise ValueError('Gate cache identity mismatch')
    offsets, gates = cache['row_offsets'].tolist(), cache['gates'].float().numpy()
    if gates.shape != (offsets[-1], 4) or not np.isfinite(gates).all():
        raise ValueError('Invalid gates')
    if offsets[0] != 0 or any(end - start != len(row['target']) for start, end, row in zip(offsets[:-1], offsets[1:], records)):
        raise ValueError('Invalid candidate offsets')
    if any(not np.isfinite(value) or value <= 0 for value in temperatures):
        raise ValueError('Invalid temperatures')
    for loop in logits:
        if len(loop) != len(records) or any(len(scores) != len(row['target']) or len(scores) < 2 or
                not np.isfinite(scores).all() for scores, row in zip(loop, records, strict=True)):
            raise ValueError('Invalid logits')
    partitions = {'policy_fit': np.array([index for index, row in enumerate(records) if selection_group(row)]),
                  'policy_check': np.array([index for index, row in enumerate(records) if not selection_group(row)])}
    destination = output / 'results'
    destination.mkdir(parents=True)
    write_json(destination / 'protocol.json', {'version': 'step5b-diagnostics-v1', 'args': vars(args),
        'source_sha256': {str(path): fingerprint(path) for path in [source, cache_path, protocol_path]},
        'script_sha256': fingerprint(Path(__file__)), 'temperature': temperatures,
        'analysis': ['hard-label transitions plus all-row NLL regret', 'causal-signal deciles and ranked coverage',
                     'label-informed oracle feasible subset and point-budget MILP upper bound'],
        'partition': 'Unchanged Step1 group hash; fit/check previously inspected, development only',
        'bounds': {'accuracy_loss': .01, 'nll_increase_raw_and_scaled': .02},
        'tie_break': 'Signal descending then group_id; ranked curves descriptive, not threshold policies',
        'limits': ['no new training or inference', 'no threshold selection or new confirmed policy',
                   'oracle uses future outputs and labels; not deployable', 'no measured acceleration',
                   'single model seed; formal calibration/test/ood untouched']})
    raw = quantities(logits, records, [1.] * 4)
    scaled = quantities(logits, records, temperatures)
    signals = causal_signals(logits, temperatures, gates, offsets)
    group_ids = [row['group_id'] for row in records]
    metrics = {'partitions': {}}
    for name, indices in partitions.items():
        metrics['partitions'][name] = {'transitions': transitions(indices, raw, scaled, signals),
            'signals': signal_analysis(indices, signals, raw, scaled, group_ids),
            'fixed_loop3': evaluate(logits, np.full(len(records), 3), indices, records, temperatures, args.bootstrap),
            'oracle': oracle(indices, raw, scaled)}
        print(f'{name}: {len(indices)} rows; diagnostics complete', flush=True)
    write_json(destination / 'metrics.json', metrics)
    plot_report(metrics, destination)
    print(f'Completed Step5B diagnostics: {destination}', flush=True)


if __name__ == '__main__':
    main()
