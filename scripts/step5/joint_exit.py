"""Step5B small joint-threshold study with matched-budget random controls."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax

from offline_exit import picked
from step1_common import fingerprint
from step1_finish import selection_group
from step2_evaluate import paired_difference, write_json


def row_features(logits, temperatures, gates, offsets):
    probability = {}
    for name, ts in [('raw', [1.] * 4), ('scaled', temperatures)]:
        values = [softmax(scores / ts[2]) for scores in logits[2]]
        probability[name] = {
            'confidence': np.array([value.max() for value in values]),
            'margin': np.array([np.sort(value)[-1] - np.sort(value)[-2] for value in values]),
        }
    stable = np.array([softmax(previous / temperatures[1]).argmax() ==
                       softmax(current / temperatures[2]).argmax()
                       for previous, current in zip(logits[1], logits[2], strict=True)])
    js = []
    for previous, current in zip(logits[1], logits[2], strict=True):
        previous, current = softmax(previous / temperatures[1]), softmax(current / temperatures[2])
        middle = (previous + current) / 2.
        js.append(.5 * np.sum(previous * np.log(np.maximum(previous, 1e-300) / middle)) +
                  .5 * np.sum(current * np.log(np.maximum(current, 1e-300) / middle)))
    cumulative = 1. - np.cumprod(1. - 1. / (1. + np.exp(-np.asarray(gates))), axis=1)
    gate = np.array([np.mean(cumulative[start:end, 2])
                     for start, end in zip(offsets[:-1], offsets[1:])])
    return probability, stable, np.array(js), gate


def policy_masks(features, confidence_threshold, stability_required=False, js_threshold=None,
                 confidence_scale='raw'):
    confidence = features[0][confidence_scale]['confidence']
    mask = confidence >= confidence_threshold
    if stability_required:
        mask &= features[1]
    if js_threshold is not None:
        mask &= features[2] <= js_threshold
    return mask


def depths_from_mask(mask):
    depths = np.full(len(mask), 4, dtype=np.int64)
    depths[mask] = 3
    return depths


def delta(logits, depths, indices, records, temperatures, bootstrap):
    return paired_difference(
        picked(logits, depths, indices, [1.] * 4),
        [logits[3][index] for index in indices],
        [records[index] for index in indices], bootstrap,
    ), paired_difference(
        picked(logits, depths, indices, temperatures),
        [logits[3][index] / temperatures[3] for index in indices],
        [records[index] for index in indices], bootstrap,
    )


def eligible(raw, scaled):
    return (raw['accuracy_one_sided_95_lower'] > -.01 and raw['nll_one_sided_95_upper'] < .02 and
            scaled['accuracy_one_sided_95_lower'] > -.01 and scaled['nll_one_sided_95_upper'] < .02)


def random_subset(indices, count, seed):
    if count == 0:
        return np.array([], dtype=int)
    return np.random.default_rng(seed).choice(indices, size=count, replace=False)


def random_summary(logits, records, depths, indices, temperatures, seeds, bootstrap):
    count = int(np.count_nonzero(depths[indices] == 3))
    point = []
    for seed in seeds:
        selected = random_subset(indices, count, seed)
        random_depths = depths.copy()
        random_depths[indices] = 4
        random_depths[selected] = 3
        raw, scaled = delta(logits, random_depths, indices, records, temperatures, bootstrap)
        point.append({'seed': seed, 'exit_rows': count, 'raw': raw, 'scaled': scaled})
    def band(scale, metric):
        values = [item[scale][metric] for item in point]
        return {'mean': float(np.mean(values)), 'p05': float(np.quantile(values, .05)),
                'p95': float(np.quantile(values, .95)), 'values': values}
    return {'exit_rows': count, 'seeds': seeds, 'raw': {metric: band('raw', metric)
        for metric in ['accuracy_difference', 'nll_difference']},
        'scaled': {metric: band('scaled', metric) for metric in ['accuracy_difference', 'nll_difference']}}


def candidate_specs():
    confidence_thresholds = [.80, .85, .90, .92, .94, .96, .98]
    specs = []
    for scale in ('raw', 'scaled'):
        for threshold in confidence_thresholds:
            specs.append({'name': f'confidence_{scale}', 'confidence_scale': scale,
                          'confidence_threshold': threshold})
            specs.append({'name': f'stable_confidence_{scale}', 'confidence_scale': scale,
                          'confidence_threshold': threshold, 'stability_required': True})
    for scale in ('raw', 'scaled'):
        for confidence_threshold in [.85, .90, .94]:
            for js_threshold in [.005, .01, .02, .04]:
                specs.append({'name': f'confidence_js_{scale}', 'confidence_scale': scale,
                              'confidence_threshold': confidence_threshold,
                              'js_threshold': js_threshold})
    return specs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='artifacts/step3/uniform4')
    parser.add_argument('--out-dir', default='artifacts/step5/joint_exit_uniform4_v1')
    parser.add_argument('--bootstrap', type=int, default=500)
    parser.add_argument('--random-seeds', type=int, nargs='+', default=[17, 29, 43, 71, 101,
                                                                          127, 151, 179, 211, 239,
                                                                          271, 307, 331, 367, 401,
                                                                          431, 463, 503, 541, 577, 613])
    args = parser.parse_args()
    if args.bootstrap < 100:
        raise ValueError('At least 100 bootstrap draws required')
    output, run = Path(args.out_dir), Path(args.run)
    if output.exists():
        raise FileExistsError(output)
    protocol_path, prediction_path = run / 'results/protocol.json', run / 'results/validation_predictions.pt'
    cache_path = run / 'cache/validation.pt'
    protocol = json.loads(protocol_path.read_text())
    data = torch.load(prediction_path, map_location='cpu', weights_only=False)['multi_loop_head']
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    records, logits = data['records'], data['logits']
    temperatures = [item['temperature'] for item in data['temperatures']]
    if protocol['version'] != 'step3-multi-loop-v1' or protocol['args']['smoke'] or protocol['args']['loops'] != 4:
        raise ValueError('Expected formal E2 Uniform-4')
    if len(records) != 2048 or len(logits) != 4 or records != cache['rows']:
        raise ValueError('Unpaired prediction and gate cache')
    if len({row['group_id'] for row in records}) != len(records):
        raise ValueError('Expected one row per group')
    adapters = {path.name: fingerprint(path) for path in (run / 'results/adapter').iterdir() if path.is_file()}
    if adapters != cache['config']['adapter_sha256'] or cache['config']['step3_scripts_sha256'] != protocol['scripts_sha256']:
        raise ValueError('Gate cache identity mismatch')
    offsets, gates = cache['row_offsets'].tolist(), cache['gates'].float().numpy()
    features = row_features(logits, temperatures, gates, offsets)
    fit = np.array([index for index, row in enumerate(records) if selection_group(row)], dtype=int)
    check = np.array([index for index, row in enumerate(records) if not selection_group(row)], dtype=int)
    output.mkdir(parents=True)
    destination = output / 'results'
    destination.mkdir()
    write_json(destination / 'protocol.json', {
        'version': 'step5b-joint-exit-v1', 'args': vars(args), 'status': 'development only',
        'source_sha256': {str(path): fingerprint(path) for path in [protocol_path, prediction_path, cache_path]},
        'script_sha256': fingerprint(Path(__file__)), 'decision': 'Only loop3 may exit; loop4 is fallback',
        'families': ['confidence', 'confidence+argmax_stability', 'confidence+JS'],
        'confidence_thresholds': [.80, .85, .90, .92, .94, .96, .98],
        'js_thresholds': [.005, .01, .02, .04], 'policy_fit_rows': len(fit), 'policy_check_rows': len(check),
        'random_seeds': args.random_seeds, 'random_definition': 'exactly match candidate policy exit count separately on fit/check',
        'selection': 'max loop3 exits among fit candidates satisfying raw and temperature-scaled bootstrap bounds; always4 fallback',
        'bounds': {'accuracy_lower': -.01, 'nll_upper': .02},
        'limits': ['policy-check is previously viewed development data, not final test',
                   'random controls are budget-matched but use finite fixed seeds',
                   'offline loop savings are theoretical; no dynamic inference or measured speedup',
                   'single E2 seed and choice rows only']})
    candidates = []
    for spec in candidate_specs():
        mask = policy_masks(features, spec['confidence_threshold'], spec.get('stability_required', False),
                            spec.get('js_threshold'), spec['confidence_scale'])
        depths = depths_from_mask(mask)
        raw, scaled = delta(logits, depths, fit, records, temperatures, args.bootstrap)
        candidates.append({**spec, 'fit_exit_rows': int(mask[fit].sum()), 'fit_exit_fraction': float(mask[fit].mean()),
                           'fit_raw': raw, 'fit_scaled': scaled, 'fit_eligible': eligible(raw, scaled)})
    eligible_candidates = [candidate for candidate in candidates if candidate['fit_eligible']]
    chosen = max(eligible_candidates, key=lambda candidate: (candidate['fit_exit_rows'],
                                                              -candidate['fit_scaled']['nll_difference'])) if eligible_candidates else None
    selected = {'name': 'always4', 'confidence_scale': 'raw', 'confidence_threshold': None,
                'fit_exit_rows': 0, 'fit_exit_fraction': 0., 'fit_eligible': True} if chosen is None else chosen
    selected_mask = np.zeros(len(records), dtype=bool) if chosen is None else policy_masks(
        features, chosen['confidence_threshold'], chosen.get('stability_required', False),
        chosen.get('js_threshold'), chosen['confidence_scale'])
    selected_depths = depths_from_mask(selected_mask)
    reports = {}
    for name, indices in [('policy_fit', fit), ('policy_check', check)]:
        raw, scaled = delta(logits, selected_depths, indices, records, temperatures, args.bootstrap)
        reports[name] = {'exit_rows': int(selected_mask[indices].sum()), 'exit_fraction': float(selected_mask[indices].mean()),
                         'policy_raw': raw, 'policy_scaled': scaled,
                         'matched_random': random_summary(logits, records, selected_depths, indices, temperatures,
                                                          args.random_seeds, args.bootstrap)}
        reports[name]['fixed_depth4'] = {'exit_rows': 0, 'accuracy_difference': 0., 'nll_difference': 0.}
    write_json(destination / 'metrics.json', {'status': 'completed development pilot; no measured acceleration',
        'candidate_count': len(candidates), 'candidates': candidates, 'selected': selected,
        'selected_reports': reports,
        'interpretation': 'Policy and random controls have identical exit counts within each partition; policy advantage is evidence of ranking, not an independent test.'})
    print(f'Selected {selected["name"]} threshold={selected.get("confidence_threshold")} '
          f'fit exits={selected["fit_exit_rows"]}; wrote {destination}', flush=True)


if __name__ == '__main__':
    main()
