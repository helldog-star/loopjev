"""Paired E2-vs-E1 comparisons; predictions only, never starts training."""
import argparse
from pathlib import Path

from step1_finish import selection_group, task_group
from step2_compare import load_run
from step2_evaluate import paired_difference, write_json


def check_pair(candidate, reference):
    c, r = candidate['protocol'], reference['protocol']
    if c['version'] != 'step3-multi-loop-v1' or r['version'] != 'step2-final-only-v1':
        raise ValueError('Expected Step3 candidate and Step2 reference')
    if c['args']['loops'] != 4 or c['args']['mode'] != r['args']['mode']:
        raise ValueError('Expected four-loop E2 and same training mode')
    expected = [.25] * 4 if c['args']['weighting'] == 'uniform' else [.1, .2, .3, .4]
    if c['alpha'] != expected:
        raise ValueError('Unexpected supervision weights')
    for key in ['rows', 'source_cache_sha256', 'lora', 'probe', 'model_provenance',
                'initialization', 'packages', 'gpu']:
        if c[key] != r[key]:
            raise ValueError(f'Unpaired protocol: {key}')
    for name, digest in r['scripts_sha256'].items():
        if c['scripts_sha256'].get(name) != digest:
            raise ValueError(f'Step2 dependency changed: {name}')
    for key in ['steps', 'batch_rows', 'head_lr', 'lora_lr', 'l2', 'seed',
                'candidate_batch', 'token_budget', 'smoke']:
        if c['args'][key] != r['args'][key]:
            raise ValueError(f'Unpaired training setting: {key}')
    for key in ['optimizer_steps', 'decision_rows', 'candidates', 'unpadded_tokens',
                'row_schedule_sha256', 'head_parameters', 'adapter_parameters']:
        if candidate['metrics']['training_cost'][key] != reference['metrics']['training_cost'][key]:
            raise ValueError(f'Unpaired training cost: {key}')
    for run in [candidate, reference]:
        loops = run['protocol']['args']['loops']
        if run['metrics']['loops'] != loops or run['metrics']['training_cost']['runtime_loops'] != loops:
            raise ValueError('Runtime depth mismatch')


def compare(candidate, reference, repeats):
    check_pair(candidate, reference)
    reference_loops = reference['protocol']['args']['loops']
    pairs = [('E2_loop2_vs_E1_final', 1, reference_loops - 1)]
    if reference_loops == 4:
        pairs += [('E2_loop4_vs_E1_loop4', 3, 3), ('E2_loop2_vs_E1_loop2', 1, 1)]
    output = {}
    for method, ref_method in [('multi_loop_head', 'final_only_head'), ('independent_probe', 'independent_probe')]:
        c, r = candidate['predictions'][method], reference['predictions'][ref_method]
        if c['records'] != r['records']:
            raise ValueError('Validation records, targets, or candidate ordering differ')
        if len(c['logits']) != 4 or len(r['logits']) != reference_loops:
            raise ValueError('Prediction depth mismatch')
        records = c['records']
        subsets = {'validation_all': list(range(len(records))),
                   'validation_check': [i for i, row in enumerate(records) if not selection_group(row)]}
        for group in ['reasoning', 'retrieval_matching', 'other']:
            subsets['validation_check/' + group] = [i for i in subsets['validation_check'] if task_group(records[i]) == group]
        output[method] = {}
        for name, cl, rl in pairs:
            output[method][name] = {}
            for subset_name, indices in subsets.items():
                if not indices:
                    continue
                result = {}
                for scale in ['raw', 'temperature_scaled']:
                    ct = 1.0 if scale == 'raw' else c['temperatures'][cl]['temperature']
                    rt = 1.0 if scale == 'raw' else r['temperatures'][rl]['temperature']
                    metrics = paired_difference([c['logits'][cl][i] / ct for i in indices],
                        [r['logits'][rl][i] / rt for i in indices], [records[i] for i in indices], repeats)
                    lower = metrics['accuracy_one_sided_95_lower']
                    metrics['quality_noninferiority_at_preregistered_bounds'] = bool(
                        lower is not None and lower > -.01 and metrics['nll_one_sided_95_upper'] < .02)
                    result[scale] = metrics
                output[method][name][subset_name] = result
    c, r = candidate['metrics']['latency'], reference['metrics']['latency']
    for key in ['row_ids', 'candidate_batch', 'repeats', 'warmup_passes', 'scope']:
        if c[key] != r[key]:
            raise ValueError(f'Unpaired latency setting: {key}')
    output['measured_full_runtime_latency_ratio'] = c['mean_seconds_per_decision'] / r['mean_seconds_per_decision']
    output['latency_runtime_loops'] = {'candidate': c['runtime_loops'], 'reference': r['runtime_loops']}
    output['latency_limitations'] = ('Candidate executes all four loops even when evaluating loop2 scores. '
        'No measured loop2 early-exit speedup; equal optimizer steps are not equal FLOPs.')
    output['training_cost'] = {key: run['metrics']['training_cost'] for key, run in [('candidate', candidate), ('reference', reference)]}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--allow-smoke', action='store_true')
    args = parser.parse_args()
    if args.bootstrap < 1:
        raise ValueError('Positive bootstrap count required')
    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(destination)
    candidate, reference = [load_run(path, args.allow_smoke) for path in [args.candidate, args.reference]]
    result = compare(candidate, reference, args.bootstrap)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, {'status': 'SMOKE ONLY' if args.allow_smoke else 'single-seed development comparison',
        'direction': 'E2 candidate minus E1 reference', 'candidate': args.candidate, 'reference': args.reference,
        'bounds': {'accuracy_lower': -.01, 'nll_upper': .02},
        'limitations': 'one-sided percentile group bootstrap; no multiplicity correction; check is development data',
        'comparison': result})
    print(f'Saved {destination}', flush=True)


if __name__ == '__main__':
    main()
