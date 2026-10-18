"""Step5A offline exits: no model training, inference, or measured speedup."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit, softmax
from step1_common import fingerprint, summarize
from step1_finish import selection_group, compact
from step2_evaluate import paired_difference, write_json

VERSION = 'step5-offline-exit-v1'
FAMILIES = ('maxprob_raw', 'maxprob_scaled', 'gate_cumulative_min', 'gate_cumulative_mean')


def confidence(logits, temperatures, gates, offsets):
    """Only model outputs enter scores; labels and future rounds are not consulted."""
    n = len(logits[0])
    values = {}
    for name, ts in [('maxprob_raw', [1.] * 4), ('maxprob_scaled', temperatures)]:
        values[name] = np.array([[softmax(logits[t][i] / ts[t]).max() for t in range(4)] for i in range(n)])
    cumulative = 1. - np.cumprod(1. - expit(gates), axis=1)
    for name, reducer in [('gate_cumulative_min', np.min), ('gate_cumulative_mean', np.mean)]:
        values[name] = np.stack([reducer(cumulative[a:b], axis=0) for a, b in zip(offsets[:-1], offsets[1:])])
    return values


def exits(scores, threshold):
    result = np.full(len(scores), 4, dtype=np.int64)
    if threshold is None:
        return result
    # Main pilot uses minimum depth 2; depth 4 is unconditional fallback.
    for t in [1, 2]:
        result[(result == 4) & (scores[:, t] >= threshold)] = t + 1
    return result


def picked(logits, depths, indices, temperatures):
    return [np.asarray(logits[int(depths[i])-1][i], dtype=np.float64) / temperatures[int(depths[i])-1] for i in indices]


def paired(logits, depths, indices, records, temperatures, repeats):
    rows = [records[i] for i in indices]
    return paired_difference(picked(logits, depths, indices, temperatures),
        [logits[3][i] / temperatures[3] for i in indices], rows, repeats)


def noninferior(d):
    return d['accuracy_one_sided_95_lower'] is not None and d['accuracy_one_sided_95_lower'] > -.01 and d['nll_one_sided_95_upper'] < .02


def cost(depths, indices):
    x = depths[indices]
    return {'rows': len(x), 'mean_loops': float(x.mean()),
            'loop_counts': {str(t): int((x == t).sum()) for t in range(1, 5)},
            'theoretical_loop_reduction_fraction': float(1.-x.mean()/4.),
            'measured_latency_or_flops': None}


def evaluate(logits, depths, indices, records, temperatures, repeats):
    result = {'cost': cost(depths, indices)}
    for name, ts in [('raw', [1.]*4), ('temperature_scaled', temperatures)]:
        scores = picked(logits, depths, indices, ts)
        delta = paired(logits, depths, indices, records, ts, repeats)
        result[name] = {'metrics': compact(summarize(scores, [records[i] for i in indices])),
                        'vs_same_model_loop4': delta, 'noninferior': noninferior(delta)}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', default='artifacts/step3/uniform4')
    p.add_argument('--reference', default='artifacts/step2/lora4')
    p.add_argument('--out-dir', default='artifacts/step5/offline_uniform4_v1')
    p.add_argument('--bootstrap', type=int, default=1000)
    args = p.parse_args()
    if args.bootstrap < 100: raise ValueError('At least 100 bootstrap draws required')
    out = Path(args.out_dir)
    if out.exists(): raise FileExistsError(out)
    run = Path(args.run); ref = Path(args.reference)
    protocol = json.loads((run/'results/protocol.json').read_text())
    if protocol['version'] != 'step3-multi-loop-v1' or protocol['args']['smoke'] or protocol['args']['loops'] != 4:
        raise ValueError('Full four-loop E2 required')
    data = torch.load(run/'results/validation_predictions.pt',map_location='cpu',weights_only=False)['multi_loop_head']
    cache = torch.load(run/'cache/validation.pt',map_location='cpu',weights_only=False)
    records, logits = data['records'], data['logits']
    temperatures = [x['temperature'] for x in data['temperatures']]
    if len(logits)!=4 or len(records)!=2048 or records!=cache['rows']: raise ValueError('Unpaired cached records')
    if any(x['kind']!='choice' for x in records) or len({x['group_id'] for x in records})!=len(records): raise ValueError('Expected one choice row per group')
    if not all(np.isfinite(t) and t>0 for t in temperatures): raise ValueError('Invalid temperature')
    gates = cache['gates'].float().numpy();offsets=cache['row_offsets'].tolist()
    if gates.shape != (offsets[-1],4) or not np.isfinite(gates).all(): raise ValueError('Bad gate cache')
    if offsets[0]!=0 or any(b-a!=len(r['target']) for a,b,r in zip(offsets[:-1],offsets[1:],records)): raise ValueError('Invalid candidate offsets')
    for t in range(4):
        if len(logits[t])!=len(records) or any(len(z)!=len(r['target']) or not np.isfinite(z).all() for z,r in zip(logits[t],records)):
            raise ValueError('Bad logits')
    # Bind gates to the exact trained adapter and experiment, not frozen Step1 gates.
    adapters={f.name:fingerprint(f) for f in (run/'results/adapter').iterdir() if f.is_file()}
    if adapters!=cache['config']['adapter_sha256'] or cache['config']['step3_scripts_sha256']!=protocol['scripts_sha256']:
        raise ValueError('Gate cache model identity mismatch')
    reference_protocol=json.loads((ref/'results/protocol.json').read_text())
    reference=torch.load(ref/'results/validation_predictions.pt',map_location='cpu',weights_only=False)['final_only_head']
    if reference['records']!=records or reference_protocol['version']!='step2-final-only-v1' or reference_protocol['args']['loops']!=4 or reference_protocol['args']['smoke']:
        raise ValueError('Unpaired E1 four-loop reference')
    if reference_protocol['source_cache_sha256']!=protocol['source_cache_sha256']: raise ValueError('Reference cache differs')
    fit=np.array([i for i,r in enumerate(records) if selection_group(r)],dtype=int)
    check=np.array([i for i,r in enumerate(records) if not selection_group(r)],dtype=int)
    if not len(fit) or not len(check):raise ValueError('Empty partition')
    # Protocol saved before examining policy outcomes. No policy-check scores enter selection.
    out.mkdir(parents=True);dest=out/'results';dest.mkdir()
    grid=[float(x) for x in np.linspace(0.,1.,21)] + [None]
    spec={'version':VERSION,'args':vars(args),'status':'offline development only',
          'source_sha256':{str(f):fingerprint(f) for f in [run/'results/protocol.json',run/'results/validation_predictions.pt',run/'cache/validation.pt',ref/'results/protocol.json',ref/'results/validation_predictions.pt']},
          'script_sha256':fingerprint(Path(__file__)), 'families':list(FAMILIES), 'threshold_grid':grid,
          'minimum_depth':2,'fallback_depth':4,'temperature':temperatures,'rows':{'policy_fit':len(fit),'policy_check':len(check)},
          'partition':'reuse Step1 SHA256(loopjev-validation-select-v1:group_id), even=fit, odd=check',
          'selection':'per-family minimum fit mean loops among policies satisfying BOTH raw/scaled single-sided group-bootstrap quality bounds vs E2 loop4; tie by scaled NLL delta then conservative threshold; fallback always4',
          'bounds':{'accuracy_lower':-.01,'nll_upper':.02},
          'gate_definition':'sigmoid(raw last-token gate logits), cumulative 1-product(1-lambda) per candidate, then min or mean over ALL candidates; min and mean are heuristics, not calibrated row correctness',
          'limits':['validation previously inspected; not independent final test','four selected families; no multiplicity correction','theoretical loops only; no actual early stop or speed measurement','per-loop calibration does not guarantee calibration after adaptive selection','learned risk head deferred until independent heldout-group or OOF fitting predictions exist','one seed; choice only; no formal calibration/test/ood access']}
    write_json(dest/'protocol.json',spec)
    scores=confidence(logits,temperatures,gates,offsets)
    curves={};selected={};decisions={}
    for family in FAMILIES:
        trials=[]
        for threshold in grid:
            depths=exits(scores[family],threshold)
            raw=paired(logits,depths,fit,records,[1.]*4,args.bootstrap)
            scaled=paired(logits,depths,fit,records,temperatures,args.bootstrap)
            trials.append({'threshold':threshold,'cost':cost(depths,fit),'raw':raw,'scaled':scaled,
                           'eligible':noninferior(raw) and noninferior(scaled)})
        valid=[t for t in trials if t['eligible']]
        chosen=min(valid,key=lambda t:(t['cost']['mean_loops'],t['scaled']['nll_difference'],-(2. if t['threshold'] is None else t['threshold'])))
        depths=exits(scores[family],chosen['threshold']);decisions[family]=depths.tolist()
        selected[family]={'threshold':chosen['threshold'],'policy_fit':chosen,'policy_check':evaluate(logits,depths,check,records,temperatures,args.bootstrap)}
        selected[family]['policy_check']['secondary_vs_E1_loop4']={}
        for scale,ts,rt in [('raw',[1.]*4,1.),('temperature_scaled',temperatures,reference['temperatures'][3]['temperature'])]:
            selected[family]['policy_check']['secondary_vs_E1_loop4'][scale]=paired_difference(picked(logits,depths,check,ts),[reference['logits'][3][i]/rt for i in check],[records[i] for i in check],args.bootstrap)
        curves[family]=trials
        print(f"{family}: threshold={chosen['threshold']}, fit loops={chosen['cost']['mean_loops']:.3f}, check loops={selected[family]['policy_check']['cost']['mean_loops']:.3f}",flush=True)
    fixed={str(t):evaluate(logits,np.full(len(records),t),check,records,temperatures,args.bootstrap) for t in range(1,5)}
    write_json(dest/'metrics.json',{'status':'completed offline development pilot; not measured acceleration','selected':selected,'fixed_depth':fixed,
        'go_candidates':[name for name,v in selected.items() if v['policy_check']['cost']['mean_loops']<4 and v['policy_check']['raw']['noninferior'] and v['policy_check']['temperature_scaled']['noninferior']],
        'selection_curves_policy_fit_only':curves})
    write_json(dest/'exit_decisions.json',{'group_ids':[r['group_id'] for r in records],'policy_fit_indices':fit.tolist(),'policy_check_indices':check.tolist(),'depths':decisions})
    print(f'Completed offline pilot: {dest}',flush=True)

if __name__=='__main__':main()
