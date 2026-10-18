"""External checkpoint comparisons: matched examples, explicitly unmatched training."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
import json
from openjev_eval import SEMANTIC_KEYS,VERSION
from step1_finish import selection_group,task_group
from step2_evaluate import paired_difference,write_json


def load(path,allow_smoke):
    root=Path(path)/'results'
    result={name:json.loads((root/f'{name}.json').read_text()) for name in ['metrics','protocol']}
    if result['metrics']['status']=='SMOKE ONLY' and not allow_smoke:raise ValueError('Smoke is not experimental evidence')
    result['predictions']=torch.load(root/'validation_predictions.pt',map_location='cpu',weights_only=False)
    return result


def compare(candidate,reference,loop,repeats):
    if candidate['protocol']['version']!=VERSION:raise ValueError('Expected Open-Jev external candidate')
    mapping={'step2-final-only-v1':'final_only_head','step3-multi-loop-v1':'multi_loop_head','step4-qwen3-dense-v1':'dense_head'}
    method=mapping[reference['protocol']['version']]
    c=candidate['predictions']['external_head'];r=reference['predictions'][method]
    for split in ['dev-cal','validation']:
        if candidate['protocol']['source_cache_sha256'][split]!=reference['protocol']['source_cache_sha256'][split]:raise ValueError('Different locked source cache')
    if candidate['protocol']['args']['smoke']!=reference['protocol']['args']['smoke']:raise ValueError('Cannot mix smoke and full runs')
    canonical=lambda rows:[{key:row[key] for key in SEMANTIC_KEYS} for row in rows]
    if canonical(c['records'])!=canonical(r['records']):raise ValueError('Unpaired rows, groups, candidate order or targets')
    loop=loop or len(r['logits'])
    if not 1<=loop<=len(r['logits']):raise ValueError('Invalid reference loop')
    records=c['records'];subsets={'validation_all':list(range(len(records))),
        'validation_check':[i for i,row in enumerate(records) if not selection_group(row)]}
    for group in ['reasoning','retrieval_matching','other']:
        subsets['validation_check/'+group]=[i for i in subsets['validation_check'] if task_group(records[i])==group]
    results={}
    for subset,indices in subsets.items():
        if not indices:continue
        results[subset]={}
        for name,ct,rt in [('raw_vs_raw',1.,1.),
            ('dev_cal_vs_dev_cal',c['temperatures'][0]['temperature'],r['temperatures'][loop-1]['temperature']),
            ('released_vs_raw',c['released_temperature'],1.)]:
            results[subset][name]=paired_difference([c['logits'][0][i]/ct for i in indices],
                [r['logits'][loop-1][i]/rt for i in indices],[records[i] for i in indices],repeats)
    return {'reference_loop':loop,'subsets':results,
        'latency_note':'No automatic speed ratio: official Open-Jev timing includes tokenization; reference timing is pretokenized.',
        'training_note':'External checkpoint, not equal training budget or controlled architecture evidence; dataset overlap not ruled out.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',required=True);p.add_argument('--reference',required=True)
    p.add_argument('--reference-loop',type=int);p.add_argument('--out',required=True)
    p.add_argument('--bootstrap',type=int,default=1000);p.add_argument('--allow-smoke',action='store_true')
    a=p.parse_args()
    if a.bootstrap<1 or (a.reference_loop is not None and a.reference_loop<1):raise ValueError('Invalid arguments')
    dest=Path(a.out)
    if dest.exists():raise FileExistsError(dest)
    result=compare(load(a.candidate,a.allow_smoke),load(a.reference,a.allow_smoke),a.reference_loop,a.bootstrap)
    dest.parent.mkdir(parents=True,exist_ok=True)
    write_json(dest,{'status':'SMOKE ONLY' if a.allow_smoke else 'external checkpoint development comparison',
        'direction':'Open-Jev minus reference','candidate':a.candidate,'reference':a.reference,
        'limitations':'group bootstrap; no multiplicity correction; no equal-training claim; released_vs_raw is a distinct calibration condition',
        'comparison':result})
    print(f'Saved {dest}',flush=True)


if __name__=='__main__':main()
