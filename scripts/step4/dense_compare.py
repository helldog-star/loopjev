"""Cross-backbone paired evaluation with explicit permitted architecture differences."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from dense_train import semantic_records
from step1_finish import selection_group,task_group
from step2_compare import load_run
from step2_evaluate import paired_difference,write_json


def check_pair(candidate,reference):
    c,r=candidate['protocol'],reference['protocol']
    if c['version']!='step4-qwen3-dense-v1' or r['version'] not in ['step2-final-only-v1','step3-multi-loop-v1']:
        raise ValueError('Expected Step4 Qwen candidate and Step2/3 Ouro reference')
    if r['args']['mode']!='lora':raise ValueError('Use LoRA reference for primary controlled comparison')
    for key in ['rows','source_cache_sha256','lora','probe','packages']:
        if c[key]!=r[key]:raise ValueError(f'Unpaired protocol: {key}')
    for key in ['steps','batch_rows','seed','head_lr','lora_lr','l2','candidate_batch','token_budget','smoke']:
        if c['args'][key]!=r['args'][key]:raise ValueError(f'Unpaired setting: {key}')
    for name in ['step2_train.py','step2_evaluate.py','step1_common.py','step1_probe.py','step1_finish.py']:
        if c['scripts_sha256'][name]!=r['scripts_sha256'][name]:raise ValueError(f'Changed shared dependency: {name}')
    for key in ['optimizer_steps','decision_rows','candidates','row_schedule_sha256']:
        if candidate['metrics']['training_cost'][key]!=reference['metrics']['training_cost'][key]:
            raise ValueError(f'Unpaired training exposure: {key}')
    if candidate['metrics']['loops']!=1 or candidate['metrics']['training_cost']['runtime_loops']!=1:
        raise ValueError('Expected one dense forward')


def compare(candidate,reference,reference_loop,repeats):
    check_pair(candidate,reference)
    ref_method='final_only_head' if reference['protocol']['version'].startswith('step2') else 'multi_loop_head'
    loops=reference['metrics']['loops'];rl=reference_loop or loops
    if not 1<=rl<=loops:raise ValueError('Reference loop unavailable')
    output={}
    for method,rm in [('dense_head',ref_method),('independent_probe','independent_probe')]:
        c,r=candidate['predictions'][method],reference['predictions'][rm]
        if len(c['logits'])!=1 or len(r['logits'])!=loops:raise ValueError('Wrong prediction depths')
        if semantic_records(c['records'])!=semantic_records(r['records']):
            raise ValueError('Rows, group ids, candidates, targets or permutations differ')
        records=c['records']
        subsets={'validation_all':list(range(len(records))),
            'validation_check':[i for i,row in enumerate(records) if not selection_group(row)]}
        for group in ['reasoning','retrieval_matching','other']:
            subsets['validation_check/'+group]=[i for i in subsets['validation_check'] if task_group(records[i])==group]
        output[method]={}
        for subset,indices in subsets.items():
            if not indices:continue
            values={}
            for scale in ['raw','temperature_scaled']:
                ct=1. if scale=='raw' else c['temperatures'][0]['temperature']
                rt=1. if scale=='raw' else r['temperatures'][rl-1]['temperature']
                delta=paired_difference([c['logits'][0][i]/ct for i in indices],
                    [r['logits'][rl-1][i]/rt for i in indices],[records[i] for i in indices],repeats)
                lower=delta['accuracy_one_sided_95_lower']
                delta['dense_noninferiority_at_existing_bounds']=bool(lower is not None and lower>-.01 and delta['nll_one_sided_95_upper']<.02)
                values[scale]=delta
            output[method][subset]=values
    c,r=candidate['metrics']['latency'],reference['metrics']['latency']
    for key in ['row_ids','candidate_batch','repeats','warmup_passes','scope']:
        if c[key]!=r[key]:raise ValueError(f'Unpaired latency benchmark: {key}')
    if candidate['protocol']['gpu']!=reference['protocol']['gpu']:raise ValueError('Different benchmark GPU')
    output['latency']={'dense_over_reference_full_runtime_ratio':c['mean_seconds_per_decision']/r['mean_seconds_per_decision'],
        'dense_seconds_per_decision':c['mean_seconds_per_decision'],'reference_seconds_per_decision':r['mean_seconds_per_decision'],
        'reference_evaluated_loop':rl,'reference_executed_loops':r['runtime_loops'],
        'limitation':'Full runtime only, even for an intermediate reference loop; different tokenizers; pretokenized inputs; no FLOP claim'}
    output['training_cost']={'dense':candidate['metrics']['training_cost'],'reference':reference['metrics']['training_cost']}
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',required=True);p.add_argument('--reference',required=True)
    p.add_argument('--reference-loop',type=int);p.add_argument('--out',required=True)
    p.add_argument('--bootstrap',type=int,default=1000);p.add_argument('--allow-smoke',action='store_true')
    args=p.parse_args()
    if args.bootstrap<1 or (args.reference_loop is not None and args.reference_loop<1):raise ValueError('Invalid arguments')
    dest=Path(args.out)
    if dest.exists():raise FileExistsError(dest)
    candidate,reference=[load_run(path,args.allow_smoke) for path in [args.candidate,args.reference]]
    result=compare(candidate,reference,args.reference_loop,args.bootstrap)
    dest.parent.mkdir(parents=True,exist_ok=True)
    write_json(dest,{'status':'SMOKE ONLY' if args.allow_smoke else 'single-seed cross-backbone development comparison',
        'direction':'Qwen dense minus Ouro reference','candidate':args.candidate,'reference':args.reference,
        'reference_loop':args.reference_loop or reference['metrics']['loops'],
        'limitations':['not a pretraining-controlled causal architecture experiment','same rows/updates, not equal tokens or FLOPs',
            'group bootstrap does not measure training-seed uncertainty','no multiplicity adjustment; raw and scaled separate'],
        'comparison':result})
    print(f'Saved {dest}',flush=True)


if __name__=='__main__':main()
