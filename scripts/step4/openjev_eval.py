"""Evaluate the released Open-Jev checkpoint, never train it or replace its head."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
BASE_ID='Qwen/Qwen3.5-2B'
REVISION='15852e8c16360a2fea060d615a32b45270f8a8fc'
SOURCE_REVISION='bd4118882f733574a3250a4b65fe4d884130c08b'
VERSION='step4-openjev-external-v1'
SEMANTIC_KEYS=('id','group_id','source','kind','options','target','permutation')


def sha(path):
    from runtime_paths import script_path
    path = script_path(path)
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def audit_package(path):
    root=Path(path);manifest=json.loads((root/'manifest.json').read_text())
    if manifest['model']!=BASE_ID or manifest['revision']!=REVISION:raise ValueError('Unexpected upstream model/revision')
    for name,entry in manifest['files'].items():
        p=root/name
        if root.resolve() not in p.resolve().parents:raise ValueError('Unsafe manifest path')
        if p.stat().st_size!=entry['bytes'] or sha(p)!=entry['sha256']:raise ValueError(f'Package integrity failed: {name}')
    config=json.loads((root/'checkpoint/model.json').read_text())
    if config['model_id']!=BASE_ID or config['revision']!=REVISION or config['lora_rank']!=8:raise ValueError('Unexpected checkpoint identity')
    temperature=json.loads((root/'checkpoint/temperature.json').read_text())['temperature']
    import math
    if not math.isfinite(temperature) or temperature<=0:raise ValueError('Invalid released temperature')
    return {'model':config,'released_temperature':temperature,'manifest_sha256':sha(root/'manifest.json'),
            'inference_files':{name:entry['sha256'] for name,entry in manifest['files'].items() if name.startswith('checkpoint/')}}


def source_root(path):
    root=Path(path)
    if (root/'SOURCE.json').is_file():
        provenance=json.loads((root/'SOURCE.json').read_text())
        if provenance['revision']!=SOURCE_REVISION:raise ValueError('Unexpected vendored source revision')
        for name,digest in provenance['files'].items():
            if sha(root/name)!=digest:raise ValueError(f'Official source modified: {name}')
    else:
        revision=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
        if revision!=SOURCE_REVISION:raise ValueError('Use the pinned official source commit')
        subprocess.run(['git','-C',str(root),'diff','--exit-code','HEAD','--'],check=True,stdout=subprocess.DEVNULL)
    parent=root/'src' if (root/'src/jev/model.py').is_file() else root
    if not (parent/'jev/model.py').is_file():raise FileNotFoundError('Complete official jev package is required')
    return parent


def audit_base(path,manifest_path):
    """Local paths ignore HF revision arguments; verify against pinned Hub file metadata."""
    root=Path(path);manifest=json.loads(Path(manifest_path).read_text())
    if manifest['model_id']!=BASE_ID or manifest['revision']!=REVISION:raise ValueError('Wrong base manifest')
    for name,entry in manifest['files'].items():
        p=root/name
        if not p.is_file():raise FileNotFoundError(p)
        if 'sha256' in entry:
            if sha(p)!=entry['sha256']:raise ValueError(f'Wrong pinned base file: {name}')
        else:
            data=p.read_bytes();digest=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
            if digest!=entry['git_blob_sha1']:raise ValueError(f'Wrong pinned tokenizer/config: {name}')
    return {'path':str(root.resolve()),'revision':REVISION,'metadata_sha256':sha(manifest_path),'verified_files':manifest['files']}


def prepare_rows(args,official_renderer):
    import torch
    import pyarrow.parquet as pq
    from step1_common import candidate_prompts
    output={};metadata={};hashes={}
    for split in ['dev-cal','validation']:
        cache=Path(args.cache_dir)/f'{split}.pt';hashes[split]=sha(cache)
        data=torch.load(cache,map_location='cpu',weights_only=True)
        expected_count=512 if split=='dev-cal' else 2048
        if len(data['rows'])!=expected_count or data['config']['split']!=split:raise ValueError('Wrong locked cache')
        records=data['rows'][:8] if args.smoke else data['rows']
        paths=sorted((Path(args.data)/'data/community-hard-mix-v2-redistributable').glob(('train' if split=='dev-cal' else split)+'-*.parquet'))
        if {p.name:sha(p) for p in paths}!={Path(p).name:v for p,v in data['config']['data_sha256'].items()}:raise ValueError('Dataset changed')
        ids={r['id'] for r in records};selected={}
        for row in pq.read_table(paths,columns=['id','group_id','source','kind','state_json','question','options','target']).to_pylist():
            if row['id'] in ids:
                if row['id'] in selected:raise ValueError('Duplicate data id')
                selected[row['id']]=row
        if len({r['group_id'] for r in records})!=len(records):raise ValueError('Repeated group')
        inputs=[]
        for cached in records:
            row=selected[cached['id']];perm=cached['permutation']
            if sorted(perm)!=list(range(len(row['options']))):raise ValueError('Invalid permutation')
            for key in ['group_id','source','kind']:
                if row[key]!=cached[key]:raise ValueError('Metadata mismatch')
            for key in ['options','target']:
                if [row[key][i] for i in perm]!=cached[key]:raise ValueError('Candidate/target mismatch')
            model_row={'state':json.loads(row['state_json']),'question':row['question'],
                       'kind':row['kind'],'options':[row['options'][i] for i in perm]}
            if official_renderer(model_row)!=candidate_prompts(model_row):raise ValueError('Official renderer differs from locked candidate semantics')
            inputs.append(model_row)
        output[split]=inputs;metadata[split]=records
    if {r['group_id'] for r in metadata['dev-cal']} & {r['group_id'] for r in metadata['validation']}:raise ValueError('Split leakage')
    return output,metadata,hashes


def metric_groups(logits,records,temperatures):
    from step1_common import summarize
    from step1_probe import grouped_metrics
    from step1_finish import selection_group,task_group
    import numpy as np
    partitions={'validation_all':list(range(len(records))),
        'validation_select':[i for i,r in enumerate(records) if selection_group(r)],
        'validation_check':[i for i,r in enumerate(records) if not selection_group(r)]}
    results={}
    for partition,indices in partitions.items():
        groups={'overall':indices}
        for group in ['reasoning','retrieval_matching','other']:
            groups[group]=[i for i in indices if task_group(records[i])==group]
        results[partition]={}
        for group,members in groups.items():
            if not members:continue
            scores=[logits[i] for i in members];rows=[records[i] for i in members]
            results[partition][group]={name:grouped_metrics(scores,rows,temp) for name,temp in temperatures.items()}
    soft=[i for i,r in enumerate(records) if not (np.count_nonzero(r['target'])==1 and np.isclose(max(r['target']),1))]
    return {'partitions':results,'soft_label_validation':{
        name:summarize([logits[i] for i in soft],[records[i] for i in soft],temp) for name,temp in temperatures.items()} if soft else {'rows':0}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',default='models/Open-Jev-2B/package')
    p.add_argument('--source',default='repos/Open-Jev')
    p.add_argument('--base-model',default='models/Qwen3.5-2B')
    p.add_argument('--base-manifest',default='scripts/step4/openjev-base-manifest.json')
    p.add_argument('--cache-dir',default='artifacts/step1_expanded/cache')
    p.add_argument('--data',default='data/Open-Jev-v1.1')
    p.add_argument('--out-dir',required=True)
    p.add_argument('--audit-only',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--benchmark-rows',type=int,default=32)
    p.add_argument('--benchmark-repeats',type=int,default=3)
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    if min(args.benchmark_rows,args.benchmark_repeats)<1:raise ValueError('Positive benchmark sizes required')
    root=Path(args.out_dir)
    if root.exists() and any(root.iterdir()):raise FileExistsError(root)
    package=audit_package(args.package);issues=[];base=None;source=None
    try:source=source_root(args.source)
    except (OSError,ValueError,subprocess.CalledProcessError) as e:issues.append('official source: '+str(e))
    try:base=audit_base(args.base_model,args.base_manifest)
    except (OSError,ValueError,KeyError) as e:issues.append('pinned base model: '+str(e))
    from importlib.metadata import version
    versions={name:version(name) for name in ['torch','transformers','peft','numpy','scipy','pyarrow']}
    if versions['transformers']!='5.10.2' or versions['peft']!='0.19.1':issues.append('use separate environment: transformers==5.10.2, peft==0.19.1')
    if args.audit_only:
        write(root/'results/audit.json',{'status':'AUDIT ONLY','package':package,'base':base,'packages':versions,'ready_for_inference':not issues,'blockers':issues})
        print(json.dumps({'package_integrity':'passed','ready_for_inference':not issues,'blockers':issues},ensure_ascii=False),flush=True)
        return
    if issues:raise RuntimeError('; '.join(issues))
    for key in ['JEV_LOAD_8BIT','JEV_LOAD_4BIT','JEV_DEVICE_MAP','JEV_MAX_MEMORY','JEV_ALLOW_OFFLOAD']:
        if os.environ.get(key):raise ValueError(f'Unset {key}; benchmark uses released BF16 single-device path')
    if os.environ.get('JEV_TORCH_DTYPE','bfloat16')!='bfloat16':raise ValueError('Released BF16 only')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    sys.path.insert(0,str(source))
    from jev.api import candidate_prompts as official_renderer
    from jev.model import DecisionModel
    import numpy as np
    import torch
    from peft import PeftModel
    from step1_common import fit_temperature
    if torch.device(args.device).type!='cuda':raise ValueError('This benchmark requires a CUDA device')
    torch.set_num_threads(4)
    inputs,records,cache_hashes=prepare_rows(args,official_renderer)
    protocol={'version':VERSION,'args':vars(args),'source_revision':SOURCE_REVISION,
        'source_sha256':{str(f.relative_to(source)):sha(f) for f in sorted((source/'jev').glob('*.py'))},
        'package':package,'base':base,'packages':versions,'source_cache_sha256':cache_hashes,
        'scripts_sha256':{name:sha(Path(__file__).resolve().parents[1]/name) for name in ['step4/openjev_eval.py','step4/openjev_compare.py','step1_common.py','step1_probe.py','step1_finish.py','step2_evaluate.py']},
        'rows':{s:len(r) for s,r in records.items()},'training':'none; released adapter/head unchanged',
        'limitations':['external pretrained decision checkpoint; different training budget/mixture; overlap not ruled out',
            'local dev-cal scaling is additional analysis, not replacement of released temperature',
            'validation is development data; no official calibration/test/ood reads',
            'latency includes tokenization, unlike existing pretokenized Ouro/Qwen timing']}
    out=root/'results';out.mkdir(parents=True,exist_ok=True);write(out/'protocol.json',protocol)
    # Same official load sequence, changing ONLY model location after exact pinned-file verification.
    model=DecisionModel(str(Path(args.base_model).resolve()),REVISION,device=args.device,lora_rank=0,max_length=package['model']['max_length'])
    checkpoint=Path(args.package)/'checkpoint'
    model.backbone=PeftModel.from_pretrained(model.backbone,checkpoint/'adapter')
    model.lora_rank=package['model']['lora_rank']
    model.head.load_state_dict(torch.load(checkpoint/'head.pt',map_location=args.device,weights_only=True),strict=True)
    model._validate_loaded_parameters();model.eval();model.requires_grad_(False)
    scores={};lengths={}
    with torch.inference_mode():
        for split,rows in inputs.items():
            values=[];lengths[split]=0
            for i,row in enumerate(rows):
                value=model([row])[0].float().cpu().numpy().astype(np.float64)
                if len(value)!=len(records[split][i]['target']) or not np.isfinite(value).all():raise ValueError('Invalid official output')
                values.append(value);lengths[split]+=model.last_input_tokens
                if (i+1)%100==0:print(f'{split}: {i+1}/{len(rows)}',flush=True)
            scores[split]=values
        n=min(args.benchmark_rows,len(inputs['validation']));samples=[]
        for repeat in range(args.benchmark_repeats+1):
            torch.cuda.synchronize(torch.device(args.device));started=time.perf_counter()
            for row in inputs['validation'][:n]:model([row])[0].softmax(0)
            torch.cuda.synchronize(torch.device(args.device))
            if repeat:samples.append(time.perf_counter()-started)
    fit=fit_temperature(scores['dev-cal'],records['dev-cal'])
    temps={'raw':1.,'released_temperature':package['released_temperature'],'temperature_scaled':fit['temperature']}
    metrics=metric_groups(scores['validation'],records['validation'],temps)
    torch.save({'external_head':{'records':records['validation'],'logits':[scores['validation']],
        'temperatures':[fit],'released_temperature':package['released_temperature']}},out/'validation_predictions.pt')
    write(out/'metrics.json',{'status':'SMOKE ONLY' if args.smoke else 'completed external development evaluation',
        'protocol_version':VERSION,'temperatures':temps,'dev_cal_temperature_fit':fit,'input_tokens':lengths,
        'methods':{'external_head':metrics},'latency':{'mean_seconds_per_decision':float(np.mean(samples)/n),
        'seconds_per_pass':samples,'row_ids':[r['id'] for r in records['validation'][:n]],
        'scope':'official model forward including tokenization, padding, device transfer and head; no generation or prefix cache'}})
    print(f'Completed external evaluation: {out}',flush=True)


if __name__=='__main__':main()
