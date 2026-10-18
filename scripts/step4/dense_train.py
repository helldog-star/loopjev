"""Step4 Qwen3 Base candidate decision baseline; no generative reasoning."""
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pyarrow.parquet as pq
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from step1_common import candidate_prompts, fingerprint
from step2_train import SPLITS, validated_caches, set_seed, row_schedule, decision_loss, candidate_batches
from step2_evaluate import evaluate, write_json

VERSION = 'step4-qwen3-dense-v1'
PROMPT = 'openjev-candidate-plain+Answer-newline-v1'
SEMANTIC_KEYS = ('id', 'group_id', 'source', 'kind', 'options', 'target', 'permutation')


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='models/Qwen3-1.7B-Base')
    p.add_argument('--source-model', default='models/Ouro-1.4B')
    p.add_argument('--data', default='data/Open-Jev-v1.1')
    p.add_argument('--cache-dir', default='artifacts/step1_expanded/cache')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--batch-rows', type=int, default=16)
    p.add_argument('--head-lr', type=float, default=.001)
    p.add_argument('--lora-lr', type=float, default=.0002)
    p.add_argument('--l2', type=float, default=.001)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--candidate-batch', type=int, default=4)
    p.add_argument('--token-budget', type=int, default=4096)
    p.add_argument('--max-length', type=int, default=4096)
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--benchmark-rows', type=int, default=32)
    p.add_argument('--benchmark-repeats', type=int, default=3)
    p.add_argument('--preflight-only', action='store_true', help='CPU model/data/tokenizer audit; never loads weights or uses CUDA')
    p.add_argument('--smoke', action='store_true')
    return p.parse_args()


def semantic_records(records):
    return [{key: row[key] for key in SEMANTIC_KEYS} for row in records]


def render_prompt(raw):
    return raw + '\nAnswer:\n'


def prepare(args):
    cfg = AutoConfig.from_pretrained(args.model, local_files_only=True)
    if cfg.model_type != 'qwen3' or cfg.hidden_size != 2048 or cfg.num_hidden_layers != 28:
        raise ValueError('This locked pilot expects Qwen3-1.7B architecture')
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tok.padding_side = 'right'
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ids = [tok.encode(word, add_special_tokens=False) for word in ['Yes', 'No']]
    if any(len(value) != 1 for value in ids):
        raise ValueError('Yes and No must each be one token')
    yes, no = ids[0][0], ids[1][0]
    # Validate the original frozen memberships/model provenance without using their hidden states.
    caches = validated_caches(SimpleNamespace(cache_dir=args.cache_dir, model=args.source_model, smoke=args.smoke))
    ouro_tok = AutoTokenizer.from_pretrained(args.source_model, local_files_only=True, trust_remote_code=True)
    prepared, inputs, reports = {}, {}, {}
    for split, payload in caches.items():
        actual_split = 'train' if split == 'dev-cal' else split
        paths = sorted((Path(args.data)/'data/community-hard-mix-v2-redistributable').glob(actual_split+'-*.parquet'))
        actual_hashes = {p.name: fingerprint(p) for p in paths}
        expected = {Path(p).name: h for p,h in payload['config']['data_sha256'].items()}
        if actual_hashes != expected:
            raise ValueError('Dataset files differ from locked Step1 provenance')
        selected = {r['id'] for r in payload['rows']}
        source = {}
        for row in pq.read_table(paths, columns=['id','group_id','source','kind','state_json','question','options','target']).to_pylist():
            if row['id'] in selected:
                if row['id'] in source:
                    raise ValueError('Duplicate selected row id')
                source[row['id']] = row
        rows, tokens, offsets, raw_hashes = [], [], [0], []
        old_lengths, lengths = [], []
        for cached in payload['rows']:
            row = source[cached['id']]
            permutation = cached['permutation']
            if sorted(permutation) != list(range(len(row['options']))):
                raise ValueError('Invalid permutation')
            for key in ['group_id', 'source', 'kind']:
                if row[key] != cached[key]:
                    raise ValueError('Metadata mismatch')
            for key in ['options','target']:
                if [row[key][i] for i in permutation] != cached[key]:
                    raise ValueError('Candidate/target mismatch')
            raw = candidate_prompts({**row,'state':json.loads(row['state_json'])})
            raw = [raw[i] for i in permutation]
            original = [ouro_tok.apply_chat_template([{'role':'user','content':s}], tokenize=False, add_generation_prompt=True) for s in raw]
            if [hashlib.sha256(s.encode()).hexdigest() for s in original] != cached['prompt_sha256']:
                raise ValueError('Original candidate renderer or question/state differs')
            prompts = [render_prompt(s) for s in raw]
            encoded = tok(prompts, add_special_tokens=False, truncation=False)['input_ids']
            for prompt, sequence in zip(prompts, encoded, strict=True):
                # Initialization must correspond to actual next-token boundaries, not guessed IDs.
                for word, token_id in [('Yes',yes),('No',no)]:
                    if tok.encode(prompt+word,add_special_tokens=False) != sequence+[token_id]:
                        raise ValueError('Contextual Yes/No boundary differs from initialization')
            if any(len(seq)>min(args.max_length,args.token_budget) for seq in encoded):
                raise ValueError(f'Qwen prompt exceeds fixed limit at {row["id"]}; no silent truncation or row dropping')
            new_lengths=list(map(len,encoded))
            rows.append({**cached,'reference_prompt_sha256':cached['prompt_sha256'],
                         'reference_token_lengths':cached['token_lengths'],
                         'prompt_sha256':[hashlib.sha256(s.encode()).hexdigest() for s in prompts],
                         'token_lengths':new_lengths})
            tokens.extend(encoded); offsets.append(len(tokens))
            raw_hashes.extend(hashlib.sha256(s.encode()).hexdigest() for s in raw)
            old_lengths.extend(cached['token_lengths']); lengths.extend(new_lengths)
        prepared[split]={'rows':rows,'row_offsets':torch.tensor(offsets,dtype=torch.long),
                         'config':{'kind':'choice','split':split,'total_ut_steps':1,'prompt_format':PROMPT}}
        inputs[split]=tokens
        reports[split]={'rows':len(rows),'candidates':len(tokens),'truncated_candidates':0,
            'qwen_tokens':sum(lengths),'ouro_tokens':sum(old_lengths),'qwen_max_length':max(lengths),
            'qwen_mean_length':float(np.mean(lengths)),
            'semantic_rows_sha256':hashlib.sha256(json.dumps(semantic_records(rows),sort_keys=True).encode()).hexdigest(),
            'raw_candidate_prompts_sha256':hashlib.sha256(json.dumps(raw_hashes).encode()).hexdigest()}
    model_root=Path(args.model)
    weights=sorted(model_root.glob('*.safetensors'))
    if not weights:
        raise FileNotFoundError('No safetensors weights found')
    names=['config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','model.safetensors.index.json']
    provenance={'model':str(model_root.resolve()),'model_sha256':{p.name:fingerprint(p) for p in weights},
        'source_sha256':{name:fingerprint(model_root/name) for name in names if (model_root/name).is_file()},
        'revision_note':'local download; upstream commit not assumed; file SHA256 records exact snapshot',
        'config':cfg.to_dict(),'yes_id':yes,'no_id':no,'prompt_format':PROMPT,
        'uses_chat_template':False,'generates_reasoning':False}
    return tok,prepared,inputs,reports,provenance


def load_dense(args, provenance):
    model=AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
        torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa')
    model.requires_grad_(False)
    head=(model.lm_head.weight[provenance['yes_id']].float()-model.lm_head.weight[provenance['no_id']].float()).detach().clone()
    names=[name for name,module in model.named_modules() if isinstance(module,torch.nn.Linear) and name.endswith(('.q_proj','.v_proj'))]
    if len(names)!=56:
        raise ValueError('Expected 28 Q/V pairs')
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.0,bias='none',target_modules=['q_proj','v_proj']))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    if any(p.requires_grad and 'lora_' not in name for name,p in model.named_parameters()):
        raise ValueError('Unexpected trainable backbone parameter')
    count=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count!=1605632:
        raise ValueError(f'Unexpected adapter parameter count: {count}')
    return model,head


def last_hidden(model,tok,sequences):
    device=next(model.parameters()).device
    encoded=tok.pad({'input_ids':sequences},padding=True,return_tensors='pt').to(device)
    decoder=model.get_base_model().model if hasattr(model,'peft_config') else model.model
    result=decoder(**encoded,use_cache=False,return_dict=True)
    positions=encoded['attention_mask'].sum(-1)-1
    features=result.last_hidden_state[torch.arange(len(sequences),device=device),positions].float()
    return features,encoded['input_ids'].numel()


def row_logits(model,tok,tokens,start,end,head,args):
    scores=[]; padded=0
    for indices in candidate_batches(range(start,end),tokens,args):
        features,count=last_hidden(model,tok,[tokens[i] for i in indices])
        scores.append(features@head);padded+=count
    return torch.cat(scores),padded


def train(args,payload,tokens,tok,model,initial):
    model.train();head=initial.clone().requires_grad_()
    adapters=[p for p in model.parameters() if p.requires_grad]
    params=[head,*adapters]
    optimizer=torch.optim.AdamW([{'params':[head],'lr':args.head_lr},{'params':adapters,'lr':args.lora_lr}],weight_decay=0)
    schedule=list(row_schedule(len(payload['rows']),args.batch_rows,args.seed,args.steps))
    budget={'optimizer_steps':args.steps,'decision_rows':0,'candidates':0,'unpadded_tokens':0,'padded_tokens':0,
        'runtime_loops':1,'row_schedule_sha256':hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),
        'head_parameters':head.numel(),'adapter_parameters':sum(p.numel() for p in adapters),
        'gradient_checkpointing':True,'trainable_head_bias':False}
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    start,end_event=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record();started=time.perf_counter();curve=[]
    for update,indices in enumerate(schedule,1):
        optimizer.zero_grad(set_to_none=True);total=0.
        for index in indices:
            row=payload['rows'][index];a,b=payload['row_offsets'][index:index+2].tolist()
            scores,padded=row_logits(model,tok,tokens,a,b,head,args)
            loss=decision_loss(scores,row['target'])/len(indices)
            if not torch.isfinite(loss):raise ValueError('Non-finite decision loss')
            loss.backward();total+=float(loss.detach())
            budget['decision_rows']+=1;budget['candidates']+=b-a
            budget['unpadded_tokens']+=sum(row['token_lengths']);budget['padded_tokens']+=padded
        penalty=args.l2*head.square().sum();penalty.backward()
        norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
        if update==1 or update%50==0 or update==args.steps:
            curve.append({'update':update,'decision_loss':total,'head_penalty':float(penalty.detach()),'gradient_norm':float(norm)})
            print(f'update={update}/{args.steps} decision_loss={total:.4f}',flush=True)
    end_event.record();torch.cuda.synchronize()
    budget.update(wall_seconds=time.perf_counter()-started,cuda_event_span_seconds=start.elapsed_time(end_event)/1000,
        cuda_event_span_definition='elapsed device timeline including host gaps; not active kernel time or FLOPs',
        peak_allocated_bytes=torch.cuda.max_memory_allocated())
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):raise ValueError('Frozen weights received gradient')
    if not torch.isfinite(head).all() or torch.equal(head.detach(),initial):raise ValueError('Head did not update')
    return head.detach().cpu(),curve,budget


@torch.no_grad()
def extract(model,tok,tokens,payload,head,args):
    model.eval();hidden=torch.empty(len(tokens),1,len(head),dtype=torch.float16)
    torch.cuda.synchronize();started=time.perf_counter();padded=0
    for indices in candidate_batches(sorted(range(len(tokens)),key=lambda i:len(tokens[i])),tokens,args):
        features,count=last_hidden(model,tok,[tokens[i] for i in indices])
        hidden[indices,0]=features.cpu().half();padded+=count
    torch.cuda.synchronize()
    if not torch.isfinite(hidden).all():raise ValueError('Non-finite extracted features')
    return {**payload,'hidden':hidden},{'wall_seconds':time.perf_counter()-started,'padded_tokens':padded,'candidates':len(tokens)}


@torch.inference_mode()
def benchmark(model,tok,tokens,payload,head,args):
    model.eval();head=head.cuda();n=min(args.benchmark_rows,len(payload['rows']));samples=[]
    for repeat in range(args.benchmark_repeats+1):
        torch.cuda.synchronize();started=time.perf_counter()
        for index in range(n):
            a,b=payload['row_offsets'][index:index+2].tolist()
            scores,_=row_logits(model,tok,tokens,a,b,head,args);scores.softmax(0)
        torch.cuda.synchronize()
        if repeat:samples.append(time.perf_counter()-started)
    return {'row_ids':[r['id'] for r in payload['rows'][:n]],'decision_rows':n,'candidate_batch':args.candidate_batch,
        'repeats':args.benchmark_repeats,'warmup_passes':1,'runtime_loops':1,'seconds_per_pass':samples,
        'mean_seconds_per_decision':float(np.mean(samples)/n),
        'scope':'GPU model forward + decision head; pretokenized; no autoregressive generation'}


def main():
    args=arguments()
    if min(args.steps,args.batch_rows,args.candidate_batch,args.token_budget,args.max_length,args.bootstrap,args.benchmark_rows,args.benchmark_repeats)<1 or min(args.head_lr,args.lora_lr)<=0 or args.l2<0 or args.seed<0:
        raise ValueError('Invalid arguments')
    if args.smoke and args.steps>2:raise ValueError('Smoke requires --steps 1 or 2')
    root=Path(args.out_dir)
    if root.exists() and any(root.iterdir()):raise FileExistsError('Use a new empty out-dir')
    torch.set_num_threads(4);set_seed(args.seed)
    tok,payloads,inputs,audit,provenance=prepare(args)
    scripts=Path(__file__).resolve().parents[1]
    source_hashes={s:fingerprint(Path(args.cache_dir)/f'{s}.pt') for s in SPLITS}
    protocol={'version':VERSION,'args':vars(args),'rows':{s:len(p['rows']) for s,p in payloads.items()},
        'source_cache_sha256':source_hashes,'model_provenance':provenance,'input_audit':audit,
        'supervision':'dense final hidden state; row-average soft CE + 0.1 Brier; L2 once on raw head; no bias',
        'initialization':'own model bare Yes-minus-No output rows; contextual token boundaries verified',
        'lora':{'rank':8,'alpha':16,'dropout':0.0,'targets':['q_proj','v_proj']},
        'probe':{'steps':300,'lr':.01,'l2':.01,'train_only_standardization':True},
        'packages':{p:version(p) for p in ['torch','transformers','peft','numpy','pyarrow']},
        'scripts_sha256':{name:fingerprint(scripts/name) for name in ['step4/dense_train.py','step4/dense_compare.py','step2_train.py','step2_evaluate.py','step1_common.py','step1_probe.py','step1_finish.py']},
        'limitations':['downstream protocol control, not controlled pretraining or architecture-only causality',
            'different tokenizer/prompt wrapper and token counts; no silent truncation',
            'equal row updates are not equal FLOPs','single-seed development; previously inspected validation',
            'dev-cal only; no official calibration/test/ood fitting or evaluation']}
    results=root/'results';results.mkdir(parents=True,exist_ok=True)
    if args.preflight_only:
        write_json(results/'preflight.json',{'status':'CPU PREFLIGHT ONLY','protocol':protocol})
        print(f'CPU preflight passed: {audit}; no weights loaded or GPU computation',flush=True)
        return
    if not torch.cuda.is_available():raise RuntimeError('Training requires CUDA; use --preflight-only for CPU audit')
    protocol['gpu']={'name':torch.cuda.get_device_name(),'memory_bytes':torch.cuda.get_device_properties(0).total_memory}
    write_json(results/'protocol.json',protocol)
    model,initial=load_dense(args,provenance)
    protocol['parameter_counts']={'total_with_adapter':sum(p.numel() for p in model.parameters()),
        'adapter':sum(p.numel() for p in model.parameters() if p.requires_grad),'head':initial.numel(),
        'initial_head_l2_norm':float(initial.norm())}
    write_json(results/'protocol.json',protocol)
    old={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    head,curve,cost=train(args,payloads['train'],inputs['train'],tok,model,initial)
    cost['changed_adapter_tensors']=sum(not torch.equal(p.detach().cpu(),old[n]) for n,p in model.named_parameters() if n in old)
    if cost['changed_adapter_tensors']==0:raise ValueError('Adapter did not update')
    model.save_pretrained(results/'adapter',safe_serialization=True,save_embedding_layers=False)
    torch.save({'weights':head,'loops':1,'protocol':protocol},results/'head.pt')
    fresh={};extraction={};cache=root/'cache';cache.mkdir()
    adapter_hashes={p.name:fingerprint(p) for p in (results/'adapter').iterdir() if p.is_file()}
    for split in SPLITS:
        fresh[split],extraction[split]=extract(model,tok,inputs[split],payloads[split],head,args)
        fresh[split]['config'].update(model_sha256=provenance['model_sha256'],source_sha256=provenance['source_sha256'],
            source_cache_sha256=source_hashes[split],adapter_sha256=adapter_hashes,scripts_sha256=protocol['scripts_sha256'])
        torch.save(fresh[split],cache/f'{split}.pt')
    latency=benchmark(model,tok,inputs['validation'],payloads['validation'],head,args)
    del model,old;torch.cuda.empty_cache()
    methods,predictions,probe=evaluate(fresh,head,SimpleNamespace(loops=1,bootstrap=args.bootstrap))
    methods['dense_head']=methods.pop('final_only_head');predictions['dense_head']=predictions.pop('final_only_head')
    torch.save(probe,results/'probe.pt');torch.save(predictions,results/'validation_predictions.pt')
    write_json(results/'metrics.json',{'status':'SMOKE ONLY' if args.smoke else 'completed single-seed development condition',
        'protocol_version':VERSION,'mode':'lora','loops':1,'training_curve':curve,'training_cost':cost,
        'extraction_cost':extraction,'latency':latency,'methods':methods})
    print(f'Completed Qwen3 dense baseline; results={results}',flush=True)


if __name__=='__main__':main()
