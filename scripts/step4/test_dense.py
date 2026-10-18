"""CPU-only regression checks with a tiny random Qwen3, never the downloaded weights."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from transformers.tokenization_utils_base import BatchEncoding
from peft import LoraConfig,PeftModel,get_peft_model
import step2_evaluate
from step1_probe import train_probe
from step2_train import decision_loss
from dense_train import last_hidden,row_logits,render_prompt
from dense_compare import compare,check_pair

class TinyTokenizer:
    def pad(self,inputs,padding,return_tensors):
        sequences=inputs['input_ids'];length=max(map(len,sequences))
        return BatchEncoding({'input_ids':torch.tensor([s+[0]*(length-len(s)) for s in sequences]),
            'attention_mask':torch.tensor([[1]*len(s)+[0]*(length-len(s)) for s in sequences])})

class DenseTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17);torch.set_num_threads(2)
        cfg=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
            num_attention_heads=2,num_key_value_heads=1,head_dim=16,max_position_embeddings=64,
            attention_dropout=0.,tie_word_embeddings=True,pad_token_id=0)
        self.base=Qwen3ForCausalLM(cfg).float().eval();self.cfg=cfg
        self.tok=TinyTokenizer();self.tokens=[[2,4,7],[3,5,2,8,9],[8,4]]
        self.args=SimpleNamespace(candidate_batch=1,token_budget=64)

    def test_padding_microbatch_and_row_loss(self):
        together,_=last_hidden(self.base,self.tok,self.tokens)
        separate=torch.cat([last_hidden(self.base,self.tok,[s])[0] for s in self.tokens])
        torch.testing.assert_close(together,separate,atol=1e-5,rtol=1e-5)
        head=torch.randn(32,requires_grad=True)
        logits,_=row_logits(self.base,self.tok,self.tokens,0,3,head,self.args)
        torch.testing.assert_close(logits,together@head,atol=1e-5,rtol=1e-5)
        target=torch.tensor([0.,.3,.7]);loss=decision_loss(logits,target.tolist())
        reference=-(target*torch.log_softmax(logits,0)).sum()+.1*(torch.softmax(logits,0)-target).square().sum()
        torch.testing.assert_close(loss,reference)
        loss.backward();self.assertGreater(float(head.grad.abs().sum()),0.)
        self.assertEqual(render_prompt('candidate'),'candidate\nAnswer:\n')

    def test_lora_backward_checkpoint_and_reload(self):
        base_state=copy.deepcopy(self.base.state_dict());self.base.requires_grad_(False)
        model=get_peft_model(self.base,LoraConfig(r=2,lora_alpha=4,lora_dropout=0.,bias='none',target_modules=['q_proj','v_proj']))
        model.train();head=torch.randn(32,requires_grad=True)
        parameters=[p for p in model.parameters() if p.requires_grad]
        old=[p.detach().clone() for p in parameters]
        def grads(checkpoint):
            if checkpoint:model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            else:model.gradient_checkpointing_disable()
            model.zero_grad(set_to_none=True);head.grad=None
            scores,_=row_logits(model,self.tok,self.tokens,0,3,head,self.args)
            loss=decision_loss(scores,[0.,0.,1.]);loss.backward()
            return {n:p.grad.clone() for n,p in model.named_parameters() if p.requires_grad}
        ordinary=grads(False);checkpoint=grads(True)
        for name in ordinary:torch.testing.assert_close(ordinary[name],checkpoint[name],atol=1e-6,rtol=1e-5)
        torch.optim.AdamW(parameters+[head],lr=.001).step()
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(old,parameters)))
        self.assertFalse(any(p.grad is not None for p in model.parameters() if not p.requires_grad))
        model.eval();expected,_=last_hidden(model,self.tok,self.tokens)
        with tempfile.TemporaryDirectory(prefix='loopjev-step4-cpu-') as d:
            model.save_pretrained(d,safe_serialization=True,save_embedding_layers=False)
            base=Qwen3ForCausalLM(self.cfg);base.load_state_dict(base_state)
            reloaded=PeftModel.from_pretrained(base,d).eval()
            actual,_=last_hidden(reloaded,self.tok,self.tokens)
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)

    def test_shared_evaluator_one_dense_depth(self):
        payload={}
        for split in ['train','dev-cal','validation']:
            rows=[{'id':f'{split}-{i}','group_id':f'{split}-group-{i}','source':'reasoning-control-v1',
                'target':[1.,0.] if i%2 else [0.,1.]} for i in range(12)]
            payload[split]={'rows':rows,'row_offsets':torch.arange(0,25,2),'hidden':torch.randn(24,1,32)}
        def cpu_probe(data,args):
            args.device='cpu';args.steps=3
            return train_probe(data,args)
        with patch.object(step2_evaluate,'train_probe',cpu_probe):
            methods,predictions,_=step2_evaluate.evaluate(payload,torch.randn(32),SimpleNamespace(loops=1,bootstrap=10))
        self.assertEqual(len(methods['final_only_head']),1)
        self.assertEqual(len(predictions['final_only_head']['logits']),1)
        self.assertTrue(np.isfinite(methods['final_only_head'][0]['partitions']['validation_all']['overall']['raw']['nll']))

    def test_comparison_controls_and_intermediate_latency(self):
        rows=[{'id':str(i),'group_id':f'g{i}','source':'reasoning-control-v1','kind':'choice',
               'options':['a','b'],'target':[1.,0.],'permutation':[0,1],'token_lengths':[3,4]} for i in range(12)]
        prediction={'records':rows,'logits':[[np.array([1.,0.]) for _ in rows]],'temperatures':[{'temperature':1.}]}
        args={k:1 for k in ['steps','batch_rows','seed','head_lr','lora_lr','l2','candidate_batch','token_budget']};args.update(smoke=True,mode='lora')
        proto={'args':args,'rows':{'train':12},'source_cache_sha256':{'train':'same'},'lora':{'rank':8},'probe':{},'packages':{},'gpu':{'name':'test'},
            'scripts_sha256':{n:'same' for n in ['step2_train.py','step2_evaluate.py','step1_common.py','step1_probe.py','step1_finish.py']}}
        cost={k:1 for k in ['optimizer_steps','decision_rows','candidates','row_schedule_sha256','runtime_loops']}
        latency={'row_ids':list(range(12)),'candidate_batch':1,'repeats':1,'warmup_passes':1,'scope':'same','runtime_loops':1,'mean_seconds_per_decision':.1}
        c={'protocol':{**copy.deepcopy(proto),'version':'step4-qwen3-dense-v1'},'metrics':{'loops':1,'training_cost':cost,'latency':latency},
           'predictions':{'dense_head':copy.deepcopy(prediction),'independent_probe':copy.deepcopy(prediction)}}
        r={'protocol':{**copy.deepcopy(proto),'version':'step3-multi-loop-v1'},
           'metrics':{'loops':4,'training_cost':{**cost,'runtime_loops':4},'latency':{**latency,'runtime_loops':4,'mean_seconds_per_decision':.2}},
           'predictions':{k:{'records':copy.deepcopy(rows),'logits':prediction['logits']*4,'temperatures':prediction['temperatures']*4} for k in ['multi_loop_head','independent_probe']}}
        # Different tokenizer metadata is allowed; changed semantic identity is not.
        for entry in c['predictions'].values():entry['records'][0]['token_lengths']=[8,9]
        result=compare(c,r,2,10)
        self.assertEqual(result['latency']['reference_executed_loops'],4)
        self.assertEqual(result['dense_head']['validation_all']['raw']['accuracy_difference'],0.)
        bad=copy.deepcopy(c);bad['protocol']['args']['seed']=999
        with self.assertRaises(ValueError):check_pair(bad,r)
        bad=copy.deepcopy(c);bad['predictions']['dense_head']['records'][0]['target']=[0.,1.]
        with self.assertRaises(ValueError):compare(bad,r,2,10)
        with self.assertRaises(ValueError):compare(c,r,5,10)

if __name__=='__main__':unittest.main()
