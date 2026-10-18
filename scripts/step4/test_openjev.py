"""CPU-only external-evaluation regression tests, no upstream weights required."""
import copy,json,tempfile,unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from openjev_eval import audit_package,metric_groups,sha,BASE_ID,REVISION,VERSION
from openjev_compare import compare

class ExternalTests(unittest.TestCase):
    def test_package_integrity_and_temperature(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'checkpoint').mkdir()
            (root/'checkpoint/model.json').write_text(json.dumps({'model_id':BASE_ID,'revision':REVISION,'lora_rank':8}))
            (root/'checkpoint/temperature.json').write_text(json.dumps({'temperature':1.5}))
            files={str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in (root/'checkpoint').iterdir()}
            (root/'manifest.json').write_text(json.dumps({'model':BASE_ID,'revision':REVISION,'files':files}))
            self.assertEqual(audit_package(root)['released_temperature'],1.5)
            (root/'checkpoint/temperature.json').write_text('{}')
            with self.assertRaises(ValueError):audit_package(root)

    def test_three_calibrations_and_comparison(self):
        rows=[{'id':str(i),'group_id':'group'+str(i),'kind':'choice','source':'reasoning-control-v1',
               'options':['a','b'],'target':[1.,0.],'permutation':[0,1]} for i in range(16)]
        logits=[np.array([2.,0.]) for _ in rows]
        scores=metric_groups(logits,rows,{'raw':1.,'released_temperature':1.5,'temperature_scaled':2.})
        values=scores['partitions']['validation_all']['overall']
        self.assertLess(values['raw']['overall']['nll'],values['released_temperature']['overall']['nll'])
        self.assertLess(values['released_temperature']['overall']['nll'],values['temperature_scaled']['overall']['nll'])
        c={'protocol':{'version':VERSION,'args':{'smoke':True},'source_cache_sha256':{'dev-cal':'a','validation':'b'}},
           'predictions':{'external_head':{'records':rows,'logits':[logits],'temperatures':[{'temperature':2.}],'released_temperature':1.5}}}
        r=copy.deepcopy(c);r['protocol']['version']='step3-multi-loop-v1'
        r['predictions']={'multi_loop_head':{'records':copy.deepcopy(rows),'logits':[logits]*4,'temperatures':[{'temperature':2.}]*4}}
        result=compare(c,r,2,20)
        self.assertEqual(result['subsets']['validation_all']['raw_vs_raw']['nll_difference'],0.)
        self.assertEqual(result['subsets']['validation_all']['dev_cal_vs_dev_cal']['nll_difference'],0.)
        bad=copy.deepcopy(r);bad['predictions']['multi_loop_head']['records'][0]['permutation']=[1,0]
        with self.assertRaises(ValueError):compare(c,bad,2,20)
        bad=copy.deepcopy(r);bad['protocol']['source_cache_sha256']['dev-cal']='x'
        with self.assertRaises(ValueError):compare(c,bad,2,20)

if __name__=='__main__':unittest.main()
