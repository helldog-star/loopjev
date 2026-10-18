"""CPU checks for causal, row-level offline exit semantics."""
import unittest
import numpy as np
from scipy.special import logit
from offline_exit import confidence, exits, picked, noninferior

class ExitTests(unittest.TestCase):
    def test_first_eligible_round_and_no_future_information(self):
        scores=np.array([[1.,.9,.1,0.],[1.,.2,.9,0.],[1.,.1,.2,1.]])
        np.testing.assert_array_equal(exits(scores,.8),[2,3,4])
        np.testing.assert_array_equal(exits(scores,None),[4,4,4])
        changed=scores.copy();changed[0,2:]=1.;changed[:,3]=0.
        np.testing.assert_array_equal(exits(changed,.8),exits(scores,.8))
    def test_gate_sigmoid_cumulative_then_candidate_aggregation(self):
        logits=[[np.array([0.,1.])]]*4
        gates=logit(np.array([[.1,.2,.3,.4],[.2,.3,.4,.5]]))
        c=confidence(logits,[1.]*4,gates,[0,2])
        cumulative=1.-np.cumprod(1.-np.array([[.1,.2,.3,.4],[.2,.3,.4,.5]]),axis=1)
        np.testing.assert_allclose(c['gate_cumulative_min'][0],cumulative.min(0))
        np.testing.assert_allclose(c['gate_cumulative_mean'][0],cumulative.mean(0))
    def test_candidate_permutation_and_whole_row_depth(self):
        logits=[[np.array([t,2*t,-t],float),np.array([-t,t],float)] for t in [1,2,3,4]]
        gates=np.arange(20).reshape(5,4)/10.-2
        original=confidence(logits,[1.,2.,3.,4.],gates,[0,3,5])
        perm=[2,0,1,4,3]
        changed=confidence([[z[::-1] if len(z)==2 else z[[2,0,1]] for z in loop] for loop in logits],[1.,2.,3.,4.],gates[perm],[0,3,5])
        for k in original:np.testing.assert_allclose(original[k],changed[k])
        output=picked(logits,np.array([2,3]),[0,1],[1.,2.,3.,4.])
        np.testing.assert_allclose(output[0],logits[1][0]/2)
        np.testing.assert_allclose(output[1],logits[2][1]/3)
    def test_both_quality_bounds_required(self):
        self.assertTrue(noninferior({'accuracy_one_sided_95_lower':0.,'nll_one_sided_95_upper':0.}))
        self.assertFalse(noninferior({'accuracy_one_sided_95_lower':-.01,'nll_one_sided_95_upper':0.}))
        self.assertFalse(noninferior({'accuracy_one_sided_95_lower':0.,'nll_one_sided_95_upper':.02}))

if __name__=='__main__':unittest.main()
