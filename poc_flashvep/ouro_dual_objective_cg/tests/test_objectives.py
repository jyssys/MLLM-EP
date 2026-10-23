"""Software checks. These fixtures do not establish pretrained quality."""
import json
import math
import unittest
from pathlib import Path
from ..common import PRIOR,ROOT,guard
from ..objective_features import prepare_items,contextualize_offline
from ..objective_families import choose,scores,NAMES
from ..adaptive_selector import DualSelector
from ...ouro_budget_joint_cg.cost_predictor import ServiceLookup
from ...ouro_budget_joint_cg.optimizer import enumerate_candidates

class ObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=ServiceLookup(json.loads((PRIOR/'cost_model_artifacts/service_lookup_v1.json').read_text()))
        cls.norm=json.loads((ROOT/'objective_normalization_v1.json').read_text())
        cls.state=dict(kind='NEW_CHUNK',id='a',C=0,start=0,depth=0,prompt_remaining=8192,token_budget=2048,
            atomic=False,issued_decode=[dict(id='b',R=2)],TBT_target_ms=200.,decode_age_max_ms=50.)

    def test_all_families_finite(self):
        cs=enumerate_candidates(self.state,math.inf,self.model)
        items=prepare_items(cs,self.state,self.norm,self.model)
        for family in NAMES:
            c,rank=choose(family,items)
            self.assertIn(c,cs);self.assertTrue(all(math.isfinite(x) for x in rank))

    def test_U2_identity(self):
        cs=enumerate_candidates(self.state,120,self.model)
        items=prepare_items(cs,self.state,self.norm,self.model)
        expected=max(cs,key=lambda c:(c.utility,c.complete_tokens_now,-c.residual_chunk_ms,c.G))
        self.assertEqual(choose('O0',items)[0],expected)

    def test_endpoint_ablation_shared(self):
        cs=enumerate_candidates(self.state,120,self.model);items=prepare_items(cs,self.state,self.norm,self.model)
        expected=choose('O2',items,endpoint='E_ONLY')[0]
        for family in NAMES:self.assertEqual(choose(family,items,weight=8,endpoint='E_ONLY')[0],expected)

    def test_O1_feature_invariance(self):
        cs=enumerate_candidates(self.state,120,self.model);items=prepare_items(cs,self.state,self.norm,self.model)
        full=scores('O1',items)
        pairs=[(pair,rank) for pair,rank in zip(items,full) if pair[0].C==2048]
        self.assertEqual(scores('O1',[p for p,_ in pairs]),[r for _,r in pairs])

    def test_no_resize_continuation(self):
        s=dict(self.state,kind='CONTINUE_CHUNK',C=512,depth=1,start=1024)
        for family in NAMES:
            selector=DualSelector(self.model,120,self.norm,family=family)
            c=selector(s);self.assertEqual(c.C,512);self.assertLessEqual(c.G,3)

    def test_G_control_changes_only_surface(self):
        selector=DualSelector(self.model,120,self.norm,mode='G_ONLY',fixed_C=512)
        cs=selector.choices(self.state)
        self.assertTrue(cs);self.assertEqual({c.C for c in cs},{512})

    def test_all_chunk_rows_are_clipped(self):
        s=dict(self.state,prompt_remaining=37,token_budget=23)
        self.assertEqual({c.C for c in enumerate_candidates(s,math.inf,self.model)},{23})

    def test_prelude_receives_no_completion_credit(self):
        for c in enumerate_candidates(self.state,math.inf,self.model):self.assertEqual(c.complete_tokens_now,0)

    def test_future_output_is_not_used(self):
        s=dict(self.state,id='a')
        data=dict(requests=[dict(id='a',token_times=[],engine_enqueue=0),dict(id='b',token_times=[1.,2.],engine_enqueue=0)])
        left=contextualize_offline(s,data,2.5)
        data['requests'][1]['token_times'].append(100.)
        self.assertEqual(contextualize_offline(s,data,2.5),left)

    def test_feedback_has_minimum_support(self):
        cs=enumerate_candidates(self.state,120,self.model)
        a=prepare_items(cs,dict(self.state,past_TBT_samples=1,past_TBT_violation_fraction=0),self.norm,self.model)
        b=prepare_items(cs,dict(self.state,past_TBT_samples=1,past_TBT_violation_fraction=1),self.norm,self.model)
        self.assertEqual(scores('O13',a),scores('O13',b))

if __name__=='__main__':unittest.main()
