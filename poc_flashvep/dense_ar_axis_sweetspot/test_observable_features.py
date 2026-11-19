import copy
import unittest
from .observable_features import pressure,scores
from .replay import Cost


class FeaturesTests(unittest.TestCase):
    def fixture(self):
        return dict(observation=dict(P_ready_external=3,P_backlog=3,state_age_ms=1.),
            pending_D_requests=['q1','q2'],active_D_requests=['q0'],active_delivered_counts={'q0':8},
            predicted_active_D_finish=99999.)

    def test_future_predictor_field_cannot_change_feature(self):
        d=self.fixture();a=pressure(d,Cost(.2,.1,.3),Cost(.2,.1,.3),16,1)
        d['predicted_active_D_finish']=-99999.;d['observation']['D_available']=1e12
        self.assertEqual(a,pressure(d,Cost(.2,.1,.3),Cost(.2,.1,.3),16,1))

    def test_no_queue_has_unit_pressure(self):
        d=self.fixture();d['observation']['P_ready_external']=0
        d['active_D_requests']=[];d['pending_D_requests']=[]
        self.assertEqual(pressure(d,Cost(.2,.1,.3),Cost(.2,.1,.3),16,1)['pressure_ratio'],1.)

    def test_pressure_can_reverse_axis_score(self):
        table={'Layer':Cost(.3,.05,.3),'Token':Cost(.2,.15,.3)}
        low=scores(.1,table,1.,1);high=scores(2.,table,1.,1)
        self.assertEqual(min(low,key=low.get),'Token')
        self.assertEqual(min(high,key=high.get),'Layer')

    def test_request_count_is_not_a_cohort_count(self):
        d=self.fixture();d['active_D_requests']=['q0','q3']
        d['active_delivered_counts']={'q0':8,'q3':8}
        result=pressure(d,Cost(.2,.1,.3),Cost(.2,.1,.3),16,2)
        self.assertEqual(result['active_D_request_count'],2)
        self.assertEqual(result['active_D_cohort_count'],1)
