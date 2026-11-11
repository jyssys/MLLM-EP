import unittest
from .gates import interference,same_window,model_contract


class Gates(unittest.TestCase):
    def test_tail_not_only_wall(self):
        ref=dict(wall=100.,ITL=80.,maximum=90.)
        self.assertFalse(interference(dict(wall=100.,ITL=88.,maximum=90.),ref,ref)['passed'])
        self.assertFalse(interference(dict(wall=100.,ITL=80.,maximum=100.),ref,ref)['passed'])
        self.assertTrue(interference(ref,ref,ref)['passed'])

    def test_same_run_and_window(self):
        a=dict(run='one',left=1.,right=2.)
        self.assertTrue(same_window(a,a))
        self.assertFalse(same_window(a,dict(a,run='two')))
        self.assertFalse(same_window(a,dict(a,right=3.)))

    def test_context_and_minimum_output(self):
        self.assertFalse(model_contract('huginn',2048,4))
        self.assertFalse(model_contract('huginn',4090,16))
        self.assertTrue(model_contract('huginn',3900,16,1))


if __name__=='__main__':unittest.main()
