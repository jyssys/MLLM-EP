"""CPU tests for input availability and generation/scoring invariants."""
import unittest
from types import SimpleNamespace
import torch
from poc_async_recurrent_pp.emulator import StageEmulator
from poc_async_recurrent_pp.scoring import numeric, score
from poc_async_recurrent_pp.contract import ROOT, read


class Model:
    config=SimpleNamespace(num_hidden_layers=24,total_ut_steps=4)
    model=SimpleNamespace(layers=[torch.nn.Identity() for _ in range(24)])


def coeff():
    return dict(alpha=[[1.0]*3 for _ in range(3)],w0_beta=[1.0]*3,
                w0_bias=[[0.0]*2048 for _ in range(3)])


def call(hook, stage, r, value):
    args,_=hook.hook(stage)(None,(value,),dict(current_ut=r,cache_position=torch.tensor([10])))
    return args[0]


class Semantics(unittest.TestCase):
    def test_freeze_does_not_read_current_preceding_output(self):
        hook=StageEmulator(Model(),'delta',coeff())
        try:
            for j in range(4):
                call(hook,j,0,torch.full((1,1,2048),float(j+1)))
                hook.output_hook(j)(None,None,torch.full((1,1,2048),float(j+2)))
            call(hook,0,1,torch.full((1,1,2048),5.0))
            frozen=hook.frozen[2].clone()
            # Serial emulator receives an intentionally corrupted preceding output.
            actual=call(hook,2,1,torch.full((1,1,2048),float('nan')))
            self.assertTrue(torch.equal(actual,frozen))
            self.assertTrue(torch.isfinite(actual).all())
        finally:
            hook.close()

    def test_w0_has_no_current_token_previous_loop(self):
        hook=StageEmulator(Model(),'w0',coeff())
        try:
            embedding=torch.ones((1,1,2048))
            call(hook,0,0,embedding)
            self.assertEqual(set(hook.frozen),{1,2,3})
            self.assertTrue(all(torch.equal(value,embedding) for value in hook.frozen.values()))
            self.assertIsNone(hook.previous)
        finally:
            hook.close()

    def test_prefill_is_exact(self):
        hook=StageEmulator(Model(),'w0',coeff())
        try:
            value=torch.ones((1,3,2048))
            for j in range(4):
                self.assertIs(call(hook,j,0,value),value)
            self.assertFalse(hook.frozen)
        finally:
            hook.close()

    def test_new_token_clears_old_token_activations(self):
        hook=StageEmulator(Model(),'stale',coeff())
        try:
            for j in range(4):
                call(hook,j,0,torch.ones((1,1,2048)))
                hook.output_hook(j)(None,None,torch.ones((1,1,2048)))
            call(hook,0,1,torch.ones((1,1,2048)))
            self.assertTrue(hook.frozen)
            call(hook,0,0,torch.ones((1,1,2048)))
            self.assertFalse(hook.frozen)
            self.assertIsNone(hook.previous)
        finally:
            hook.close()

    def test_numeric_parser(self):
        for text,expected in [('1,234','1234'),('-5','-5'),('2.50','2.5'),('3.0','3')]:
            self.assertEqual(numeric(text),numeric(expected))
        self.assertIsNone(numeric('the answer is perhaps 9'))

    def test_explicit_decimal_answer_is_not_cut_at_decimal_point(self):
        value=score(dict(dataset='gsm8k',answer='1234.5'),
                    dict(text='The final answer is $1,234.50$.',finish_reason='eos'))
        self.assertTrue(value['answer_matches'])

    def test_last_box_and_completion(self):
        row=dict(dataset='gsm8k',answer='42')
        result=dict(text=r'Work: \boxed{40}. Final: \boxed{42}',finish_reason='eos')
        self.assertTrue(score(row,result)['correct_completed'])
        result['finish_reason']='length'
        self.assertFalse(score(row,result)['correct_completed'])
        self.assertTrue(score(row,result)['answer_matches'])
        self.assertTrue(score(row,dict(text='42<|im_end|>',finish_reason='eos'))['correct_completed'])

    def test_hand_checked_math_equivalence(self):
        cases=[('42','42',True),(r'\frac{1}{2}','0.5',True),
               (r'\sqrt{4}','2',True),('x^2+2x+1','(x+1)^2',True),
               (r'\frac{6}{8}',r'\frac{3}{4}',True),(r'\pi',r'\pi',True),
               ('-7','7',False),('13','12',False),('0','0',True),('2.5',r'\frac{5}{2}',True)]
        for prediction,reference,expected in cases:
            with self.subTest(prediction=prediction,reference=reference):
                value=score(dict(dataset='math500',answer=reference),
                            dict(text='Final: \\boxed{'+prediction+'}',finish_reason='eos'))
                self.assertEqual(value['correct_completed'],expected)

    def test_calibration_disjoint(self):
        data=read(ROOT/'outputs/manifests/data.json')
        self.assertFalse({r['id'] for r in data['calibration']} & {r['id'] for r in data['full']})
        self.assertEqual(len(data['full']),1819)
        self.assertEqual(len({r['id'] for r in data['full']}),1819)

    def test_previous_round_uses_computed_output_not_recycled_prediction(self):
        hook=StageEmulator(Model(),'stale',coeff())
        try:
            for r in range(3):
                call(hook,0,r,torch.ones((1,1,2048)))
                for j in range(4):
                    if j:
                        value=call(hook,j,r,torch.zeros((1,1,2048)))
                        if r:
                            self.assertTrue(torch.equal(value,torch.full_like(value,float(10*r+j-1))))
                    hook.output_hook(j)(None,None,torch.full((1,1,2048),float(10*(r+1)+j)))
        finally:
            hook.close()

    def test_symbolic_stale_path_needs_three_neighbor_rounds(self):
        # Structural graph test only. These sets are not pretrained activations.
        previous=[set() for _ in range(4)]
        reached={}
        for recurrence in range(2,6):
            current=[{'fresh_F1_R2'} if recurrence==2 else set()]
            current += [set(previous[j-1]) for j in range(1,4)]
            reached[recurrence]='fresh_F1_R2' in current[3]
            previous=current
        self.assertFalse(reached[4])
        self.assertTrue(reached[5])


if __name__=='__main__':
    unittest.main()
