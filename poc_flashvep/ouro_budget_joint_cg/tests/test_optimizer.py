import unittest
from ..optimizer import enumerate_candidates,BudgetSelector

class MeasuredModelFixture:
    # Synthetic definition fixture only. No performance evidence.
    def estimate(self,path,c,g,context,depth):
        return ((2 if path=='staged_prelude' else 5+g*(10+c/256)),True)

class OptimizerTests(unittest.TestCase):
    def state(self,**updates):
        return dict(kind='NEW_CHUNK',C=0,depth=0,start=0,prompt_remaining=8192,token_budget=2048,issued_decode=[],**updates)

    def test_utility_pays_all_four_loops(self):
        rows=enumerate_candidates(self.state(),80,MeasuredModelFixture())
        half=next(c for c in rows if c.C==512 and c.G==2)
        self.assertEqual(half.complete_tokens_now,0)
        self.assertEqual(half.residual_chunk_ms,2+2*(5+2*(10+512/256)))

    def test_staged_prelude_cannot_claim_planned_R4_completion_now(self):
        rows=enumerate_candidates(self.state(),200,MeasuredModelFixture())
        self.assertTrue(all(r.complete_tokens_now==0 for r in rows))

    def test_continuation_keeps_span_and_clips_group(self):
        s=dict(kind='CONTINUE_CHUNK',C=512,depth=2,start=0,prompt_remaining=8192,token_budget=2048,issued_decode=[])
        rows=enumerate_candidates(s,100,MeasuredModelFixture())
        self.assertTrue(all(c.C==512 and c.G<=2 for c in rows))
        self.assertEqual({c.G for c in rows},{1,2})

    def test_C_only_atomic_cannot_choose_half_work(self):
        rows=enumerate_candidates(self.state(atomic=True),200,MeasuredModelFixture(),'C_ONLY')
        self.assertTrue(all(c.G==4 and c.complete_tokens_now==c.C for c in rows))

    def test_unknown_future_labels_do_not_change_choice(self):
        a=self.state();b=dict(a,future_output_ids=[3],actual_next_token_ms=100000)
        selector=BudgetSelector(MeasuredModelFixture(),80)
        self.assertEqual(selector(a),selector(b))

if __name__=='__main__':
    unittest.main()
