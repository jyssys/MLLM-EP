"""Test static cohort ordering and timestamp preservation on CPU."""
import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest
from .fixed_matched_cohort import FirstTokenBarrier, defer_running, has_selected_prefill, transform_schedule, transform_coordinator


def request(computed):
    return SimpleNamespace(num_computed_tokens=computed,num_prompt_tokens=8000,
                           kv_transfer_params=dict(dense_cohort_id='g',dense_cohort_size=2))


class FixedMatchedCohortTests(unittest.TestCase):
    def test_new_scope_does_not_modify_frozen_batch1_configs(self):
        from .fixed_cohort_probe import configuration
        from .batch_matched import state_configuration
        cfg=configuration('Layer',4)
        old=state_configuration('Layer',4)
        self.assertEqual(old['D_admission_batch'],1)
        self.assertEqual(cfg['D_admission_batch'],2)
        self.assertEqual(cfg['P_group_size'],1)
        self.assertTrue(cfg['correctness_only'])
        self.assertEqual(cfg['token_budget'],8192)
        self.assertEqual(cfg['D_token_budget'],8192)

    def test_coordinator_waits_for_the_fixed_size_without_changing_B(self):
        from .serving import main
        code=transform_coordinator(inspect.getsource(main))
        self.assertIn('len(pending_D)>=cfg[',code)
        self.assertIn('query_budget=8192',code)
        compile(code,'CPU_fixed_cohort_coordinator','exec')

    def test_ready_decode_waits_for_the_other_prompt(self):
        a,b=request(8000),request(0)
        scheduler=SimpleNamespace(requests=dict(a=a,b=b))
        self.assertTrue(defer_running(scheduler,a,{}))
        self.assertFalse(defer_running(scheduler,b,{}))
        b.num_computed_tokens=8000
        self.assertFalse(defer_running(scheduler,a,{}))
        del scheduler.requests['b']
        self.assertFalse(defer_running(scheduler,a,{}))

    def test_one_selected_prefill_does_not_change_the_token_budget(self):
        scheduler=SimpleNamespace(requests=dict(a=request(0),b=request(0)))
        self.assertTrue(has_selected_prefill(scheduler,dict(a=8000)))
        self.assertTrue(defer_running(scheduler,scheduler.requests['b'],dict(a=8000)))
        self.assertFalse(has_selected_prefill(scheduler,{}))

    def test_first_token_hold_preserves_all_message_times_and_counts(self):
        values=[];audit=[]
        commands=dict(kind='add_batch',commands=[dict(kind='add',request_id=r,
            kv_params=dict(dense_cohort_id='g',dense_cohort_size=2)) for r in ('q0','q1')])
        raw=SimpleNamespace(recv=lambda:commands,send=values.append,close=lambda:None)
        channel=FirstTokenBarrier(raw,audit.append)
        self.assertIs(channel.recv(),commands)
        a=dict(kind='output',request_id='q0',count=1,time=11.)
        b=dict(kind='output',request_id='q1',count=1,time=13.)
        channel.send(a);self.assertEqual(values,[])
        channel.send(b);self.assertEqual(values,[a,b])
        self.assertEqual([r['time'] for r in values],[11.,13.])
        c=dict(kind='output',request_id='q0',count=2,time=15.)
        channel.send(c);self.assertEqual(values,[a,b,c])
        self.assertEqual(len(audit),1)
        channel.close()

    def test_transform_installed_scheduler_without_importing_cuda(self):
        path=Path('/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/core/sched/scheduler.py')
        source=path.read_text();tree=ast.parse(source)
        method=next(n for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='Scheduler'
                    for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='schedule')
        original=''.join(source.splitlines(keepends=True)[method.lineno-1:method.end_lineno])
        changed=transform_schedule(original)
        self.assertEqual(changed.count('if _matched_defer_running('),1)
        self.assertEqual(changed.count('if _matched_has_selected_prefill('),1)
        self.assertIn('token_budget = self.max_num_scheduled_tokens',changed)
        compile(changed,'CPU_fixed_cohort_schedule','exec')


if __name__=='__main__':
    unittest.main()
