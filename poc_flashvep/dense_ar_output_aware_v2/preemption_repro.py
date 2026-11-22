"""Replay the failed observer's actual query sequence at its execute hook."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from .common import ROOT,write

FAILED='V2_C3Q_Agg_Natural400'

def replay(source_text,require_witness=True):
    tree=ast.parse(source_text)
    function=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='execute')
    node=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
    raw=ROOT/'raw'/FAILED
    rows=[r for r in json.loads((raw/'actual_queries_replica0.json').read_text())['rows'] if r['request_id']=='q90']
    witnesses=json.loads((raw/'scheduler_preemptions_replica0.json').read_text())['rows']
    frames={};scope=dict(original_execute=lambda *a,**k:None,
        torch=SimpleNamespace(cuda=SimpleNamespace(is_current_stream_capturing=lambda:False)),
        info={'q90':dict(original_length=31397)},state_rows={},pending_hidden={},cfg={'state_capture':True},
        role='replica0',preemptions=witnesses if require_witness else [],observer_preemption_resets=[],time=__import__('time'))
    exec(compile(node,'actual_frozen_execute_hook','exec'),scope)
    error=None
    try:
        for r in rows:
            batch=SimpleNamespace(req_ids=['q90-8e6defc3'],num_computed_tokens_np=[r['start']],
                num_scheduled_tokens=[r['count']],query_start_loc_np=[0])
            frame=SimpleNamespace(input_batch=batch,hidden_states=torch.zeros((r['count'],1)))
            scope['execute'](SimpleNamespace(execute_model_state=frame))
    except AssertionError as exc:error=repr(exc)
    return dict(passed=error is None,error=error,query_intervals=[(r['start'],r['count']) for r in rows],
        exact_preemption_witnesses=witnesses if require_witness else [],resets=scope['observer_preemption_resets'],
        scope='CPU replay of the actual failed state-observer execute hook. This is not a model or GPU correctness proof.')

def main():
    path=ROOT/'raw'/FAILED/'worker_source/poc_flashvep/dense_ar_longbench400_pd/standard_worker.py'
    result=replay(path.read_text());write(ROOT/'analysis/agg_partial_preemption_observer_red_replay_v2.json',result)
    print('ACTUAL_OBSERVER_Q90_REPLAY',result,flush=True)
    assert result['passed'],'The original observer fails on the witnessed preemption sequence'

if __name__=='__main__':main()
