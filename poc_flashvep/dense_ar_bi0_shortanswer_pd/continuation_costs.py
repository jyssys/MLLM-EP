"""Measure serial real-request services before a state-updating replay."""
import json
import time
from .common import ROOT,write,save
from .qualification import configuration
from .run import execute
from .axis_timeline import analyze


def main(fast=False):
    qualification_tag='fast128_v3' if fast else 'shared128_v2'
    teacher_flag='token_geometry_teacher_fast128_v3_complete.json' if fast else 'token_geometry_teacher_complete.json'
    flags=[f'axis_qualification_{qualification_tag}_complete.json',teacher_flag]
    if fast:flags.append('fast_native_control_state_v3_complete.json')
    while not all((ROOT/'state'/name).exists() for name in flags):
        for marker in ('shared_axis_qualification_hold.json','axis_state_mismatch_shared128_v2.json',
                       f'axis_qualification_{qualification_tag}_failed.json','token_geometry_teacher_hold.json',
                       'async_qualification_state_hold.json','async_qualification_teacher_hold.json',
                       f'token_geometry_mismatch_{"fast128_v3" if fast else "v1"}.json','STOP_REQUESTED.json'):
            assert not (ROOT/'state'/marker).exists(),marker
        time.sleep(3)
    declaration=ROOT/'configs'/f'frozen_continuation_cost_protocol_{"fast" if fast else "slow"}_v1.json'
    if not declaration.exists():
        write(declaration,dict(time=time.time(),calibration='cost_calibration20.json',validation='cost_validation15.json',
            actions=['Native']+[f'{axis}{cut}' for cut in (1,2,4,6,8,12) for axis in ('Layer','Token')],
            actual_batch=[1,1,1],B=8192,S=1024,KV_block=128,official_UCX_init=True,
            execution=f'Compile0, full decode graph, async={fast}, BI0, Natural EOS.',
            scope='Serial service envelopes with real prompts. No other request queue. No CPU KV/hidden capture. '
            'Costs include host, model, transport, sampling and delivery. They are not exclusive GPU kernel times. '
            'Calibration and validation prompts are disjoint. The fixed input split precedes cost measurements.'))
    results=[]
    for split,count in (('calibration',20),('validation',15)):
        for axis,cut in [('Native',0)]+[(a,c) for c in (1,2,4,6,8,12) for a in ('Layer','Token')]:
            name=f'ContinuationCost_{axis}{cut}_{split}{count}_BI0_v1'
            extra=dict(axis_kind=axis,axis_cut=cut,axis_state_capture=False,isolated_continuation_cost=True,
                hidden_source_slots=4,runtime_qualification_tag=qualification_tag,
                token_teacher_gate_tag='fast128_v3' if fast else 'v1',qualified_async_runtime=fast) if cut else {}
            cfg=configuration(name,axis,f'cost_{split}{count}.json',count=count,state_capture=False,
                correctness_only=False,execution_family='MATCHED_NO_COMPILE',async_scheduling=fast,
                KV_block_tokens=128,restore_official_UCX_init=True,phase='ISOLATED_SERIAL_SERVICE_COST',**extra)
            assert execute(save(cfg)),name
            value=analyze(name);assert value['all_prompt_work_accounted'],name
            results.append(dict(name=name,split=split,axis=axis,cut=cut,summary=value['summary']))
            write(ROOT/'analysis/BI0_continuation_serial_costs.json',dict(results=results,
                evidence='ISOLATED_SERIAL_SERVICE_MEASUREMENT',scope='Host service envelopes, not GPU-exclusive time or serving speedup.'))
    write(ROOT/'state/continuation_costs_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--fast',action='store_true');args=parser.parse_args()
    try:main(args.fast)
    except BaseException as error:
        write(ROOT/'state/continuation_costs_hold.json',dict(time=time.time(),error=repr(error)));raise
