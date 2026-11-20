"""Freeze discovery winners, then run new starts on paired baseline traces."""
import json
import time
from .common import ROOT,write
from .campaign import serving
from .serving_analysis import analyze
from .axis_timeline import analyze as axis_analyze


def main(fast=False):
    version='v2' if fast else 'v1';discovery_version='v3' if fast else 'v2'
    qualification_tag='fast128_v3' if fast else 'shared128_v2'
    design=ROOT/'configs'/f'frozen_controlled_confirmation_protocol_{version}.json'
    if not design.exists():
        write(design,dict(time=time.time(),discovery_seed=2026112061,QPS=2.35,
            heldout_arrival_seeds=[2026112051,2026112052,2026112053],N=200,
            selection=f'Highest measured SLO-qualified output tokens/s among full200 task-quality-CI-passing cuts{[1,2,4,6,8,12] if fast else [4,8,12]}, independently per axis. TTFT p95 breaks a tie.',
            decode_gate='Final paired-start CI upper bound <=1.05 vs controlled Native and separately continuous Native. Not used to hide a failed confirmation.',
            actual_control_batches=[1,1,1],source_credit=4,B=8192,S=1024,KV_block=128,official_UCX_init=True,async_scheduling=fast,
            external_baselines='Ordinary continuous Native and Agg at the same shared KV128 and exact held-out arrival traces. C3 external vs C0 mechanism control is disclosed.',
            scope='Discovery start61 is not a held-out start. Numerical variation and actual output work remain explicit. No dynamic policy or sustainable-capacity claim.'))
    while not (ROOT/'state'/f'controlled_fixed_axis_discovery_{discovery_version}_complete.json').exists():
        for failed in ('STOP_REQUESTED.json','controlled_discovery_hold.json'):
            assert not (ROOT/'state'/failed).exists(),failed
        time.sleep(5)
    selection=ROOT/'configs'/f'frozen_static_axis_selection_{version}.json'
    if not selection.exists():
        discovery=json.loads((ROOT/'analysis'/f'BI0_controlled_fixed_axis_discovery_{discovery_version}.json').read_text())['results']
        winners={}
        for axis in ('Layer','Token'):
            choices=[r for r in discovery if r['axis']==axis and r['quality']['CI_passed']]
            if not choices:winners[axis]=dict(status='NO_FULL200_QUALITY_QUALIFIED_CANDIDATE');continue
            best=max(choices,key=lambda r:(r['metrics']['SLO_qualified_output_tokens_per_second'],-r['metrics']['TTFT_p95_ms']))
            winners[axis]=dict(cut=best['cut'],discovery_name=best['name'],discovery_metrics=best['metrics'])
        write(selection,dict(time=time.time(),winners=winners,scope=f'Best tested cuts{[1,2,4,6,8,12] if fast else [4,8,12]}, not a global best over all fractions or policies. Frozen before held-out starts.'))
    winners=json.loads(selection.read_text())['winners'];results=[]
    for seed in (2026112051,2026112052,2026112053):
        arms=[('ContinuousNative','Native',None),('Agg','Agg',None),('ControlledNative','Native',None)]
        arms.extend((axis,axis,value['cut']) for axis,value in winners.items() if 'cut' in value)
        for label,method,cut in arms:
            name=f'Confirmation_{label}{cut or 0}_N200_q2.35_s{seed}_BI0_{version}'
            extra={}
            if label not in ('ContinuousNative','Agg'):
                extra.update(controlled_pipeline=True,source_credit_limit=4,
                    execution_family='MATCHED_NO_COMPILE',async_scheduling=fast)
            if cut:extra.update(axis_kind=method,axis_cut=cut,axis_state_capture=False,
                qualified_controlled_performance=True,hidden_source_slots=4,axis_mode='STATIC',
                runtime_qualification_tag=qualification_tag,token_teacher_gate_tag='fast128_v3' if fast else 'v1',
                burst_gate_tag=qualification_tag,qualified_async_runtime=fast)
            measured=serving(name,method,'longbench_shortanswer200_v2.json',200,2.35,seed,
                KV_block_tokens=128,restore_official_UCX_init=True,quiescent_witness=fast,
                phase='PAIRED_SHARED_KV128_CONFIRMATION',**extra)
            trace=analyze(name)
            if label not in ('ContinuousNative','Agg'):
                geometry=axis_analyze(name);assert geometry['all_prompt_work_accounted'],name
                certs=[json.loads(line) for line in (ROOT/'raw'/name/'idle_certificates.jsonl').read_text().splitlines()]
                assert len(certs)==200 and all(r['passed'] for r in certs),name
                assert all(set(map(int,v['actual_request_batch_histogram']))=={1} for v in trace['geometry'].values()),name
            results.append(dict(name=name,label=label,cut=cut,seed=seed,metrics=measured))
            write(ROOT/'analysis'/f'BI0_shared_KV128_confirmation_{version}.json',dict(results=results,
                evidence='MEASURED_ONLINE',scope='Full200 Natural EOS. Discovery-frozen static choices. Actual-batch1 controls and ordinary continuous baselines remain separate.'))
    write(ROOT/'state'/f'paired_shared_KV128_confirmation_{version}_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--fast',action='store_true');args=parser.parse_args()
    try:main(args.fast)
    except BaseException as error:
        write(ROOT/'state/confirmation_hold.json',dict(time=time.time(),error=repr(error)));raise
