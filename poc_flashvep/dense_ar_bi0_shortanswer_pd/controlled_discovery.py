"""Test strong fixed cuts before a window or axis policy is proposed."""
import time
from .common import ROOT,write
from .controlled_preflight import main as preflight
from .campaign import serving
from .serving_analysis import analyze
from .axis_timeline import analyze as axis_analyze
from .quality import quality_gate


def main(fast=False):
    version='v3' if fast else 'v2';qualification_tag='fast128_v3' if fast else 'shared128_v2'
    teacher_flag='token_geometry_teacher_fast128_v3_complete.json' if fast else 'token_geometry_teacher_complete.json'
    cuts=(1,2,4,6,8,12) if fast else (4,8,12)
    declaration=ROOT/'configs'/f'frozen_controlled_fixed_axis_discovery_{version}.json'
    if not declaration.exists():
        write(declaration,dict(time=time.time(),dataset='longbench_shortanswer200_v2.json',N=200,
            QPS=2.35,arrival_seed=2026112061,cuts=list(cuts),actual_batch=[1,1,1],source_credit=4,
            B=8192,S=1024,KV_block=128,restore_official_UCX_init=True,
            execution=f'Compile0, full decode graph, async={fast}, Natural EOS.',
            primary_comparator='Controlled Native under the same quiescent D admission and source-credit contract.',
            external_comparators='Ordinary continuous Native/Agg use a different scheduling contract and remain separate.',
            scope='One independent discovery start per cell. Full200 task quality is checked after each run. '
            'No result is a held-out winner or sustainable-capacity estimate. No dynamic policy is evaluated here.'))
    while not (ROOT/'state'/teacher_flag).exists() or (fast and not (ROOT/'state/fast_native_control_state_v3_complete.json').exists()):
        for failed in ('token_geometry_teacher_hold.json',f'token_geometry_mismatch_{"fast128_v3" if fast else "v1"}.json',
                       'async_qualification_state_hold.json','async_qualification_teacher_hold.json','STOP_REQUESTED.json'):
            assert not (ROOT/'state'/failed).exists(),failed
        time.sleep(3)
    preflight(tag=qualification_tag,block=128,restore=True,asynchronous=fast,qualification_tag=qualification_tag)
    results=[]
    for axis,cut in [('Native',0)]+[(a,c) for c in cuts for a in ('Layer','Token')]:
        name=f'ControlledDiscovery_{axis}{cut}_N200_q2.35_s2026112061_BI0_{version}'
        extra=dict(axis_kind=axis,axis_cut=cut,axis_state_capture=False,hidden_source_slots=4,
                   qualified_controlled_performance=True,runtime_qualification_tag=qualification_tag,
                   token_teacher_gate_tag='fast128_v3' if fast else 'v1',burst_gate_tag=qualification_tag,
                   qualified_async_runtime=fast) if cut else {}
        measured=serving(name,axis,'longbench_shortanswer200_v2.json',200,2.35,2026112061,
            phase='CONTROLLED_FIXED_AXIS_DISCOVERY',controlled_pipeline=True,source_credit_limit=4,
            axis_mode='STATIC',execution_family='MATCHED_NO_COMPILE',async_scheduling=fast,
            KV_block_tokens=128,restore_official_UCX_init=True,**extra)
        analyze(name);timing=axis_analyze(name);assert timing['all_prompt_work_accounted'],name
        assert all(key in measured for key in ('TTFT_p95_ms','SLO_qualified_output_tokens_per_second','quality_gate'))
        quality=quality_gate(name)
        results.append(dict(name=name,axis=axis,cut=cut,metrics=measured,quality=quality))
        write(ROOT/'analysis'/f'BI0_controlled_fixed_axis_discovery_{version}.json',dict(results=results,
            evidence='MEASURED_ONLINE',independent_starts_per_cell=1,
            scope='Natural EOS finite cohorts. Matched actual request batches1, shared runtime. '
            'Full200 token-layer rows are nearly matched, not exactly matched under 16-token alignment. '
            'Point/CI task-quality gates are reported separately from performance. No causal policy gain is established.'))
    write(ROOT/'state'/f'controlled_fixed_axis_discovery_{version}_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--fast',action='store_true');args=parser.parse_args()
    try:main(args.fast)
    except BaseException as error:
        write(ROOT/'state/controlled_discovery_hold.json',dict(time=time.time(),error=repr(error)));raise
