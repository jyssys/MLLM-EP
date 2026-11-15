"""Small, resumable stages with no implicit performance acceptance."""
import argparse,fcntl
from .common import *
from .configuration import base
from .frontier import run
from .run import execute
from .sustainable import duration_config,assess
from .qualification import against_previous
from .analyze import metrics,records


def tuning():
    jobs=[]
    for budget,slots in ((384,16),(512,2),(8192,1),(512,4),(1024,2)):
        jobs.append(base(f'frontier_Colocated_N64_B{budget}_S{slots}_O16_q3_v1','Colocated',
            chunk=budget,token_budget=budget,D_slots=slots))
    for budget,slots in ((384,16),(512,2),(8192,1)):
        jobs.append(base(f'frontier_Colocated_N64_B{budget}_S{slots}_O64_q2_v1','Colocated',
            cap=64,rate=2.,chunk=budget,token_budget=budget,D_slots=slots))
    return jobs


def state_gate(name,count):
    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text());folder=ROOT/'raw'/name
    commits=records(folder/'actual_cut_commits.jsonl');cuts={r:x['cut'] for x in commits for r in x['requests']}
    P={r:0 for r in cuts};suffix={r:0 for r in cuts};decode={r:0 for r in cuts};failures=[]
    for role in ('prefill','decode'):
        for path in folder.glob(f'extension_{role}_*.jsonl'):
            for row in records(path):
                if row['event']=='ACTUAL_ATTENTION_LAYER_EXECUTION' and row.get('measured'):
                    for rid in row['requests']:
                        if rid not in cuts:continue
                        if role=='prefill':expected=list(range(cuts[rid]));P[rid]+=1
                        elif row.get('real_suffix',row.get('prefilling') and cuts[rid]<36):
                            expected=list(range(cuts[rid],36));suffix[rid]+=1
                        else:expected=list(range(36));decode[rid]+=1
                        if row['actual']!=expected:failures.append(dict(request_id=rid,reason='INCORRECT_LAYER_VISITS'))
                elif row['event']=='FULL_GRAPH_REPLAY' and role=='decode':
                    for rid in row['requests']:
                        if rid in cuts:decode[rid]+=1
    for rid,cut in cuts.items():
        cap=cfg['request_caps'][int(rid[1:])]
        if P[rid]!=1 or suffix[rid]!=int(cut<36) or decode[rid]!=cap-1:
            failures.append(dict(request_id=rid,reason='WORK_NOT_EXACTLY_ONCE',P=P[rid],suffix=suffix[rid],decode=decode[rid],cap=cap))
    checks={}
    for role in ('prefill','decode'):
        for kind in ('KV_lifetime_','reservation_lifetime_'):
            checks[kind+role]=json.loads((folder/(kind+role+'.json')).read_text())['passed']
    result=dict(name=name,count=len(cuts),failures=failures,lifetime_checks=checks,cuts=cuts,
        passed=len(cuts)==count and not failures and all(checks.values()),evidence='CORRECTNESS_ONLY')
    write(ROOT/'analysis'/(name+'_work_lifetime.json'),result);return result


def qualify():
    jobs=[base('qual_Forced_Last024812_N32_full_state_v2','FORCED_QUALIFICATION',count=32,cap=16,
        request_caps=[2]*32,
        arrival_offsets=[0.]*32,arrival_rate=None,forced_layers=[0,2,4,8,12,0,4,8],
        verify_layer_execution=True,hidden_verify_hash=True,full_prompt_state_witness=True,
        full_prompt_state_witness_roles=['prefill','decode'],verify_native_prefix_KV=True,
        correctness_only=True,negotiation_timeout_s=120,
        phase='FORCED_PROTOCOL_AND_FULL_STATE_NOT_PERFORMANCE')]
    jobs.append(base('qual_DIRECT_N32_O16_q5_v1','DIRECT',count=32,cap=16,rate=5.,
        verify_layer_execution=True,correctness_only=True,direct_wait_limit_ms=1000.,
        phase='ACTUAL_DEADLINE_POLICY_WORK_AND_LIFETIME_NOT_PERFORMANCE'))
    for cfg in jobs:
        run([cfg]);name=cfg['name'];gate=state_gate(name,cfg['count']);assert gate['passed'],gate
        if cfg.get('full_prompt_state_witness'):
            reference=LEGACY/'raw/qual_D16_independent_full_state_N32_v2'
            expected={r['external_request_id']:r for p in reference.glob('extension_prefill_*.jsonl')
                for r in records(p) if r['event']=='VALID_QUERY_STATE' and r.get('query_tokens')==8000
                and r['external_request_id'].startswith('q')}
            assert len(expected)==32
            states=json.loads((ROOT/'raw'/name/'full_prompt_state_decode.json').read_text())['requests']
            source_states=json.loads((ROOT/'raw'/name/'full_prompt_state_prefill.json').read_text())['requests']
            Native_imports={r['external_request_id']:r for path in (ROOT/'raw'/name).glob('extension_decode_*.jsonl')
                for r in records(path) if r['event']=='NATIVE_IMPORTED_PREFIX_IDENTITY'}
            checks={}
            for rid in expected:
                if gate['cuts'][rid]==36:
                    value=source_states[rid]
                    checks[rid]=dict(KV_equal=Native_imports.get(rid,{}).get('passed',False) and
                        value['KV']==expected[rid]['KV'],Hidden_equal=value['final_hidden_SHA256']==expected[rid]['final_hidden_SHA256'])
                else:
                    value=states[rid]
                    checks[rid]=dict(KV_equal=value['KV']==expected[rid]['KV'],
                        Hidden_equal=value['final_hidden_SHA256']==expected[rid]['final_hidden_SHA256'])
            passed=all(all(v.values()) for v in checks.values())
            write(ROOT/'analysis'/(name+'_full_state_gate.json'),dict(checks=checks,passed=passed,
                evidence='CORRECTNESS_ONLY',scope='All36 valid prompt KV planes and all8000 final Hidden rows against the frozen independent full model.'))
            assert passed,'Full-state parity failed'


def duration_initial():
    jobs=[duration_config('duration_Native_O16_q3_s41_v1','Native',3.),
        duration_config('duration_Colocated_B8192_S1_O16_q3_s41_v1','Colocated',3.,
            chunk=8192,token_budget=8192,D_slots=1),
        duration_config('duration_Static_Last8_O16_q3_s41_v1','STATIC',3.,static_layers=8,migration_fraction=1.),
        duration_config('duration_DIRECT_O16_q3_s41_v1','DIRECT',3.,direct_wait_limit_ms=1000.)]
    for cfg in jobs:
        name=save(cfg)
        if not execute(name,restore=True):raise RuntimeError('Duration run failed: '+name)
        gate=against_previous(name);assert gate['passed'],gate
        result=assess(name);write(ROOT/'analysis'/(name+'_metrics.json'),metrics(name))
        print('FIXED_WINDOW',name,{k:(v['qualified'],v['arrival_cohort_joint_attainment'])
            for k,v in result['SLO_regimes'].items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['tuning','qualify','duration_initial'])
    args=parser.parse_args()
    if args.stage=='tuning':run(tuning())
    elif args.stage=='qualify':qualify()
    else:duration_initial()
