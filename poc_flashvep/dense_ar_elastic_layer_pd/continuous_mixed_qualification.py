"""Audit Full36 cached joins after partial-layer requests resume Decode."""
import fcntl
from .common import *
from .configuration import base
from .run import execute
from .analyze import records,compare


def main():
    cfg=base('continuous_native_mixed_Last024812_N32_O64_v1',arm='FORCED_QUALIFICATION',count=32,cap=64,
        arrival_offsets=[0.]*32,arrival_rate=None,native_continuous_D_ingress=True,
        defer_completion_log=True,verify_layer_execution=True,numerical_native_reference=False,
        forced_layers=[0,2,4,8,12,0,4,8],correctness_only=True,negotiation_timeout_s=120,
        D_admission_batch=16,D_token_budget=128000,max_reserved_requests=16,
        phase='FORCED_MIXED_CUT_CACHED_NATIVE_JOIN_CORRECTNESS_NOT_ONLINE_POLICY')
    try:
        if not execute(save(cfg),restore=False):return
        name=cfg['name'];folder=ROOT/'raw'/name;commits=records(folder/'actual_cut_commits.jsonl')
        cuts={rid:r['cut'] for r in commits for rid in r['requests']}
        prefix={r:0 for r in cuts};suffix={r:0 for r in cuts};decode={r:0 for r in cuts};checks=[]
        for role in ('prefill','decode'):
            for path in folder.glob(f'extension_{role}_*.jsonl'):
                for row in records(path):
                    if row['event']=='ACTUAL_ATTENTION_LAYER_EXECUTION' and row.get('measured'):
                        for rid in row['requests']:
                            if rid not in cuts:continue
                            if role=='prefill':assert row['actual']==list(range(cuts[rid]));prefix[rid]+=1
                            elif row.get('real_suffix'):
                                assert row['actual']==list(range(cuts[rid],36));suffix[rid]+=1
                            else:assert row['actual']==list(range(36));decode[rid]+=1
                    if row['event']=='FULL_GRAPH_REPLAY' and role=='decode':
                        for rid in row['requests']:
                            if rid in cuts:decode[rid]+=1
            for kind in ('KV_lifetime_','reservation_lifetime_'):
                value=json.loads((folder/(kind+role+'.json')).read_text());checks.append(dict(check=kind+role,passed=value['passed']))
        work=len(cuts)==32 and all(prefix[r]==1 and suffix[r]==int(cuts[r]<36) and decode[r]==63 for r in cuts)
        output=compare('baseline_Native_O200_N64_q4p8_s21_v1',name,True)
        joins=sum(r['reason']=='NATIVE_CACHED_QUERY_JOIN_ACTIVE_DECODE' for r in records(folder/'D_host_admission.jsonl'))
        result=dict(name=name,output=output,work_once=work,checks=checks,cuts=cuts,
            Full36_Decode_steps=decode,actual_native_active_joins=joins,
            passed=output['passed'] and work and all(r['passed'] for r in checks) and joins>0,
            evidence='CORRECTNESS_ONLY',scope='Forced mixed-cut qualification. Cached Native queries can join ordinary Full36 Decode across request cut identities. Genuine suffix queries remain isolated from existing Decode.')
        write(ROOT/'analysis/continuous_native_mixed_N32_qualification.json',result)
        print('CONTINUOUS_NATIVE_MIXED_GATE',result['passed'],joins,flush=True)
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
