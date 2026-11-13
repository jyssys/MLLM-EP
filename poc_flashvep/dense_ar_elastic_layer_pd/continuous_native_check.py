"""Qualify ordinary cached Native ingress before using its timing results."""
import fcntl
from .common import *
from .configuration import base
from .run import execute
from .analyze import records,compare,metrics


def main():
    reference=PREVIOUS/'raw/qual_D16_independent_full_state_N32_v2'
    expected={r['external_request_id']:r for p in reference.glob('extension_prefill_*.jsonl')
        for r in records(p) if r['event']=='VALID_QUERY_STATE' and r.get('query_tokens')==8000
        and r['external_request_id'].startswith('q')}
    rows=[]
    try:
        jobs=[]
        for count in (8,32):
            jobs.append(base(f'native_continuous_exact_N{count}_O64_v1',count=count,cap=64,rate=6.,
                native_continuous_D_ingress=True,defer_completion_log=True,
                imported_native_state_witness=True,full_prompt_state_witness=True,
                full_prompt_state_witness_roles=['prefill'],correctness_only=True,
                phase='NATIVE_CONTINUOUS_FULL_PREFIX_STATE_CORRECTNESS'))
        for cap in (16,64,200):
            jobs.append(base(f'native_continuous_O{cap}_N64_q4p8_v1',cap=cap,
                native_continuous_D_ingress=True,defer_completion_log=True,
                D_admission_batch=16,D_token_budget=128000,
                phase='STANDARD_CACHED_NATIVE_INGRESS_DISCOVERY'))
        for cfg in jobs:
            name=save(cfg)
            if not execute(name,restore=False):write(ROOT/'state/native_continuous_progress.json',dict(results=rows,failed=name));return
            output=compare('baseline_Native_O200_N64_q4p8_s21_v1',name,True)
            row=dict(name=name,output=output,metrics=metrics(name))
            if cfg.get('correctness_only'):
                folder=ROOT/'raw'/name
                source=json.loads((folder/'full_prompt_state_prefill.json').read_text())['requests']
                destination=json.loads((folder/'imported_native_prompt_KV.json').read_text())['requests']
                checks=[]
                for index in range(cfg['count']):
                    rid='q'+str(index);a=source.get(rid);b=destination.get(rid);e=expected[rid]
                    checks.append(dict(request_id=rid,source_hidden_exact=a is not None and a['final_hidden_SHA256']==e['final_hidden_SHA256'],
                        source_KV_exact=a is not None and a['KV']==e['KV'],destination_KV_exact=b is not None and b['KV']==e['KV']))
                row['state_checks']=checks
                row['state_pass']=all(all(c[k] for k in ('source_hidden_exact','source_KV_exact','destination_KV_exact')) for c in checks)
            row['passed']=output['passed'] and row.get('state_pass',True)
            row['actual_active_decode_joins']=sum(r['reason']=='NATIVE_CACHED_QUERY_JOIN_ACTIVE_DECODE'
                for r in records(ROOT/'raw'/name/'D_host_admission.jsonl'))
            rows.append(row);write(ROOT/'state/native_continuous_progress.json',dict(results=rows))
            print('NATIVE_CONTINUOUS',name,row['passed'],row['actual_active_decode_joins'],flush=True)
            if not row['passed']:return
        write(ROOT/'analysis/native_continuous_qualification_and_discovery.json',dict(results=rows,passed=True,
            scope='Native cached prompt queries use Full36 Decode. No partial prefill or layer migration is counted in these runs.'))
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
