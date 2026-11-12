"""Use independent full-model execution to resolve a PD reference mismatch."""
import fcntl,json
from .common import *
from .prepare import save
from .run import execute
from .analyze import compare,records
from .state_qualification import events


def main():
    c=json.loads((ROOT/'configs/colocated_qwen8_L8000_O16_full_cohort_N32_qual_v4.json').read_text())
    c.update(name='eager_qwen8_L8000_O16_N32_independent_full_cohort_reference_v21',
        ar_enforce_eager=True,ar_execution_family='BI_EAGER_NO_GRAPH',
        phase='INDEPENDENT_FULL_MODEL_REFERENCE_AUDIT')
    full=save(c)
    source=json.loads((ROOT/'data/qualification_qwen8_L8000.json').read_text());target='qualification_q20_q23_L8000.json'
    path=ROOT/'data'/target
    if not path.exists():write(path,source[20:24])
    c=json.loads((ROOT/'configs/eager_qwen8_L8000_O16_N8_reference_qual_v16.json').read_text())
    c.update(name='eager_qwen8_L8000_O16_q20q23_last_query_replay_audit_v21',dataset=target,count=4,
        seed=c['seed']+20,request_caps=[16]*4,arrival_offsets=[0.]*4,calibration_serial=False,
        capture_last_prompt_state=True,phase='LAST_PROMPT_QUERY_RECOMPUTATION_DIAGNOSTIC')
    diagnostic=save(c)
    try:
        if execute(full,restore=False):
            names=['eager_qwen8_L8000_O16_N32_reference_qual_v16']+[f'eager_qwen8_L8000_O16_N32_c{k}_qual_v16' for k in (35,34,32,28)]
            checks=[compare(full,n) for n in names]
            write(ROOT/'analysis/independent_full_reference_output_audit.json',dict(reference=full,checks=checks,
                scope='Independent local full-model Prefill and Decode. No Native KV transfer or intermediate-state continuation. Do not select a reference merely to make a candidate pass.'))
            print('FULL_REFERENCE_CHECKS',[(x['candidate'],x['passed'],len(x['failures'])) for x in checks],flush=True)
        if execute(diagnostic,restore=False):
            src={x['external_request_id']:x for x in events(diagnostic,'prefill') if x['event']=='LAST_QUERY_STATE' and x['external_request_id'].startswith('q')}
            dst={x['external_request_id']:x for x in events(diagnostic,'decode') if x['event']=='LAST_QUERY_STATE' and x['external_request_id'].startswith('q')}
            values=[]
            for rid in src:
                a,b=src[rid],dst[rid]
                values.append(dict(request=rid,source_sample_index=20+int(rid[1:]),
                    source_position=a['absolute_position'],destination_position=b['absolute_position'],
                    different_KV_layers=[i for i,(x,y) in enumerate(zip(a['KV'],b['KV'])) if x!=y],
                    final_hidden_exact=a['final_hidden_SHA256']==b['final_hidden_SHA256']))
            write(ROOT/'analysis/last_prompt_query_replay_state_audit.json',dict(rows=values,
                evidence='CORRECTNESS_ONLY',scope='Hash differences locate numerical or state differences. They do not identify a specific arithmetic kernel without further evidence.'))
            print('LAST_QUERY_STATES',values,flush=True)
    finally:
        with (ROOT/'state/quiet_lane.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
