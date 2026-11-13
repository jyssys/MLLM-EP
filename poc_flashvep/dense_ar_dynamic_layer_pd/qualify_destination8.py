"""Test a wider legal suffix cohort before using it for strong static tuning."""
import fcntl
from .common import *
from .qualify import static_config,state_gate
from .run import execute
from .analyze import compare,records


def main():
    initialize()
    c=static_config(12,8)
    c.update(name='qual_D8_Last12_Source4_N8_v1',D_admission_batch=8,D_token_budget=64000,
        D_cohort_collect_ms=10000,hidden_destination_slots=8,native_ACK_hold=True,
        phase='DESTINATION_BATCH8_STATE_AND_TOKEN_QUALIFICATION_NOT_SERVING')
    try:
        if not execute(save(c),restore=False):return
        output=compare('qual_native_O200_N8_A_v1',c['name'],True);state=state_gate(c['name'],8)
        source=records(next((ROOT/'raw'/c['name']).glob('extension_decode_*.jsonl')))
        actual=[r for r in source if r['event']=='ACTUAL_ATTENTION_LAYER_EXECUTION' and r.get('measured') and r['prefilling']]
        B8=any(len(r['requests'])==8 for r in actual)
        passed=output['passed'] and state and B8
        write(ROOT/'state/D8_suffix_qualified.json',dict(name=c['name'],passed=passed,
            output=output,state_pass=state,actual_batch8=B8,
            scope='Source remains Batch4. Only the destination budget and maximum legal suffix cohort increase. The10s collection delay is diagnostic and is not a Serving policy.'))
        assert passed
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
