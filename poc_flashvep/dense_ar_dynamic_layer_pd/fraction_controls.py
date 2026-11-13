"""Qualify a fixed request-ID fraction without future arrival information."""
import fcntl
from .common import *
from .online_pilots import serving_config
from .run import execute
from .analyze import compare,metrics,records


def main():
    initialize();rows=[]
    try:
        for count in (32,64):
            c=serving_config(f'validation_IDblocks_Static12_f31p25_O16_N{count}_v1',cap=16,count=count,
                policy='STATIC',static_layers=12,migration_fraction=.3125,static_fraction_axis='REQUEST_ID_BLOCKS',
                static_request_block_size=4,verify_layer_execution=count==32,
                phase='FIXED_REQUEST_ID_FRACTION_QUALIFICATION' if count==32 else 'EXACT_DISCOVERY_REQUEST_FRACTION_STATIC_CONTROL')
            if not execute(save(c),restore=False):return
            gate=compare('coarse_O16_N64_q4.777_native16_s1_v1',c['name'],count==32)
            requests=records(ROOT/'raw'/c['name']/'requests.jsonl')
            actual=sum(r['completed_D_prefill_layers']>0 for r in requests)
            expected=sum(4 for block in range(count//4) if int((block+1)*.3125)>int(block*.3125))
            row=dict(name=c['name'],output_gate=gate,metrics=metrics(c['name']),expected_migrated_requests=expected,
                actual_migrated_requests=actual,exact_request_fraction_pass=actual==expected,
                scope='The fixed ID mask does not observe D state or future arrivals. A source group can split at a mask boundary. Homogeneous full Static alternatives remain in the wider sweep.')
            rows.append(row);write(ROOT/'state/fraction_controls_progress.json',dict(results=rows))
            print('ID_FRACTION',count,gate['passed'],expected,actual,flush=True)
            if not gate['passed'] or actual!=expected:return
        write(ROOT/'analysis/fraction_control_results.json',dict(results=rows,passed=True))
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
