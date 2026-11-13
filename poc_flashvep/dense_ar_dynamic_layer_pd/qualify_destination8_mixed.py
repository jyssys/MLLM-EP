"""Qualify mixed frontiers at a destination budget that fits eight prompts."""
import argparse,fcntl
from .common import *
from .qualify import dynamic_config,state_gate
from .analyze import compare,records
from .run import execute


def main():
    p=argparse.ArgumentParser();p.add_argument('--version',type=int,default=1);p.add_argument('--max-reserved',type=int,default=16)
    a=p.parse_args()
    initialize();assert json.loads((ROOT/'state/D8_suffix_qualified.json').read_text())['passed']
    c=dynamic_config(f'qual_D8_mixed_Last024812_N32_v{a.version}',32)
    c.update(dataset='discovery.json',D_token_budget=64000,D_admission_batch=8,hidden_destination_slots=8,
        D_cohort_collect_ms=10000,negotiation_timeout_s=90,native_ACK_hold=True,
        max_reserved_requests=a.max_reserved,
        forced_layers=[0,2,2,4,4,8,8,12],phase='DESTINATION_BATCH8_MIXED_FRONTIER_EXACTNESS_NOT_SERVING')
    try:
        if not execute(save(c),restore=False):return
        output=compare('coarse_O16_N64_q4.777_native16_s1_v1',c['name'],True)
        state=state_gate(c['name'],32,True)
        rows=[r for p in (ROOT/'raw'/c['name']).glob('extension_decode_*.jsonl') for r in records(p)]
        widths=sorted({len(r['requests']) for r in rows if r['event']=='ACTUAL_ATTENTION_LAYER_EXECUTION' and r.get('measured') and r['prefilling']})
        counts=sorted({r['offloaded_layers'] for r in records(ROOT/'raw'/c['name']/'actual_cut_commits.jsonl')})
        passed=output['passed'] and state and 8 in widths and counts==[0,2,4,8,12]
        target=ROOT/'state/D8_mixed_protocol_qualified.json'
        if target.exists() and json.loads(target.read_text()).get('name')!=c['name']:
            old=json.loads(target.read_text());write(ROOT/'state'/(old['name']+'_coverage_result.json'),old)
        write(target,dict(name=c['name'],passed=passed,
            output=output,state_pass=state,actual_suffix_batches=widths,actual_offloaded_layer_counts=counts,
            scope='Forced correctness only. Wider destination batches must fit the real token budget. No performance uses the10s collection delay.'))
        assert passed
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
