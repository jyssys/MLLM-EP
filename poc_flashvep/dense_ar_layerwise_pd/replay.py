"""Retain the red-capable prior-output replay without rewriting old evidence."""
import argparse,json,time
from .common import *


def main():
    initialize();p=argparse.ArgumentParser();p.add_argument('--assert-original-parity',action='store_true');a=p.parse_args()
    from ..looped_pd_atomic_prefill.ar_debug import evidence
    from ..looped_pd_atomic_prefill.evidence import compare
    original=evidence();reference='ext_qwen_bi_compiled_no_graph_qualification32_reference_v1'
    controls={label:compare(reference,'ext_qwen_bi_compiled_no_graph_qualification32_'+label+'_v1')
        for label in ('repeat','dp2','dp2_repeat','c4096')}
    write(ROOT/'analysis/prior_output_replay.json',dict(time=time.time(),original=original,qualified_family=controls,
        scope='CPU replay of saved actual GPU token outputs. This catches the original symptom. It is not a fresh Native or Layer correctness run.'))
    for r in original:print(r['label'],r['exact_sequences'],r['compared'],r['passed'])
    for label,r in controls.items():print('BI_COMPILED_NO_GRAPH',label,r['compared'],r['passed'])
    if a.assert_original_parity:assert all(r['passed'] for r in original),'Captured original Qwen token divergence remains detectable'


if __name__=='__main__':main()
