"""Qualify source-credit backpressure before the remaining bounded runs."""
import json
from .common import *
from .prepare import save
from .run import execute
from .actual_layer_gate import gate
from .resume_campaign import restore,diagnostic,confirmation,record


def main():
    try:
        for count in (8,32):
            c=json.loads((ROOT/'configs'/f'eager_qwen8_L8000_O16_N{count}_c28_qual_v16.json').read_text())
            c.update(name=f'eager_qwen8_L8000_O16_N{count}_source_credit4_qual_v25',
                hidden_source_slots=4,hidden_source_backpressure=True,D_admission_batch=2,
                phase='BOUNDED_HIDDEN_EXPORT_CREDIT_CORRECTNESS',
                scope='Stress four real Source export slots against two-request D admission. Unstarted Prefix work must wait rather than overwrite or exhaust live state. Normal ACK and exact state remain required.')
            n=save(c)
            if not execute(n,restore=False):return
            output='eager_qwen8_L8000_O16_N32_independent_full_cohort_reference_v21' if count==32 else 'eager_qwen8_L8000_O16_N8_local_full_qual_v16'
            state=f'eager_qwen8_L8000_O16_N{count}_reference_qual_v16'
            g=gate(n,state,output)
            record('source_credit_qualification',dict(name=n,passed=g['passed']))
            if not g['passed']:return
        diagnostic()
        confirmation()
    finally:restore()


if __name__=='__main__':main()
