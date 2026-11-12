"""Separate Graph and sampling enclosures without changing SLO metrics."""
import json,statistics
from .common import *
from .prepare import save
from .run import execute
from .analyze import compare,metrics,percentile
from .resume_campaign import restore,record


def main():
    results=[]
    try:
        for label in ('native','c28'):
            c=json.loads((ROOT/'configs'/f'graph_serving_qwen8_L8000_O16_N64_s3_{label}_v28.json').read_text())
            c.update(name=f'graph_qwen8_L8000_O16_N32_{label}_event_diagnostic_v29',
                count=32,request_caps=[16]*32,arrival_offsets=[0.]*32,execution_events=True,
                runtime_observation=True,runtime_observation_roles=['prefill','decode'],
                verify_layer_execution=True,state_capture=False,hidden_source_backpressure=label=='c28',
                phase='GRAPH_FIRST_TRANSITION_AND_SAMPLER_DIAGNOSTIC',clean_timing=False,
                scope='Actual model/graph and sampler stream-event enclosures. First transition ITL remains in primary metrics. No interval is excluded to pass a Decode gate.')
            name=save(c)
            if not execute(name,restore=False):return
            check=compare('graph_serving_qwen8_L8000_O16_N64_s3_native_v28',name,True)
            folder=ROOT/'raw'/name
            core=json.loads((folder/'CUDA_event_enclosures_decode.json').read_text())['rows']
            sampler=json.loads((folder/'CUDA_sampler_enclosures_decode.json').read_text())['rows']
            example={kind:[r for r in rows if 'q4' in r['requests']] for kind,rows in [('core',core),('sampler',sampler)]}
            row=dict(name=name,output_gate=check,metrics=metrics(name),q4_example=example,
                evidence='DIAGNOSTIC_STREAM_EVENT_AND_HOST_ENCLOSURES')
            results.append(row);write(ROOT/'analysis/graph_first_transition_diagnostic_v30.json',dict(rows=results,
                scope='Enclosure times can include launch gaps and waits. They are not exclusive CUDA kernel durations. Client and Worker ITL remain separately available in raw token events.'))
            record('graph_cost_diagnostic',dict(name=name,output_gate=check))
            print('GRAPH_COST',label,{k:[round(x['elapsed_ms'],2) for x in v] for k,v in example.items()},flush=True)
            if not check['passed']:return
    finally:restore()


if __name__=='__main__':main()
