"""Test earlier exact frontiers only after the Native graph gate passes."""
import json
from .common import *
from .prepare import save
from .run import execute
from .analyze import compare,metrics
from .graph_feasibility import graph_config,gate
from .resume_campaign import restore


def main():
    progress=json.loads((ROOT/'state/graph_feasibility_progress_v28.json').read_text())['results']
    assert len([r for r in progress if r.get('passed')])==6
    pilots=[r for r in progress if 'metrics' in r];native=next(r for r in pilots if '_native_' in r['name'])
    if not native['metrics']['Decode_absolute_SLO_pass']:return
    last8=next(r for r in pilots if '_c28_' in r['name']);a=native['metrics'];b=last8['metrics']
    protected=last8['output_gate']['passed'] and b['Decode_absolute_SLO_pass']
    protected=protected and b['ITL_p99_ms']<=1.05*a['ITL_p99_ms'] and b['TPOT_p95_ms']<=1.05*a['TPOT_p95_ms']
    if not protected:
        write(ROOT/'analysis/earlier_layer_frontier_gate.json',dict(passed=False,evidence='NOT_RUN',
            reason='LAST8_PILOT_DID_NOT_PASS_NATIVE_RELATIVE_DECODE_PROTECTION'))
        return
    # Last-eight Layers cannot remove the full measured P overload in a linear
    # layer-count approximation. This is a selection hypothesis, not capacity.
    evidence=dict(P_capacity_reference_requests_s=3.4547610425479833,
        offered_QPS=native['metrics']['offered_QPS'],
        proportional_last8_P_capacity_estimate=3.4547610425479833*36/28,
        evidence='STRUCTURAL_SELECTION_HYPOTHESIS_NOT_MEASURED_CAPACITY',
        scope='Only measured exact frontiers can enter performance. The proportional estimate ignores Attention, transfer, scheduling, and overlap.')
    write(ROOT/'analysis/earlier_layer_frontier_hypothesis.json',evidence)
    results=[]
    try:
        for cut in (24,20):
            for count in (8,32):
                c=json.loads((ROOT/'configs'/f'eager_qwen8_L8000_O16_N{count}_c28_qual_v16.json').read_text())
                c=graph_config(c,f'graph_qwen8_L8000_O16_N{count}_c{cut}_qual_v27')
                c.update(layer_cut=cut,execution_events=False,verify_layer_execution=True,
                    hidden_source_backpressure=True,phase='EARLIER_EXACT_LAYER_FRONTIER_QUALIFICATION')
                name=save(c);complete=execute(name,restore=False)
                passed=gate(name,count,True) if complete else False
                results.append(dict(name=name,complete=complete,passed=passed))
                write(ROOT/'state/graph_extended_cuts_progress_v27.json',dict(results=results))
                if not passed:return
            c=json.loads((ROOT/'configs/graph_serving_qwen8_L8000_O16_N64_s3_c28_v28.json').read_text())
            c.update(name=f'graph_serving_qwen8_L8000_O16_N64_s3_c{cut}_v27',layer_cut=cut,
                hidden_source_backpressure=True,phase='EARLIER_LAYER_FRONTIER_SELECTION',
                scope='Earlier Static Cut, not an adaptive predictor. Same Native graph policy, fixed KV budget, prompt pool, and selection arrival seed. Exact N8/N32 precedes this N64 pilot.')
            name=save(c)
            if not execute(name,restore=False):return
            output=compare(native['name'],name);m=metrics(name)
            row=dict(name=name,metrics=m,output_gate=output)
            results.append(row);write(ROOT/'state/graph_extended_cuts_progress_v27.json',dict(results=results))
            print('EARLIER_LAYER_PILOT',json.dumps(row),flush=True)
            if not output['passed'] or not m['Decode_absolute_SLO_pass']:return
    finally:restore()


if __name__=='__main__':main()
