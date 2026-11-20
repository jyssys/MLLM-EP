"""Measure matched-request serial and online paths with equal output work."""
import json
import time
from .common import ROOT,write,initialize,digest
from .campaign import cell


def main():
    initialize()
    data=json.loads((ROOT/'data/longbench_shortanswer200_v2.json').read_text())
    ref_path=ROOT/'raw/BI0_Standard_ShortAnswer200_v2_Natural_C1_cap2048_v1/natural_reference.json'
    ref={r['prompt_id']:r for r in json.loads(ref_path.read_text())['rows']}
    lengths=[ref[r['prompt_id']]['output_tokens'] for r in data]
    path=ROOT/'configs/frozen_serial_online_causal_protocol_v1.json'
    if not path.exists():
        write(path,dict(time=time.time(),cut=4,seed=2026112082,QPS=2.35,N=200,
            actual_batch=[1,1,1],source_credit=4,hidden_source_slots=4,KV_block=128,
            output_contract='DIAGNOSTIC_FIXED_LENGTH_REPLAY',total_output_tokens=sum(lengths),
            reference_SHA256=digest(ref_path),
            modes=['ONLINE','SERIAL_ADMISSION'],
            arrival='Both modes enqueue the same full Poisson trace. Serial admission does not reset arrival timestamps.',
            serial='Wait for the prior D request to finish before the next P dispatch. Source credit is still4. This admission barrier differs from source credit1.',
            credit_intervention='Repeat online Layer/Token with credit1 and the same hidden pool4. The separate Last8 sweep retains the full1/4/8/16/32 design.',
            service_intervention='Use a state-updating FIFO replay to exchange per-request source, destination suffix and decode service vectors. This is an offline counterfactual, not a measured GPU speedup.',
            validation='Before attributing modeled effects, require online p95 relative error <=10% and request TTFT MAE <=max(100ms,10% of observed mean). Do not fit to the online result.',
            order=[['Layer','SERIAL_ADMISSION',4],['Token','ONLINE',4],['Token','SERIAL_ADMISSION',4],
                   ['Layer','ONLINE',4],['Layer','ONLINE',1],['Token','ONLINE',1]],
            scope='Same request IDs, input hashes, arrival trace and exact output counts within the causal diagnostic. No frozen production SLO or Natural200 main result is replaced.'))
    while not (ROOT/'state/new_overlay_runtime_smoke_complete.json').exists():
        assert not (ROOT/'state/STOP_REQUESTED.json').exists()
        time.sleep(3)
    results=[]
    for axis,mode,credit in [('Layer','SERIAL_ADMISSION',4),('Token','ONLINE',4),
        ('Token','SERIAL_ADMISSION',4),('Layer','ONLINE',4),('Layer','ONLINE',1),('Token','ONLINE',1)]:
        name=f'CausalLast4_{axis}_{mode}_q2.35_c{credit}_s2026112082_N200_v1'
        row=cell(name,axis,4,seed=2026112082,credit=credit,
            entry_module='poc_flashvep.dense_ar_bi0_axis_rootcause.causal_serving',
            strict_serial_pipeline=mode=='SERIAL_ADMISSION',diagnostic_controlled_output_replay=True,
            replay_output_lengths=lengths,phase='SERIAL_ONLINE_CAUSAL_CONTROL')
        row['causal_mode']=mode
        results.append(row)
        write(ROOT/'analysis/serial_online_causal_control_v1.json',dict(results=results,
            evidence='MEASURED_ONLINE',diagnostic=True,
            scope='Fixed output4723tokens. Serial and online have identical arrivals. Serial admission is not an online serving optimization.'))
        assert row['status']!='FAILED',name
        assert row['metrics']['output_tokens']==sum(lengths)
    write(ROOT/'state/serial_online_causal_control_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/serial_online_causal_control_hold.json',dict(time=time.time(),error=repr(error)))
        raise
