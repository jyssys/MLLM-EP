"""Preserve Natural EOS in the serial/online Last4 follow-up."""
import json
import time
import numpy as np
from .common import ROOT,write
from .campaign import cell
from .causal_replay import extract


def main():
    declaration=ROOT/'configs/frozen_Natural_Last4_serial_online_v1.json'
    if not declaration.exists():
        write(declaration,dict(time=time.time(),cut=4,N=200,QPS=2.35,seed=2026112061,
            source_credit=4,hidden_source_slots=4,actual_batch=[1,1,1],output='Natural EOS cap2048',
            reason='The fixed-output full200 controls did not reproduce the old serial Layer advantage. Test the original Natural EOS contract separately.',
            rule='Do not pool fixed-output controls with Natural EOS. Preserve same arrivals even under serial admission. Both axes run regardless of sign.',
            scope='One fresh serial start per axis. Online clean controls use the same original trace. This is an exploratory causal follow-up, not a new3-start performance confirmation.'))
    rows=[]
    for axis in ('Layer','Token'):
        name=f'NaturalSerial_{axis}4_q2.35_c4_s2026112061_N200_v1'
        result=cell(name,axis,4,seed=2026112061,
            entry_module='poc_flashvep.dense_ar_bi0_axis_rootcause.causal_serving',
            strict_serial_pipeline=True,phase='NATURAL_EOS_SERIAL_ONLINE_CAUSAL_FOLLOWUP')
        rows.append(result);write(ROOT/'analysis/Natural_Last4_serial_controls_v1.json',dict(results=rows))
        assert result['status']!='FAILED'
    while not (ROOT/'state/natural_Last4_reproduction_complete.json').exists():time.sleep(5)
    comparisons=[]
    for row in rows:
        axis=row['axis'];names=[row['name'],f'NaturalLast4Repro_{axis}_clean_before_q2.35_c4_s2026112061_N200_v1']
        values=[extract(name) for name in names]
        comparisons.append(dict(axis=axis,names=names,
            same_prompt_ids=[r['prompt_id'] for r in values[0]['rows']]==[r['prompt_id'] for r in values[1]['rows']],
            same_arrival_offsets=values[0]['arrival_offsets']==values[1]['arrival_offsets'],
            strict_output_sequences_equal=[r['output_ids'] for r in values[0]['rows']]==[r['output_ids'] for r in values[1]['rows']],
            summaries=[dict(name=v['name'],P_mean_ms=1000*float(np.mean([r['P_service'] for r in v['rows']])),
                D_first_mean_ms=1000*float(np.mean([r['D_first_service'] for r in v['rows']])),
                D_decode_mean_ms=1000*float(np.mean([r['D_decode'] for r in v['rows']])),
                P_plus_D_first_mean_ms=1000*float(np.mean([r['P_service']+r['D_first_service'] for r in v['rows']])),
                TTFT_mean_ms=v['measured_mean_TTFT_ms'],TTFT_p95_ms=v['measured_p95_TTFT_ms']) for v in values]))
    write(ROOT/'analysis/Natural_Last4_serial_online_causal_comparison_v1.json',dict(results=comparisons,
        evidence='MEASURED_ONLINE',scope='Natural EOS and fixed-length diagnostics remain separate. Serial admission keeps the original Poisson arrivals. Host service is not GPU-exclusive time.'))
    write(ROOT/'state/Natural_Last4_serial_controls_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':main()
