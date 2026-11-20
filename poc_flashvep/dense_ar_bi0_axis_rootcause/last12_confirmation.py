"""Confirm cut12 before selecting new high-value interventions."""
import json
import time
from .common import ROOT,write,initialize
from .campaign import cell,smoke


def main():
    initialize()
    declaration=ROOT/'configs/frozen_last12_confirmation_v1.json'
    if not declaration.exists():
        write(declaration,dict(time=time.time(),N=200,cut=12,QPS=2.35,seeds=[2026112071,2026112072,2026112073],
            actual_batches=[1,1,1],source_credit=4,hidden_source_slots=4,KV_block=128,
            order=[['Native','Layer','Token'],['Token','Native','Layer'],['Layer','Token','Native']],
            SLO=dict(TTFT_ms=5000,every_ITL_ms=34,TPOT_ms=33),
            scope='New independent paired arrival starts. Prior discovery is historical and is not pooled into confirmation CI. No outcome-based cancellation.'))
    if not (ROOT/'state/new_overlay_runtime_smoke_complete.json').exists():
        smoke()
    results=[]
    orders=(('Native','Layer','Token'),('Token','Native','Layer'),('Layer','Token','Native'))
    for seed,order in zip((2026112071,2026112072,2026112073),orders):
        for axis in order:
            name=f'Last12Confirm_{axis}_q2.35_c4_s{seed}_N200_v1'
            row=cell(name,axis,0 if axis=='Native' else 12,seed=seed,
                phase='LAST12_INDEPENDENT_CONFIRMATION')
            results.append(row)
            write(ROOT/'analysis/last12_confirmation_v1.json',dict(results=results,
                evidence='MEASURED_ONLINE',scope='Every scheduled arm remains visible, including failures and quality failures.'))
            assert row['status']!='FAILED',name
    write(ROOT/'state/last12_confirmation_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/last12_confirmation_hold.json',dict(time=time.time(),error=repr(error)))
        raise
