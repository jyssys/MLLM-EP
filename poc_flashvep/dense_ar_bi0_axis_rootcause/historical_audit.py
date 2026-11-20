"""Reproduce the reported paradox without changing frozen evidence."""
import json
from .common import ROOT,PREVIOUS,initialize,write,digest


def main():
    initialize()
    path=PREVIOUS/'analysis/BI0_controlled_fixed_axis_discovery_v3.json'
    values=json.loads(path.read_text())['results']
    cells={(r['axis'],r['cut']):r for r in values}
    rows=[]
    for cut in (2,4,8,12):
        sides={}
        for axis in ('Layer','Token'):
            row=cells[(axis,cut)]
            trace=json.loads((PREVIOUS/'analysis'/(row['name']+'_axis_timeline.json')).read_text())
            sides[axis]=dict(name=row['name'],TTFT_p95_ms=row['metrics']['TTFT_p95_ms'],
                P_queue_ms=trace['summary']['P_queue_ms']['mean'],
                P_service_host_ms=trace['summary']['P_prepare_to_ready_ms']['mean'],
                D_ready_to_first_host_ms=trace['summary']['ready_to_first_worker_output_ms']['mean'])
        rows.append(dict(cut=cut,sides=sides,TTFT_Layer_over_Token=sides['Layer']['TTFT_p95_ms']/sides['Token']['TTFT_p95_ms'],
            D_host_Layer_over_Token=sides['Layer']['D_ready_to_first_host_ms']/sides['Token']['D_ready_to_first_host_ms']))
    assert rows[1]['TTFT_Layer_over_Token']>1 and rows[1]['D_host_Layer_over_Token']<1
    result=dict(evidence='HISTORICAL_TRACE_ANALYSIS',source=str(path),source_SHA256=digest(path),rows=rows,
        paradox_reproduced=True,scope='A smaller destination host service does not imply a lower online TTFT. No new GPU measurement or causal conclusion.')
    write(ROOT/'analysis/historical_paradox_audit.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
