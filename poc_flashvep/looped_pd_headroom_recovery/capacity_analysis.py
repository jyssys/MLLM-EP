"""Finite independent-service tables. Exclude preparation from D timings."""
import statistics
from .common import *
from .analyze import records,table


def percentile(values,q):
    if not values:return None
    values=sorted(values)
    i=(len(values)-1)*q
    lo=int(i);hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(i-lo)


def main():
    rows=[]
    for path in (ROOT/'configs').glob('*.json'):
        cfg=json.loads(path.read_text())
        if not isinstance(cfg,dict) or 'name' not in cfg:continue
        folder=ROOT/'raw'/cfg['name']
        for value in records(folder/'cohorts.jsonl'):
            if value.get('status')!='COMPLETE':continue
            rows.append(dict(name=cfg['name'],model=cfg['model'],length=cfg['length'],cap=value['job'].get('cap',1),
                chunk=cfg['chunk'],role='P_ONLY',batch=value['job']['batch'],repeat=value['repeat'],
                host_service_seconds=value['prefill_host_service_s'],requests_per_second=value['job']['batch']/value['prefill_host_service_s'],
                output_parity=value['native_output_parity'],evidence='MEASURED_UNPROFILED_FINITE_LOCAL_SERVICE',
                scope='No D reservation or NIXL wait. Three repeats within one process are not independent starts. Not a sustainable online rate.'))
        values=records(folder/'prepared_D_requests.jsonl')
        window=folder/'prepared_D_window.json'
        if values and window.exists():
            w=json.loads(window.read_text());span=w['end']-w['start']
            rows.append(dict(name=cfg['name'],model=cfg['model'],length=cfg['length'],cap=cfg['cap'],chunk=cfg['chunk'],
                role='PREPARED_D',batch=len(values),repeat=0,host_service_seconds=span,
                requests_per_second=len(values)/span,output_tokens_per_second=sum(len(r['output_ids']) for r in values)/span,
                TPOT_p95_ms=percentile([r['tpot_ms'] for r in values],.95),ready_pattern=cfg['D_ready_pattern'],
                evidence='MEASURED_NATIVE_VALID_KV_PREPARED_D',
                scope='Native import and valid-state synchronization precede the D timer. Finite all-ready or staggered cohorts are not online capacity.'))
    table('independent_capacity.csv',rows)
    print(json.dumps(dict(rows=len(rows))))


if __name__=='__main__':main()
