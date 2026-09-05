"""Summarise safe runtime configuration and phase-regime envelopes."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--aggregate', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); rows = list(csv.DictReader(Path(a.aggregate).open(newline='', encoding='utf-8')))
    def f(r,k):
        try:return float(r.get(k,0) or 0)
        except:return 0.0
    groups={}
    for r in rows: groups.setdefault((r.get('phase',''), int(f(r,'M'))), []).append(r)
    envelopes=[]
    for (phase,m), z in sorted(groups.items()):
        by={}
        for sms in sorted({int(f(r,'sms')) for r in z if r.get('sms') not in ('',None)}):
            q=[f(r,'critical_cuda_ms') for r in z if int(f(r,'sms'))==sms]
            if q: by[sms]=float(np.median(q))
        if not by: continue
        best=min(by,key=by.get); static=by.get(20,min(by.values()))
        envelopes.append({'phase':phase,'M':m,'n':len(z),'best_sms':best,'best_ms':by[best],
                          'static_sms20_ms':static,'static_to_oracle_pct':100*(static-by[best])/(static+1e-12),
                          'config_medians':json.dumps(by,sort_keys=True)})
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    fields=sorted(envelopes[0]) if envelopes else ['phase','M']
    with out.open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(envelopes)
    summary={}
    for phase in sorted({x['phase'] for x in envelopes}):
        z=[x for x in envelopes if x['phase']==phase]
        # Invocation-regime medians are descriptive; absent matched routes they
        # are not a causal claim.
        summary[phase]={'rows':len(z),'median_headroom_pct':float(np.median([x['static_to_oracle_pct'] for x in z])),
                        'p90_headroom_pct':float(np.quantile([x['static_to_oracle_pct'] for x in z],.9)),
                        'max_headroom_pct':float(max(x['static_to_oracle_pct'] for x in z))}
    out.with_name('runtime_oracle_summary.json').write_text(json.dumps({'rows':len(rows),'envelope_rows':len(envelopes),'by_phase':summary},indent=2),encoding='utf-8')
    print(json.dumps({'rows':len(rows),'envelope_rows':len(envelopes),'by_phase':summary},indent=2))
if __name__ == '__main__': main()
