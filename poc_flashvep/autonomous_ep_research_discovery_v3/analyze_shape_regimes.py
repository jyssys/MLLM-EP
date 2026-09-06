#!/usr/bin/env python3
"""Aggregate fresh logical rows by token shape for regime/anomaly mining."""
from __future__ import annotations
import argparse,csv,statistics
from collections import defaultdict
from pathlib import Path

def q(v,p):
    v=sorted(v)
    if not v:return None
    x=(len(v)-1)*p; lo=int(x); hi=min(len(v)-1,lo+1)
    return v[lo]+(v[hi]-v[lo])*(x-lo)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--atlas',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    groups=defaultdict(list)
    with open(a.atlas) as f:
        for r in csv.DictReader(f):
            groups[(r['trace'],r['phase'],int(float(r['M'])))].append(r)
    rows=[]
    for (tr,ph,M),rs in sorted(groups.items()):
        vals=lambda k:[float(x.get(k,0) or 0) for x in rs]
        rows.append({'trace':tr,'phase':ph,'M':M,'n':len(rs),
          'tmoe_p50_ms':q(vals('cuda_ms'),.5),'tmoe_p90_ms':q(vals('cuda_ms'),.9),
          'tmoe_p99_ms':q(vals('cuda_ms'),.99),'dispatch_p50_ms':q(vals('deepep_dispatch_ms'),.5),
          'expert_p50_ms':q(vals('expert_ms'),.5),'combine_p50_ms':q(vals('deepep_combine_ms'),.5),
          'wait_p50_ms':q(vals('deepep_event_wait_ms'),.5),
          'active_p50':q(vals('active_experts'),.5),'rank_cv_p50':q(vals('rank_cv'),.5),
          'expert_cv_p50':q(vals('expert_cv'),.5)})
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    md=out.with_suffix('.md');lines=['# Shape-regime atlas','', 'Fresh logical observations grouped by exact `(trace, phase, M)`; rank rows are already collapsed by the upstream parser.', '', '|trace|phase|M|n|TMoE p50/p90/p99 (ms)|dispatch|expert|combine|wait|active|rank CV|expert CV|','|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows: lines.append(f"|{r['trace']}|{r['phase']}|{r['M']}|{r['n']}|{r['tmoe_p50_ms']:.3f}/{r['tmoe_p90_ms']:.3f}/{r['tmoe_p99_ms']:.3f}|{r['dispatch_p50_ms']:.3f}|{r['expert_p50_ms']:.3f}|{r['combine_p50_ms']:.3f}|{r['wait_p50_ms']:.3f}|{r['active_p50']:.0f}|{r['rank_cv_p50']:.3f}|{r['expert_cv_p50']:.3f}|")
    md.write_text('\n'.join(lines)+'\n')
    print({'groups':len(rows),'csv':str(out),'md':str(md)})
if __name__=='__main__':main()
