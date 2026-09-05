"""Aggregate four EP-rank observer rows into invocation-level summaries.

Rank timestamps are only used to cluster rows that are within a small local
window; no cross-device absolute timestamp subtraction is performed.  The
result uses max rank CUDA interval as a conservative critical-path measure.
"""
from __future__ import annotations
import argparse,csv,gzip,json,re
from pathlib import Path
import numpy as np

def load(paths):
 out=[]
 for root in paths:
  files = list(Path(root).rglob('invocations.jsonl')) + list(Path(root).rglob('invocations.jsonl.gz'))
  for p in sorted(files):
   src=p.parent.name
   opener = gzip.open if p.suffix == '.gz' else open
   with opener(p, 'rt', encoding='utf-8', errors='ignore') as fh:
    for line in fh:
     try:r=json.loads(line)
     except:continue
     if int(r.get('layer',-1))<0 or float(r.get('M',0))>2048:continue
     r['source']=src; r['sms']=int(re.search(r'sms(\d+)',str(r.get('request_context',''))).group(1)) if re.search(r'sms(\d+)',str(r.get('request_context',''))) else (20 if 'sms20' in str(r.get('request_context','')) else None)
     for k in ('M','cuda_ms','wall_ms','fanout_mean','fanout_f4','rank_max_mean','expert_max_mean','expert_cv','active_experts','total_assignments'):
      try:r[k]=float(r.get(k,0) or 0)
      except:r[k]=0.0
     out.append(r)
 return out

def aggregate(rows, window_ns=2_000_000):
 out=[]
 buckets={}
 for r in rows: buckets.setdefault((r['source'],r['phase'],int(r['layer']),int(r['M'])),[]).append(r)
 for key,z0 in sorted(buckets.items()):
  z=sorted(z0,key=lambda r:int(r.get('timestamp_ns',0)))
  cur=[]; last=None
  def flush(g):
   if not g:return
   vals=np.asarray([r['cuda_ms'] for r in g]); ranks={int(r['ep_rank']):r for r in g}; rr=[r['cuda_ms'] for r in ranks.values()]
   # A valid group normally contains one row per EP rank.  Incomplete groups
   # are retained but marked so they cannot be mistaken for a full critical span.
   a=g[0]; out.append({'source':a['source'],'phase':a['phase'],'layer':int(a['layer']),'M':int(a['M']),'sms':a.get('sms'),'n_ranks':len(ranks),'complete_ranks':len(ranks)==4,'critical_cuda_ms':float(max(rr)),'mean_rank_cuda_ms':float(np.mean(rr)),'rank_imbalance':float(max(rr)/(np.mean(rr)+1e-12)),'wall_max_ms':float(max(r.get('wall_ms',0) for r in ranks.values())),'fanout_mean':float(np.mean([r.get('fanout_mean',0) for r in ranks.values()])), 'fanout_f4':float(np.mean([r.get('fanout_f4',0) for r in ranks.values()])), 'rank_max_mean':float(np.mean([r.get('rank_max_mean',0) for r in ranks.values()])), 'expert_max_mean':float(np.mean([r.get('expert_max_mean',0) for r in ranks.values()])), 'active_experts':float(np.mean([r.get('active_experts',0) for r in ranks.values()])), 'total_assignments':float(np.mean([r.get('total_assignments',0) for r in ranks.values()])), 'timestamp_ns':int(np.median([int(r.get('timestamp_ns',0)) for r in g]))})
  for r in z:
   t=int(r.get('timestamp_ns',0))
   if last is None or t-last<=window_ns:cur.append(r)
   else:flush(cur);cur=[r]
   last=t
  flush(cur)
 return [r for r in out if r['complete_ranks']]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--trace',action='append',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();rows=aggregate(load(a.trace));out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 with (out/'aggregate_invocations.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=sorted(rows[0]) if rows else ['phase']);w.writeheader();w.writerows(rows)
 summary={'rows':len(rows),'sources':sorted({r['source'] for r in rows}),'phase_counts':{p:sum(r['phase']==p for r in rows) for p in sorted({r['phase'] for r in rows})},'critical_quantiles':{p:[float(np.quantile([r['critical_cuda_ms'] for r in rows if r['phase']==p],q)) for q in (.5,.9,.95,.99)] for p in sorted({r['phase'] for r in rows})}}
 (out/'aggregate_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
