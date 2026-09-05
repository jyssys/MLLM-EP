"""Streaming invocation aggregation using the driver's route id.

The rank observers in the online harness write four EP-rank records for one
route/layer invocation.  Their CUDA clocks are rank-local, so this script
never subtracts timestamps; it only joins records sharing the normalized
route id and reports the maximum rank interval as the conservative critical
path.  Unlike the older timestamp-window aggregator this is linear in the
trace size and does not retain all JSON objects in memory.
"""
from __future__ import annotations
import argparse,csv,gzip,json,re
from pathlib import Path
import numpy as np

def files(paths):
    for root in paths:
        for p in sorted(Path(root).rglob("invocations.jsonl")) + sorted(Path(root).rglob("invocations.jsonl.gz")):
            yield p

def norm_route(r):
    rf=str(r.get("route_file", ""))
    # route_00000001_dp0_l8.npz and dp1 records refer to the same invocation.
    m=re.search(r"(route_\d+)_dp\d+_l(\d+)",rf)
    return (m.group(1)+"_l"+m.group(2)) if m else None

def as_float(r,k):
    try:return float(r.get(k,0) or 0)
    except Exception:return 0.0

def aggregate(paths):
    out=[]; current=None; group=[]
    def flush(g):
        if not g:return
        ranks={int(r.get("ep_rank",-1)):r for r in g}
        vals=[as_float(r,"cuda_ms") for r in ranks.values()]
        if not vals:return
        a=g[0]; sms=a.get("sms")
        if sms is None:
            m=re.search(r"sms(\d+)",str(a.get("request_context",""))); sms=int(m.group(1)) if m else (20 if "sms20" in str(a.get("request_context","")) else None)
        out.append({"source":a.get("source", ""),"phase":a.get("phase",""),"layer":int(a.get("layer",-1)),"M":int(as_float(a,"M")),"sms":sms,"route_id":a.get("_route_id"),"n_ranks":len(ranks),"complete_ranks":len(ranks)==4,"critical_cuda_ms":max(vals),"mean_rank_cuda_ms":float(np.mean(vals)),"rank_imbalance":max(vals)/(float(np.mean(vals))+1e-12),"wall_max_ms":max(as_float(r,"wall_ms") for r in ranks.values()),"fanout_mean":float(np.mean([as_float(r,"fanout_mean") for r in ranks.values()])),"fanout_f4":float(np.mean([as_float(r,"fanout_f4") for r in ranks.values()])),"rank_max_mean":float(np.mean([as_float(r,"rank_max_mean") for r in ranks.values()])),"expert_max_mean":float(np.mean([as_float(r,"expert_max_mean") for r in ranks.values()])),"active_experts":float(np.mean([as_float(r,"active_experts") for r in ranks.values()])),"total_assignments":float(np.mean([as_float(r,"total_assignments") for r in ranks.values()])),"timestamp_ns":int(np.median([int(r.get("timestamp_ns",0)) for r in ranks.values()]))})
    for p in files(paths):
        src=p.parent.name; opener=gzip.open if p.suffix==".gz" else open
        current=None; group=[]
        with opener(p,"rt",encoding="utf-8",errors="ignore") as fh:
            for line in fh:
                try:r=json.loads(line)
                except Exception:continue
                layer=int(r.get("layer",-1)); M=as_float(r,"M")
                if layer<0 or M>2048:continue
                rid=norm_route(r)
                if rid is None:
                    # Fallback to a local timestamp bucket for old hook rows.
                    rid=f"ts{int(r.get('timestamp_ns',0))//2_000_000}"
                key=(rid,str(r.get("phase","")),layer,int(M))
                r["source"]=src; r["_route_id"]=rid
                if current is None: current=key
                if key!=current:
                    flush(group); group=[]; current=key
                group.append(r)
        flush(group)
    return [r for r in out if r["complete_ranks"]]

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--trace",action="append",required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
    rows=aggregate(a.trace); out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    fields=list(rows[0]) if rows else ["phase"]
    with out.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    print(json.dumps({"rows":len(rows),"phase_counts":{p:sum(r["phase"]==p for r in rows) for p in sorted({r["phase"] for r in rows})}},indent=2))
if __name__=="__main__":main()
