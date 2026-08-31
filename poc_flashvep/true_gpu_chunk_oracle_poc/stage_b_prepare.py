"""Prepare a fixed, bounded measured-interval candidate table for Stage B."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SHORT = ROOT / "poc_flashvep/deepep_revalidation/results/live_prefill_execution_regime_20260821_111609"
LONG = ROOT / "poc_flashvep/deepep_revalidation/results/chunk_oracle_gpu_scale_validation_20260831_223000"
REPRESENTATIVE = ("coffee_rocket", "model_card", "retina", "method")
LAYERS = (0, 24, 47); BUDGETS = (128, 256); STEP = 16; OFFSET = 32
FAIR_CUTS = ROOT / "poc_flashvep/deepep_revalidation/results/fair_chunk_oracle_decomposition_20260831_235900/strategy_cuts.json"

def load_route(sid: str) -> tuple[str, np.ndarray]:
    if sid in {"coffee_rocket", "model_card", "retina", "method"}:
        man=json.loads((SHORT/"workload_manifest.json").read_text()); item=next(x["vision"] for x in man["pairs"] if x["vision"]["request_id"]==sid)
        with np.load(SHORT/item["route_file"]) as z: return "short", z["routed_experts"].astype(np.int64)
    with np.load(LONG/f"routing.{sid}.npz") as z: return "long", z["routed_experts"].astype(np.int64)

def fixed(n:int,b:int)->list[int]: return list(range(0,n,b))+([n] if n%b else [])
def balanced(n:int,b:int)->list[int]:
    k=(n+b-1)//b; q,r=divmod(n,k); return [0]+list(np.cumsum([q+(i<r) for i in range(k)],dtype=np.int64))

def intervals(n:int,b:int)->tuple[list[int],list[list[int]]]:
    # Candidate boundaries are fixed cuts +/- 0/16/32 tokens, plus endpoints.
    cuts=fixed(n,b); pts={0,n}
    for c in cuts:
        for d in range(-OFFSET,OFFSET+1,STEP): pts.add(max(0,min(n,c+d)))
    pts=sorted(pts); out=[]
    for i,s in enumerate(pts[:-1]):
        for e in pts[i+1:]:
            if e-s>b: break
            if e>s: out.append([int(s),int(e)])
    return pts,out

def main()->None:
    ap=argparse.ArgumentParser(); ap.add_argument("--result",required=True); args=ap.parse_args(); out=Path(args.result); out.mkdir(parents=True,exist_ok=True)
    tasks=[]
    for sid in REPRESENTATIVE:
        source,route=load_route(sid); n=len(route)
        for layer in LAYERS:
            for b in BUDGETS:
                pts,ints=intervals(n,b)
                old=json.loads(FAIR_CUTS.read_text())["samples"].get(sid, {}).get(str(b), {})
                tile=old.get("same_count", fixed(n,b))
                tasks.append({"task_id":f"{sid}:{layer}:{b}","request_id":sid,"source":source,"layer":layer,"budget":b,"n":n,"grid_step":STEP,"boundary_offsets":list(range(-OFFSET,OFFSET+1,STEP)),"candidate_boundaries":pts,"intervals":ints,"fixed_cuts":fixed(n,b),"balanced_cuts":list(map(int,balanced(n,b))),"tile_same_count_cuts":[int(x) for x in tile]})
    out_file=out/"stage_b_intervals.json"; out_file.write_text(json.dumps({"status":"ok","tasks":tasks,"selection_policy":"four fixed representative requests; layers 0/24/47; B=128/256; boundaries fixed-cut +/-32 on 16-token grid"},separators=(",",":"))+"\n")
    print("tasks",len(tasks),"intervals",sum(len(t["intervals"]) for t in tasks))
if __name__=="__main__": main()
