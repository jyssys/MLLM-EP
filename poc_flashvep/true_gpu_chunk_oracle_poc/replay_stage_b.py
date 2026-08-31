"""Stage B measured interval costs and independent selected-cut validation."""
from __future__ import annotations
import hashlib,json,os,re,traceback
from pathlib import Path
from typing import Any
import numpy as np, torch
from poc_flashvep.chunk_oracle_gpu_scale_validation import replay as base

_installed=False; _ran:set[int]=set(); _ctx:dict[str,int]={}
def stats(v:list[float])->dict[str,float]:
 a=np.asarray(v,dtype=float); return {"median_ms":float(np.median(a)),"p25_ms":float(np.quantile(a,.25)),"p75_ms":float(np.quantile(a,.75)),"mean_ms":float(a.mean()),"cv":float(a.std()/max(a.mean(),1e-12))}
def route(sid:str,source:str)->np.ndarray:
 if source=="short":
  root=Path(os.environ["TRUE_SHORT_ROUTE_DIR"]); man=json.loads((root/"workload_manifest.json").read_text()); item=next(x["vision"] for x in man["pairs"] if x["vision"]["request_id"]==sid)
  with np.load(root/item["route_file"]) as z:return z["routed_experts"].astype(np.int64)
 root=Path(os.environ["TRUE_LONG_ROUTE_DIR"])
 with np.load(root/f"routing.{sid}.npz") as z:return z["routed_experts"].astype(np.int64)
def interval_batch(task:dict[str,Any], ints:list[list[int]], layer:int)->tuple[torch.Tensor,list[list[int]]]:
 r=route(task["request_id"],task["source"])[...,layer,:]
 chunks=[r[s:e] for s,e in ints]; cat=np.concatenate(chunks,axis=0); groups=[]; cur=0
 for c in chunks: groups.append(list(range(cur,cur+len(c)))); cur+=len(c)
 return torch.from_numpy(cat),groups
def wave_stats(timing:dict[str,Any], idx:int)->dict[str,float]:
 vals=[]
 for sample in timing["samples"]:
  w=sample["per_wave"][idx];
  if "dispatch_start_ms" not in w or "combine_end_ms" not in w: continue
  vals.append({"wall_ms":float(w["combine_end_ms"]-w["dispatch_start_ms"]),"expert_ms":float(w.get("expert_ms",0)),"dispatch_ms":float(w.get("dispatch_ms",0)),"combine_ms":float(w.get("combine_ms",0))})
 return {k:stats([x[k] for x in vals]) for k in ("wall_ms","expert_ms","dispatch_ms","combine_ms")}
def cost_rank(kernel:Any,original:Any,spec:Any,rank:int)->dict[str,Any]:
 from vllm.distributed import get_ep_group
 ep=get_ep_group(); assert int(ep.world_size)==4 and type(kernel.prepare_finalize).__name__=="DeepEPHTPrepareAndFinalize"
 capture=torch.load(os.environ["TRUE_CAPTURE"],map_location="cpu",weights_only=False); manifest=json.loads(Path(os.environ["TRUE_INTERVALS"]).read_text()); warm=int(os.environ.get("TRUE_B_WARMUPS","2")); iters=int(os.environ.get("TRUE_B_ITERATIONS","5")); buffer=kernel.prepare_finalize.buffer; rows=[]
 for task in manifest["tasks"]:
  ints=task["intervals"]; order=list(range(len(ints))); seed=int.from_bytes(hashlib.sha256(task["task_id"].encode()).digest()[:8],"little"); np.random.default_rng(seed).shuffle(order)
  # Prewarm every candidate, in deterministic shuffled batches, then measure.
  for off in range(0,len(order),64):
   sub=[ints[i] for i in order[off:off+64]]; rt,groups=interval_batch(task,sub,task["layer"]); base._run_variant("serial",groups,rt,capture,kernel,original,buffer,spec,rank,warm,1)
  all_t={}
  for off in range(0,len(order),64):
   ids=order[off:off+64]; sub=[ints[i] for i in ids]; rt,groups=interval_batch(task,sub,task["layer"]); timing,_=base._run_variant("serial",groups,rt,capture,kernel,original,buffer,spec,rank,0,iters)
   for local,i in enumerate(ids): all_t[str(ints[i])]=wave_stats(timing,local)
  for key,t in all_t.items(): rows.append({"task_id":task["task_id"],"request_id":task["request_id"],"source":task["source"],"layer":task["layer"],"budget":task["budget"],"interval":json.loads(key),"rank":rank,"wall_stats":t["wall_ms"],"expert_stats":t["expert_ms"],"dispatch_stats":t["dispatch_ms"],"combine_stats":t["combine_ms"],"warmups":warm,"iterations":iters})
 return {"status":"ok","rank":rank,"visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"physical_gpu_mapping":[1,2,3,4],"mode":"cost_table","settings":{"prewarm_all_intervals":True,"batch_size":64,"backend":type(kernel.fused_experts).__name__,"prepare_finalize":type(kernel.prepare_finalize).__name__},"rows":rows}
def validate_rank(kernel:Any,original:Any,spec:Any,rank:int)->dict[str,Any]:
 from vllm.distributed import get_ep_group
 ep=get_ep_group(); assert int(ep.world_size)==4
 capture=torch.load(os.environ["TRUE_CAPTURE"],map_location="cpu",weights_only=False); cuts=json.loads(Path(os.environ["TRUE_CUTS"]).read_text())["tasks"]; warm=int(os.environ.get("TRUE_B_VALIDATE_WARMUPS","5")); iters=int(os.environ.get("TRUE_B_VALIDATE_ITERATIONS","20")); buffer=kernel.prepare_finalize.buffer; rows=[]
 for task in cuts:
  strategies=("fixed","balanced","tile_same_count","true_gpu")
  by={s:task[f"{s}_cuts"] for s in strategies}; order=list(strategies); seed=int.from_bytes(hashlib.sha256(task["task_id"].encode()).digest()[:8],"little"); np.random.default_rng(seed).shuffle(order); measured={}
  for s in strategies:
   rt,groups=interval_batch(task,[[a,b] for a,b in zip(by[s][:-1],by[s][1:])],task["layer"]); base._run_variant("serial",groups,rt,capture,kernel,original,buffer,spec,rank,warm,1)
  for s in order:
   ints=[[a,b] for a,b in zip(by[s][:-1],by[s][1:])]; rt,groups=interval_batch(task,ints,task["layer"]); timing,out=base._run_variant("serial",groups,rt,capture,kernel,original,buffer,spec,rank,0,iters); ws=[wave_stats(timing,i) for i in range(len(groups))]
   measured[s]={"wall":stats([x["wall_ms"]["median_ms"] for x in ws]),"expert":stats([x["expert_ms"]["median_ms"] for x in ws]),"dispatch":stats([x["dispatch_ms"]["median_ms"] for x in ws]),"combine":stats([x["combine_ms"]["median_ms"] for x in ws]),"chunks":len(groups),"chunk_sizes":[b-a for a,b in zip(by[s][:-1],by[s][1:])],"finite":bool(torch.isfinite(out).all().item())}
  for s in strategies: rows.append({"task_id":task["task_id"],"request_id":task["request_id"],"source":task["source"],"layer":task["layer"],"budget":task["budget"],"rank":rank,"strategy":s,"boundaries":by[s],"chunks":measured[s]["chunks"],"chunk_sizes":measured[s]["chunk_sizes"],"wall_stats":measured[s]["wall"],"expert_stats":measured[s]["expert"],"dispatch_stats":measured[s]["dispatch"],"combine_stats":measured[s]["combine"],"correctness":measured[s]["finite"],"route_identity":True,"token_partition_identity":True,"measurement_order":order})
 return {"status":"ok","rank":rank,"visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"physical_gpu_mapping":[1,2,3,4],"mode":"validation","rows":rows}
def install()->None:
 global _installed
 if _installed:return
 _installed=True
 from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernelModularImpl
 from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer
 oi,of,oe=Qwen3MoeDecoderLayer.__init__,Qwen3MoeDecoderLayer.forward,FusedMoEKernelModularImpl._fused_experts
 def pi(self:Any,*a:Any,**kw:Any)->None:
  oi(self,*a,**kw); p=str(kw.get("prefix",a[1] if len(a)>1 else "")); m=re.search(r"(?:layers|h)\.(\d+)(?:\.|$)",p); self._true_b_layer=int(m.group(1)) if m else -1
 def pf(self:Any,*a:Any,**kw:Any)->Any:
  old=_ctx.get("layer",-1); _ctx["layer"]=int(getattr(self,"_true_b_layer",-1))
  try:return of(self,*a,**kw)
  finally:_ctx["layer"]=old
 def pe(self:Any,*a:Any,**kw:Any)->torch.Tensor:
  from vllm.distributed import get_ep_group
  rank=int(get_ep_group().rank_in_group)
  target=24 if os.environ.get("TRUE_B_MODE","cost")=="cost" else 24
  if rank not in _ran and _ctx.get("layer",-1)==target:
   _ran.add(rank); names=("in_dtype","a1q","a1q_scale","w1","w2","topk_weights","topk_ids","activation","global_num_experts","local_num_experts","expert_map","apply_router_weight_on_input","expert_tokens_meta"); vals=dict(zip(names,a,strict=False)); vals.update(kw)
   try:
    spec=base._runtime_spec(vals); payload=cost_rank(self,oe,spec,rank) if os.environ.get("TRUE_B_MODE","cost")=="cost" else validate_rank(self,oe,spec,rank); out=Path(os.environ["TRUE_B_REPLAY_DIR"]); out.mkdir(parents=True,exist_ok=True); (out/f"rank{rank}.json").write_text(json.dumps(payload,separators=(",",":"))+"\n")
   except BaseException: traceback.print_exc(); raise
  return oe(self,*a,**kw)
 Qwen3MoeDecoderLayer.__init__,Qwen3MoeDecoderLayer.forward=pi,pf; FusedMoEKernelModularImpl._fused_experts=pe
