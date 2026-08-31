"""GPU hook for same-M exact-route window replay using production DeepEP/Triton."""
from __future__ import annotations
import hashlib, json, os, re, traceback
from pathlib import Path
from typing import Any
import numpy as np
import torch
from poc_flashvep.chunk_oracle_gpu_scale_validation import replay as base

_installed = False; _ran: set[int] = set(); _context: dict[str, int] = {}

def _stats(v: list[float]) -> dict[str, float]:
    a = np.asarray(v, dtype=np.float64)
    return {"median_ms": float(np.median(a)), "p25_ms": float(np.quantile(a,.25)), "p75_ms": float(np.quantile(a,.75)), "mean_ms": float(a.mean()), "cv": float(a.std()/max(a.mean(),1e-12))}

def _load_route(sample_id: str, source: str) -> tuple[np.ndarray, np.ndarray]:
    if source == "short":
        root = Path(os.environ["TRUE_SHORT_ROUTE_DIR"]); man = json.loads((root/"workload_manifest.json").read_text()); item = next(x["vision"] for x in man["pairs"] if x["vision"]["request_id"] == sample_id)
        with np.load(root/item["route_file"]) as z: return z["routed_experts"].astype(np.int64), z["prompt_token_ids"].astype(np.int64)
    root = Path(os.environ["TRUE_LONG_ROUTE_DIR"])
    with np.load(root/f"routing.{sample_id}.npz") as z: return z["routed_experts"].astype(np.int64), z["prompt_token_ids"].astype(np.int64)

def _one_rank(kernel: Any, original_experts: Any, spec: Any, rank: int) -> dict[str, Any]:
    from vllm.distributed import get_ep_group
    ep = get_ep_group()
    if int(ep.world_size) != 4 or type(kernel.prepare_finalize).__name__ != "DeepEPHTPrepareAndFinalize": raise AssertionError((ep.world_size, type(kernel.prepare_finalize).__name__))
    capture = torch.load(os.environ["TRUE_CAPTURE"], map_location="cpu", weights_only=False)
    candidates = json.loads(Path(os.environ["TRUE_CANDIDATES"]).read_text())["pairs"]
    warmups, iterations = int(os.environ.get("TRUE_WARMUPS", "5")), int(os.environ.get("TRUE_ITERATIONS", "20"))
    buffer = kernel.prepare_finalize.buffer; observations=[]
    cache: dict[tuple[str,str], tuple[np.ndarray,np.ndarray]] = {}
    for pair in candidates:
        route_np, token_ids = cache.setdefault((pair["request_id"], pair["source"]), _load_route(pair["request_id"], pair["source"]))
        route_np = route_np[:, int(pair["layer"]), :]
        routes_a = torch.from_numpy(route_np[int(pair["a"]["start"]):int(pair["a"]["end"])]).to(spec.w1.device, dtype=torch.int64, non_blocking=True).contiguous()
        routes_b = torch.from_numpy(route_np[int(pair["b"]["start"]):int(pair["b"]["end"])]).to(spec.w1.device, dtype=torch.int64, non_blocking=True).contiguous()
        routes_by = {"a": routes_a, "b": routes_b}; groups = {k:[list(range(len(v)))] for k,v in routes_by.items()}
        # Prewarm both candidates before any timed block; then deterministic shuffle.
        for label in ("a", "b"):
            base._run_variant("serial", groups[label], routes_by[label], capture, kernel, original_experts, buffer, spec, rank, warmups, 1)
        seed = int.from_bytes(hashlib.sha256(f"{pair['pair_id']}:{rank}".encode()).digest()[:8], "little")
        order = ["a", "b"]; np.random.default_rng(seed).shuffle(order)
        measured={}; outputs={}
        for label in order:
            timing, output = base._run_variant("serial", groups[label], routes_by[label], capture, kernel, original_experts, buffer, spec, rank, 0, iterations)
            measured[label]=timing; outputs[label]=output
        for label in ("a", "b"):
            t=measured[label]; f=pair[label]
            observations.append({"pair_id":pair["pair_id"],"request_id":pair["request_id"],"category":pair["category"],"source":pair["source"],"layer":pair["layer"],"M":pair["M"],"candidate":label,"start":f["start"],"end":f["end"],"measurement_order":order,"warmups":warmups,"iterations":iterations,"features":{k:f[k] for k in f if k not in ("hist","rank_counts","start","end")},"wall_stats":_stats([x["wall_ms"] for x in t["samples"]]),"expert_stats":_stats([x["expert_ms"] for x in t["samples"]]),"dispatch_stats":_stats([x["dispatch_ms"] for x in t["samples"]]),"combine_stats":_stats([x["combine_ms"] for x in t["samples"]]),"correctness":{"passed":bool(torch.isfinite(outputs[label]).all().item()),"output_shape":list(outputs[label].shape)},"route_identity":True,"token_partition_identity":True})
        del routes_a, routes_b, outputs; torch.cuda.empty_cache()
    out=Path(os.environ["TRUE_REPLAY_DIR"]); out.mkdir(parents=True, exist_ok=True)
    (out/f"rank{rank}.json").write_text(json.dumps({"status":"ok","rank":rank,"visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"physical_gpu_mapping":[1,2,3,4],"settings":{"backend":type(kernel.fused_experts).__name__,"prepare_finalize":type(kernel.prepare_finalize).__name__,"communication":"DeepEP high-throughput","warmups":warmups,"iterations":iterations,"prewarm_all_candidates":True,"measurement_order":"deterministic_sha256_pair_shuffle","route_identity":"exact immutable window IDs"},"observations":observations}, separators=(",",":"))+"\n")

def install() -> None:
    global _installed
    if _installed: return
    _installed=True
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernelModularImpl
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer
    oi,of,oe=Qwen3MoeDecoderLayer.__init__,Qwen3MoeDecoderLayer.forward,FusedMoEKernelModularImpl._fused_experts
    def pi(self:Any,*a:Any,**kw:Any)->None:
        oi(self,*a,**kw); p=str(kw.get("prefix",a[1] if len(a)>1 else "")); m=re.search(r"(?:layers|h)\.(\d+)(?:\.|$)",p); self._true_layer=int(m.group(1)) if m else -1
    def pf(self:Any,*a:Any,**kw:Any)->Any:
        old=_context.get("layer",-1); _context["layer"]=int(getattr(self,"_true_layer",-1))
        try:return of(self,*a,**kw)
        finally:_context["layer"]=old
    def pe(self:Any,*a:Any,**kw:Any)->torch.Tensor:
        from vllm.distributed import get_ep_group
        rank=int(get_ep_group().rank_in_group)
        if rank not in _ran and _context.get("layer",-1)==24:
            _ran.add(rank); names=("in_dtype","a1q","a1q_scale","w1","w2","topk_weights","topk_ids","activation","global_num_experts","local_num_experts","expert_map","apply_router_weight_on_input","expert_tokens_meta"); vals=dict(zip(names,a,strict=False)); vals.update(kw)
            try:_one_rank(self,oe,base._runtime_spec(vals),rank)
            except BaseException: traceback.print_exc(); raise
        return oe(self,*a,**kw)
    Qwen3MoeDecoderLayer.__init__,Qwen3MoeDecoderLayer.forward=pi,pf; FusedMoEKernelModularImpl._fused_experts=pe
