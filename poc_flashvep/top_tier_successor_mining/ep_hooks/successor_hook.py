"""Bounded SERE/MoDES baseline ports plus optional same-device stage observation.

No scheduler change. SERE's primary union is over the full DP-local batch before
TP sequence sharding; original CUDA rerouter is used after TP route all-gather.
MoDES only changes dropped IDs to DeepEP's existing -1 sentinel and zeros weights.
Quality-only zero-weight mode is available to test sentinel output equivalence.
"""
import atexit
import functools
import hashlib
import importlib.util
import json
import os
import re
import threading
import time
from pathlib import Path

_INSTALLED=False
_TLS=threading.local()
_POLICY={"method":"vanilla","name":"vanilla"}
_CACHE={}
_PENDING=[]
_COUNTER=0
_PROOF=False


def layer_id(prefix):
    m=re.search(r"layers\.(\d+)",prefix)
    return int(m.group(1)) if m else -1


def emit(name,record):
    out=Path(os.environ["SUCCESSOR_EP_OUT"])
    out.mkdir(parents=True,exist_ok=True)
    with (out/f"{name}.pid{os.getpid()}.jsonl").open("a") as f:
        f.write(json.dumps(record)+"\n")


def drain(force=False):
    while _PENDING:
        record,events,route=_PENDING[0]
        if force:
            events[-1].synchronize()
        elif not events[-1].query():
            break
        _PENDING.pop(0)
        for name,a,b in [("dispatch",0,1),("expert",1,2),("combine",2,3),("moe",0,3)]:
            record[name+"_ms"]=float(events[a].elapsed_time(events[b]))
        if route is not None:
            import torch
            path=Path(os.environ["SUCCESSOR_EP_OUT"])/"routes"
            path.mkdir(exist_ok=True)
            filename=f"step{record['step']}_ep{record['ep_rank']}_l{record['layer']}.pt"
            torch.save({k:v.cpu() for k,v in route.items()},path/filename)
            record["route_file"]=str(path/filename)
        emit("stages",record)


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED=True
    import faulthandler
    if "SUCCESSOR_EP_OUT" in os.environ:
        root=Path(os.environ["SUCCESSOR_EP_OUT"])
        root.mkdir(parents=True,exist_ok=True)
        _CACHE["stack_log"]=open(root/f"stack_pid{os.getpid()}.log","w")
        faulthandler.dump_traceback_later(90,repeat=True,file=_CACHE["stack_log"])
    import torch
    from vllm.distributed import get_ep_group,get_tp_group,get_dp_group
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernelModularImpl
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original_init=Qwen3MoeSparseMoeBlock.__init__
    @functools.wraps(original_init)
    def init(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        prefix=kwargs.get("prefix",args[1] if len(args)>1 else "")
        self.experts.router._successor_layer=layer_id(prefix)
        self.experts.router._successor_sp=self.is_sequence_parallel
    Qwen3MoeSparseMoeBlock.__init__=init

    original_execute=GPUModelRunner.execute_model
    @functools.wraps(original_execute)
    def execute(self,scheduler_output,*args,**kwargs):
        global _POLICY
        drain()
        _TLS.step=getattr(_TLS,"step",-1)+1
        _TLS.scheduled=dict(scheduler_output.num_scheduled_tokens)
        # Identity is immutable for this scheduler invocation. A mutable global
        # policy file is unsafe when TP workers process queued steps at different
        # host times: one may see a newer policy and enqueue a different collective.
        if "policy_catalog" not in _CACHE:
            _CACHE["policy_catalog"]={p["name"]:p for p in
                json.loads(Path(os.environ["SUCCESSOR_EP_CATALOG"]).read_text())}
        if _TLS.scheduled:
            labels={rid.split("__")[0][1:] for rid in _TLS.scheduled}
            assert len(labels)==1,labels
            name=next(iter(labels))
            request_id=next(iter(_TLS.scheduled))
            _POLICY={**_CACHE["policy_catalog"][name],"cohort":request_id.split("__")[1][1:]}
        else:
            _POLICY={"name":"dummy","method":"vanilla"}
        if os.environ.get("SUCCESSOR_EP_ROUTES")=="1":
            steps=_CACHE.setdefault("route_policy_steps",{})
            _TLS.policy_route_step=steps.get(_POLICY["name"],0)
            steps[_POLICY["name"]]=_TLS.policy_route_step+1
        _TLS.running=True
        _TLS.runner=self
        if os.environ.get("SUCCESSOR_EP_SCHEDULER_CONTEXT")=="1" and get_tp_group().rank_in_group==0:
            counts=list(_TLS.scheduled.values())
            emit("scheduler_context",dict(step=_TLS.step,dp_rank=get_dp_group().rank_in_group,
                policy=_POLICY["name"],cohort=_POLICY.get("cohort"),scheduled=_TLS.scheduled,
                active_requests=len(counts),scheduled_tokens=sum(counts),host_monotonic=time.monotonic(),
                phase="dummy" if not counts else "decode" if max(counts)<=1 else "prefill" if min(counts)>1 else "mixed"))
        try:
            return original_execute(self,scheduler_output,*args,**kwargs)
        finally:
            _TLS.running=False
            drain()
    GPUModelRunner.execute_model=execute

    original_forward=GPUModelRunner._model_forward
    @functools.wraps(original_forward)
    def forward(self,*args,**kwargs):
        input_ids=kwargs.get("input_ids",args[0] if args else None)
        positions=kwargs.get("positions",args[1] if len(args)>1 else None)
        embeds=kwargs.get("inputs_embeds")
        n=(input_ids.shape[0] if input_ids is not None else
           embeds.shape[0] if embeds is not None else positions.shape[-1])
        _TLS.full_ids=self.input_ids.gpu[:n]
        _TLS.token_metadata=None
        return original_forward(self,*args,**kwargs)
    GPUModelRunner._model_forward=forward

    original_select=BaseRouter.select_experts
    @functools.wraps(original_select)
    def select(self,hidden_states,router_logits,*args,**kwargs):
        weights,ids=original_select(self,hidden_states,router_logits,*args,**kwargs)
        _TLS.layer=getattr(self,"_successor_layer",-1)
        _TLS.route=None
        _TLS.current_m=ids.shape[0]
        if not getattr(_TLS,"running",False):
            return weights,ids
        assert not self.enable_eplb
        l=getattr(self,"_successor_layer",-1)
        _TLS.layer=l
        _TLS.route=None
        _TLS.current_m=ids.shape[0]
        original_ids=ids
        capture_routes=(os.environ.get("SUCCESSOR_EP_ROUTES")=="1"
                        and getattr(_TLS,"policy_route_step",0)<8
                        and l in (0,1,2,3,12,24,36,47))
        scheduled=getattr(_TLS,"scheduled",{})
        is_decode=bool(scheduled) and max(scheduled.values())<=1
        phase_enabled=(_POLICY.get("phase","all")=="all" or
                       _POLICY.get("phase")==("decode" if is_decode else "prefill"))
        if _POLICY["method"]=="vanilla" and not capture_routes:
            return weights,ids
        if _POLICY["method"] in ("sere","modes") and not phase_enabled and not capture_routes:
            return weights,ids
        full=getattr(_TLS,"full_ids",None)
        if full is None:
            raise RuntimeError("Missing token identity at runtime router")
        sp=getattr(self,"_successor_sp",False)
        cached=getattr(_TLS,"token_metadata",None) if _POLICY.get("cache_token_metadata",False) else None
        if cached is not None:
            assert cached["sp"]==sp and cached["full_n"]==full.numel()
            token_ids,chunk=cached["token_ids"],cached["chunk"]
        elif sp:
            tp=get_tp_group()
            chunk=(full.numel()+tp.world_size-1)//tp.world_size
            padded=torch.nn.functional.pad(full,(0,chunk*tp.world_size-full.numel()),value=-1)
            token_ids=padded[tp.rank_in_group*chunk:(tp.rank_in_group+1)*chunk]
        else:
            token_ids=full
            chunk=None
        if token_ids.numel()!=ids.shape[0]:
            raise RuntimeError(f"Modality identity mismatch {token_ids.shape} vs {ids.shape}, SP={sp}")
        if "token_metadata" not in _CACHE:
            _CACHE["token_metadata"]=json.loads(Path(os.environ["SUCCESSOR_EP_TOKEN_IDS"]).read_text())
        tm=_CACHE["token_metadata"]
        if "special" not in _CACHE:
            _CACHE["special"]=torch.tensor(tm["special_ids"]+[-1],device=ids.device)
        if _POLICY["method"]=="modes" or capture_routes:
            if cached is not None and cached["text"] is not None:
                vision,text=cached["vision"],cached["text"]
            else:
                vision=(token_ids==tm["image_id"]) | (token_ids==tm["video_id"])
                text=~torch.isin(token_ids,_CACHE["special"])
        else:
            vision=text=None
        if _POLICY.get("cache_token_metadata",False):
            _TLS.token_metadata=dict(sp=sp,full_n=full.numel(),token_ids=token_ids,chunk=chunk,vision=vision,text=text)
        if _POLICY["method"]=="sere" and phase_enabled:
            if l<0:
                raise RuntimeError("Unknown SERE layer")
            path=_POLICY["similarity"]
            if path not in _CACHE:
                _CACHE[path]=torch.load(path,map_location=ids.device,weights_only=True)
            if "sere_extension" not in _CACHE:
                path_so=os.environ["SUCCESSOR_SERE_EXTENSION"]
                spec=importlib.util.spec_from_file_location("sere_successor_reference",path_so)
                module=importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                _CACHE["sere_extension"]=module
            # The paper retains a batch-global primary union. TP's SP partition
            # must not silently replace this with separate smaller unions.
            gathered=get_tp_group().all_gather(ids,dim=0) if sp else ids
            # Exclude TP padding (which is only at the end of the full sequence).
            gathered=gathered[:full.numel()].contiguous()
            k=(_POLICY.get("retain",2))
            all_weights=torch.empty(gathered.shape,device=ids.device,dtype=weights.dtype)
            # Official CUDA uses weights only for shape/device, not values.
            if "sere_masks" not in _CACHE:
                _CACHE["sere_masks"]=[(torch.zeros(128,device=ids.device,dtype=torch.bool),
                                       torch.zeros(128,device=ids.device,dtype=torch.long)) for _ in range(48)]
            rerouted=_CACHE["sere_extension"].reroute(all_weights,gathered.clone(),_CACHE[path][l],k,
                *_CACHE["sere_masks"][l],_POLICY.get("rho",.5))
            if sp:
                padded=torch.nn.functional.pad(rerouted,(0,0,0,chunk*get_tp_group().world_size-rerouted.shape[0]),value=-1)
                ids=padded[get_tp_group().rank_in_group*chunk:(get_tp_group().rank_in_group+1)*chunk].contiguous()
                weights=weights.masked_fill(token_ids[:,None]==-1,0)
            else:
                ids=rerouted
        elif _POLICY["method"]=="modes" and phase_enabled:
            path=_POLICY["alpha"]
            if path not in _CACHE:
                _CACHE[path]=torch.tensor(json.loads(Path(path).read_text())["alpha"],device=ids.device)
            at,av=_CACHE[path][l]
            # Official MoDES casts normalized top-k probabilities to the BF16
            # router dtype BEFORE scaling/thresholding. vLLM returns FP32 top-k
            # weights, so use BF16 for the decision, preserving the unmasked
            # stock weights for EP baseline math (including the no-op control).
            decision_weights=weights.to(torch.bfloat16)
            drop=((text[:,None] & (decision_weights*at<_POLICY["tau_text"])) |
                  (vision[:,None] & (decision_weights*av<_POLICY["tau_vision"])))
            weights=weights.masked_fill(drop,0)
            if _POLICY.get("fast_skip",True):
                ids=ids.masked_fill(drop,-1)
        elif _POLICY["method"] not in ("vanilla","sere","modes"):
            raise ValueError(_POLICY)
        if capture_routes:
            _TLS.route={"ids":ids.detach().clone(),"original_ids":original_ids.detach().clone(),
                        "weights":weights.detach().clone(),"token_ids":token_ids.detach().clone(),
                        "vision":vision.detach().clone(),"text":text.detach().clone()}
        return weights,ids
    BaseRouter.select_experts=select

    original_apply=FusedMoEKernelModularImpl.apply
    @functools.wraps(original_apply)
    def apply(self,*args,**kwargs):
        global _COUNTER,_PROOF
        # Every real OR dummy collective advances this ordinal, so asynchronous
        # DP frontend step numbering cannot incorrectly join different EP calls.
        _COUNTER+=1
        is_real=getattr(_TLS,"running",False)
        if not is_real and not hasattr(_TLS,"runner"):
            return original_apply(self,*args,**kwargs)
        if not _PROOF:
            cfg=_TLS.runner.vllm_config
            pf=self.prepare_finalize
            record={"vllm_source":__import__("vllm").__file__,
                    "vllm_version":__import__("vllm").__version__,
                    "deepep_source":__import__("deep_ep").__file__,
                    "prepare_finalize":type(pf).__name__,"experts":type(self.fused_experts).__name__,
                    "dp_size":cfg.parallel_config.data_parallel_size,
                    "tp_size":cfg.parallel_config.tensor_parallel_size,
                    "ep_size":get_ep_group().world_size,"ep_rank":get_ep_group().rank_in_group,
                    "enable_ep":cfg.parallel_config.enable_expert_parallel,
                    "dbo":cfg.parallel_config.enable_dbo,
                    "sequence_parallel":cfg.parallel_config.use_sequence_parallel_moe,
                    "all2all_backend":cfg.parallel_config.all2all_backend,
                    "physical_visible_devices":os.environ["CUDA_VISIBLE_DEVICES"]}
            assert record["prepare_finalize"]=="DeepEPHTPrepareAndFinalize",record
            assert record["dp_size"]==2 and record["tp_size"]==2 and record["ep_size"]==4,record
            emit("runtime_proof",record)
            _PROOF=True
        if os.environ.get("SUCCESSOR_EP_TIMING")!="1":
            return original_apply(self,*args,**kwargs)
        events=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
        _TLS.events=events
        scheduled=getattr(_TLS,"scheduled",{}) if is_real else {}
        counts=list(scheduled.values())
        phase="dummy" if not is_real else "decode" if counts and max(counts)<=1 else "prefill" if counts and min(counts)>1 else "mixed"
        record={"step":_TLS.step,"layer":getattr(_TLS,"layer",-1),"invocation":_COUNTER,
                "ep_rank":get_ep_group().rank_in_group,"tp_rank":get_tp_group().rank_in_group,
                "dp_rank":get_dp_group().rank_in_group,"M":getattr(_TLS,"current_m",-1),
                "request_ids":list(scheduled),"scheduled":scheduled,"phase":phase,
                "policy":_POLICY["name"],"cohort":_POLICY.get("cohort"),
                "host_start_ns":time.monotonic_ns()}
        output=original_apply(self,*args,**kwargs)
        _PENDING.append((record,events,getattr(_TLS,"route",None)))
        _TLS.events=None
        return output
    FusedMoEKernelModularImpl.apply=apply
    for method,start,end in [("_prepare",0,1),("_fused_experts",1,2),("_finalize",2,3)]:
        original=getattr(FusedMoEKernelModularImpl,method)
        def wrap(original=original,start=start,end=end):
            @functools.wraps(original)
            def timed(self,*args,**kwargs):
                events=getattr(_TLS,"events",None)
                if events is not None and start==0:
                    events[start].record()
                output=original(self,*args,**kwargs)
                if events is not None:
                    events[end].record()
                return output
            return timed
        setattr(FusedMoEKernelModularImpl,method,wrap())
    atexit.register(lambda:drain(force=True))
