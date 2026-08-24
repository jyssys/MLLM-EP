"""Low-synchronization live D/E/C timing and exact-input expert replay."""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

import torch


_INSTALLED = False
_CONTEXT = threading.local()
_PENDING: list[dict[str, Any]] = []
_LAST_WAVE = -1
_FLUSHED = False
_ORIGIN: torch.cuda.Event | None = None


def _layer(prefix: str) -> int:
    match = re.search(r"(?:layers|h)\.(\d+)(?:\.|$)", prefix)
    return int(match.group(1)) if match else -1


def _event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


def _control() -> dict[str, Any]:
    path = Path(os.environ["FLASHVEP_RUNTIME_CONTROL"])
    return json.loads(path.read_text()) if path.exists() else {}


def _origin() -> torch.cuda.Event:
    global _ORIGIN
    if _ORIGIN is None:
        _ORIGIN = _event(); _ORIGIN.record(torch.cuda.current_stream())
    return _ORIGIN


def _runtime_config(kernel: Any, values: dict[str, Any]) -> dict[str, int]:
    from vllm.model_executor.layers.fused_moe.fused_moe import try_get_optimal_moe_config
    experts = kernel.fused_experts
    config = try_get_optimal_moe_config(
        values["w1"].size(), values["w2"].size(), 8,
        experts.quant_config.config_name(values["a1q"].dtype),
        int(values["a1q"].shape[0]), block_shape=experts.block_shape,
    )
    return {key: int(value) for key, value in config.items() if isinstance(value, (int, bool))}


def _resolve(stage: dict[str, Any], origin: torch.cuda.Event) -> dict[str, Any]:
    result = {}
    for prefix in ("compute", "comm"):
        start, end = stage[f"{prefix}_start"], stage[f"{prefix}_end"]
        result[f"{prefix}_ms"] = float(start.elapsed_time(end))
        result[f"{prefix}_start_ms"] = float(origin.elapsed_time(start))
        result[f"{prefix}_end_ms"] = float(origin.elapsed_time(end))
    return result


def _flush() -> None:
    global _FLUSHED
    if _FLUSHED:
        return
    _FLUSHED = True
    if not _PENDING:
        return
    _PENDING[-1]["combine"]["comm_end"].synchronize()
    origin = _origin()
    from vllm.distributed import get_ep_group
    rank = int(get_ep_group().rank_in_group)
    output = Path(os.environ["FLASHVEP_RUNTIME_RAW_DIR"]); output.mkdir(parents=True, exist_ok=True)
    with (output / f"rank{rank}.jsonl").open("w") as handle:
        for item in _PENDING:
            row = {key: value for key, value in item.items() if key not in ("dispatch", "expert", "combine")}
            row["dispatch"] = _resolve(item["dispatch"], origin)
            row["expert"] = _resolve(item["expert"], origin)
            row["combine"] = _resolve(item["combine"], origin)
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    (output / f"rank{rank}.proof.json").write_text(json.dumps({
        "status":"ok", "rank":rank, "events":len(_PENDING),
        "timing":"same-stream CUDA events; one final bounded synchronization",
        "dispatch_combine":"compute-stream exposed span plus DeepEP comm-stream span",
        "visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
    },indent=2)+"\n")


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    from vllm.distributed import get_ep_group
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernelModularImpl
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer

    original_init = Qwen3MoeDecoderLayer.__init__
    original_forward = Qwen3MoeDecoderLayer.forward
    original_prepare = FusedMoEKernelModularImpl._prepare
    original_experts = FusedMoEKernelModularImpl._fused_experts
    original_finalize = FusedMoEKernelModularImpl._finalize

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        prefix = str(kwargs.get("prefix", args[1] if len(args)>1 else ""))
        self._flashvep_runtime_layer = _layer(prefix)

    def patched_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        global _LAST_WAVE
        prior_layer = getattr(_CONTEXT,"layer",-1); prior_entry=getattr(_CONTEXT,"entry",{})
        layer = int(getattr(self,"_flashvep_runtime_layer",-1))
        if layer == 0:
            entry = _control(); wave=int(entry.get("wave",-1))
            if entry.get("flush"):
                _flush()
            _CONTEXT.entry = entry if wave != _LAST_WAVE else {**entry,"instrument":False}
            _LAST_WAVE = wave
        _CONTEXT.layer=layer
        try:
            return original_forward(self,*args,**kwargs)
        finally:
            _CONTEXT.layer=prior_layer
            if layer == 47: _CONTEXT.entry=prior_entry

    def patched_prepare(self: Any, *args: Any, **kwargs: Any) -> Any:
        entry=dict(getattr(_CONTEXT,"entry",{})); layer=int(getattr(_CONTEXT,"layer",-1))
        if not entry.get("instrument") or layer < 0:
            return original_prepare(self,*args,**kwargs)
        compute=torch.cuda.current_stream(); comm=self.prepare_finalize.buffer.get_comm_stream(); _origin()
        cs,ce,ms,me=_event(),_event(),_event(),_event(); cs.record(compute); ms.record(comm)
        value=original_prepare(self,*args,**kwargs)
        ce.record(compute); me.record(comm)
        _CONTEXT.record={
            "wave":int(entry["wave"]),"context":entry["context"],"iteration":int(entry["iteration"]),
            "measured":bool(entry["measured"]),"request_id":entry["request_id"],
            "layer":layer,"rank":int(get_ep_group().rank_in_group),
            "dispatch":{"compute_start":cs,"compute_end":ce,"comm_start":ms,"comm_end":me},
        }
        return value

    def patched_experts(self: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        record=getattr(_CONTEXT,"record",None)
        if record is None:
            return original_experts(self,*args,**kwargs)
        names=("in_dtype","a1q","a1q_scale","w1","w2","topk_weights","topk_ids","activation","global_num_experts","local_num_experts","expert_map","apply_router_weight_on_input","expert_tokens_meta")
        values=dict(zip(names,args,strict=False)); values.update(kwargs)
        histogram=[int(x) for x in values["expert_tokens_meta"].expert_num_tokens_cpu.tolist()]
        stream=torch.cuda.current_stream(); start,end=_event(),_event(); start.record(stream)
        output=original_experts(self,*args,**kwargs); end.record(stream)
        record.update({"histogram":histogram,"n":sum(histogram),"g":sum(x>0 for x in histogram),
                       "dispatched_rows":int(values["a1q"].shape[0]),"runtime_config":_runtime_config(self,values)})
        record["expert"]={"compute_start":start,"compute_end":end,"comm_start":start,"comm_end":end}
        entry=dict(getattr(_CONTEXT,"entry",{}))
        if os.environ.get("FLASHVEP_RUNTIME_MODE")=="isolated" and record["layer"]==int(entry.get("target_layer",-1)) and record["rank"]==int(entry.get("target_rank",-1)):
            warmups=int(os.environ.get("FLASHVEP_RUNTIME_ISOLATED_WARMUPS","20")); iterations=int(os.environ.get("FLASHVEP_RUNTIME_ISOLATED_ITERATIONS","100"))
            for _ in range(warmups): original_experts(self,*args,**kwargs)
            starts=[_event() for _ in range(iterations)]; ends=[_event() for _ in range(iterations)]
            for a,b in zip(starts,ends): a.record(stream); original_experts(self,*args,**kwargs); b.record(stream)
            ends[-1].synchronize(); samples=[float(a.elapsed_time(b)) for a,b in zip(starts,ends)]
            path=Path(os.environ["FLASHVEP_RUNTIME_ISOLATED_OUTPUT"]); path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps({"layer":record["layer"],"rank":record["rank"],"n":record["n"],"g":record["g"],"histogram":histogram,"runtime_config":record["runtime_config"],"warmups":warmups,"iterations":iterations,"samples_ms":samples,"routing_changed":False},indent=2)+"\n")
        return output

    def patched_finalize(self: Any, *args: Any, **kwargs: Any) -> Any:
        record=getattr(_CONTEXT,"record",None)
        if record is None:
            return original_finalize(self,*args,**kwargs)
        compute=torch.cuda.current_stream(); comm=self.prepare_finalize.buffer.get_comm_stream()
        cs,ce,ms,me=_event(),_event(),_event(),_event(); cs.record(compute); ms.record(comm)
        value=original_finalize(self,*args,**kwargs); ce.record(compute); me.record(comm)
        record["combine"]={"compute_start":cs,"compute_end":ce,"comm_start":ms,"comm_end":me}
        _PENDING.append(record); _CONTEXT.record=None
        return value

    Qwen3MoeDecoderLayer.__init__=patched_init
    Qwen3MoeDecoderLayer.forward=patched_forward
    FusedMoEKernelModularImpl._prepare=patched_prepare
    FusedMoEKernelModularImpl._fused_experts=patched_experts
    FusedMoEKernelModularImpl._finalize=patched_finalize
