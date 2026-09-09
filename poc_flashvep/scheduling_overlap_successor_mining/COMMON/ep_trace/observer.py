"""Clean vLLM 0.20 operation-cost observation for labelled transfer diagnostics.

No policy, routing, placement or model-output changes. Reuses validated source
seams, not earlier studies' experimental policy hooks. Request identities are
frozen per execute call. Same-device event durations only; sampling overhead is
not silently treated as clean request latency.
"""
import functools
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
from collections import deque

_INSTALLED = False
_TLS = threading.local()
_PENDING = deque()
_INVOCATION = 0
_RUNNER = None
_PROOF = False
_SINK = None


def emit(record):
    global _SINK
    if _SINK is None:
        root = Path(os.environ["SCHEDULING_EP_TRACE_OUT"])
        root.mkdir(parents=True, exist_ok=True)
        _SINK = (root / f"worker_pid{os.getpid()}.jsonl").open("a", buffering=1)
    _SINK.write(json.dumps(record) + "\n")


def drain():
    while _PENDING and _PENDING[0][2][-1].query():
        record, names, events = _PENDING.popleft()
        for name, a, b in names:
            record[name + "_ms"] = events[a].elapsed_time(events[b])
        emit(record)


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "4,5,6,7"
    import torch
    from vllm.distributed import get_dp_group, get_ep_group, get_tp_group
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer, Qwen3MoeAttention
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernelModularImpl

    def ranks():
        return {"ep_rank": get_ep_group().rank_in_group,
                "dp_rank": get_dp_group().rank_in_group,
                "tp_rank": get_tp_group().rank_in_group}

    def timed(function, stage):
        @functools.wraps(function)
        def call(*args, **kwargs):
            context = getattr(_TLS, "context", None)
            if not context or not context["selected"]:
                return function(*args, **kwargs)
            a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            record = {**context, **ranks(), "kind": "operation", "stage": stage,
                      "layer": getattr(_TLS, "layer", -1), "host_start": time.monotonic()}
            a.record()
            result = function(*args, **kwargs)
            b.record()
            record["host_end"] = time.monotonic()
            _PENDING.append((record, [(stage, 0, 1)], [a, b]))
            return result
        return call

    original_execute = GPUModelRunner.execute_model

    @functools.wraps(original_execute)
    def execute(self, scheduler_output, *args, **kwargs):
        global _RUNNER
        _RUNNER = self
        drain()
        step = getattr(_TLS, "step", -1) + 1
        _TLS.step = step
        scheduled = dict(scheduler_output.num_scheduled_tokens)
        counts = list(scheduled.values())
        previous = getattr(_TLS, "context", None)
        every = int(os.environ.get("SCHEDULING_EP_STEP_EVERY", "4"))
        _TLS.context = {"step": step, "scheduled": scheduled, "request_ids": list(scheduled),
                        "scheduled_tokens": sum(counts), "active_requests": len(counts),
                        "phase": "dummy" if not counts else "decode_or_single_prefill" if max(counts) == 1
                        else "prefill" if min(counts) > 1 else "mixed",
                        "selected": step % every == 0 and step < 2000}
        try:
            return original_execute(self, scheduler_output, *args, **kwargs)
        finally:
            _TLS.context = previous
            drain()
    GPUModelRunner.execute_model = execute

    original_forward = GPUModelRunner._model_forward

    def forward(self, *args, **kwargs):
        context = getattr(_TLS, "context", None)
        if context and context["selected"]:
            states = []
            for rid, n in context["scheduled"].items():
                index = self.input_batch.req_id_to_index.get(rid)
                if index is None:
                    continue
                states.append({"request_id": rid, "scheduled": n,
                               "computed": int(self.input_batch.num_computed_tokens_cpu[index]),
                               "prompt_length": int(self.input_batch.num_prompt_tokens[index])})
            flags = [s["computed"] < s["prompt_length"] for s in states]
            if flags:
                context["phase"] = "prefill" if all(flags) else "decode" if not any(flags) else "mixed"
            context["request_states"] = states
            emit({**context, **ranks(), "kind": "scheduler_context", "host_s": time.monotonic()})
        return timed(original_forward, "llm_forward")(self, *args, **kwargs)
    GPUModelRunner._model_forward = forward
    GPUModelRunner._execute_mm_encoder = timed(GPUModelRunner._execute_mm_encoder, "vision_encoder")

    original_init = Qwen3MoeDecoderLayer.__init__
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        prefix = str(kwargs.get("prefix", args[1] if len(args) > 1 else ""))
        match = re.search(r"layers\.(\d+)", prefix)
        self._scheduling_layer = int(match.group(1)) if match else -1
    Qwen3MoeDecoderLayer.__init__ = init
    original_layer = timed(Qwen3MoeDecoderLayer.forward, "layer")
    def layer(self, *args, **kwargs):
        previous = getattr(_TLS, "layer", -1)
        _TLS.layer = self._scheduling_layer
        try:
            return original_layer(self, *args, **kwargs)
        finally:
            _TLS.layer = previous
    Qwen3MoeDecoderLayer.forward = layer
    Qwen3MoeAttention.forward = timed(Qwen3MoeAttention.forward, "attention")
    BaseRouter.select_experts = timed(BaseRouter.select_experts, "router")

    original_apply = FusedMoEKernelModularImpl.apply
    def apply(self, *args, **kwargs):
        global _INVOCATION, _PROOF
        _INVOCATION += 1
        if _RUNNER is None:
            return original_apply(self, *args, **kwargs)
        if not _PROOF:
            import vllm, deep_ep
            pc = _RUNNER.vllm_config.parallel_config
            proof = {"kind": "runtime_proof", **ranks(), "tp": pc.tensor_parallel_size,
                     "dp": pc.data_parallel_size, "ep": get_ep_group().world_size,
                     "enable_ep": pc.enable_expert_parallel, "dbo": pc.enable_dbo,
                     "prepare_finalize": type(self.prepare_finalize).__name__,
                     "experts": type(self.fused_experts).__name__,
                     "backend": pc.all2all_backend, "vllm_version": vllm.__version__,
                     "vllm_source": vllm.__file__, "deep_ep_source": deep_ep.__file__,
                     "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]}
            emit(proof)
            assert (proof["tp"], proof["dp"], proof["ep"]) == (2, 2, 4)
            assert proof["prepare_finalize"] == "DeepEPHTPrepareAndFinalize"
            assert proof["enable_ep"] and not proof["dbo"]
            _PROOF = True
        # Count EVERY collective, including dummy participants. The ordinal is
        # retained to validate cross-rank joins rather than trusting DP step IDs.
        every = int(os.environ.get("SCHEDULING_EP_MOE_CYCLE_EVERY", "4"))
        if (_INVOCATION // 48) % every or _INVOCATION > 96000:
            return original_apply(self, *args, **kwargs)
        context = dict(getattr(_TLS, "context", None) or {"phase": "dummy", "request_ids": []})
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        record = {**context, **ranks(), "kind": "moe", "invocation": _INVOCATION,
                  "layer": getattr(_TLS, "layer", -1), "local_M": int(hidden.shape[0]),
                  "host_start": time.monotonic()}
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        _TLS.moe = (record, events)
        try:
            result = original_apply(self, *args, **kwargs)
            record["host_end"] = time.monotonic()
            _PENDING.append((record, [("dispatch", 0, 1), ("expert", 1, 2),
                                      ("combine", 2, 3), ("moe", 0, 3)], events))
            return result
        finally:
            _TLS.moe = None
    FusedMoEKernelModularImpl.apply = apply

    for method, first, last in [("_prepare", 0, 1), ("_fused_experts", 1, 2), ("_finalize", 2, 3)]:
        original = getattr(FusedMoEKernelModularImpl, method)
        def wrap(original=original, first=first, last=last):
            def call(self, *args, **kwargs):
                observation = getattr(_TLS, "moe", None)
                if observation and first == 0:
                    observation[1][0].record()
                if observation and first == 1:
                    meta = kwargs.get("expert_tokens_meta", args[12] if len(args) > 12 else None)
                    if meta is not None and meta.expert_num_tokens_cpu is not None:
                        observation[0]["local_expert_histogram"] = meta.expert_num_tokens_cpu.tolist()
                result = original(self, *args, **kwargs)
                if observation:
                    observation[1][last].record()
                return result
            return call
        setattr(FusedMoEKernelModularImpl, method, wrap())
