"""Bounded native LP extension for vLLM 0.20/Qwen3-VL/DeepEP.

The real scheduler keeps a request pending across stages. GPU hidden/residual
state survives between execute_model calls. Decode runs all 48 layers and is
sampled normally on each call. Only attention metadata and DP row-size metadata
are narrowed outside the active prefill range; expert/router semantics stay
with the installed implementation.
"""
from __future__ import annotations

import atexit
import copy
import functools
import json
import os
import threading
import time
from pathlib import Path

import torch
import torch.distributed as dist

from .contract import partition

TLS = threading.local()
EVENTS = []
CUDA_EVENTS = []
INSTALLED = False


def flush():
    if CUDA_EVENTS:
        torch.cuda.synchronize()
        for record, begin, end in CUDA_EVENTS:
            record["cuda_ms"] = float(begin.elapsed_time(end))
            EVENTS.append(record)
        CUDA_EVENTS.clear()
    if not EVENTS:
        return
    root = Path(os.environ["NATIVE_LP_OUTPUT"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"native.pid{os.getpid()}.jsonl").open("a") as f:
        for row in EVENTS:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    EVENTS.clear()


def extra(request):
    params = getattr(request, "sampling_params", None)
    return (getattr(params, "extra_args", None) or {})


def decode_metadata(metadata, request_indices, row_indices):
    """Select whole decode requests, retaining their exact KV block tables."""
    if metadata.__class__.__name__ != "FlashAttentionMetadata":
        raise RuntimeError(f"unsupported attention contract: {type(metadata)}")
    assert not metadata.use_cascade
    result = copy.copy(metadata)
    result.num_actual_tokens = len(request_indices)
    result.max_query_len = 1
    result.query_start_loc = torch.arange(
        len(request_indices) + 1, device=row_indices.device, dtype=torch.int32
    )
    result.seq_lens = metadata.seq_lens[request_indices]
    result.block_table = metadata.block_table[request_indices]
    result.slot_mapping = metadata.slot_mapping[row_indices]
    result.scheduler_metadata = None
    result.prefix_scheduler_metadata = None
    return result


def install():
    global INSTALLED
    if INSTALLED:
        return
    INSTALLED = True
    trace = os.environ.get("NATIVE_LP_TRACE") == "1"
    correctness = os.environ.get("NATIVE_LP_CORRECTNESS") == "1"
    light_trace = os.environ.get('NATIVE_LP_LIGHT_TRACE') == '1'
    if trace:
        if light_trace:
            from .light_trace import install as install_trace
        else:
            from poc_mllm_ep_collective_granularity.instrumentation import install as install_trace
        install_trace()
    from poc_mllm_ep_communication_motivation import instrumentation as base
    from vllm.distributed import get_dp_group, get_ep_group, get_tp_group
    from vllm.forward_context import DPMetadata, get_forward_context
    from vllm.model_executor.models.qwen3_vl_moe import Qwen3MoeLLMModel
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker import dp_utils

    atexit.register(flush)
    original_update = Scheduler.update_from_output
    original_add = Scheduler.add_request
    original_schedule = Scheduler.schedule
    original_free = Scheduler._free_encoder_inputs
    original_runner = GPUModelRunner._model_forward
    original_execute = GPUModelRunner.execute_model
    original_model = Qwen3MoeLLMModel.forward

    def batch_states(worker):
        scheduled = getattr(TLS, 'scheduled', None)
        if worker is None or not scheduled:
            return []
        result=[]
        for index, req_id in enumerate(worker.input_batch.req_ids):
            if req_id not in scheduled:
                continue
            req=worker.requests[req_id]
            result.append({'id':req_id,'index':index,'count':int(scheduled[req_id]),
                'computed':int(worker.input_batch.num_computed_tokens_cpu[index]),
                'prompt':int(worker.input_batch.num_prompt_tokens[index]),
                'groups':int(extra(req).get('native_lp_groups',1)),
                'injection':'native_lp_groups' in extra(req),
                'capture':bool(extra(req).get('native_lp_capture',False))})
        return result

    def coordinated_header(should_ubatch, orig_num_tokens_per_ubatch,
                           padded_num_tokens_per_ubatch, cudagraph_mode, parallel_config):
        # Piggyback on the runtime's mandatory DP padding synchronization.
        # A second conditional collective inside model.forward can race the
        # engine's empty/dummy path at a wave transition. Real and dummy calls
        # now execute precisely the stock number/order of DP collectives.
        states=batch_states(getattr(TLS,'current_runner',None))
        candidates=[s for s in states if s['groups']>1 and s['computed']<s['prompt']]
        assert len(candidates)<=1
        injected=candidates[0] if candidates else None
        marked=next((s for s in states if s['injection'] and s['computed']<s['prompt']),None)
        decode_min=min((s['computed']-s['prompt']+1 for s in states
                        if s['computed']>=s['prompt']),default=-1)
        registry=getattr(TLS,'stage_registry',{})
        values=[orig_num_tokens_per_ubatch,padded_num_tokens_per_ubatch,
            int(should_ubatch),cudagraph_mode,
            injected['groups'] if injected else 0,
            registry.get(injected['id'],0) if injected else 0,
            marked['count'] if marked else 0,
            sum(s['count'] for s in states if marked is None or s['id']!=marked['id']),
            marked['computed'] if marked else -1,decode_min]
        device,group=dp_utils._get_device_and_group(parallel_config)
        tensor=torch.zeros(10,parallel_config.data_parallel_size,device=device,dtype=torch.int32)
        tensor[:,parallel_config.data_parallel_rank]=torch.tensor(values,device=device,dtype=torch.int32)
        dist.all_reduce(tensor,group=group)
        TLS.coordinated_headers=tensor[4:].cpu().T.contiguous()
        return tensor[:4]

    dp_utils._run_ar=coordinated_header

    def timeline(kind, request_id=None, **fields):
        control=base._control()
        root=Path(os.environ['NATIVE_LP_OUTPUT'])
        row={'diagnostic':kind,'request':request_id,'wave':control.get('wave'),
             'host_monotonic_ns':time.monotonic_ns(),'clock':'HOST_LIFECYCLE_NOT_GPU_CWT',**fields}
        with (root/f'lifecycle.pid{os.getpid()}.jsonl').open('a') as f:
            f.write(json.dumps(row,separators=(',',':'))+'\n')

    def add_request(self, request, *args, **kwargs):
        result=original_add(self,request,*args,**kwargs)
        if trace and 'native_lp_groups' in extra(request):
            timeline('scheduler_request_received',request.request_id)
        return result

    def schedule(self):
        # Between frozen waves, use the installed CP scheduler with its
        # ordinary token budget. Worker capacity stays at the full budget.
        control = base._control()
        arm = control.get("arm", "FULL")
        chunk = int(arm[2:]) if arm.startswith("CP") else 0
        self.max_num_scheduled_tokens = chunk + 4 if chunk else 20000
        self.scheduler_config.enable_chunked_prefill = bool(chunk)
        result=original_schedule(self)
        if trace:
            for req_id,count in result.num_scheduled_tokens.items():
                req=self.requests.get(req_id)
                if req is not None and 'native_lp_groups' in extra(req) and not getattr(req,'_native_admission_logged',False):
                    timeline('scheduler_first_admission',req_id,rows=count)
                    req._native_admission_logged=True
        return result

    def free_encoder(self, request):
        groups = int(extra(request).get("native_lp_groups", 1))
        if groups > 1 and getattr(request, "_native_lp_stage", 0) < groups - 1:
            return
        return original_free(self, request)

    @functools.wraps(original_update)
    def update(self, scheduler_output, model_runner_output):
        pending = []
        advanced = []
        markers=[]
        for req_id in scheduler_output.num_scheduled_tokens:
            req = self.requests.get(req_id)
            if req is None:
                continue
            groups = int(extra(req).get("native_lp_groups", 1))
            if 'native_lp_groups' in extra(req) and not req.num_output_tokens:
                markers.append({'request':req_id,'stage':getattr(req,'_native_lp_stage',0),
                                'groups':groups,'scheduled_rows':scheduler_output.num_scheduled_tokens[req_id]})
            if groups <= 1 or req.num_output_tokens:
                continue
            stage = getattr(req, "_native_lp_stage", 0)
            if stage < groups - 1:
                index = model_runner_output.req_id_to_index[req_id]
                model_runner_output.sampled_token_ids[index] = []
                # schedule() has optimistically advanced this count. Roll it
                # back until the complete 48-layer prefill exists.
                req.num_computed_tokens = 0
                pending.append(req)
            advanced.append((req, stage + 1))
        result = original_update(self, scheduler_output, model_runner_output)
        for req, new_stage in advanced:
            req._native_lp_stage = new_stage
        for req in pending:
            req.num_computed_tokens = 0
        if markers:
            control=base._control()
            root=Path(os.environ['NATIVE_LP_OUTPUT'])
            row={'kind':'prefill_stage_completion','completion_ns':time.monotonic_ns(),
                 'wave':control.get('wave'),'arm':control.get('arm'),'markers':markers,
                 'decode_tokens_completed':sum(len(tokens) for tokens in model_runner_output.sampled_token_ids)}
            # One CPU record per active stage, AFTER GPU/model completion;
            # no router trace, tensor transfer, CUDA event or GPU sync.
            with (root/f'completion.pid{os.getpid()}.jsonl').open('a') as f:
                f.write(json.dumps(row,separators=(',',':'))+'\n')
        return result

    @functools.wraps(original_runner)
    def runner(self, *args, **kwargs):
        if not self.input_batch.req_ids or not getattr(TLS, "scheduled", None):
            return original_runner(self, *args, **kwargs)
        states = []
        for index, req_id in enumerate(self.input_batch.req_ids):
            req = self.requests[req_id]
            count = int(TLS.scheduled[req_id])
            states.append({
                "id": req_id, "index": index, "count": count,
                "computed": int(self.input_batch.num_computed_tokens_cpu[index]),
                "prompt": int(self.input_batch.num_prompt_tokens[index]),
                "groups": int(extra(req).get("native_lp_groups", 1)),
                "injection": "native_lp_groups" in extra(req),
                "capture": bool(extra(req).get("native_lp_capture", False)),
            })
        TLS.runner = self
        TLS.states = states
        TLS.last_states = states
        TLS.capture_logits = True
        if trace and getattr(TLS,'input_pipeline_begin',None) is not None:
            end=torch.cuda.Event(enable_timing=True);end.record()
            control=base._control()
            CUDA_EVENTS.append(({'diagnostic':'pre_lm_input_pipeline','wave':control.get('wave'),
                'iteration':control.get('outer_iteration'),'ep_rank':get_ep_group().rank_in_group,
                'is_injection':any(s['injection'] and s['computed']<s['prompt'] for s in states),
                'fresh_encoder':bool(getattr(TLS,'vision_key',())),
                'native_stage':max((getattr(TLS,'stage_registry',{}).get(s['id'],0) for s in states if s['injection']),default=0),
                'host_ms':(time.monotonic_ns()-TLS.input_pipeline_host)/1e6},TLS.input_pipeline_begin,end))
            TLS.input_pipeline_begin=None
        if correctness and any(s["capture"] for s in states):
            EVENTS.append({"diagnostic": "runner_states", "states": states,
                "dp_rank": get_dp_group().rank_in_group})
            if get_tp_group().rank_in_group==0:
                import hashlib
                import numpy as np
                control=base._control()
                key=control.get('wave')
                seen=getattr(self,'_native_input_identity_seen',set())
                if key not in seen:
                    cursor=0
                    for s in states:
                        if s['capture'] and s['computed']<s['prompt']:
                            def digest(t):
                                return None if t is None else hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                            request=self.requests[s['id']]
                            ids=np.asarray(request.prompt_token_ids,dtype=np.int64)
                            pos=kwargs.get('positions',args[1] if len(args)>1 else None)
                            emb=kwargs.get('inputs_embeds',args[3] if len(args)>3 else None)
                            EVENTS.append({'diagnostic':'prefill_input_identity','wave':key,
                                'ep_rank':get_ep_group().rank_in_group,'token_count':len(ids),
                                'ids_sha256':hashlib.sha256(ids.tobytes()).hexdigest(),
                                'positions_sha256':digest(None if pos is None else pos[...,cursor:cursor+s['count']]),
                                'embeddings_sha256':digest(None if emb is None else emb[cursor:cursor+s['count']])})
                        cursor+=s['count']
                    seen.add(key);self._native_input_identity_seen=seen
        try:
            return original_runner(self, *args, **kwargs)
        finally:
            TLS.states = None

    @functools.wraps(original_execute)
    def execute(self, scheduler_output, *args, **kwargs):
        TLS.scheduled = scheduler_output.num_scheduled_tokens
        TLS.current_runner=self
        new = {r.req_id:r for r in scheduler_output.scheduled_new_reqs}
        keys = []
        TLS.vision_is_injection=False
        for req_id, indices in scheduler_output.scheduled_encoder_inputs.items():
            req = new.get(req_id, self.requests.get(req_id))
            if req is not None:
                if os.environ.get('NATIVE_LP_VISION_CACHE')=='1':
                    import hashlib
                    for i in indices:
                        digest=hashlib.sha256()
                        for name,field in sorted(req.mm_features[i].data.items()):
                            value=field.data
                            if not isinstance(value,torch.Tensor) or value.device.type!='cpu':
                                raise RuntimeError('diagnostic vision cache needs exact CPU processor tensors')
                            value=value.detach().contiguous()
                            digest.update((name+str(value.dtype)+str(tuple(value.shape))).encode())
                            digest.update(value.view(torch.uint8).numpy().tobytes())
                        keys.append(digest.hexdigest())
                else:
                    keys.extend(req.mm_features[i].identifier for i in indices)
                TLS.vision_is_injection |= 'native_lp_groups' in extra(req)
        TLS.vision_key = tuple(keys)
        TLS.vision_call = 0
        if trace and any('native_lp_groups' in extra(req) for req in new.values()):
            timeline('worker_first_execution',ep_rank=get_ep_group().rank_in_group)
        TLS.input_pipeline_begin=None
        if trace and base._control().get('capture_stage',False):
            TLS.input_pipeline_host=time.monotonic_ns()
            TLS.input_pipeline_begin=torch.cuda.Event(enable_timing=True)
            TLS.input_pipeline_begin.record()
        try:
            return original_execute(self, scheduler_output, *args, **kwargs)
        finally:
            TLS.scheduled = None
            TLS.current_runner=None
            # External-DP executes dummy batches after the real request set
            # drains. Never classify those rows with a previous batch's IDs.
            TLS.last_states = []
            TLS.lp_stage = None
            TLS.active_prefill = None
            if trace:
                flush()

    @functools.wraps(original_model)
    def model(self, input_ids, positions, intermediate_tensors=None,
              inputs_embeds=None, deepstack_input_embeds=None):
        states = getattr(TLS, "states", None)
        runtime_dummy = not states and hasattr(self,'_native_lp_cache')
        if not states and not runtime_dummy:
            return original_model(self, input_ids, positions=positions,
                intermediate_tensors=intermediate_tensors, inputs_embeds=inputs_embeds,
                deepstack_input_embeds=deepstack_input_embeds)
        fc = get_forward_context()
        if fc.attn_metadata is None and not runtime_dummy:
            return original_model(self, input_ids, positions=positions,
                intermediate_tensors=intermediate_tensors, inputs_embeds=inputs_embeds,
                deepstack_input_embeds=deepstack_input_embeds)
        if not hasattr(self, "_native_lp_cache"):
            self._native_lp_cache = {}
        states=states or []
        candidates = [s for s in states if s["groups"] > 1 and s["computed"] < s["prompt"]]
        if len(candidates) > 1:
            raise RuntimeError("bounded harness supports one injected LP prefill")
        injected = candidates[0] if candidates else None
        stage = self._native_lp_cache.get(injected["id"], (0, None, None))[0] if injected else 0
        marked=next((s for s in states if s['injection'] and s['computed']<s['prompt']),None)
        decode_min=min((s['computed']-s['prompt']+1 for s in states if s['computed']>=s['prompt']),default=-1)
        header = torch.tensor([
            injected["groups"] if injected else 0,
            stage if injected else 0,
            marked["count"] if marked else 0,
            sum(s["count"] for s in states if marked is None or s["id"] != marked["id"]),
            marked['computed'] if marked else -1,decode_min,
        ], dtype=torch.int64)
        headers = [torch.empty_like(header) for _ in range(2)]
        coordinated=getattr(TLS,'coordinated_headers',None)
        if coordinated is None:
            raise RuntimeError('missing mandatory DP coordination header')
        headers=list(coordinated)
        assert torch.equal(headers[get_dp_group().rank_in_group],header.to(torch.int32)), (headers,header,states)
        TLS.global_prefill_rows=sum(int(h[2]) for h in headers)
        TLS.prefill_offset=max(int(h[4]) for h in headers)
        TLS.global_decode_step=min((int(h[5]) for h in headers if int(h[5])>=0),default=-1)
        groups = max(int(h[0]) for h in headers)
        if marked and marked['computed']==0 and stage==0 and get_tp_group().rank_in_group==0:
            control=base._control()
            # One CPU timestamp per injection, after multimodal preparation
            # and mandatory DP coordination. No tensor transfer or GPU sync.
            record={'kind':'injected_lm_entry','wave':control.get('wave'),
                    'entry_ns':time.monotonic_ns(),'dp_rank':get_dp_group().rank_in_group,
                    'rows':marked['count'],'clock':'HOST_MONOTONIC_LM_ENTRY_NOT_CWT'}
            root=Path(os.environ['NATIVE_LP_OUTPUT'])
            with (root/f'model_entry.pid{os.getpid()}.jsonl').open('a') as f:
                f.write(json.dumps(record,separators=(',',':'))+'\n')
        if correctness and any(s["capture"] for s in states):
            EVENTS.append({"diagnostic": "stage_headers", "headers": [h.tolist() for h in headers],
                "states": states, "dp_rank": get_dp_group().rank_in_group})
        if groups == 0:
            TLS.active_prefill = None
            TLS.lp_stage = None
            result = original_model(self, input_ids, positions=positions,
                intermediate_tensors=intermediate_tensors, inputs_embeds=inputs_embeds,
                deepstack_input_embeds=deepstack_input_embeds)
            capture_full(self, states, result, correctness)
            if correctness:
                flush()
            return result
        if runtime_dummy:
            raise RuntimeError('LP active while a DP has no resident decode: outside bounded B8 contract')
        active_headers = [h for h in headers if int(h[0])]
        assert len(active_headers) == 1, "one injected request globally"
        stage = int(active_headers[0][1])
        TLS.capture_logits = stage == groups - 1
        lo, hi = partition(groups)[stage]
        hidden = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        hidden = hidden.clone()
        residual = None
        cursor = 0
        injected_rows = []
        decode_rows = []
        decode_requests = []
        for s in states:
            rows = list(range(cursor, cursor + s["count"]))
            if injected and s["id"] == injected["id"]:
                injected_rows = rows
            else:
                assert s["computed"] >= s["prompt"] and s["count"] == 1, states
                decode_rows.extend(rows)
                decode_requests.append(s["index"])
            cursor += s["count"]
        idx = torch.tensor(injected_rows, device=hidden.device, dtype=torch.long)
        didx = torch.tensor(decode_rows, device=hidden.device, dtype=torch.long)
        if injected and stage:
            cached_stage, saved_h, saved_r = self._native_lp_cache[injected["id"]]
            assert cached_stage == stage
            hidden[idx] = saved_h
            residual = torch.zeros_like(hidden)
            residual[idx] = saved_r
        full_attn, full_slot, full_dp = fc.attn_metadata, fc.slot_mapping, fc.dp_metadata
        d_attn = {k: decode_metadata(v, decode_requests, didx) for k, v in full_attn.items()}
        d_slots = {k: v[didx] for k, v in full_slot.items()}
        counts = torch.tensor([int(h[3]) for h in headers], dtype=full_dp.num_tokens_across_dp_cpu.dtype)
        d_dp = DPMetadata(counts)
        route_masks=trace and not light_trace
        visual=injected_mask=None
        if route_masks:
            pending_visual = getattr(base._TLS, "pending_visual", None)
            visual = (torch.zeros(hidden.shape[0], dtype=torch.bool, device=hidden.device)
                      if pending_visual is None else pending_visual)
            visual = visual.to(device=hidden.device, non_blocking=True)
            if visual.shape[0] != hidden.shape[0]:
                visual = torch.nn.functional.pad(visual, (0, hidden.shape[0]-visual.shape[0]))
            injected_mask = torch.zeros_like(visual)
            injected_mask[idx] = True
        prior_masks = [getattr(base._TLS, key, None) for key in ("visual", "text", "injected")]
        ppos = positions[..., didx]
        layer_execution = []
        try:
            for layer_idx, layer in enumerate(self.layers):
                active = lo <= layer_idx < hi
                TLS.active_prefill = active
                TLS.lp_stage = stage
                fc.attn_metadata = full_attn if active else d_attn
                fc.slot_mapping = full_slot if active else d_slots
                fc.dp_metadata = full_dp if active else d_dp
                fc.moe_layer_index = layer_idx
                if route_masks:
                    base._TLS.visual = visual if active else visual[didx]
                    base._TLS.text = ~base._TLS.visual
                    base._TLS.injected = injected_mask if active else injected_mask[didx]
                if active:
                    hidden, residual = layer(positions, hidden, residual)
                    if deepstack_input_embeds is not None and layer_idx < len(deepstack_input_embeds):
                        hidden = hidden + deepstack_input_embeds[f"deepstack_input_embeds_{layer_idx}"]
                else:
                    dh = hidden[didx]
                    dr = None if layer_idx == 0 else residual[didx]
                    dh, dr = layer(ppos, dh, dr)
                    if residual is None:
                        residual = torch.zeros_like(hidden)
                    hidden[didx], residual[didx] = dh, dr
                if trace or correctness:
                    layer_execution.append({"layer": layer_idx, "prefill": active,
                                            "decode_rows": len(decode_rows)})
            if injected:
                self._native_lp_cache[injected["id"]] = (
                    stage + 1, hidden[idx].clone(), residual[idx].clone()
                )
                registry=getattr(TLS,'stage_registry',{})
                registry[injected['id']]=stage+1
                TLS.stage_registry=registry
            result, _ = self.norm(hidden, residual)
            capture_full(self, states, result, correctness and stage == groups-1)
            if injected and stage == groups - 1:
                del self._native_lp_cache[injected["id"]]
                TLS.stage_registry.pop(injected['id'],None)
            if trace or correctness:
                from poc_mllm_ep_communication_motivation.instrumentation import _control
                control = _control()
                EVENTS.append({"wave": control.get("wave"), "arm": control.get("arm"),
                    "iteration": control.get("outer_iteration"), "stage": stage,
                    "groups": groups, "range": [lo, hi], "ep_rank": get_ep_group().rank_in_group,
                    "request": None if injected is None else injected["id"],
                    "prefill_rows": len(injected_rows), "decode_rows": len(decode_rows),
                    "layers": layer_execution, "counts_full": full_dp.num_tokens_across_dp_cpu.tolist(),
                    "counts_decode": counts.tolist(), "scope": "CONTROLLED_NATIVE_LP_EP4_HARNESS"})
                flush()
            return result
        finally:
            fc.attn_metadata, fc.slot_mapping, fc.dp_metadata = full_attn, full_slot, full_dp
            for key, value in zip(("visual", "text", "injected"), prior_masks):
                setattr(base._TLS, key, value)

    def capture_full(self, states, result, enabled):
        if not enabled or get_tp_group().rank_in_group != 0:
            return
        cursor = 0
        for s in states:
            if s["capture"] and s["computed"] < s["prompt"]:
                root = Path(os.environ["NATIVE_LP_OUTPUT"])
                root.mkdir(parents=True, exist_ok=True)
                # Final normalized language hidden before the LM head, all
                # prompt rows; bounded correctness-only transfer.
                value = result[cursor:cursor+s["count"]].detach().cpu()
                from poc_mllm_ep_communication_motivation.instrumentation import _control
                control = _control()
                torch.save(value, root / f"hidden.w{control.get('wave')}.dp{get_dp_group().rank_in_group}.pt")
            cursor += s["count"]

    Scheduler._free_encoder_inputs = free_encoder
    Scheduler.add_request = add_request
    Scheduler.schedule = schedule
    Scheduler.update_from_output = update
    GPUModelRunner._model_forward = runner
    GPUModelRunner.execute_model = execute
    Qwen3MoeLLMModel.forward = model

    from vllm.model_executor.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration
    original_logits = Qwen3VLMoeForConditionalGeneration.compute_logits

    def logits(self, hidden_states, *args, **kwargs):
        result = original_logits(self, hidden_states, *args, **kwargs)
        if correctness and getattr(TLS, "capture_logits", False) and get_tp_group().rank_in_group == 0:
            for s in getattr(TLS, "last_states", []):
                if s["capture"] and s["computed"] < s["prompt"] and result is not None:
                    control = base._control()
                    root = Path(os.environ["NATIVE_LP_OUTPUT"])
                    torch.save(result[s["index"]].detach().cpu(), root / f"logits.w{control.get('wave')}.dp{get_dp_group().rank_in_group}.pt")
        return result

    Qwen3VLMoeForConditionalGeneration.compute_logits = logits

    if os.environ.get("NATIVE_LP_VISION_CACHE") == "1":
        from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer
        uncached_vision = Qwen3_VisionTransformer.forward

        def cached_vision(self, *args, **kwargs):
            key = (getattr(TLS,"vision_key",()),getattr(TLS,"vision_call",0))
            TLS.vision_call = key[1] + 1
            if not key[0]:
                return uncached_vision(self,*args,**kwargs)
            if not hasattr(self,"_native_vision_cache"):
                self._native_vision_cache = {}
            if key in self._native_vision_cache:
                if trace or correctness:
                    control=base._control()
                    EVENTS.append({'diagnostic':'vision_cache_hit','wave':control.get('wave'),
                                   'ep_rank':get_ep_group().rank_in_group})
                return self._native_vision_cache[key].clone()
            result = uncached_vision(self,*args,**kwargs)
            assert isinstance(result,torch.Tensor)
            self._native_vision_cache[key] = result.detach().clone()
            if len(self._native_vision_cache)>16:
                self._native_vision_cache.pop(next(iter(self._native_vision_cache)))
            return result

        Qwen3_VisionTransformer.forward = cached_vision

    if trace:
        from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
        original_select = BaseRouter.select_experts

        def select(self, hidden_states, router_logits, **kwargs):
            if getattr(TLS,"scheduled",None) and getattr(TLS,"lp_stage",None) is None:
                states=getattr(TLS,"last_states",[]) or []
                raw=torch.zeros(sum(s["count"] for s in states),dtype=torch.bool,device=hidden_states.device)
                cursor=0
                for s in states:
                    if s["injection"] and s["computed"]<s["prompt"]:
                        raw[cursor:cursor+s["count"]]=True
                    cursor+=s["count"]
                if raw.numel():
                    base._TLS.router_injected=base._fit_mask(raw,hidden_states.shape[0],hidden_states.device)
            weights, ids = original_select(self, hidden_states, router_logits, **kwargs)
            record = getattr(base._TLS, "route_record", None)
            if record is not None:
                record["native_lp_active_prefill"] = getattr(TLS, "active_prefill", None)
                record["native_lp_stage"] = getattr(TLS, "lp_stage", None)
            if (correctness or base._control().get('route_stats',False)) and getattr(TLS,"scheduled",None):
                mask = getattr(base._TLS, "router_injected", None)
                if mask is not None and bool(mask.any()):
                    mask = base._fit_mask(mask, ids.shape[0], ids.device)
                    selected_ids = ids[mask].detach().cpu().contiguous()
                    selected_weights = weights[mask].detach().cpu().contiguous()
                    import hashlib
                    control = base._control()
                    EVENTS.append({"diagnostic":"route_identity", "wave":control.get("wave"),
                        "layer":getattr(base._TLS,"layer",-1),"ep_rank":get_ep_group().rank_in_group,
                        "rows":selected_ids.shape[0],
                        "ids_sha256":hashlib.sha256(selected_ids.numpy().tobytes()).hexdigest(),
                        "weights_sha256":hashlib.sha256(selected_weights.view(torch.uint8).numpy().tobytes()).hexdigest()})
            return weights, ids

        if not light_trace:
            BaseRouter.select_experts = select
        from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernelModularImpl
        original_experts = FusedMoEKernelModularImpl._fused_experts

        def experts(self, *args, **kwargs):
            result = original_experts(self, *args, **kwargs)
            stage = getattr(base._TLS, "stage", None)
            if stage is not None:
                stage[0]["native_lp_active_prefill"] = getattr(TLS,"active_prefill",None)
                stage[0]["native_lp_stage"] = getattr(TLS,"lp_stage",None)
                states=getattr(TLS,'states',None) or []
                stage[0]['injected_rows']=sum(s['count'] for s in states if s['injection'] and s['computed']<s['prompt'])
                stage[0]['global_prefill_rows']=getattr(TLS,'global_prefill_rows',0)
                stage[0]['prefill_offset']=getattr(TLS,'prefill_offset',-1)
                stage[0]['global_decode_step']=getattr(TLS,'global_decode_step',-1)
                w1 = kwargs.get("w1", args[3] if len(args)>3 else None)
                w2 = kwargs.get("w2", args[4] if len(args)>4 else None)
                if w1 is not None and w2 is not None:
                    assert tuple(w1.shape)==(32,1536,2048),tuple(w1.shape)
                    assert tuple(w2.shape)==(32,2048,768),tuple(w2.shape)
                    stage[0]["loaded_expert_weight_shapes"]=[list(w1.shape),list(w2.shape)]
                    stage[0]["expert_weight_bytes_per_expert"]=(w1.numel()*w1.element_size()+w2.numel()*w2.element_size())//32
                    stage[0]["group_membership"]={"tp":get_tp_group().ranks,
                        "dp":get_dp_group().ranks,"ep":get_ep_group().ranks}
                    emap=kwargs.get('expert_map',args[10] if len(args)>10 else None)
                    if emap is not None and not light_trace:
                        if not hasattr(self,'_native_verified_expert_map'):
                            actual=emap.detach().cpu().tolist()
                            rank=get_ep_group().rank_in_group
                            expected=[e-rank*32 if rank*32<=e<(rank+1)*32 else -1 for e in range(128)]
                            assert actual==expected, (rank,actual)
                            self._native_verified_expert_map=actual
                        stage[0]['actual_expert_map']=self._native_verified_expert_map
            return result

        FusedMoEKernelModularImpl._fused_experts = experts
        from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer
        original_vision = Qwen3_VisionTransformer.forward

        def vision(self, *args, **kwargs):
            control = base._control()
            if not control.get("capture_stage", False):
                return original_vision(self, *args, **kwargs)
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            result = original_vision(self, *args, **kwargs)
            end.record()
            CUDA_EVENTS.append(({"diagnostic":"vision_encoder", "wave":control.get("wave"),
                "iteration":control.get("outer_iteration"),"ep_rank":get_ep_group().rank_in_group,
                'is_injection':getattr(TLS,'vision_is_injection',False)},begin,end))
            return result

        Qwen3_VisionTransformer.forward = vision
