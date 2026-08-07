"""Low-volume tracing and a narrow vLLM 0.20 DBO correctness workaround."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any


_LOCK = threading.Lock()
_STATE = threading.local()
_INSTALLED = False
_FIX_INSTALLED = False


def _jsonable_slice(value: slice) -> list[int | None]:
    return [
        int(item) if item is not None else None
        for item in (value.start, value.stop, value.step)
    ]


def _write(kind: str, payload: dict[str, Any]) -> None:
    directory = Path(os.environ["FLASHVEP_DBO_CORRECTNESS_TRACE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"trace_pid{os.getpid()}.jsonl"
    row = {
        "time_ns": time.time_ns(),
        "kind": kind,
        "pid": os.getpid(),
        "dp_rank": os.environ.get("VLLM_DP_RANK"),
        "local_rank": os.environ.get("LOCAL_RANK"),
        **payload,
    }
    with _LOCK:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def _maybe_write(kind: str, payload: dict[str, Any]) -> None:
    if os.environ.get("FLASHVEP_DBO_CORRECTNESS_TRACE_DIR"):
        _write(kind, payload)


def _install_metadata_cache_workaround() -> None:
    """Do not share FA3 mutable scheduler metadata between DBO builders.

    vLLM 0.20 caches attention metadata by KV spec and builder type. DBO uses
    one builder per ubatch, but the cache key does not include the ubatch ID.
    FlashAttention metadata includes a mutable scheduler/semaphore buffer, so
    reusing ubatch 0 metadata in ubatch 1 is unsafe under concurrent execution.
    """

    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadataBuilder

    original_init = FlashAttentionMetadataBuilder.__init__

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        vllm_config = kwargs.get("vllm_config")
        if vllm_config is None and len(args) >= 3:
            vllm_config = args[2]
        parallel = getattr(vllm_config, "parallel_config", None)
        if bool(getattr(parallel, "use_ubatching", False)):
            self.supports_update_block_table = False

    FlashAttentionMetadataBuilder.__init__ = patched_init


def _install_qwen3_vl_deepstack_workaround() -> None:
    """Slice the shared Qwen3-VL DeepStack buffer for each DBO ubatch.

    Qwen3-VL stores DeepStack vision embeddings outside the model arguments.
    The stock forward path therefore gives both concurrent ubatches the prefix
    of the same buffer and lets each one clear it. DBO currently creates two
    ubatches, so the second slice starts at ``total_tokens - num_tokens``.
    """

    from vllm.model_executor.models.qwen3_vl import (
        Qwen3VLForConditionalGeneration,
    )
    from vllm.sequence import IntermediateTensors
    from vllm.v1.worker.ubatching import dbo_current_ubatch_id, dbo_enabled

    original_get = Qwen3VLForConditionalGeneration._get_deepstack_input_embeds
    original_clear = Qwen3VLForConditionalGeneration._clear_deepstack_input_embeds

    def patched_get(self: Any, num_tokens: int) -> Any:
        if not dbo_enabled():
            return original_get(self, num_tokens)
        buffers = getattr(self, "deepstack_input_embeds", None)
        total_tokens = int(
            getattr(self, "deepstack_input_embeds_num_tokens", 0)
        )
        if not buffers or total_tokens == 0:
            return None
        ubatch_id = int(dbo_current_ubatch_id())
        start = 0 if ubatch_id == 0 else total_tokens - num_tokens
        stop = start + num_tokens
        if start < 0 or stop > total_tokens:
            raise ValueError(
                "Invalid DBO DeepStack slice: "
                f"{ubatch_id=}, {start=}, {stop=}, {total_tokens=}"
            )
        _maybe_write(
            "deepstack_slice",
            {
                "ubatch_id": ubatch_id,
                "token_slice": [start, stop],
                "total_tokens": total_tokens,
            },
        )
        return IntermediateTensors(
            {
                f"deepstack_input_embeds_{idx}": buffers[idx][start:stop]
                for idx in range(self.deepstack_num_level)
            }
        )

    def patched_clear(self: Any, num_tokens: int) -> None:
        if not dbo_enabled():
            original_clear(self, num_tokens)
            return
        completed = getattr(self, "_flashvep_dbo_deepstack_completed", None)
        if completed is None:
            completed = set()
            self._flashvep_dbo_deepstack_completed = completed
        completed.add(int(dbo_current_ubatch_id()))
        if len(completed) == 2:
            # Do not enqueue a zero-fill that can race the other CUDA stream.
            # The next vision prefill overwrites the buffer before reactivating
            # it; setting the valid length to zero prevents stale decode use.
            self.deepstack_input_embeds_num_tokens = 0
            completed.clear()

    Qwen3VLForConditionalGeneration._get_deepstack_input_embeds = patched_get
    Qwen3VLForConditionalGeneration._clear_deepstack_input_embeds = patched_clear


def install_dbo_correctness_fix() -> None:
    global _FIX_INSTALLED
    if _FIX_INSTALLED:
        return
    _FIX_INSTALLED = True
    _install_metadata_cache_workaround()
    _install_qwen3_vl_deepstack_workaround()


def install_dbo_correctness_probe() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    import vllm.v1.worker.gpu_model_runner as gpu_model_runner_module
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    from vllm.v1.worker.gpu_model_runner import (
        AsyncGPUModelRunnerOutput,
        GPUModelRunner,
    )
    from vllm.v1.worker.ubatching import UBatchContext
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.models.qwen3_vl import (
        Qwen3VLForConditionalGeneration,
    )

    original_execute_model = GPUModelRunner.execute_model
    original_make_slices = gpu_model_runner_module.maybe_create_ubatch_slices
    original_ubatch_enter = UBatchContext.__enter__
    original_attention_forward = FlashAttentionImpl.forward
    original_async_get_output = AsyncGPUModelRunnerOutput.get_output
    original_deepstack_get = (
        Qwen3VLForConditionalGeneration._get_deepstack_input_embeds
    )

    def execute_model(self: Any, scheduler_output: Any, *args: Any, **kwargs: Any):
        req_ids = list(scheduler_output.num_scheduled_tokens)
        _STATE.req_ids = req_ids
        _STATE.model_attention_logged = False
        _STATE.scheduled_tokens = [
            int(scheduler_output.num_scheduled_tokens[req_id]) for req_id in req_ids
        ]
        _write(
            "before_dbo_split",
            {
                "request_ids": req_ids,
                "scheduled_tokens": _STATE.scheduled_tokens,
                "total_scheduled_tokens": int(
                    scheduler_output.total_num_scheduled_tokens
                ),
                "scheduled_encoder_request_ids": list(
                    scheduler_output.scheduled_encoder_inputs
                ),
            },
        )
        result = original_execute_model(self, scheduler_output, *args, **kwargs)
        if hasattr(result, "req_ids"):
            _write("model_output_order", {"request_ids": list(result.req_ids)})
        return result

    def maybe_create_ubatch_slices(*args: Any, **kwargs: Any):
        result = original_make_slices(*args, **kwargs)
        unpadded, padded = result
        req_ids = list(getattr(_STATE, "req_ids", []))

        def describe(slices: Any) -> list[dict[str, Any]] | None:
            if slices is None:
                return None
            rows = []
            for ubatch_id, ubatch_slice in enumerate(slices):
                request_slice = ubatch_slice.request_slice
                rows.append(
                    {
                        "ubatch_id": ubatch_id,
                        "request_slice": _jsonable_slice(request_slice),
                        "token_slice": _jsonable_slice(ubatch_slice.token_slice),
                        "request_ids": req_ids[request_slice],
                    }
                )
            return rows

        _write(
            "after_dbo_split",
            {
                "request_ids_before_split": req_ids,
                "unpadded_ubatches": describe(unpadded),
                "padded_ubatches": describe(padded),
            },
        )
        return result

    def ubatch_enter(self: Any):
        result = original_ubatch_enter(self)
        _STATE.ubatch_id = int(self.id)
        _STATE.logged_attention = False
        return result

    def attention_forward(
        self: Any,
        layer: Any,
        query: Any,
        key: Any,
        value: Any,
        kv_cache: Any,
        attn_metadata: Any,
        output: Any,
        *args: Any,
        **kwargs: Any,
    ):
        in_ubatch = hasattr(_STATE, "ubatch_id")
        should_log = (
            not bool(getattr(_STATE, "logged_attention", False))
            if in_ubatch
            else not bool(getattr(_STATE, "model_attention_logged", False))
        )
        if should_log and attn_metadata is not None:
            if in_ubatch:
                _STATE.logged_attention = True
            else:
                _STATE.model_attention_logged = True
            query_start_loc = attn_metadata.query_start_loc
            seq_lens = attn_metadata.seq_lens
            _write(
                "attention_shape",
                {
                    "ubatch_id": (
                        int(_STATE.ubatch_id) if in_ubatch else None
                    ),
                    "query_shape": list(query.shape),
                    "key_shape": list(key.shape),
                    "value_shape": list(value.shape),
                    "num_actual_tokens": int(attn_metadata.num_actual_tokens),
                    "batch_size_q": int(query_start_loc.shape[0] - 1),
                    "batch_size_k": int(seq_lens.shape[0]),
                    "query_start_loc": [int(v) for v in query_start_loc.tolist()],
                    "sequence_lengths": [int(v) for v in seq_lens.tolist()],
                    "max_query_len": int(attn_metadata.max_query_len),
                    "max_sequence_len": int(attn_metadata.max_seq_len),
                    "scheduler_metadata_ptr": (
                        int(attn_metadata.scheduler_metadata.data_ptr())
                        if attn_metadata.scheduler_metadata is not None
                        else None
                    ),
                },
            )
        try:
            return original_attention_forward(
                self,
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                *args,
                **kwargs,
            )
        except BaseException as exc:
            _write(
                "attention_error",
                {
                    "ubatch_id": int(getattr(_STATE, "ubatch_id", -1)),
                    "error": repr(exc),
                },
            )
            raise

    def async_get_output(self: Any):
        output = original_async_get_output(self)
        _write(
            "model_output_order",
            {
                "request_ids": list(output.req_ids),
                "sampled_token_ids": output.sampled_token_ids,
            },
        )
        return output

    def deepstack_get(self: Any, num_tokens: int):
        result = original_deepstack_get(self, num_tokens)
        if result is not None:
            token_slice = get_forward_context().additional_kwargs.get(
                "ubatch_token_slice"
            )
            _write(
                "deepstack_source_slice",
                {
                    "ubatch_id": int(getattr(_STATE, "ubatch_id", -1)),
                    "num_tokens": int(num_tokens),
                    "token_slice": (
                        _jsonable_slice(token_slice)
                        if token_slice is not None
                        else None
                    ),
                    "valid_tokens": int(
                        getattr(self, "deepstack_input_embeds_num_tokens", 0)
                    ),
                },
            )
        return result

    GPUModelRunner.execute_model = execute_model
    gpu_model_runner_module.maybe_create_ubatch_slices = maybe_create_ubatch_slices
    UBatchContext.__enter__ = ubatch_enter
    FlashAttentionImpl.forward = attention_forward
    AsyncGPUModelRunnerOutput.get_output = async_get_output
    Qwen3VLForConditionalGeneration._get_deepstack_input_embeds = deepstack_get
