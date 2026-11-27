"""Local vLLM worker RPC for one bounded CUPTI kernel activity window."""
import ctypes
import json
import os
from pathlib import Path
import time

import torch


def clock(lib):
    before = time.monotonic_ns()
    tick = ctypes.c_uint64()
    rc = lib.pd_activity_timestamp(ctypes.byref(tick))
    after = time.monotonic_ns()
    if rc:
        raise RuntimeError(f"CUPTI clock failed with {rc}")
    return {"cupti_ns": tick.value, "mono_before_ns": before, "mono_after_ns": after}


class KernelActivityWorker:
    def study_cache_layout(self):
        """Read allocated cache layout after warmup with no tensor copy."""
        from vllm.distributed.kv_transfer import get_kv_transfer_group
        from vllm.v1.attention.backends.utils import get_kv_cache_layout

        runner = self.model_runner
        tensor = getattr(runner, "cross_layers_kv_cache", None)
        connector = get_kv_transfer_group()
        return {
            "layout": get_kv_cache_layout(),
            "runner": type(runner).__module__,
            "connector": type(connector).__name__,
            "connector_prefers_cross_layer": connector.prefer_cross_layer_blocks,
            "has_cross_layer_field": hasattr(runner, "cross_layers_kv_cache"),
            "allocated_cross_layer": tensor is not None,
            "shape": list(tensor.shape) if tensor is not None else None,
            "stride": list(tensor.stride()) if tensor is not None else None,
        }

    def study_shape_start(self):
        """Collect CPU scheduler rows at a fixed stride with no GPU sync."""
        if getattr(self, "_pd_shape_original", None) is not None:
            raise RuntimeError("Shape observer already enabled")
        self._pd_shape_original = self.execute_model
        samples = []
        self._pd_shapes = samples
        index = 0

        def observe(scheduled):
            nonlocal index
            index += 1
            if index % 4 == 0 and scheduled.total_num_scheduled_tokens:
                rows = scheduled.num_scheduled_tokens
                batch = scheduled.scheduled_cached_reqs
                new = scheduled.scheduled_new_reqs
                positions = dict(zip(batch.req_ids, batch.num_computed_tokens))
                positions.update((row.req_id, row.num_computed_tokens) for row in new)
                new_ids = {row.req_id for row in new}
                samples.append({
                    "host_s": time.monotonic(), "rows": list(rows.values()),
                    "kv_rows": [positions.get(req, 0) + q for req, q in rows.items()],
                    "context": [req in new_ids or batch.is_context_phase(req) for req in rows],
                })
            return self._pd_shape_original(scheduled)

        self.execute_model = observe
        return {"stride": 4}

    def study_shape_stop(self):
        self.execute_model = self._pd_shape_original
        self._pd_shape_original = None
        result, self._pd_shapes = self._pd_shapes, []
        return {"samples": result, "stride": 4}

    def study_activity_start(self):
        lib = ctypes.CDLL(os.environ["PD_ACTIVITY_LIBRARY"])
        lib.pd_activity_start.argtypes = [ctypes.c_size_t]
        lib.pd_activity_timestamp.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
        lib.pd_activity_stop.argtypes = [ctypes.c_char_p]
        for tag in ("exported", "dropped", "invalid"):
            getattr(lib, "pd_activity_" + tag).restype = ctypes.c_uint64
        torch.cuda.synchronize()
        rc = lib.pd_activity_start(256)
        if rc:
            raise RuntimeError(f"CUPTI start failed with {rc}")
        self._pd_kernel_activity = (lib, clock(lib))
        return self._pd_kernel_activity[1]

    def study_activity_stop(self):
        lib, start = self._pd_kernel_activity
        torch.cuda.synchronize()
        end = clock(lib)
        directory = Path(os.environ["PD_ACTIVITY_OUTPUT"])
        directory.mkdir(parents=True, exist_ok=True)
        root = directory / f"worker-{os.getpid()}"
        status = lib.pd_activity_stop(os.fsencode(root.with_suffix(".csv")))
        info = {"returncode": status, "clock_start": start, "clock_end": end,
                **{tag: getattr(lib, "pd_activity_" + tag)() for tag in ("exported", "dropped", "invalid")}}
        root.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
        return info
