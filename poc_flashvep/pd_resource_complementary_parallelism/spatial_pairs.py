"""Time real Qwen operators on explicitly scoped CUDA Green Context streams."""
import argparse
from contextlib import contextmanager, ExitStack
from hashlib import sha256
import json
import os
from pathlib import Path
from statistics import median
import time

import torch
from cuda.bindings import driver
from flashinfer.green_ctx import get_cudevice, get_device_resource, split_resource

from .operators import Layer, errors, load_layer, prepare_decode

PARTITIONS = ((112, 16), (96, 32), (64, 64), (32, 96))
SM_RESOURCE = driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM


def call(operation, *args):
    status, *output = operation(*args)
    if status != driver.CUresult.CUDA_SUCCESS:
        name_status, name = driver.cuGetErrorName(status)
        if name_status != driver.CUresult.CUDA_SUCCESS:
            name = str(status)
        raise RuntimeError(f"{operation.__name__}: {name!r}")
    if len(output) == 1:
        return output[0]
    return tuple(output) if output else None


@contextmanager
def current(context):
    call(driver.cuCtxPushCurrent, context)
    try:
        yield
    finally:
        popped = call(driver.cuCtxPopCurrent)
        if int(popped) != int(context):
            raise RuntimeError("CUDA runtime switched the current context during a scoped call")


def create_green(device, resource, stack):
    desc = call(driver.cuDevResourceGenerateDesc, [resource], 1)
    green = call(driver.cuGreenCtxCreate, desc, device, driver.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM)
    stack.callback(call, driver.cuGreenCtxDestroy, green)
    return green


def partition(device, counts, stack):
    cuda_device = get_cudevice(device)
    resource = get_device_resource(cuda_device)
    result, remaining = split_resource(resource, 1, counts[0])
    # Re-query the remainder through a context, following FlashInfer's split
    # path. Keep that temporary resource owner alive for this pair's lifetime.
    temporary = create_green(cuda_device, remaining, stack)
    available = call(driver.cuGreenCtxGetDevResource, temporary, SM_RESOURCE)
    second, _ = split_resource(available, 1, counts[1])
    scopes = [create_green(cuda_device, item, stack) for item in (*result, *second)]
    actual = [int(call(driver.cuGreenCtxGetDevResource, scope, SM_RESOURCE).sm.smCount) for scope in scopes]
    if actual != list(counts):
        raise RuntimeError(f"Requested {counts} SMs, driver provisioned {actual}")
    return scopes


def make_stream(context, green, stack):
    flags = driver.CUstream_flags.CU_STREAM_NON_BLOCKING
    with current(context):
        if green is None:
            stream = call(driver.cuStreamCreate, flags)
        else:
            stream = call(driver.cuGreenCtxStreamCreate, green, flags, 0)
    stack.callback(destroy_stream, context, stream)
    return stream


def destroy_stream(context, stream):
    with current(context):
        call(driver.cuStreamDestroy, stream)


def make_event(context, stack, flags=driver.CUevent_flags.CU_EVENT_DEFAULT):
    with current(context):
        event = call(driver.cuEventCreate, flags)
    stack.callback(destroy_event, context, event)
    return event


def destroy_event(context, event):
    with current(context):
        call(driver.cuEventDestroy, event)


class OperatorStream:
    """Own an external stream and reusable timing events for one CUDA scope."""

    def __init__(self, device, primary, stack, green=None):
        self.device, self.primary, self.green = device, primary, green
        self.context = primary if green is None else call(driver.cuCtxFromGreenCtx, green)
        self.handle = make_stream(self.context, green, stack)
        if green is not None:
            associated = call(driver.cuStreamGetGreenCtx, self.handle)
            if int(associated) != int(green):
                raise RuntimeError("External driver stream is missing its requested green context")
        self.stream = torch.cuda.get_stream_from_external(int(self.handle), self.device)
        mode = "stream_scope_events"
        try:
            self.event_context = self.context
            self.begin, self.end = [make_event(self.context, stack) for _ in range(2)]
            self.record_same_stream(self.begin)
            self.record_same_stream(self.end)
            self.finish()
        except RuntimeError:
            if green is None:
                raise
            mode = "primary_events_recorded_on_green_scope"
            self.event_context = primary
            self.begin, self.end = [make_event(primary, stack) for _ in range(2)]
            self.record_green(self.begin)
            self.record_green(self.end)
            self.finish()
        self.event_mode = mode

    def record_same_stream(self, event):
        with current(self.context):
            call(driver.cuEventRecord, event, self.handle)

    def record_green(self, event):
        with current(self.primary):
            call(driver.cuGreenCtxRecordEvent, self.green, event)

    def enqueue(self, operation, gate):
        if gate is not None:
            with current(self.context):
                call(driver.cuStreamWaitEvent, self.handle, gate, 0)
        record = self.record_same_stream if self.event_mode == "stream_scope_events" else self.record_green
        record(self.begin)
        with torch.cuda.stream(self.stream), current(self.context):
            answer = operation()
        record(self.end)
        return answer

    def finish(self):
        with current(self.context):
            call(driver.cuStreamSynchronize, self.handle)
        with current(self.event_context):
            call(driver.cuEventSynchronize, self.end)
            return float(call(driver.cuEventElapsedTime, self.begin, self.end))


class StartGate:
    """Release already-enqueued operators together with a CUDA stream memop."""

    def __init__(self, device, primary, stack):
        self.primary = primary
        cuda_device = get_cudevice(device)
        attr = driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1
        if not call(driver.cuDeviceGetAttribute, attr, cuda_device):
            raise RuntimeError("CUDA driver reports no stream memory operations on this device")
        self.wait_stream, self.release_stream = [make_stream(primary, None, stack) for _ in range(2)]
        self.event = make_event(primary, stack, driver.CUevent_flags.CU_EVENT_DISABLE_TIMING)
        self.state = torch.zeros(1, dtype=torch.int32, device=device)
        torch.cuda.synchronize(device)
        self.address = driver.CUdeviceptr(self.state.data_ptr())
        self.generation = 0

    def arm(self):
        self.generation += 1
        with current(self.primary):
            call(driver.cuStreamWaitValue32, self.wait_stream, self.address, self.generation,
                 driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ)
            call(driver.cuEventRecord, self.event, self.wait_stream)
        return self.event

    def release(self):
        with current(self.primary):
            call(driver.cuStreamWriteValue32, self.release_stream, self.address, self.generation,
                 driver.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT)


def execute(operations, gate):
    begin = time.perf_counter()
    mark = None
    result = []
    try:
        if gate is not None:
            mark = gate.arm()
        result = [stream.enqueue(op, mark) for stream, op in operations]
    finally:
        submit = time.perf_counter()
        if gate is not None:
            gate.release()
    times = [stream.finish() for stream, _ in operations]
    stop = time.perf_counter()
    return result, {"op_gpu_ms": times, "coordinator_wall_ms": 1000 * (stop - begin),
                    "release_to_completion_ms": 1000 * (stop - submit) if gate is not None else None}


def measure(operations, gate, count):
    # Initialize each operator on the actual context before arming an event
    # whose release depends on all host-side launches having returned.
    execute(operations, None)
    for _ in range(4):
        execute(operations, gate)
    timings = []
    for _ in range(count):
        outputs, row = execute(operations, gate)
        timings.append(row)
    return outputs, timings


def compare(streams, ops, gate, count, reference):
    one, one_rows = measure([(streams[0], ops[0])], gate, count)
    two, two_rows = measure([(streams[1], ops[1])], gate, count)
    joint, pairs = measure(list(zip(streams, ops, strict=True)), gate, count)
    isolation_errors = [errors(x, y) for x, y in zip(reference, (one[0], two[0]), strict=True)]
    paired_errors = [errors(x, y) for x, y in zip(reference, joint, strict=True)]
    isolated_d, isolated_p = [median(row["op_gpu_ms"][0] for row in samples)
                              for samples in (one_rows, two_rows)]
    pair_d = median(row["op_gpu_ms"][0] for row in pairs)
    pair_p = median(row["op_gpu_ms"][1] for row in pairs)
    return {
        "isolated_d": one_rows, "isolated_p": two_rows, "paired": pairs,
        "event_scope": [stream.event_mode for stream in streams],
        "correctness": {"isolated": isolation_errors, "paired": paired_errors},
        "isolated_d_gpu_p50_ms": isolated_d, "isolated_p_gpu_p50_ms": isolated_p,
        "pair_d_gpu_p50_ms": pair_d, "pair_p_gpu_p50_ms": pair_p,
        "d_slowdown_pct": 100 * (pair_d / isolated_d - 1),
        "p_slowdown_pct": 100 * (pair_p / isolated_p - 1),
    }


def emit(out, row):
    with out.open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    print(row["kind"], row.get("sm", ""), row.get("shape", ""), row.get("pair", ""), flush=True)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--ungated", action="store_true", help="Measure the host-enqueued diagnostic fallback")
    args = parser.parse_args()
    if len(set(args.rows)) != len(args.rows) or min(args.rows + args.batch + [args.context, args.repeats]) <= 0:
        parser.error("Use positive operator geometry and distinct prefill row sizes")
    if torch.cuda.device_count() != 1:
        parser.error("Run with one approved CUDA device")
    rows = json.loads(args.trace.read_text())["requests"]
    if len(rows) < max(args.batch) or any(len(row["prompt_token_ids"]) < max(args.context, *args.rows)
                                        for row in rows[:max(args.batch)]):
        parser.error("The frozen trace must cover the required token rows")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    cfg, weights = load_layer(args.model, 0)
    layer = Layer(cfg, weights, device)
    del weights
    prefill = {}
    for size in args.rows:
        tokens = rows[0]["prompt_token_ids"][:size]
        prefill[size] = layer.mlp_input(tokens)
    decode = {batch: prepare_decode(layer, rows, batch, args.context) for batch in args.batch}
    torch.cuda.synchronize()
    primary = call(driver.cuCtxGetCurrent)
    matrices = {}
    for size, hidden in prefill.items():
        for batch, (_, decode_rows, decode_attention, lengths) in decode.items():
            def mlp(value):
                return lambda: layer.mlp(value)
            for pair, op in (("attention", decode_attention), ("MLP", mlp(decode_rows))):
                operations = [op, mlp(hidden)]
                matrices[size, batch, pair] = (operations, [fn() for fn in operations], lengths)
    torch.cuda.synchronize()

    try:
        version = call(driver.cuDriverGetVersion)
    except RuntimeError:
        version = None
    emit(args.out, {"kind": "spatial_setup", "model": str(args.model), "model_type": cfg["model_type"],
                    "trace_sha256": sha256(args.trace.read_bytes()).hexdigest(), "physical": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "driver_api": version, "device_sm": torch.cuda.get_device_properties(device).multi_processor_count,
                    "evidence_boundary": "Valid per-operator events within the owning context and host launch/fence envelopes. Exact overlap was outside the event contract."})

    with ExitStack() as stack:
        gate = None
        if not args.ungated:
            try:
                gate = StartGate(device, primary, stack)
            except RuntimeError as error:
                emit(args.out, {"kind": "spatial_gate_unavailable", "status": "DIAGNOSTIC",
                                "reason": str(error)[:1600]})
        full_sm = {}
        for counts in (None, *PARTITIONS):
            with ExitStack() as scope:
                try:
                    greens = [None, None] if counts is None else partition(device, counts, scope)
                    streams = [OperatorStream(device, primary, scope, green) for green in greens]
                    for (size, batch, op_name), (operations, reference, lengths) in matrices.items():
                        results = compare(streams, operations, gate, args.repeats, reference)
                        key = (size, batch, op_name)
                        if counts is None:
                            full_sm[key] = results
                        results["d_slowdown_vs_unpartitioned_isolated_pct"] = 100 * (
                            results["pair_d_gpu_p50_ms"] / full_sm[key]["isolated_d_gpu_p50_ms"] - 1)
                        results["p_slowdown_vs_unpartitioned_isolated_pct"] = 100 * (
                            results["pair_p_gpu_p50_ms"] / full_sm[key]["isolated_p_gpu_p50_ms"] - 1)
                        emit(args.out, {"kind": "spatial_pair", "status": "MEASURED_PHYSICAL",
                                        "sm": counts, "shape": [size, batch, args.context], "pair": [op_name, "MLP"],
                                        "actual_kv_rows": lengths, "submission": "cuda_gate" if gate else "host_enqueue",
                                        **results})
                except (RuntimeError, ValueError) as error:
                    emit(args.out, {"kind": "spatial_unavailable", "status": "INCONCLUSIVE_RUNTIME",
                                    "sm": counts, "reason": str(error)[:1600]})
                    if counts is None:
                        raise


if __name__ == "__main__":
    main()
