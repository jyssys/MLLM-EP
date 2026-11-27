"""Real-weight Qwen operator concurrency and physical cross-GPU placement."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from safetensors import safe_open
from vllm import _custom_ops
from vllm.vllm_flash_attn import flash_attn_varlen_func


def load_layer(snapshot, layer):
    model = json.loads((snapshot / "config.json").read_text())
    weights = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.layers.{layer}."
    keys = [key for key in weights if key.startswith(prefix)] + ["model.embed_tokens.weight"]
    handles, result = {}, {}
    with ExitStack() as stack:
        for key in keys:
            shard = weights[key]
            if shard not in handles:
                handles[shard] = stack.enter_context(safe_open(snapshot / shard, framework="pt", device="cpu"))
            result[key.removeprefix(prefix)] = handles[shard].get_tensor(key)
    return model, result


def rms(tensor, w, epsilon):
    return F.rms_norm(tensor, (tensor.shape[-1],), w, epsilon)


def rotate(tensor, positions, theta):
    n = tensor.shape[-1]
    frequency = theta ** (-torch.arange(0, n, 2, dtype=torch.float32, device=tensor.device) / n)
    phase = positions.float().reshape(-1, 1) * frequency
    both = torch.cat((phase, phase), dim=-1).unsqueeze(1)
    x, y = tensor.chunk(2, dim=-1)
    return tensor * both.cos().to(tensor.dtype) + torch.cat((-y, x), dim=-1) * both.sin().to(tensor.dtype)


class Layer:
    def __init__(self, cfg, weights, device):
        self.cfg, self.device = cfg, torch.device(device)
        self.storage = {key: weight.to(self.device) for key, weight in weights.items()}
        b = [self.storage.get(f"self_attn.{name}_proj.bias") for name in ("q", "k", "v")]
        self.qkv_weight = torch.cat([self.storage[f"self_attn.{name}_proj.weight"] for name in ("q", "k", "v")])
        self.qkv_bias = torch.cat(b) if all(t is not None for t in b) else None
        self.gate_up = torch.cat([self.storage[f"mlp.{name}_proj.weight"] for name in ("gate", "up")])
        self.head_dim = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
        self.q_rows = cfg["num_attention_heads"] * self.head_dim
        self.k_rows = cfg["num_key_value_heads"] * self.head_dim

    def input(self, tokens):
        tensor = F.embedding(torch.as_tensor(tokens, dtype=torch.long, device=self.device),
                             self.storage["model.embed_tokens.weight"])
        return rms(tensor, self.storage["input_layernorm.weight"], self.cfg["rms_norm_eps"])

    def qkv(self, rows):
        return F.linear(rows, self.qkv_weight, self.qkv_bias)

    def project(self, rows):
        return F.linear(rows, self.storage["self_attn.o_proj.weight"])

    def mlp_input(self, tokens):
        hidden = F.embedding(torch.as_tensor(tokens, dtype=torch.long, device=self.device),
                             self.storage["model.embed_tokens.weight"])
        q, k, v = self.qkvs(self.input(tokens))
        look = attention(q, k, v, [len(tokens)], [len(tokens)])
        residual = hidden + self.project(look.flatten(1))
        return rms(residual, self.storage["post_attention_layernorm.weight"], self.cfg["rms_norm_eps"])

    def mlp(self, rows):
        paired = F.linear(rows, self.gate_up)
        state = torch.empty((*paired.shape[:-1], paired.shape[-1] // 2), device=rows.device, dtype=rows.dtype)
        torch.ops._C.silu_and_mul(state, paired)
        return F.linear(state, self.storage["mlp.down_proj.weight"])

    def kv(self, rows):
        bias = self.qkv_bias[self.q_rows:] if self.qkv_bias is not None else None
        values = F.linear(rows, self.qkv_weight[self.q_rows:], bias)
        k, v = values.split(self.k_rows, dim=-1)
        k = k.reshape(-1, self.cfg["num_key_value_heads"], self.head_dim)
        weight = self.storage.get("self_attn.k_norm.weight")
        if weight is not None:
            k = rms(k, weight, self.cfg["rms_norm_eps"])
        pos = torch.arange(rows.shape[0], device=rows.device)
        return rotate(k, pos, self.cfg["rope_theta"]), v.reshape_as(k).contiguous()

    def qkvs(self, rows):
        q, k, v = self.qkv(rows).split((self.q_rows, self.k_rows, self.k_rows), dim=-1)
        q, k, v = [part.reshape(rows.shape[0], -1, self.head_dim) for part in (q, k, v)]
        for name, state in (("q", q), ("k", k)):
            weight = self.storage.get(f"self_attn.{name}_norm.weight")
            if weight is not None:
                norm = rms(state, weight, self.cfg["rms_norm_eps"])
                if name == "q":
                    q = norm
                else:
                    k = norm
        pos = torch.arange(rows.shape[0], device=rows.device)
        return (rotate(q, pos, self.cfg["rope_theta"]).contiguous(),
                rotate(k, pos, self.cfg["rope_theta"]).contiguous(), v.contiguous())


def cumulative(values, device):
    import itertools

    return torch.as_tensor(list(itertools.accumulate(values, initial=0)), device=device, dtype=torch.int32)


def attention_fn(q, k, v, queries, keys):
    q_len = cumulative(queries, q.device)
    kv_len = cumulative(keys, q.device)

    def calculate():
        return flash_attn_varlen_func(q, k, v, max(queries), q_len, max(keys), kv_len,
                                     causal=True, fa_version=3)
    return calculate


def attention(q, k, v, queries, keys):
    return attention_fn(q, k, v, queries, keys)()


def errors(ref, alt):
    exact = torch.equal(ref, alt)
    if exact:
        return dict(exact=True, relative_l2=0.0, max_abs=0.0)
    reference, delta = ref.float(), ref.float() - alt.float()
    return dict(exact=False, relative_l2=float(delta.norm() / reference.norm().clamp_min(1e-12)),
                max_abs=float(delta.abs().max()))


def events():
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def timed(fn, device, repeats, warm=4):
    result = []
    with torch.cuda.device(device):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize(device)
        for _ in range(repeats):
            start, finish = events()
            start.record()
            now = time.perf_counter()
            output = fn()
            finish.record()
            finish.synchronize()
            result.append({"gpu_ms": start.elapsed_time(finish), "wall_ms": 1000 * (time.perf_counter() - now)})
            del output
    return result


def resource_window(fn, device, duration=0.40):
    """Measure physical counters while repeatedly replaying the same operator."""
    import pynvml as nv
    from .telemetry import get

    with torch.cuda.device(device):
        graph = torch.cuda.CUDAGraph()
        for _ in range(4):
            fn()
        torch.cuda.synchronize(device)
        with torch.cuda.graph(graph):
            outputs = [fn() for _ in range(8)]
        graph.replay()
        torch.cuda.synchronize(device)
        nv.nvmlInit()
        handle = nv.nvmlDeviceGetHandleByUUID(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[device.index])
        before, after = nv.nvmlGpmSampleAlloc(), nv.nvmlGpmSampleAlloc()
        try:
            nv.nvmlGpmSampleGet(handle, before)
            start = time.monotonic()
            count = 0
            while time.monotonic() - start < duration:
                for _ in range(8):
                    graph.replay()
                torch.cuda.synchronize(device)
                count += len(outputs) * 8
            nv.nvmlGpmSampleGet(handle, after)
            return {"calls": count, "seconds": time.monotonic() - start, **get(handle, before, after)}
        finally:
            nv.nvmlGpmSampleFree(before)
            nv.nvmlGpmSampleFree(after)
            nv.nvmlShutdown()


def stream_modes(device, spatial):
    result = {
        "sequential": [torch.cuda.Stream(device=device)] * 2,
        "streams": [torch.cuda.Stream(device=device) for _ in range(2)],
        "priority": [torch.cuda.Stream(device=device, priority=priority) for priority in (-1, 0)],
    }
    statuses = {}
    if spatial:
        try:
            from flashinfer.green_ctx import split_device_green_ctx_by_sm_count
        except (ImportError, RuntimeError) as error:
            return result, {"flashinfer_green": {"status": "UNAVAILABLE", "reason": str(error)}}

        number = torch.cuda.get_device_properties(device).multi_processor_count
        for ratio, parts in [(0.90, (112, 16)), (0.75, (96, 32)), (0.50, (64, 64)), (0.25, (32, 96))]:
            if sum(parts) > number:
                statuses[str(ratio)] = {"status": "UNAVAILABLE", "requested_sm": parts}
                continue
            try:
                streams, resources = split_device_green_ctx_by_sm_count(device, list(parts))
                stdout = [int(entry.sm.smCount) for entry in resources]
                for stream in streams[:2]:
                    with torch.cuda.stream(stream):
                        torch.ones((64, 64), device=device, dtype=torch.bfloat16) @ torch.ones(
                            (64, 64), device=device, dtype=torch.bfloat16)
                    stream.synchronize()
                result[f"green_{ratio}"] = streams[:2]
                statuses[str(ratio)] = {"status": "ALLOCATED", "actual_sm": stdout, "requested_sm": parts}
            except (RuntimeError, ImportError, AttributeError, ValueError) as error:
                statuses[str(ratio)] = {"status": "UNAVAILABLE", "reason": str(error)[:1200], "requested_sm": parts}
    return result, statuses


def paired(fn_d, fn_p, device, streams, repetitions):
    samples = []
    with torch.cuda.device(device):
        for iteration in range(3 + repetitions):
            stamp = torch.cuda.Event(enable_timing=True)
            stamp.record()
            start = time.perf_counter()
            bounds = []
            for fn, stream in zip((fn_d, fn_p), streams, strict=True):
                with torch.cuda.stream(stream):
                    stream.wait_event(stamp)
                    first, last = events()
                    first.record()
                    data = fn()
                    last.record()
                    bounds.append((first, last, data))
            for _, end, _ in bounds:
                end.synchronize()
            end_ms = 1000 * (time.perf_counter() - start)
            intervals = [[stamp.elapsed_time(a), stamp.elapsed_time(b)] for a, b, _ in bounds]
            if iteration >= 3:
                a, b = intervals
                samples.append({"d_ms": a[1] - a[0], "p_ms": b[1] - b[0],
                                "makespan_ms": max(a[1], b[1]), "wall_ms": end_ms,
                                "overlap_ms": max(0, min(a[1], b[1]) - max(a[0], b[0])),
                                "intervals_ms": intervals})
    return samples


def prepare_decode(layer, rows, batch, context):
    inputs, qs, ks, vs = [], [], [], []
    offsets = []
    for r in range(batch):
        count = context - (r % 3) * max(1, context // 16)
        x = layer.input(rows[r]["prompt_token_ids"][:count])
        q, k, v = layer.qkvs(x)
        offsets.append(len(x))
        inputs.append(x[-1:])
        qs.append(q[-1:])
        ks.append(k)
        vs.append(v)
    q, k, v = map(torch.cat, (qs, ks, vs))
    hidden = torch.cat(inputs)
    attended = attention(q, k, v, [1] * batch, offsets)
    last_tokens = [rows[r]["prompt_token_ids"][offsets[r] - 1] for r in range(batch)]
    residual = F.embedding(torch.tensor(last_tokens, device=layer.device), layer.storage["model.embed_tokens.weight"])
    next_rows = rms(residual + layer.project(attended.flatten(1)),
                    layer.storage["post_attention_layernorm.weight"], layer.cfg["rms_norm_eps"])
    return hidden, next_rows, attention_fn(q, k, v, [1] * batch, offsets), offsets


def p2p_phase(P, D, input_rows, repetitions):
    p, d = P.device, D.device
    reference_k, reference_v = P.kv(input_rows)
    dbuf = torch.empty_like(input_rows, device=d)
    dk = torch.empty_like(reference_k, device=d)
    dv = torch.empty_like(reference_v, device=d)
    host = torch.empty_like(input_rows, device="cpu", pin_memory=True)
    host.copy_(input_rows)
    host_k, host_v = [torch.empty_like(tensor, device="cpu", pin_memory=True)
                      for tensor in (reference_k, reference_v)]
    host_k.copy_(reference_k)
    host_v.copy_(reference_v)
    with torch.cuda.device(d):
        def compact_handoff():
            dk.copy_(reference_k, non_blocking=True)
            dv.copy_(reference_v, non_blocking=True)
            return dk, dv

        def projected_handoff():
            dbuf.copy_(input_rows, non_blocking=True)
            return D.kv(dbuf)

        a = timed(compact_handoff, d, repetitions)
        b = timed(projected_handoff, d, repetitions)
        h2d = timed(lambda: dbuf.copy_(host, non_blocking=True), d, repetitions)
        def compact_host_handoff():
            dk.copy_(host_k, non_blocking=True)
            dv.copy_(host_v, non_blocking=True)
        h2d_kv = timed(compact_host_handoff, d, repetitions)
        c = projected_handoff()
        correctness = [errors(ref.to(d), alt) for ref, alt in zip((reference_k, reference_v), c, strict=True)]
    isolated_projection = timed(lambda: P.kv(input_rows), p, repetitions)
    with torch.cuda.device(p):
        returned = torch.empty_like(input_rows)
        reverse = timed(lambda: returned.copy_(dbuf, non_blocking=True), p, repetitions)
    return {"hidden_B": input_rows.numel() * input_rows.element_size(),
            "kv_B": sum(t.numel() * t.element_size() for t in (reference_k, reference_v)),
            "project_P": isolated_projection, "direct_P_to_D": a,
            "hidden_to_D_plus_project": b, "pinned_host_to_D": h2d, "pinned_host_KV_to_D": h2d_kv,
            "hidden_D_to_P": reverse, "correctness": correctness}


def split_mlp(P, D, source, dec, ratio, repetitions, quantum=None, produce=None, streams=None, coincident=False):
    p, d = P.device, D.device
    count = int(len(source) * ratio)
    left = len(source) - count
    local = P.mlp(source)
    with torch.cuda.device(d):
        staging = torch.empty_like(source[left:], device=d)
        if streams is None:
            s_decode, sd = torch.cuda.Stream(device=d, priority=-1), torch.cuda.Stream(device=d)
        else:
            s_decode, sd = streams
    with torch.cuda.device(p):
        sp = torch.cuda.Stream(device=p)
        out = torch.empty_like(local)
    modes = ("local_with_D", "split_D_idle", "split_with_D")
    data = {variant: [] for variant in modes}
    checks = {}
    for index in range(4 + repetitions):
        rotate_order = index % len(modes)
        for variant in modes[rotate_order:] + modes[:rotate_order]:
            torch.cuda.synchronize(p)
            torch.cuda.synchronize(d)
            begin = time.perf_counter()
            rows, front = source, None
            if produce is not None:
                with torch.cuda.device(p), torch.cuda.stream(sp):
                    rows = produce()
                    front = torch.cuda.Event()
                    front.record()
            d_bound = None
            if variant != "split_D_idle":
                with torch.cuda.device(d), torch.cuda.stream(s_decode):
                    if coincident and front is not None:
                        s_decode.wait_event(front)
                    a, b = events()
                    a.record()
                    d_result = dec()
                    b.record()
                    d_bound = (a, b, d_result)
            remote_bounds = None
            if variant != "local_with_D":
                with torch.cuda.device(d), torch.cuda.stream(sd):
                    if front is not None:
                        sd.wait_event(front)
                    staging.copy_(rows[left:], non_blocking=True)
                    remote_start, valid = events()
                    remote_start.record()
                    if quantum and len(staging) > quantum:
                        remote = torch.cat([D.mlp(tile) for tile in staging.split(quantum)])
                    else:
                        remote = D.mlp(staging)
                    valid.record()
                    remote_bounds = remote_start, valid
            with torch.cuda.device(p), torch.cuda.stream(sp):
                if variant == "local_with_D":
                    out.copy_(P.mlp(rows))
                else:
                    out[:left].copy_(P.mlp(rows[:left]))
                    sp.wait_event(valid)
                    out[left:].copy_(remote, non_blocking=True)
                complete = torch.cuda.Event()
                complete.record()
            complete.synchronize()
            elapsed = 1000 * (time.perf_counter() - begin)
            d_ms, overlap = None, None
            if d_bound:
                first, last, _ = d_bound
                last.synchronize()
                d_ms = first.elapsed_time(last)
                if remote_bounds:
                    start_offset, end_offset = [first.elapsed_time(t) for t in remote_bounds]
                    overlap = max(0, min(d_ms, end_offset) - max(0, start_offset))
            if index >= 4:
                data[variant].append({"p_critical_wall_ms": elapsed, "d_ms": d_ms, "remote_compute_overlap_ms": overlap})
            if index == 3:
                checks[variant] = errors(local, out)
    samples = [{"mode": variant, "trials": data[variant], "correctness": checks[variant]} for variant in modes]
    return {"rows": len(source), "D_rows": count, "bytes_round_trip": 2 * count * source.shape[-1] * 2,
            "remote_M": quantum or count, "included_prefill_front": produce is not None,
            "coincident_release": coincident, "data": samples}


def put(out, row):
    with out.open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    print(row["kind"], row.get("shape", row.get("rows", "")), flush=True)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[1024, 4096, 8192])
    parser.add_argument("--contexts", type=int, nargs="+", default=[8192, 16384])
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--spatial", action="store_true")
    parser.add_argument("--targeted", action="store_true")
    parser.add_argument("--overlap-quanta", type=int, nargs="+",
                        help="Test these remote GEMM row quanta at concurrent release")
    args = parser.parse_args()
    if args.layer != 0:
        parser.error("The real-token block input capture starts at model layer zero")
    trace = json.loads(args.trace.read_text())["requests"]
    if min(len(row["prompt_token_ids"]) for row in trace) < max(args.rows + args.contexts):
        parser.error("Use original/pinned controlled input rows long enough for every shape")
    cfg, weights = load_layer(args.model, args.layer)
    P, D = [Layer(cfg, weights, f"cuda:{n}") for n in (0, 1)]
    if not torch.cuda.can_device_access_peer(0, 1):
        raise RuntimeError("Qualified direction B needs NVLink peer access")
    modes, spatial = stream_modes(D.device, args.spatial)
    put(args.out, {"kind": "setup", "model": str(args.model), "dtype": "BF16", "timing": "cuda_event_and_wall",
                   "head_dim": P.head_dim, "head_shape": [cfg["num_attention_heads"], cfg["num_key_value_heads"]],
                   "hidden": cfg["hidden_size"], "layers": cfg["num_hidden_layers"], "spatial": spatial,
                   "cuda_version": torch.version.cuda, "torch": torch.__version__, "transfer": "cuda_p2p_copy"})
    for size in args.rows:
        tokens = trace[0]["prompt_token_ids"][:size]
        xp = P.input(tokens)
        mp = P.mlp_input(tokens)
        inputs = xp.to(D.device)
        md = mp.to(D.device)
        q, k, v = D.qkvs(inputs)
        peer_input = torch.empty_like(inputs)
        collective_in = [tensor[:min(512, size)].contiguous() for tensor in (xp, inputs)]
        collective_out = [torch.empty_like(tensor) for tensor in collective_in]
        def dma():
            return peer_input.copy_(xp, non_blocking=True)
        def allreduce():
            torch.cuda.nccl.all_reduce(collective_in, collective_out, streams=[
                torch.cuda.current_stream(P.device), torch.cuda.current_stream(D.device)])
            return collective_out
        pre = {"MLP": lambda: D.mlp(md), "QKV": lambda: D.qkv(inputs),
               "O": lambda: D.project(inputs), "attention": attention_fn(q, k, v, [size], [size])}
        front = timed(lambda: P.mlp_input(tokens), P.device, args.repeats)
        p_mlp = timed(lambda: P.mlp(mp), P.device, args.repeats)
        full = timed(lambda: P.mlp(P.mlp_input(tokens)), P.device, args.repeats)
        put(args.out, {"kind": "layer_cost", "rows": size, "front": front, "mlp": p_mlp, "full": full})
        put(args.out, {"kind": "destination", "rows": size, **p2p_phase(P, D, xp, args.repeats)})
        for ctx in args.contexts:
            for batch in args.batch:
                dl, dm, da, live_lengths = prepare_decode(D, trace, batch, ctx)
                dec = {"attention": da, "MLP": lambda: D.mlp(dm), "QKV": lambda: D.qkv(dl),
                       "KV_DMA": dma, "AllReduce": allreduce}
                run = [("attention", name) for name in pre] + [
                    ("MLP", "attention"), ("MLP", "MLP"), ("QKV", "MLP"), ("KV_DMA", "MLP"), ("AllReduce", "MLP")]
                if args.targeted or args.overlap_quanta:
                    def decode_step():
                        da()
                        return D.mlp(dm)
                    settings = ([(fraction, True, quantum)
                                 for fraction in (0.25, 0.5) for quantum in args.overlap_quanta]
                                if args.overlap_quanta else
                                [(fraction, coincident, None) for fraction in (0.125, 0.25, 0.5)
                                 for coincident in (False, True)])
                    for fraction, simultaneous, quantum in settings:
                        decision = split_mlp(P, D, mp, decode_step, fraction, args.repeats, quantum=quantum,
                                             produce=lambda: P.mlp_input(tokens), coincident=simultaneous)
                        put(args.out, {"kind": "round_trip", "shape": [size, batch, ctx],
                                       "allocation": "priority_streams", "fraction": fraction, **decision})
                    continue
                if size == 4096 and ctx == 16384 and batch == 4:
                    for role, group in (("decode", dec), ("prefill", pre)):
                        for name, operator in group.items():
                            if name in ("AllReduce", "KV_DMA"):
                                continue
                            try:
                                reading = resource_window(operator, D.device)
                            except (RuntimeError, ImportError) as error:
                                reading = {"error": str(error)}
                            put(args.out, {"kind": "isolated_resources", "role": role, "operator": name,
                                           "shape": [size, batch, ctx], **reading})
                joint = torch.cat((dm, md))
                mixed = timed(lambda: D.mlp(joint), D.device, args.repeats)
                pure = timed(lambda: D.mlp(dm), D.device, args.repeats)
                check = errors(D.mlp(dm), D.mlp(joint)[:batch])
                put(args.out, {"kind": "mixed_mlp", "shape": [size, batch, ctx],
                               "joint": mixed, "decode_iso": pure, "correctness": check})
                for d_name, p_name in run:
                    try:
                        isolated_d = timed(dec[d_name], D.device, args.repeats)
                        isolated_p = timed(pre[p_name], D.device, args.repeats)
                    except (RuntimeError, AttributeError) as error:
                        if d_name != "AllReduce":
                            raise
                        put(args.out, {"kind": "unavailable", "shape": [size, batch, ctx],
                                       "pair": [d_name, p_name], "reason": str(error)})
                        continue
                    for mode, streams in tuple(modes.items()):
                        if mode.startswith("green") and (batch not in (1, 4) or d_name not in ("attention", "MLP")
                                                              or p_name not in ("MLP", "attention")):
                            continue
                        try:
                            result = paired(dec[d_name], pre[p_name], D.device, streams, args.repeats)
                        except (RuntimeError, AttributeError) as error:
                            if not mode.startswith("green") and d_name != "AllReduce":
                                raise
                            put(args.out, {"kind": "unavailable", "shape": [size, batch, ctx],
                                           "pair": [d_name, p_name], "mode": mode, "reason": str(error)})
                            if mode.startswith("green"):
                                del modes[mode]
                            continue
                        put(args.out, {"kind": "pair", "shape": [size, batch, ctx], "actual_kv_rows": live_lengths,
                                       "pair": [d_name, p_name], "mode": mode, "isolated_D": isolated_d,
                                       "isolated_P": isolated_p, "concurrent": result})
                if batch in (1, 4):
                    for fraction in (0.125, 0.25, 0.5):
                        data = split_mlp(P, D, mp, da, fraction, args.repeats)
                        put(args.out, {"kind": "round_trip", "shape": [size, batch, ctx], "fraction": fraction, **data})
                    if size >= 4096:
                        def decode_front():
                            da()
                            return D.mlp(dm)
                        if size != 8192 or ctx == 16384:
                            full_round_trip = split_mlp(P, D, mp, decode_front, 0.25, args.repeats,
                                                        produce=lambda: P.mlp_input(tokens))
                            put(args.out, {"kind": "round_trip", "shape": [size, batch, ctx], "fraction": 0.25,
                                           "allocation": "priority_streams", **full_round_trip})
                            if ctx == 16384 and batch == 4:
                                for mode, allocation in modes.items():
                                    if not mode.startswith("green"):
                                        continue
                                    try:
                                        bounded = split_mlp(P, D, mp, decode_front, 0.25, args.repeats,
                                                            produce=lambda: P.mlp_input(tokens), streams=allocation)
                                    except (RuntimeError, AttributeError) as error:
                                        put(args.out, {"kind": "unavailable", "shape": [size, batch, ctx],
                                                       "mode": mode, "pair": ["full_block", "remote_MLP"],
                                                       "reason": str(error)})
                                        continue
                                    put(args.out, {"kind": "round_trip", "shape": [size, batch, ctx],
                                                   "fraction": 0.25, "allocation": mode, **bounded})
                        data = split_mlp(P, D, mp, da, 0.25, args.repeats, quantum=256)
                        put(args.out, {"kind": "round_trip", "shape": [size, batch, ctx], "fraction": 0.25,
                                       "chunked": True, **data})


if __name__ == "__main__":
    main()
