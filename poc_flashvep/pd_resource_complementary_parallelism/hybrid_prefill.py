"""Measure exact Qwen prefill layers under two-GPU attention and MLP layouts."""
import argparse
from contextlib import contextmanager
from datetime import timedelta
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
import time

from safetensors import safe_open
import torch
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F
from vllm import _custom_ops as vops
from vllm.vllm_flash_attn import flash_attn_varlen_func

WORLD = 2
ARMS = (
    "tp2", "cp2_contiguous", "cp2_zigzag", "tp_attention_row_mlp",
    "cp_attention_tp_mlp", "local_control",
)


class Times:
    def __init__(self, enabled):
        self.enabled = enabled
        self.events = {}

    @contextmanager
    def stage(self, name):
        if not self.enabled:
            yield
            return
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        yield
        end.record()
        self.events.setdefault(name, []).append((begin, end))

    def results(self):
        return {name: sum(begin.elapsed_time(end) for begin, end in segments)
                for name, segments in self.events.items()}


class Checkpoint:
    def __init__(self, path, device):
        self.path = path
        self.device = device
        self.config = json.loads((path / "config.json").read_text())
        indices = list(path.glob("*.safetensors.index.json"))
        if indices:
            self.files = json.loads(indices[0].read_text())["weight_map"]
        else:
            files = list(path.glob("*.safetensors"))
            if len(files) != 1:
                raise ValueError("Expected a sharded index or one safetensors file")
            with safe_open(files[0], framework="pt") as shard:
                self.files = dict.fromkeys(shard.keys(), files[0].name)

    def tensor(self, key, required=True):
        if key not in self.files and not required:
            return None
        with safe_open(self.path / self.files[key], framework="pt", device=str(self.device)) as shard:
            result = shard.get_tensor(key)
        if result.dtype != torch.bfloat16:
            raise ValueError(f"Expected BF16 checkpoint tensor {key}, got {result.dtype}")
        return result


class Layer(nn.Module):
    """Fused column projections and independently selectable attention/MLP shards."""

    def __init__(self, ckpt, layer, rank):
        super().__init__()
        prefix = f"model.layers.{layer}."
        cfg, device = ckpt.config, ckpt.device
        self.hidden = cfg["hidden_size"]
        self.dim = cfg.get("head_dim", self.hidden // cfg["num_attention_heads"])
        self.qheads = cfg["num_attention_heads"]
        self.kheads = cfg["num_key_value_heads"]
        self.eps = cfg["rms_norm_eps"]
        if cfg["model_type"] not in ("qwen2", "qwen3") or cfg.get("use_sliding_window"):
            raise ValueError("Expected dense Qwen with full causal attention")
        if self.qheads % WORLD or self.kheads % WORLD:
            raise ValueError("Each TP rank must own complete Q/KV groups")
        if cfg.get("rope_scaling") is not None or cfg.get("rope_parameters", {}).get("rope_type", "default") != "default":
            raise ValueError("This screen implements original unscaled RoPE")
        self.max_positions = cfg["max_position_embeddings"]
        self.theta = cfg.get("rope_theta", cfg.get("rope_parameters", {}).get("rope_theta", 10000.0))
        qkv = [ckpt.tensor(prefix + f"self_attn.{axis}_proj.weight") for axis in ("q", "k", "v")]
        biases = [ckpt.tensor(prefix + f"self_attn.{axis}_proj.bias", False) for axis in ("q", "k", "v")]
        self.qkv = torch.cat(qkv)
        self.qkv_tp = torch.cat([weight.chunk(WORLD, dim=0)[rank] for weight in qkv])
        self.qkv_bias = self.qkv_tp_bias = None
        if any(value is not None for value in biases):
            if any(value is None for value in biases):
                raise ValueError("Asymmetric QKV bias geometry")
            self.qkv_bias = torch.cat(biases)
            self.qkv_tp_bias = torch.cat([bias.chunk(WORLD, dim=0)[rank] for bias in biases])
        self.out = ckpt.tensor(prefix + "self_attn.o_proj.weight")
        self.out_tp = self.out.chunk(WORLD, dim=1)[rank].contiguous()
        if ckpt.tensor(prefix + "self_attn.o_proj.bias", False) is not None:
            raise ValueError("Expected a bias-free attention output projection")
        gate = ckpt.tensor(prefix + "mlp.gate_proj.weight")
        up = ckpt.tensor(prefix + "mlp.up_proj.weight")
        self.gate_up = torch.cat((gate, up))
        self.gate_up_tp = torch.cat((gate.chunk(WORLD)[rank], up.chunk(WORLD)[rank]))
        self.down = ckpt.tensor(prefix + "mlp.down_proj.weight")
        self.down_tp = self.down.chunk(WORLD, dim=1)[rank].contiguous()
        self.pre = ckpt.tensor(prefix + "input_layernorm.weight")
        self.post = ckpt.tensor(prefix + "post_attention_layernorm.weight")
        self.q_norm = ckpt.tensor(prefix + "self_attn.q_norm.weight", False)
        self.k_norm = ckpt.tensor(prefix + "self_attn.k_norm.weight", False)
        self.device = device

    def rope(self, length):
        powered = torch.arange(0, self.dim, 2, dtype=torch.float32, device=self.device) / self.dim
        freq = 1.0 / (self.theta ** powered)
        angles = torch.arange(length, dtype=torch.float32, device=self.device)[:, None] * freq[None, :]
        degrees = torch.cat((angles, angles), dim=-1)
        return degrees.cos().to(torch.bfloat16), degrees.sin().to(torch.bfloat16)

    def rms(self, value, weight):
        if weight is None:
            return value
        flat = value.contiguous()
        result = torch.empty_like(flat)
        width = flat.shape[-1]
        vops.rms_norm(result.view(-1, width), flat.view(-1, width), weight, self.eps)
        return result

    def qkv_for(self, value, cos, sin, tp):
        qheads = self.qheads // WORLD if tp else self.qheads
        kheads = self.kheads // WORLD if tp else self.kheads
        before = self.rms(value, self.pre)
        mixed = F.linear(before, self.qkv_tp if tp else self.qkv, self.qkv_tp_bias if tp else self.qkv_bias)
        q, k, v = mixed.split((qheads * self.dim, kheads * self.dim, kheads * self.dim), dim=-1)
        q = self.rms(q.contiguous().view(*value.shape[:-1], qheads, self.dim), self.q_norm)
        k = self.rms(k.contiguous().view(*value.shape[:-1], kheads, self.dim), self.k_norm)
        v = v.contiguous().view(*value.shape[:-1], kheads, self.dim)
        return rotary(q, cos, sin), rotary(k, cos, sin), v

    def mlp(self, x, tp):
        hidden = self.rms(x, self.post)
        gate_up = F.linear(hidden, self.gate_up_tp if tp else self.gate_up)
        inner = torch.empty((*gate_up.shape[:-1], gate_up.shape[-1] // 2), device=x.device, dtype=x.dtype)
        torch.ops._C.silu_and_mul(inner, gate_up)
        return F.linear(inner, self.down_tp if tp else self.down)

    def weight_bytes(self, tp_attn, tp_mlp):
        tensors = (
            self.qkv_tp if tp_attn else self.qkv, self.qkv_tp_bias if tp_attn else self.qkv_bias,
            self.out_tp if tp_attn else self.out, self.pre, self.post, self.q_norm, self.k_norm,
            self.gate_up_tp if tp_mlp else self.gate_up, self.down_tp if tp_mlp else self.down,
        )
        return sum(x.numel() * x.element_size() for x in tensors if x is not None)


def rotary(x, cos, sin):
    first, last = x.chunk(2, dim=-1)
    rotated = torch.cat((-last, first), dim=-1)
    return (x * cos[None, :, None, :] + rotated * sin[None, :, None, :]).contiguous()


@lru_cache(maxsize=128)
def indptr(batch, length, device):
    return torch.arange(batch + 1, dtype=torch.int32, device=torch.device(device)) * length


def fa(q, k, v):
    batch, q_len = q.shape[:2]
    k_len = k.shape[1]
    out = flash_attn_varlen_func(
        q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1),
        max_seqlen_q=q_len, cu_seqlens_q=indptr(batch, q_len, str(q.device)),
        max_seqlen_k=k_len, cu_seqlens_k=indptr(batch, k_len, str(q.device)),
        causal=True, softmax_scale=q.shape[-1] ** -0.5, fa_version=3,
    )
    return out.reshape(batch, q_len, -1)


def storage_bytes(tensor):
    return tensor.numel() * tensor.element_size()


class RowPlan:
    def __init__(self, rank, length, device, zigzag=False):
        self.rank = rank
        self.length = length
        self.half = length // WORLD
        if zigzag:
            self.segments = (((0, length // 4), (3 * length // 4, length)), ((length // 4, 3 * length // 4),))
        else:
            self.segments = (((0, self.half),), ((self.half, length),))
        ordered = [[i for start, stop in chunks for i in range(start, stop)] for chunks in self.segments]
        self.indices = torch.tensor(ordered[rank], device=device)
        self.inverse = torch.tensor([i for _, i in sorted((row, i) for i, row in enumerate(sum(ordered, [])))],
                                    device=device) if zigzag else None

    def local(self, x):
        if self.inverse is None:
            begin = self.rank * self.half
            return x[:, begin:begin + self.half].contiguous()
        return x.index_select(1, self.indices)

    def transport(self, x):
        packed = torch.empty((WORLD * x.shape[0], *x.shape[1:]), dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(packed, x)
        return packed

    def merge(self, packed):
        batch = packed.shape[0] // WORLD
        tail = packed.shape[2:]
        combined = packed.view(WORLD, batch, self.half, *tail).transpose(0, 1).reshape(batch, self.length, *tail)
        return combined if self.inverse is None else combined.index_select(1, self.inverse)

    def scatter_input(self, x):
        batch = x.shape[0]
        return x.view(batch, WORLD, self.half, *x.shape[2:]).transpose(0, 1).reshape(
            batch * WORLD, self.half, *x.shape[2:])


class Run:
    def __init__(self, layers, x, rank, arm):
        self.layers, self.rank, self.arm = layers, rank, arm
        self.x = x
        self.batch, self.length = x.shape[:2]
        self.contiguous = RowPlan(rank, self.length, x.device)
        self.zigzag = RowPlan(rank, self.length, x.device, zigzag=True)
        self.rope = [layer.rope(self.length) for layer in layers]
        self.uses_tp_attn = arm in ("tp2", "tp_attention_row_mlp")
        self.uses_tp_mlp = arm in ("tp2", "cp_attention_tp_mlp")
        self.is_cp = arm.startswith("cp")
        self.payload = self.payload_bytes()

    def payload_bytes(self):
        hidden = storage_bytes(self.x)
        layer = self.layers[0]
        docs = self.batch * self.length
        kv_full = docs * layer.kheads * layer.dim * 2 * 2
        one_link = {
            "tp2": {"attention_output_allreduce": hidden, "mlp_allreduce": hidden},
            "cp2_contiguous": {"causal_kv_send": kv_full // 2 if self.rank == 0 else 0},
            "cp2_zigzag": {"kv_allgather": kv_full // 2},
            "tp_attention_row_mlp": {"attention_reduce_scatter": hidden // 2, "hidden_allgather": hidden // 2},
            "cp_attention_tp_mlp": {"kv_allgather": kv_full // 2, "hidden_allgather": hidden // 2,
                                    "mlp_reduce_scatter": hidden // 2},
            "local_control": {},
        }[self.arm]
        if self.is_cp:
            one_link["final_output_allgather"] = hidden // 2
        return {key: value * (1 if key == "final_output_allgather" else len(self.layers))
                for key, value in one_link.items()}

    def cp_attention(self, q, k, v, plan, meter):
        with meter.stage("kv_layout"):
            sends = torch.cat((k, v), dim=-2)
        if plan.inverse is None:
            with meter.stage("kv_communication"):
                if self.rank == 0:
                    dist.send(sends, dst=1)
                    received = None
                else:
                    received = torch.empty_like(sends)
                    dist.recv(received, src=0)
            with meter.stage("kv_layout"):
                if received is not None:
                    prev_k, prev_v = received.chunk(2, dim=-2)
                    k = torch.cat((prev_k, k), dim=1)
                    v = torch.cat((prev_v, v), dim=1)
            with meter.stage("attention_fa3"):
                return fa(q, k, v)
        with meter.stage("kv_communication"):
            gathered = plan.transport(sends)
        with meter.stage("kv_layout"):
            full_k, full_v = [t.contiguous() for t in plan.merge(gathered).chunk(2, dim=-2)]
        parts = []
        offset = 0
        with meter.stage("attention_fa3"):
            for start, stop in plan.segments[self.rank]:
                rows = stop - start
                parts.append(fa(q[:, offset:offset + rows], full_k[:, :stop].contiguous(),
                                full_v[:, :stop].contiguous()))
                offset += rows
        with meter.stage("kv_layout"):
            return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

    def gather(self, x, plan, meter, name):
        with meter.stage(name):
            payload = plan.transport(x)
        with meter.stage("hidden_layout"):
            return plan.merge(payload)

    def reduce_scatter(self, x, meter, plan):
        with meter.stage("hidden_layout"):
            send = plan.scatter_input(x)
            target = torch.empty((self.batch, plan.half, x.shape[-1]), dtype=x.dtype, device=x.device)
        with meter.stage("hidden_communication"):
            dist.reduce_scatter_tensor(target, send)
        return target

    def cp_tp_mlp(self, layer, hidden, meter, plan):
        # TP contracts the MLP for all rows. Reduce-scatter in rank ordering
        # so that next layer keeps the same (potentially zig-zag) row layout.
        with meter.stage("hidden_communication"):
            hauled = plan.transport(hidden)
        with meter.stage("hidden_layout"):
            ordered = hauled.view(WORLD, self.batch, plan.half, hidden.shape[-1]).transpose(0, 1).reshape(
                self.batch, self.length, hidden.shape[-1])
        with meter.stage("mlp"):
            partial = layer.mlp(ordered, tp=True)
        return hidden + self.reduce_scatter(partial, meter, plan)

    def forward(self, meter):
        x = self.x
        plan = self.contiguous if self.arm == "cp2_contiguous" else self.zigzag
        if self.arm == "local_control" and self.batch >= WORLD:
            x = self.x[self.rank:self.rank + 1].contiguous()
        if self.is_cp:
            with meter.stage("hidden_layout"):
                x = plan.local(x)
        kv = None
        for layer, (cos, sin) in zip(self.layers, self.rope, strict=True):
            with meter.stage("attention_qkv_rope"):
                if self.is_cp:
                    cos, sin = cos.index_select(0, plan.indices), sin.index_select(0, plan.indices)
                q, k, v = layer.qkv_for(x, cos, sin, self.uses_tp_attn)
            kv = (k, v)
            if self.is_cp:
                a = self.cp_attention(q, k, v, plan, meter)
            else:
                with meter.stage("attention_fa3"):
                    a = fa(q, k, v)
            with meter.stage("attention_out_projection"):
                out = F.linear(a, layer.out_tp if self.uses_tp_attn else layer.out)
            if self.arm == "tp_attention_row_mlp":
                out = self.reduce_scatter(out, meter, self.contiguous)
                with meter.stage("mlp"):
                    z = self.contiguous.local(x) + out
                    local = z + layer.mlp(z, tp=False)
                x = self.gather(local, self.contiguous, meter, "hidden_communication")
                continue
            if self.uses_tp_attn:
                with meter.stage("hidden_communication"):
                    dist.all_reduce(out)
            hidden = x + out
            if self.arm == "cp_attention_tp_mlp":
                x = self.cp_tp_mlp(layer, hidden, meter, plan)
            else:
                with meter.stage("mlp"):
                    out = layer.mlp(hidden, self.uses_tp_mlp)
                if self.uses_tp_mlp:
                    with meter.stage("hidden_communication"):
                        dist.all_reduce(out)
                x = hidden + out
        if self.is_cp:
            x = self.gather(x, plan, meter, "final_output_communication")
        return x, kv

    def kv_view(self, kv):
        if self.uses_tp_attn:
            view = []
            for item in kv:
                others = [torch.empty_like(item) for _ in range(WORLD)]
                dist.all_gather(others, item.contiguous())
                view.append(torch.cat(others, dim=-2))
            return view
        if self.is_cp:
            plan = self.contiguous if self.arm == "cp2_contiguous" else self.zigzag
            return [plan.merge(plan.transport(item)) for item in kv]
        return kv


def validate(reference, test):
    left, right = reference.float(), test.float()
    delta = left - right
    return {
        "relative_l2": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(left)),
        "max_abs": float(delta.abs().max()),
        "cosine": float(F.cosine_similarity(left.reshape(1, -1), right.reshape(1, -1))),
        "bitwise": bool(torch.equal(reference, test)),
    }


def forward_full(layers, x):
    for layer in layers:
        cos, sin = layer.rope(x.shape[1])
        q, k, v = layer.qkv_for(x, cos, sin, tp=False)
        x = x + F.linear(fa(q, k, v), layer.out)
        x = x + layer.mlp(x, tp=False)
    return x, (k, v)


def timing(run, args, instrument):
    for _ in range(args.warmup):
        run.forward(Times(False))
    torch.cuda.synchronize()
    if args.graph and not instrument:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            graph_output = run.forward(Times(False))
        torch.cuda.synchronize()
        step = graph.replay
    else:
        graph_output = None
        step = None
    measured = []
    for i in range(args.repeats):
        dist.barrier()
        torch.cuda.synchronize()
        meter = Times(instrument)
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall = time.perf_counter()
        start.record()
        if step is None:
            ans = run.forward(meter)
        else:
            step()
            ans = graph_output
        stop.record()
        stop.synchronize()
        measured.append({"cuda_ms": start.elapsed_time(stop), "wall_ms": (time.perf_counter() - wall) * 1000,
                         "stages_cuda_ms": meter.results()})
        if i == 0:
            result = ans
    return result, measured


def load_tokens(args, model, model_max):
    rows = json.loads(args.workload.read_text())["requests"]
    target = max(args.lengths)
    if target > model_max:
        raise ValueError(f"Requested length {target} exceeds the checkpoint context {model_max}")
    if "prompt_token_ids" in rows[0]:
        matching = [row for row in rows if len(row["prompt_token_ids"]) >= target][:args.batch]
        if len(matching) != args.batch:
            raise ValueError(f"Need {args.batch} frozen prompts of at least {target} tokens")
        return [row["prompt_token_ids"][:target] for row in matching], [row["request_id"] for row in matching]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    longest = sorted(rows, key=lambda row: row["total_input_tokens"], reverse=True)[: args.batch]
    tokens = []
    for row in longest:
        messages = [{"role": "user", "content": row["prompt"]}]
        kwargs = {"enable_thinking": False} if model == "qwen3" else {}
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        if len(ids) < target or target > model_max:
            raise ValueError(f"Requested length {target}, available real prompt {len(ids)}, model {model_max}")
        tokens.append(ids[:target])
    return tokens, [row["request_id"] for row in longest]


@torch.inference_mode()
def worker(rank, address, args):
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", init_method=address, rank=rank, world_size=WORLD, timeout=timedelta(minutes=6))
    try:
        ckpt = Checkpoint(args.model, device)
        layers = [Layer(ckpt, i, rank) for i in range(args.layers)]
        input_tokens, workload_ids = load_tokens(args, ckpt.config["model_type"], layers[0].max_positions)
        embed = ckpt.tensor("model.embed_tokens.weight")
        hidden = F.embedding(torch.tensor(input_tokens, dtype=torch.int64, device=device), embed)
        del embed
        torch.cuda.synchronize()
        for length in args.lengths:
            x = hidden[:, :length].contiguous()
            expected, full_kv = forward_full(layers, x)
            for arm in args.arms:
                run = Run(layers, x, rank, arm)
                (answer, kv), clean = timing(run, args, instrument=False)
                result_kv = run.kv_view(kv)
                control = arm == "local_control" and args.batch == WORLD
                target = expected[rank:rank + 1] if control else expected
                expected_kv = tuple(val[rank:rank + 1] for val in full_kv) if control else full_kv
                correctness = {"hidden": validate(target, answer), "k": validate(expected_kv[0], result_kv[0]),
                               "v": validate(expected_kv[1], result_kv[1])}
                _, obs = timing(run, args, instrument=True)
                packet = {"rank": rank, "clean": clean, "stage_events_eager": obs, "correctness": correctness,
                          "payload_tx_bytes": run.payload, "weights_bytes_used": sum(
                              layer.weight_bytes(run.uses_tp_attn, run.uses_tp_mlp) for layer in layers)}
                receipt = [None] * WORLD
                dist.all_gather_object(receipt, packet)
                if rank == 0:
                    critical = [max(receipt[j]["clean"][i]["cuda_ms"] for j in range(WORLD))
                                for i in range(args.repeats)]
                    wall = [max(receipt[j]["clean"][i]["wall_ms"] for j in range(WORLD))
                            for i in range(args.repeats)]
                    diag = [max(receipt[j]["stage_events_eager"][i]["cuda_ms"] for j in range(WORLD))
                            for i in range(args.repeats)]
                    report = {
                        "evidence": "MEASURED_PHYSICAL", "checkpoint": str(args.model),
                        "config_sha256": sha256((args.model / "config.json").read_bytes()).hexdigest(),
                        "torch": torch.__version__, "geometry": {
                            "layers": args.layers, "batch": args.batch, "input_length": length,
                            "hidden_size": layers[0].hidden, "query_heads": layers[0].qheads,
                            "kv_heads": layers[0].kheads, "head_dim": layers[0].dim,
                        }, "p_rank_count": WORLD,
                        "workload_ids": workload_ids,
                        "token_sha256": sha256(json.dumps([t[:length] for t in input_tokens]).encode()).hexdigest(),
                        "arm": arm, "mode": "cuda_graph" if args.graph else "eager",
                        "critical_cuda_ms_p50": median(critical), "critical_wall_ms_p50": median(wall),
                        "observed_stage_runs_cuda_ms_p50": median(diag),
                        "event_instrumentation_overhead_pct": None if args.graph else 100 * (median(diag) / median(critical) - 1),
                        "physical_transmit_payload_bytes_per_rank": [sum(r["payload_tx_bytes"].values()) for r in receipt],
                        "correctness_screen_relative_l2_max": args.relative_l2,
                        "correctness_screen_pass": all(
                            values["relative_l2"] <= args.relative_l2
                            for item in receipt for values in item["correctness"].values()
                        ),
                        "ranks": receipt,
                    }
                    with args.out.open("a") as stream:
                        stream.write(json.dumps(report) + "\n")
                    print(length, arm, round(median(critical), 3), correctness["hidden"]["relative_l2"], flush=True)
                dist.barrier()
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=(2048, 8192, 16384, 32768))
    parser.add_argument("--arms", choices=ARMS, nargs="+", default=ARMS)
    parser.add_argument("--batch", type=int, choices=(1, 2), default=1)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--relative-l2", type=float, default=0.02)
    args = parser.parse_args()
    if min(args.lengths) <= 0 or any(size % 4 for size in args.lengths) or min(args.layers, args.repeats) < 1:
        parser.error("Positive sequence length divisible by four and positive layer/repeat counts required")
    if args.out.exists():
        parser.error("Output path already exists")
    if torch.cuda.device_count() != WORLD:
        parser.error("Two CUDA devices required")
    with TemporaryDirectory(prefix="hybrid-prefill-") as temporary:
        address = f"file://{Path(temporary) / 'rendezvous'}"
        torch.multiprocessing.spawn(worker, args=(address, args), nprocs=WORLD, join=True)


if __name__ == "__main__":
    main()
