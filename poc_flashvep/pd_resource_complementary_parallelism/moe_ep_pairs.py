"""Physical two-rank expert dispatch, grouped expert compute, and combine on Qwen3-30B-A3B.

Run with torchrun in a two-device GPU lease. The replay fixes routing and counts
before timing, then issues NCCL all-to-all or DeepEP intranode payloads inside the
measured path. The scope excludes online scheduling, router layout construction
and cross-P/D transport. BF16 expert ownership is linear across both ranks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer
from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.vllm_flash_attn import flash_attn_varlen_func


@dataclass
class Checkpoint:
    path: Path
    device: torch.device

    def __post_init__(self):
        self.index_path = self.path / "model.safetensors.index.json"
        self.weight_index = json.loads(self.index_path.read_text())["weight_map"]
        self.config = json.loads((self.path / "config.json").read_text())
        if self.config["model_type"] != "qwen3_moe":
            raise ValueError("The replay requires the Qwen3 MoE weight layout")

    def load(self, stack, files, name):
        shard = self.weight_index[name]
        if shard not in files:
            files[shard] = stack.enter_context(safe_open(self.path / shard, framework="pt", device="cpu"))
        weight = files[shard].get_tensor(name)
        if weight.dtype != torch.bfloat16:
            raise TypeError(f"Expected BF16 for {name}, got {weight.dtype}")
        return weight.to(self.device)

    def attention_weights(self, layer=0):
        primary = (
            "input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_proj.weight",
            "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
            "self_attn.q_norm.weight", "self_attn.k_norm.weight", "mlp.gate.weight",
        )
        with ExitStack() as stack:
            files = {}
            weights = {key: self.load(stack, files, f"model.layers.{layer}.{key}") for key in primary}
            weights["embedding"] = self.load(stack, files, "model.embed_tokens.weight")
        return weights

    def expert_weights(self, start, stop, layer=0):
        w1, w2 = [], []
        with ExitStack() as stack:
            files = {}
            for expert in range(start, stop):
                prefix = f"model.layers.{layer}.mlp.experts.{expert}."
                gate = self.load(stack, files, f"{prefix}gate_proj.weight")
                up = self.load(stack, files, f"{prefix}up_proj.weight")
                w1.append(torch.cat((gate, up)))
                w2.append(self.load(stack, files, f"{prefix}down_proj.weight"))
        return torch.stack(w1), torch.stack(w2)


def rms_norm(x, weight, epsilon):
    x_float = x.float()
    x_float *= (x_float.square().mean(-1, keepdim=True) + epsilon).rsqrt()
    return weight * x_float.to(x.dtype)


def rotate(x, cos, sin):
    a, b = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-b, a), dim=-1) * sin


def attention(q, k, v):
    """FA3 on equal-length sequence segments with immutable real model Q/K/V."""
    batch, query, heads, dim = q.shape
    length = k.shape[1]
    cu_q = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * query
    cu_k = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * length
    result = flash_attn_varlen_func(
        q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1),
        query, cu_q, length, cu_k, causal=True, fa_version=3,
    )
    return result.reshape(batch, query, heads * dim)


def layer_inputs(weights, config, token_ids):
    x = F.embedding(token_ids, weights["embedding"])
    eps = config["rms_norm_eps"]
    h = rms_norm(x, weights["input_layernorm.weight"], eps)
    head_dim = config["head_dim"]

    def project(letter):
        result = F.linear(h, weights[f"self_attn.{letter}_proj.weight"]).unflatten(-1, (-1, head_dim))
        if letter == "v":
            return result
        return rms_norm(result, weights[f"self_attn.{letter}_norm.weight"], eps)

    q, k, v = (project(letter) for letter in "qkv")
    length = token_ids.shape[-1]
    freq = config["rope_theta"] ** (-torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32) / head_dim)
    theta = torch.arange(length, device=x.device, dtype=torch.float32)[:, None] * freq
    theta = torch.cat((theta, theta), dim=-1)[:, None, :]
    cos, sin = theta.cos().to(x.dtype), theta.sin().to(x.dtype)
    q, k = rotate(q, cos, sin), rotate(k, cos, sin)

    def normalized_residual(residual, q, k, v):
        output = F.linear(attention(q, k, v), weights["self_attn.o_proj.weight"])
        return rms_norm(residual + output, weights["post_attention_layernorm.weight"], eps)

    return x, (q, k, v), normalized_residual


def choose_tokens(path, tokenizer, rank, prefill_rows, decode_batch, decode_context):
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or "requests" not in data:
        raise ValueError("Expected a workload with a requests array")
    size = max(max(prefill_rows), decode_context + 1)
    prompts = []
    seen = set()
    for row in data["requests"]:
        identity = row.get("context_sha256", row["request_id"])
        if identity in seen:
            continue
        encoded = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        if len(encoded) >= size:
            seen.add(identity)
            prompts.append((row["request_id"], encoded[:size]))
    chosen = prompts[rank * decode_batch:(rank + 1) * decode_batch]
    if len(chosen) != decode_batch:
        raise ValueError(f"Need {2 * decode_batch} distinct documents of at least {size} tokens, got {len(prompts)}")
    ids, tokens = zip(*chosen, strict=True)
    return list(ids), torch.tensor(tokens)


def route(states, weight, config):
    probs = F.linear(states, weight).softmax(dim=-1, dtype=torch.float32)
    weights, ids = probs.topk(config["num_experts_per_tok"], dim=-1)
    if config["norm_topk_prob"]:
        weights /= weights.sum(-1, keepdim=True)
    return ids.to(torch.int32).contiguous(), weights.contiguous()


@dataclass
class EP:
    activations: torch.Tensor
    ids: torch.Tensor
    weights: torch.Tensor
    rank: int
    w1: torch.Tensor
    w2: torch.Tensor
    group_size: int

    def __post_init__(self):
        total_experts = self.w1.shape[0] * self.group_size
        local_count = self.w1.shape[0]
        self.expert_map = torch.full((total_experts,), -1, dtype=torch.int32, device=self.activations.device)
        self.expert_map[self.rank * local_count:(self.rank + 1) * local_count] = torch.arange(
            local_count, dtype=torch.int32, device=self.activations.device,
        )
        self.total_experts = total_experts
        self.local_count = local_count
        sent_rows = tuple(
            torch.where((self.ids // local_count == owner).any(dim=1))[0]
            for owner in range(self.group_size)
        )
        self.send_counts = [len(indices) for indices in sent_rows]
        all_counts = [None] * self.group_size
        dist.all_gather_object(all_counts, self.send_counts)
        self.recv_counts = [row[self.rank] for row in all_counts]
        self.send_order = torch.cat(sent_rows)
        self.send_features = self.activations[self.send_order].contiguous()
        self.received_features = self.activations.new_empty((sum(self.recv_counts), self.activations.shape[-1]))
        self.received_ids = self.ids.new_empty((sum(self.recv_counts), self.ids.shape[-1]))
        self.received_weights = self.weights.new_empty((sum(self.recv_counts), self.weights.shape[-1]))
        self.combined_chunks = torch.empty_like(self.send_features)
        self.local_part = torch.empty_like(self.received_features)
        self.output = torch.empty_like(self.activations)
        self.dispatch()
        self.local_part = self.expert()

    @staticmethod
    def exchange(dest, source, recv, send):
        work = dist.all_to_all_single(dest, source, recv, send, async_op=True)
        work.wait()

    def dispatch_input_only(self):
        self.exchange(self.received_features, self.send_features, self.recv_counts, self.send_counts)
        return self.received_features

    def dispatch(self):
        self.dispatch_input_only()
        self.exchange(self.received_ids, self.ids[self.send_order].contiguous(), self.recv_counts, self.send_counts)
        self.exchange(self.received_weights, self.weights[self.send_order].contiguous(), self.recv_counts, self.send_counts)
        return self.received_features

    def expert(self):
        return fused_experts(
            self.received_features, self.w1, self.w2, self.received_weights, self.received_ids,
            global_num_experts=self.total_experts, expert_map=self.expert_map,
        )

    def combine(self, local_part=None):
        if local_part is None:
            local_part = self.local_part
        self.exchange(self.combined_chunks, local_part, self.send_counts, self.recv_counts)
        self.output.zero_().index_add_(0, self.send_order, self.combined_chunks)
        return self.output

    def run(self):
        self.dispatch()
        return self.combine(self.expert())

    def evidence(self):
        per_expert = torch.bincount(self.ids.flatten(), minlength=self.total_experts)
        dist.all_reduce(per_expert)
        counts = per_expert.cpu().tolist()
        owner_rows = [sum(counts[i:i + self.local_count]) for i in range(0, self.total_experts, self.local_count)]
        remote = sum(self.send_counts) - self.send_counts[self.rank]
        payload = self.activations.shape[-1] * self.activations.element_size()
        metadata = self.ids.shape[-1] * (self.ids.element_size() + self.weights.element_size())
        return {
            "local_token_rows": self.activations.shape[0],
            "owned_expert_ids": list(range(self.rank * self.local_count, (self.rank + 1) * self.local_count)),
            "local_expert_weight_bytes": sum(t.numel() * t.element_size() for t in (self.w1, self.w2)),
            "expert_rows_global": counts,
            "active_experts_global": sum(n > 0 for n in counts),
            "owner_rows_global": owner_rows,
            "owner_imbalance_max_div_mean": max(owner_rows) / statistics.mean(owner_rows),
            "send_rows_by_rank": self.send_counts,
            "recv_rows_by_rank": self.recv_counts,
            "dispatch_remote_activation_bytes": remote * payload,
            "dispatch_remote_routing_bytes": remote * metadata,
            "combine_remote_bytes": (sum(self.recv_counts) - self.recv_counts[self.rank]) * payload,
        }


class DeepEP(EP):
    """Decode/prefill operator replay using the original intranode HT transport."""

    def __init__(self, *args, buffer, **kwargs):
        self.buffer = buffer
        self.layout = None
        super().__init__(*args, **kwargs)

    def dispatch(self):
        if self.layout is None:
            import deep_ep

            self.transport_ids = self.ids.to(deep_ep.topk_idx_t)
            self.layout = self.buffer.get_dispatch_layout(self.transport_ids, self.total_experts)
        rank_rows, rdma_rows, expert_rows, in_rank, _ = self.layout
        x, ids, weights, _, self.handle, event = self.buffer.dispatch(
            self.activations, num_tokens_per_rank=rank_rows, num_tokens_per_rdma_rank=rdma_rows,
            is_token_in_rank=in_rank, num_tokens_per_expert=expert_rows,
            topk_idx=self.transport_ids, topk_weights=self.weights, expert_alignment=1,
            async_finish=True,
        )
        event.current_stream_wait()
        self.received_features = x
        outside = ((self.rank + 1) % self.group_size) * self.local_count
        self.received_ids = torch.where(ids >= 0, ids + self.rank * self.local_count, outside).to(torch.int32)
        self.received_weights = weights
        return x

    def combine(self, local_part=None):
        returned, _, event = self.buffer.combine(
            self.local_part if local_part is None else local_part, self.handle, async_finish=True,
        )
        event.current_stream_wait()
        return returned

    def evidence(self):
        details = super().evidence()
        metadata = self.ids.shape[-1] * (self.transport_ids.element_size() + self.weights.element_size())
        remote = sum(self.send_counts) - self.send_counts[self.rank]
        details["dispatch_remote_routing_bytes"] = remote * metadata
        return details


def deep_ep_buffer(hidden_bytes, group_size):
    import deep_ep

    dispatch = deep_ep.Buffer.get_dispatch_config(group_size)
    combine = deep_ep.Buffer.get_combine_config(group_size)
    nvlink_bytes = max(c.get_nvl_buffer_size_hint(hidden_bytes, group_size) for c in (dispatch, combine))
    return deep_ep.Buffer(dist.group.WORLD, num_nvl_bytes=nvlink_bytes, explicitly_destroy=True)


def backend_receipt(backend, buffer):
    if backend == "nccl_a2a":
        return {"process_group": dist.get_backend(), "primitive": "all_to_all_single"}
    import deep_ep
    import deep_ep_cpp

    binary = Path(deep_ep_cpp.__file__)
    return {
        "process_group": dist.get_backend(),
        "primitive": "deep_ep.Buffer.dispatch/combine",
        "package": deep_ep.__file__,
        "binary": str(binary),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "nvlink_buffer_bytes": buffer.num_nvl_bytes,
        "rdma_buffer_bytes": buffer.num_rdma_bytes,
        "num_sms": buffer.num_sms,
    }


def correctness(checkpoint, plans):
    """Temporarily compare against the unsharded fused experts, then free all reference weights."""
    count = checkpoint.config["num_experts"]
    # The unsharded reference exists outside all measured trials. The study
    # weights remain jointly sharded across the disjoint timed owners.
    full_w1, full_w2 = checkpoint.expert_weights(0, count)
    out = []
    for name, plan in plans.items():
        actual = plan.run().clone().float()
        expected = fused_experts(plan.activations, full_w1, full_w2, plan.weights, plan.ids).float()
        error = actual - expected
        out.append({
            "phase": name,
            "relative_l2": error.norm().item() / max(1e-12, expected.norm().item()),
            "max_abs": error.abs().max().item(),
            "bitwise_equal": torch.equal(actual, expected),
        })
    del full_w1, full_w2
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return out


def measure(fn_first, fn_second=None, mode="isolated", samples=30, warmup=8):
    stream_first = torch.cuda.Stream(priority=-1 if mode == "priority" else 0)
    stream_second = torch.cuda.Stream(priority=0)
    gate_stream = torch.cuda.Stream()
    if mode == "serial":
        stream_second = stream_first
    result = []
    for sample in range(samples + warmup):
        torch.cuda.synchronize()
        dist.barrier()
        gate = torch.cuda.Event(enable_timing=True)
        first_start, first_end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        second_start, second_end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        with torch.cuda.stream(gate_stream):
            # Keep Python enqueue skew outside the overlap window.
            torch.cuda._sleep(3000000)
            gate.record()
        with torch.cuda.stream(stream_first):
            stream_first.wait_event(gate)
            first_start.record()
            host_enqueue_start = time.perf_counter()
            held_first = fn_first()
            host_enqueue_ms = 1000 * (time.perf_counter() - host_enqueue_start)
            first_end.record()
        if fn_second is not None:
            with torch.cuda.stream(stream_second):
                stream_second.wait_event(gate)
                second_start.record()
                held_second = fn_second()
                second_end.record()
        first_end.synchronize()
        if fn_second is not None:
            second_end.synchronize()
        if sample >= warmup:
            first_interval = [gate.elapsed_time(event) for event in (first_start, first_end)]
            vals = {"first_interval_ms": first_interval, "first_ms": first_start.elapsed_time(first_end),
                    "first_host_enqueue_ms": host_enqueue_ms}
            if fn_second is not None:
                second_interval = [gate.elapsed_time(event) for event in (second_start, second_end)]
                vals.update(second_interval_ms=second_interval, second_ms=second_start.elapsed_time(second_end),
                            makespan_ms=max(first_interval[1], second_interval[1]) - min(first_interval[0], second_interval[0]))
            result.append(vals)
        del held_first
        if fn_second is not None:
            del held_second
    return result


@torch.inference_mode()
def run(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    group_size = dist.get_world_size()
    if group_size != 2:
        raise ValueError("Use exactly two physical GPU ranks for the pre-registered EP2 test")
    rank = dist.get_rank()
    checkpoint = Checkpoint(args.model, device)
    config = checkpoint.config
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    request_ids, tokens = choose_tokens(args.workload, tokenizer, rank, args.prefill_rows, args.decode_batch, args.decode_context)
    tokens = tokens.to(device)
    weights = checkpoint.attention_weights()
    prefix_tokens = tokens[0:1, :max(args.prefill_rows)]
    pref_x, (pref_q, pref_k, pref_v), norm_prefill = layer_inputs(weights, config, prefix_tokens)
    pref = norm_prefill(pref_x, pref_q, pref_k, pref_v).flatten(0, 1)
    dec_tokens = tokens[:, :args.decode_context + 1]
    dec_x, (dec_q, dec_k, dec_v), norm_decode = layer_inputs(weights, config, dec_tokens)
    decode_qkv = (dec_q[:, -1:].contiguous(), dec_k, dec_v)
    decode = norm_decode(dec_x[:, -1:], *decode_qkv).flatten(0, 1)
    local_count = config["num_experts"] // group_size
    w1, w2 = checkpoint.expert_weights(rank * local_count, (rank + 1) * local_count)
    phase_inputs = {"decode": decode} | {f"prefill_{m}": pref[:m] for m in args.prefill_rows}
    plans = {}
    buffer = deep_ep_buffer(config["hidden_size"] * 2, group_size) if args.backend == "deepep_ht" else None
    for name, states in phase_inputs.items():
        states = states.contiguous()
        ids, score = route(states, weights["mlp.gate.weight"], config)
        if buffer is None:
            plans[name] = EP(states, ids, score, rank, w1, w2, group_size)
        else:
            plans[name] = DeepEP(states, ids, score, rank, w1, w2, group_size, buffer=buffer)
    checks = correctness(checkpoint, plans)
    qualified = all(row["relative_l2"] <= 0.02 for row in checks)
    routing = {name: plan.evidence() for name, plan in plans.items()}
    digest = hashlib.sha256(checkpoint.index_path.read_bytes()).hexdigest()
    receipt = {
        "scope": "Real layer-0 Qwen3-30B-A3B BF16 EP2 fixed-routing replay",
        "physical_ep": True,
        "backend": args.backend,
        "backend_receipt": backend_receipt(args.backend, buffer),
        "model_revision": args.revision,
        "weight_index_sha256": digest,
        "workload_sha256": hashlib.sha256(args.workload.read_bytes()).hexdigest(),
        "rank": rank,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(local_rank),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "nccl": torch.cuda.nccl.version(),
        "peer_access": torch.cuda.can_device_access_peer(local_rank, 1 - local_rank),
        "request_ids": request_ids,
        "decode_context": args.decode_context,
        "decode_batch": args.decode_batch,
        "prefill_rows": args.prefill_rows,
        "model_config": config,
        "routing": routing,
        "correctness": checks,
        "qualified": qualified,
        "isolated": {},
        "pairs": [],
    }
    if qualified:
        dec = plans["decode"]
        dec_ops = {
            "decode_dispatch": dec.dispatch,
            "decode_combine": dec.combine,
            "decode_experts": dec.expert,
            "decode_attention": lambda: attention(*decode_qkv),
            "decode_ep_full": dec.run,
        }
        cases = list(dec_ops.items())
        for m in args.prefill_rows:
            plan = plans[f"prefill_{m}"]
            cases += [
                (f"prefill_{m}_experts", plan.expert),
                (f"prefill_{m}_dispatch", plan.dispatch),
                (f"prefill_{m}_combine", plan.combine),
                (f"prefill_{m}_ep_full", plan.run),
            ]
        for key, fn in cases:
            receipt["isolated"][key] = measure(fn, samples=args.samples)
        for m in args.prefill_rows:
            plan = plans[f"prefill_{m}"]
            combinations = [
                ("decode_dispatch", f"prefill_{m}_experts", dec.dispatch, plan.expert),
                ("decode_combine", f"prefill_{m}_experts", dec.combine, plan.expert),
                ("decode_attention", f"prefill_{m}_experts", dec_ops["decode_attention"], plan.expert),
                ("decode_experts", f"prefill_{m}_dispatch", dec.expert, plan.dispatch),
            ]
            for first, second, fn_first, fn_second in combinations:
                row = {"first": first, "second": second}
                for mode in ("serial", "streams", "priority"):
                    row[mode] = measure(fn_first, fn_second, mode=mode, samples=args.samples)
                receipt["pairs"].append(row)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"moe_ep_rank{rank}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if buffer is not None:
        buffer.destroy()
    dist.destroy_process_group()
    if not qualified:
        raise SystemExit("BF16 ownership/transport correctness gate failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--revision", default="ad44e777bcd18fa416d9da3bd8f70d33ebb85d39")
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefill-rows", nargs="+", type=int, default=[128, 512, 2048])
    parser.add_argument("--decode-batch", type=int, default=8)
    parser.add_argument("--decode-context", type=int, default=8192)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--backend", choices=("nccl_a2a", "deepep_ht"), default="nccl_a2a")
    args = parser.parse_args()
    if min(*args.prefill_rows, args.decode_batch, args.decode_context, args.samples) <= 0:
        parser.error("All dimensions and trial counts must be positive")
    run(args)


if __name__ == "__main__":
    main()
