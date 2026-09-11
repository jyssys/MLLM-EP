#!/usr/bin/env python3
"""Live audit of stock dInfer and the minimal dispatch-capable EP4 substrate."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import socket
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


EXPECTED_VISIBLE = "4,5,6,7"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class StageEvents:
    def __init__(self, device: int):
        self.device = device
        self.events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}

    def wrap(self, name, function):
        def timed(*args, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            value = function(*args, **kwargs)
            end.record()
            self.events.setdefault(name, []).append((start, end))
            return value
        return timed

    def finish(self):
        torch.cuda.synchronize(self.device)
        return {
            name: [float(start.elapsed_time(end)) for start, end in pairs]
            for name, pairs in self.events.items()
        }


def worker(rank: int, world_size: int, port: int, mode: str, model_path: str, output_dir: str, sequence_length: int):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != EXPECTED_VISIBLE:
        raise RuntimeError("unsafe CUDA_VISIBLE_DEVICES")
    torch.cuda.set_device(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    from transformers import AutoConfig
    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from vllm.distributed import get_dp_group, get_ep_group, get_pp_group, get_tp_group
    from dinfer.model import LLaDAMoeModelLM

    if mode == "single_reference":
        tp_size, dp_size = 1, 1
    elif mode == "stock":
        tp_size, dp_size = world_size, 1
    elif mode == "true_ep4":
        tp_size, dp_size = 1, world_size
    else:
        raise ValueError(mode)

    parallel = ParallelConfig(
        tensor_parallel_size=tp_size,
        data_parallel_size=dp_size,
        data_parallel_size_local=dp_size,
        data_parallel_rank=rank if dp_size > 1 else 0,
        data_parallel_rank_local=rank if dp_size > 1 else 0,
        enable_expert_parallel=mode != "single_reference",
        disable_custom_all_reduce=True,
    )
    vllm_config = VllmConfig(parallel_config=parallel)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Initialize the physical four-rank WORLD before installing a DP-aware
    # vLLM config.  vLLM treats DP present during world initialization as
    # *external* data parallelism and would otherwise rewrite ranks 0..3 to
    # 0,5,10,15 in a virtual world of 16.  Model-parallel group construction
    # below still sees the config and creates the intended in-world DP4/EP4.
    distributed.init_distributed_environment(world_size, rank, "env://", rank, "nccl")
    with set_current_vllm_config(vllm_config):
        distributed.initialize_model_parallel(tp_size, backend="nccl")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = LLaDAMoeModelLM(config=config).eval()
        model.load_weights(model_path, torch_dtype=torch.bfloat16)
        if tp_size > 1:
            model.tensor_parallel(tp_size)
        model = model.to(torch.device(f"cuda:{rank}"))

        first_moe = model.model.layers[0].mlp.experts
        expert_map = first_moe.expert_map.detach().cpu() if first_moe.expert_map is not None else None
        local_ids = (torch.where(expert_map >= 0)[0].tolist() if expert_map is not None else list(range(config.num_experts)))
        expert_weight_bytes = int(sum(t.numel() * t.element_size() for t in first_moe.get_expert_weights()))
        one_expert_bytes = expert_weight_bytes // first_moe.local_num_experts

        stage = StageEvents(rank)
        ep_communicator = get_ep_group().device_communicator
        manager = (ep_communicator.all2all_manager
                   if ep_communicator is not None else None)
        manager_class = type(manager).__name__ if manager is not None else None
        if manager is not None:
            manager.dispatch = stage.wrap("dispatch", manager.dispatch)
            manager.combine = stage.wrap("combine", manager.combine)
        for layer in model.model.layers:
            layer.mlp.experts.quant_method.apply = stage.wrap(
                f"expert_layer_{layer.self_attn.layer_idx}", layer.mlp.experts.quant_method.apply
            )

        generator = torch.Generator(device=f"cuda:{rank}")
        generator.manual_seed(20260911)
        input_ids = torch.randint(0, config.vocab_size - 3, (1, sequence_length), generator=generator, device=f"cuda:{rank}")
        dist.broadcast(input_ids, src=0)

        def context():
            if dp_size == 1:
                return nullcontext()
            return set_forward_context(None, vllm_config, num_tokens=input_ids.numel())

        with torch.inference_mode(), context():
            model(input_ids, use_cache=False, num_logits_to_keep=1)
        stage.events.clear()

        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=False,
        )
        start_wall = time.perf_counter()
        with torch.inference_mode(), profiler, context():
            result = model(input_ids, use_cache=False, num_logits_to_keep=1,
                           output_hidden_states=True)
        torch.cuda.synchronize(rank)
        wall_ms = (time.perf_counter() - start_wall) * 1000.0
        stage_ms = stage.finish()

        logits = result.logits.detach().float()
        gathered = [torch.empty_like(logits) for _ in range(world_size)]
        dist.all_gather(gathered, logits)
        rank_agreement = max(float((item - gathered[0]).abs().max().item()) for item in gathered)

        hidden_rank_agreement = []
        for hidden in result.hidden_states:
            hidden_float = hidden.detach().float()
            hidden_gathered = [torch.empty_like(hidden_float) for _ in range(world_size)]
            dist.all_gather(hidden_gathered, hidden_float)
            hidden_rank_agreement.append(max(
                float((item - hidden_gathered[0]).abs().max().item())
                for item in hidden_gathered))

        # Replicated parameters must be bit-identical in TP1/DP4.  A compact
        # checksum makes a weight-loading fault distinguishable from an EP
        # dispatch/combine fault without serializing whole tensors.
        replicated_checksums = {}
        for name, tensor in {
            "embedding": model.model.embed_tokens.weight,
            "gate0": model.model.layers[0].mlp.gate.weight,
            "lm_head": model.lm_head.weight,
        }.items():
            checksum = torch.stack((tensor.float().sum(), tensor.float().square().sum()))
            checksum_gathered = [torch.empty_like(checksum) for _ in range(world_size)]
            dist.all_gather(checksum_gathered, checksum)
            replicated_checksums[name] = [item.cpu().tolist() for item in checksum_gathered]
        if rank == 0:
            torch.save(gathered[0].cpu(), output / f"{mode}_reference_logits.pt")
            torch.save([hidden.detach().cpu() for hidden in result.hidden_states],
                       output / f"{mode}_hidden_boundaries.pt")

        profiler.export_chrome_trace(str(output / f"{mode}_rank{rank}_trace.json"))
        event_rows = []
        for event in profiler.key_averages():
            cuda_total = float(getattr(event, "device_time_total", 0.0))
            cpu_total = float(getattr(event, "cpu_time_total", 0.0))
            if cuda_total > 0 or "nccl" in event.key.lower() or "all" in event.key.lower():
                event_rows.append({"key": event.key, "cuda_total_us": cuda_total, "cpu_total_us": cpu_total, "count": event.count})
        event_rows.sort(key=lambda row: row["cuda_total_us"], reverse=True)

        audit = {
            "mode": mode,
            "rank": rank,
            "physical_gpu": rank + 4,
            "world_size": world_size,
            "tp_size": get_tp_group().world_size,
            "tp_rank": get_tp_group().rank_in_group,
            "dp_size": get_dp_group().world_size,
            "dp_rank": get_dp_group().rank_in_group,
            "pp_size": get_pp_group().world_size,
            "ep_size": get_ep_group().world_size,
            "ep_rank": get_ep_group().rank_in_group,
            "fused_moe_tp_size": first_moe.tp_size,
            "fused_moe_dp_size": first_moe.dp_size,
            "fused_moe_ep_size": first_moe.ep_size,
            "local_expert_ids": local_ids,
            "local_expert_count": first_moe.local_num_experts,
            "local_expert_weight_bytes": expert_weight_bytes,
            "one_expert_weight_bytes": one_expert_bytes,
            "w13_shape": list(first_moe.w13_weight.shape),
            "w2_shape": list(first_moe.w2_weight.shape),
            "all2all_manager": manager_class,
            "all2all_backend": os.environ.get("VLLM_ALL2ALL_BACKEND", "naive"),
            "dispatch_calls": len(stage_ms.get("dispatch", [])),
            "combine_calls": len(stage_ms.get("combine", [])),
            "stage_ms": stage_ms,
            "forward_wall_ms_instrumented": wall_ms,
            "cross_rank_logit_max_abs": rank_agreement,
            "cross_rank_hidden_max_abs_by_boundary": hidden_rank_agreement,
            "replicated_parameter_checksums": replicated_checksums,
            "profiler_events": event_rows[:100],
        }
        (output / f"{mode}_rank{rank}_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        dist.barrier()
        distributed.destroy_model_parallel()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("single_reference", "stock", "true_ep4"), required=True)
    parser.add_argument("--sequence-length", type=int, default=32)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != EXPECTED_VISIBLE:
        raise SystemExit(f"refusing launch: CUDA_VISIBLE_DEVICES must be exactly {EXPECTED_VISIBLE}")
    if torch.cuda.device_count() != 4:
        raise SystemExit("exactly four visible GPUs required")
    world_size = 1 if args.mode == "single_reference" else 4
    mp.spawn(worker, args=(world_size, free_port(), args.mode, args.model, args.output, args.sequence_length), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
