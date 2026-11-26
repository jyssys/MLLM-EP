"""Control for host launch overhead in the real Full28 remote-attention replay."""
import argparse
import ctypes
import json
from pathlib import Path
import random
import time

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .remote_attention import PeerCopies, bytes_of
from .track_a import DecodeState, Decoder, collect_prompt, errors, positions
from ..mllm_epd_h200_prefill_heavy_batch_matched import runtime as rt


class GraphPeerCopies(PeerCopies):
    """Use UVA memcpy nodes, which can be recorded in a capture."""
    def __init__(self):
        super().__init__()
        self.memcpy = self.library.cudaMemcpyAsync
        self.memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                                ctypes.c_void_p]
        self.memcpy.restype = ctypes.c_int

    def __call__(self, destination, source, stream):
        if (destination.shape != source.shape or destination.dtype != source.dtype
                or not source.is_contiguous() or not destination.is_contiguous()):
            raise ValueError("Graph copy buffers must share contiguous shape and dtype")
        code = self.memcpy(destination.data_ptr(), source.data_ptr(), bytes_of(source),
                           4, stream.cuda_stream)  # cudaMemcpyDefault uses the UVA pointer kinds
        if code:
            raise RuntimeError(f"cudaMemcpyAsync(UVA) rc={code}")


def capture(decoder, source_cache, visual, token, position, layout, stream):
    graph = torch.cuda.CUDAGraph()
    length = source_cache.get_seq_length()
    state = DecodeState(source_cache.layers, length, visual, layout, reserve=1)
    stream.wait_stream(torch.cuda.current_stream(0))
    with torch.cuda.stream(stream):
        for _ in range(4):
            reference, _ = decoder.step(token, position, state, 0)
        stream.synchronize()
    if state.remote:
        ready = torch.cuda.Event(external=True)
        ready.record(torch.cuda.current_stream(1))
        ready.synchronize()
        # The preloaded shards are immutable in this fixed-address control.
        # An external dependency lets their producer remain outside the graph.
        for shard in state.remote:
            shard.ready = ready
    with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
        captured, _ = decoder.step(token, position, state, 0)
    graph.replay()
    torch.cuda.synchronize(0)
    return graph, captured, state, errors(reference, captured), decoder


def time_replay(graph, stream, steps):
    start = torch.cuda.Event(enable_timing=True)
    finish = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        begin = time.perf_counter()
        start.record()
        for _ in range(steps):
            graph.replay()
        finish.record()
        finish.synchronize()
    return {"cuda_ms_per_step": start.elapsed_time(finish) / steps,
            "wall_ms_per_step": (time.perf_counter() - begin) * 1000 / steps}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tier", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--id", required=True)
    args = parser.parse_args()
    peer = GraphPeerCopies()
    torch.cuda.set_device(0)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
    ).eval().to("cuda:0")
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    request, = [row for row in rt.read(args.tier)["requests"] if row["request_id"] == args.id]
    with args.native.open() as stream:
        answer, = [row for row in map(json.loads, stream) if row["request_id"] == args.id]
    prefill, visual, rope = collect_prompt(model, processor, request, "cuda:0")
    cache = prefill.past_key_values
    token = torch.tensor(answer["token_ids"][0], device="cuda:0", dtype=torch.long)
    pos = positions(cache.get_seq_length(), rope, 1)[0]
    stream = torch.cuda.Stream(device=0)
    captured = {}
    layouts = ("local", "visual", "balanced", "equal_count_prefix", "prefix")
    args.out.mkdir()
    for mode in layouts:
        try:
            # Cross-device buffers and internal event dependencies belong to one graph.
            captured[mode] = capture(Decoder(model, peer), cache, visual, token, pos, mode, stream)
            print(f"{mode} graph captured", flush=True)
        except RuntimeError as error:
            rt.write(args.out / "graph_failure.json", {"request_id": args.id, "layout": mode,
                                                       "error": str(error), "captured": list(captured)})
            raise
    timings = {layout: [] for layout in layouts}
    rng = random.Random(2026112706)
    for _ in range(20):
        order = list(layouts)
        rng.shuffle(order)
        for mode in order:
            timings[mode].append(time_replay(captured[mode][0], torch.cuda.default_stream(0), 1))
    local_logits = captured["local"][1]
    rt.write(args.out / "measurements.json", {
        "request_id": args.id, "prompt_rows": cache.get_seq_length(), "visual_rows": len(visual),
        "actual_native_output_tokens": answer["output_tokens"],
        "recurrent_payload": captured["local"][4].kernel.link_bytes,
        "source_SHA256": {name: rt.sha(Path(__file__).with_name(name)) for name in
                           ("attention_graph_control.py", "track_a.py", "remote_attention.py")},
        "layouts": {
            mode: {"samples": timings[mode], "direct_vs_graph_same_layout": details[3],
                   "logit_difference_vs_local_graph": errors(local_logits, details[1]),
                   "greedy_matches_local_graph": int(local_logits.argmax()) == int(details[1].argmax())}
            for mode, details in captured.items()
        },
        "scope": "Fixed-address one-token replay, Full28, original BF16 weights, FA3 and physical NVLink, "
                 "captured graph including KV writes, device copies and merge. "
                 "One synchronized replay per timing sample. No online scheduler or variable length.",
    })


if __name__ == "__main__":
    main()
