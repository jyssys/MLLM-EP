"""Measure Qwen representations, physical copies, and concurrent transfer costs on 3 H200s."""

import argparse
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
from statistics import median
import threading
import time

from cuda.bindings import runtime as cuda
import torch

from ..encoder_assisted_decode_attention.cache import Arena, DecodePages, Placement
from ..encoder_assisted_decode_attention.model import Checkpoint, TokenDecoder, stage_weights
from ..encoder_assisted_decode_attention.peer_attention import PeerAccess, _cuda


MIB = 2**20
GIB = 2**30
ROLES = {"E": 0, "P": 1, "D": 2}
DEFAULT_CASES = (
    "lmms-lab__DocVQA__45142",
    "TIGER-Lab__Mantis-Eval__wc_22",
    "yubo2333__MMLongBench-Doc__658",
    "lmms-lab__MP-DocVQA__6771",
)
CONCURRENT_ROUTES = {
    "E": (("P", "E"), ("E", "P"), ("E", "D"), ("D", "E")),
    "P": (("E", "P"), ("P", "E"), ("P", "D")),
    "D": (("E", "D"), ("D", "E"), ("P", "D"), ("CPU", "D"), ("D", "CPU")),
}


def size(tensor):
    return tensor.numel() * tensor.element_size()


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def event():
    return torch.cuda.Event(enable_timing=True)


def timed(device, fn):
    with torch.cuda.device(device):
        begin, end = event(), event()
        start = time.perf_counter()
        begin.record()
        result = fn()
        end.record()
        end.synchronize()
        return result, {"gpu_ms": begin.elapsed_time(end), "wall_ms": 1e3 * (time.perf_counter() - start)}


def median_timing(records):
    return {key: median(record[key] for record in records) for key in records[0]}


def select_cases(manifest, names):
    rows = {row["request_id"]: row for row in json.loads(manifest.read_text())["requests"]}
    return [rows[name] for name in names]


class Copies:
    def __init__(self, nbytes):
        self.send = {role: torch.full((nbytes,), ordinal + 33, device=f"cuda:{ordinal}", dtype=torch.uint8)
                     for role, ordinal in ROLES.items()}
        self.send["CPU"] = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True).fill_(93)
        self.receive = {role: torch.empty(nbytes, dtype=torch.uint8, device=payload.device, pin_memory=(role == "CPU"))
                        for role, payload in self.send.items()}
        self.streams = {role: torch.cuda.Stream(device=f"cuda:{ordinal}") for role, ordinal in ROLES.items()}
        for first, second in ((0, 1), (0, 2), (1, 2)):
            PeerAccess(first, second)
        for ordinal in ROLES.values():
            torch.cuda.synchronize(ordinal)

    @staticmethod
    def stream_role(source, destination):
        return destination if destination != "CPU" else source

    def submit(self, source, destination, nbytes, stream):
        src = self.send[source].data_ptr()
        dst = self.receive[destination].data_ptr()
        handle = cuda.cudaStream_t(stream.cuda_stream)
        if source == "CPU":
            _cuda(cuda.cudaMemcpyAsync, dst, src, nbytes, cuda.cudaMemcpyKind.cudaMemcpyHostToDevice, handle)
        elif destination == "CPU":
            _cuda(cuda.cudaMemcpyAsync, dst, src, nbytes, cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost, handle)
        else:
            _cuda(cuda.cudaMemcpyPeerAsync, dst, ROLES[destination], src, ROLES[source], nbytes, handle)

    def measure(self, source, destination, nbytes, count, *, stream=None, origin=None):
        stream = stream or self.streams[self.stream_role(source, destination)]
        with torch.cuda.device(stream.device), torch.cuda.stream(stream):
            if origin is not None:
                stream.wait_event(origin)
            begin, end = event(), event()
            start = time.perf_counter()
            begin.record()
            for _ in range(count):
                self.submit(source, destination, nbytes, stream)
            end.record()
            end.synchronize()
            wall_ms = 1e3 * (time.perf_counter() - start)
        result = {"gpu_ms": begin.elapsed_time(end), "wall_ms": wall_ms, "copies": count}
        if origin is not None:
            result["window_ms"] = (origin.elapsed_time(begin), origin.elapsed_time(end))
        return result

    def verify(self, source, destination, nbytes):
        value = 93 if source == "CPU" else ROLES[source] + 33
        dest = self.receive[destination]
        for offset in (0, nbytes // 2, nbytes - 1):
            if dest[offset].item() != value:
                raise RuntimeError(f"Payload check failed for {source}→{destination} at byte {offset}")


class Models:
    def __init__(self, path):
        self.checkpoint = Checkpoint(path, "qwen-vl")
        self.devices = {role: torch.device(f"cuda:{i}") for role, i in ROLES.items()}
        self.prefill = self.checkpoint.replica(self.devices["P"], vision_device=self.devices["E"])
        self.encoder = self.prefill.model.visual
        self.decoder = TokenDecoder(self.checkpoint, self.checkpoint.replica(self.devices["D"]), self.devices["D"])
        self.arena = Arena(self.devices["D"], self.decoder.n_layers, self.decoder.kv_heads, self.decoder.dim, 3)

    def weight_bytes(self):
        return {"E": stage_weights(self.encoder),
                "P": stage_weights(self.prefill.model.language_model) + stage_weights(self.prefill.lm_head),
                "D": stage_weights(self.decoder.text) + stage_weights(self.decoder.lm_head)}

    def prepare(self, row):
        prepared = self.checkpoint.preprocess(row, workload="realistic")
        e, p = (self.devices[role] for role in ("E", "P"))
        pixel = prepared["pixel"].to(e, dtype=torch.bfloat16)
        grid = prepared["grid"].to(e)

        def encode():
            return self.encoder(pixel, grid_thw=grid).pooler_output

        pre_merge = []
        hook = self.encoder.merger.register_forward_pre_hook(
            lambda _module, args: pre_merge.append((args[0].shape, args[0].element_size(), str(args[0].dtype))))
        try:
            visual = encode()
        finally:
            hook.remove()
        if len(pre_merge) != 1:
            raise RuntimeError("Expected one vision merger input")
        torch.cuda.synchronize(e)
        tokens = prepared["ids"].to(p)
        positions = prepared["positions"].to(p)
        native_pos, native_delta = self.prefill.model.get_rope_index(
            tokens, prepared["modes"].to(p), image_grid_thw=prepared["grid"].to(p))
        if not torch.equal(positions, native_pos) or int(native_delta[0]) != prepared["delta"]:
            raise RuntimeError(f"MRoPE mismatch for {row['request_id']}")
        sequence = self.prefill.model.language_model.embed_tokens(tokens)
        sequence[0].index_copy_(0, prepared["visual"].to(p), visual.to(p))
        torch.cuda.synchronize(p)
        prefix_rows = int(prepared["visual"][-1]) + 2
        if tokens.shape[-1] <= prefix_rows or tokens[0, prefix_rows - 1].item() != (
                self.checkpoint.tokenizer.convert_tokens_to_ids("<|vision_end|>")):
            raise RuntimeError(f"No image-complete prefix with a remaining suffix for {row['request_id']}")

        def prefill(rows=None):
            model = self.prefill.model.language_model
            return model(inputs_embeds=sequence[:, :rows], position_ids=positions[..., :rows], use_cache=True)

        return prepared, visual, pre_merge[0], prefix_rows, encode, prefill

    def decode_clip(self, cache, row_count, rope_delta, initial):
        pages = self.arena.acquire(row_count + 16)
        self.arena.wait(self.arena.enqueue(cache, torch.arange(row_count), pages))
        placement = Placement(pages, [], row_count, 0, 1, "local")

        def decode():
            request = DecodeRequest(placement, [initial])
            for index in range(16):
                state = DecodePages(self.arena, None, [request], None, self.decoder.heads, self.decoder.dim)
                sampled, _, _ = self.decoder(request.tokens[-1:], [row_count + rope_delta + index], state)
                request.tokens.extend(sampled)
            return request.tokens

        return decode, pages


@dataclass
class DecodeRequest:
    placement: Placement
    tokens: list[int]


def case_economics(models, row, repeats):
    prepared, visual, (pre_merge, pre_merge_elem, pre_merge_dtype), prefix_rows, encode, prefill = models.prepare(row)
    e, p = (models.devices[role] for role in ("E", "P"))
    uncached = {key: [] for key in ("encode", "full_prefill", "image_complete_prefix_prefill")}
    for _ in range(repeats):
        encoded, timed_encode = timed(e, encode)
        uncached["encode"].append(timed_encode)
        full, pf = timed(p, prefill)
        uncached["full_prefill"].append(pf)
        prefix, pp = timed(p, lambda: prefill(prefix_rows))
        uncached["image_complete_prefix_prefill"].append(pp)
    individual, full_bytes = [], 0
    elem = visual.element_size()
    for item, dims in zip(row["images"], prepared["grid"].tolist(), strict=True):
        patch_rows = dims[0] * dims[1] * dims[2]
        expected_visual_rows = item["visual_tokens"]
        raw_bytes = Path(item["path"]).stat().st_size
        full_bytes += raw_bytes
        individual.append({"raw_file_bytes": raw_bytes, "processed_pixel_float32_bytes": (
            patch_rows * prepared["pixel"].shape[-1] * prepared["pixel"].element_size()),
            "processed_pixel_bf16_bytes": patch_rows * prepared["pixel"].shape[-1] * elem,
            "encoder_pre_merger_bytes": patch_rows * pre_merge[-1] * pre_merge_elem,
            "lm_ready_embedding_bytes": expected_visual_rows * visual.shape[-1] * elem,
            "visual_rows": expected_visual_rows, "grid_thw": dims})
    if sum(item["lm_ready_embedding_bytes"] for item in individual) != size(visual):
        raise RuntimeError("Measured visual embedding shape did not agree with manifest grids")
    if pre_merge[0] != sum(item["grid_thw"][0] * item["grid_thw"][1] * item["grid_thw"][2] for item in individual):
        raise RuntimeError("Measured vision merger input did not agree with patch grids")
    full_cache = full.past_key_values
    part_cache = prefix.past_key_values
    all_kv_bytes = sum(size(value) for layer in full_cache.layers for value in (layer.keys, layer.values))
    prefix_kv_bytes = sum(size(value) for layer in part_cache.layers for value in (layer.keys, layer.values))
    for kv_bytes, count in ((all_kv_bytes, prepared["ids"].shape[-1]), (prefix_kv_bytes, prefix_rows)):
        if kv_bytes != count * models.arena.bytes_per_row:
            raise RuntimeError("KV state geometry and actual tensors disagree")
    cached_repeat = {"bitwise_equal": torch.equal(visual, encoded),
                     "max_abs": float((visual.float() - encoded.float()).abs().max())}
    isolated = []
    for i in sorted({0, len(individual) - 1}):
        pixel_offset = sum(item["processed_pixel_float32_bytes"] for item in individual[:i]) // (
            prepared["pixel"].shape[-1] * prepared["pixel"].element_size())
        patch_rows = int(prepared["grid"][i].prod())
        pixel_slice = prepared["pixel"][pixel_offset:pixel_offset + patch_rows].to(e, dtype=torch.bfloat16)
        grid_slice = prepared["grid"][i:i + 1].to(e)
        output = models.encoder(pixel_slice, grid_thw=grid_slice).pooler_output
        visual_offset = sum(item["visual_rows"] for item in individual[:i])
        bundled = visual[visual_offset:visual_offset + len(output)]
        residual = (output.float() - bundled.float()).flatten()
        isolated.append({"image_index": i, "bitwise_equal": torch.equal(output, bundled),
                         "relative_l2": float(residual.norm() / bundled.float().norm().clamp_min(1e-12)),
                         "max_abs": float(residual.abs().max())})
    first = int(models.prefill.lm_head(full.last_hidden_state[:, -1]).argmax(-1)[0])
    receipt = {
        "evidence": "MEASURED_PHYSICAL", "request_id": row["request_id"], "images": individual,
        "prompt_rows": prepared["ids"].shape[-1], "image_complete_prefix_rows": prefix_rows,
        "R0_raw_file_bytes": full_bytes, "R1_processed_pixel_float32_bytes": size(prepared["pixel"]),
        "R1_processed_pixel_bf16_bytes": sum(item["processed_pixel_bf16_bytes"] for item in individual),
        "R2_encoder_pre_merger_bytes": sum(item["encoder_pre_merger_bytes"] for item in individual),
        "R2_encoder_pre_merger_dtype": pre_merge_dtype,
        "R3_lm_ready_embedding_bytes": size(visual), "R4_full_prompt_kv_bytes": all_kv_bytes,
        "R4_image_complete_prefix_kv_bytes": prefix_kv_bytes,
        "R5_decode_kv_bytes_per_row": models.arena.bytes_per_row,
        "R5_16_decode_rows_bytes": 16 * models.arena.bytes_per_row,
        "cached_repeat": cached_repeat, "image_only_vs_bundled": isolated,
        "recompute": uncached, "median_recompute": {name: median_timing(values) for name, values in uncached.items()},
    }
    return receipt, (encode, prefill, full_cache, prepared["delta"], first)


def run_overlap(copies, stage, compute, route=None, nbytes=None):
    device = torch.device(f"cuda:{ROLES[stage]}")
    stream = torch.cuda.Stream(device=device)
    transfer_stream = copies.streams[stage]
    origin = event()
    origin.record(torch.cuda.current_stream(device))
    gate = threading.Barrier(2, timeout=60)
    recorded, errors = [], []

    def worker():
        try:
            with torch.inference_mode(), torch.cuda.device(device), torch.cuda.stream(stream):
                stream.wait_event(origin)
                begin, end = event(), event()
                gate.wait()
                then = time.perf_counter()
                begin.record()
                result = compute()
                end.record()
                end.synchronize()
                recorded.append({"gpu_ms": begin.elapsed_time(end), "wall_ms": 1e3 * (time.perf_counter() - then),
                                 "window_ms": (origin.elapsed_time(begin), origin.elapsed_time(end))})
                del result
        except BaseException as error:
            errors.append(error)
            gate.abort()

    task = threading.Thread(target=worker, daemon=True)
    task.start()
    try:
        gate.wait()
        copy = None
        if route is not None:
            count = max(1, GIB // nbytes)
            copy = copies.measure(*route, nbytes, count, stream=transfer_stream, origin=origin)
    finally:
        task.join(120)
    if task.is_alive() or errors:
        raise RuntimeError("Concurrent compute worker did not finish") from (errors[0] if errors else None)
    work, = recorded
    overlap = 0.
    if copy is not None:
        overlap = max(0., min(work["window_ms"][1], copy["window_ms"][1]) -
                      max(work["window_ms"][0], copy["window_ms"][0]))
    return {"compute": work, "copy": copy, "enclosure_overlap_ms": overlap}


def sizes_with_representations(case_receipts):
    sizes = {multiple * MIB: [f"{multiple} MiB"] for multiple in (1, 16, 64, 256, 1024)}
    for case in case_receipts:
        for key in ("R3_lm_ready_embedding_bytes", "R4_image_complete_prefix_kv_bytes"):
            sizes.setdefault(case[key], []).append(f"{case['request_id']}:{key}")
    return sizes


def benchmark(args, write):
    models = Models(args.model)
    rows = select_cases(args.manifest, args.request_id or DEFAULT_CASES)
    results = []
    clip = None
    for i, row in enumerate(rows):
        receipt, measured = case_economics(models, row, args.repeats)
        results.append(receipt)
        write({"kind": "representation", **receipt})
        if i == min(2, len(rows) - 1):
            encode, prefill, cache, delta, first = measured
            decode, pages = models.decode_clip(cache, receipt["prompt_rows"], delta, first)
            _out, metrics = timed(models.devices["D"], decode)
            write({"kind": "decode_16", "evidence": "MEASURED_PHYSICAL", "request_id": row["request_id"],
                   "active_kv": "D only, original Full28 + paged FA3", **metrics})
            clip = {"E": encode, "P": prefill, "D": decode}, pages, row["request_id"]
    if clip is None:
        raise RuntimeError("At least one representative row is required")
    callbacks, pages, compute_case = clip
    sizes = sizes_with_representations(results)
    devices = {role: {"name": torch.cuda.get_device_name(ordinal),
                      "total_bytes": torch.cuda.get_device_properties(ordinal).total_memory}
               for role, ordinal in ROLES.items()}
    write({"kind": "model_allocation", "evidence": "MEASURED_PHYSICAL", "devices": devices,
           "weight_bytes": models.weight_bytes(), "arena_bytes": size(models.arena.cache),
           "boundary": "Before transfer benchmark buffers, weights and D page arena coexist",
           "torch_memory": {role: {"allocated": torch.cuda.memory_allocated(ordinal),
                                   "reserved": torch.cuda.memory_reserved(ordinal),
                                   "peak_allocated": torch.cuda.max_memory_allocated(ordinal)}
                            for role, ordinal in ROLES.items()}})
    copies = Copies(max(sizes))
    for source in copies.send:
        for destination in copies.receive:
            if source == destination:
                continue
            for nbytes, labels in sorted(sizes.items()):
                count = min(16, max(1, 256 * MIB // nbytes))
                copies.measure(source, destination, nbytes, 2)
                timings = [copies.measure(source, destination, nbytes, count) for _ in range(args.repeats)]
                copies.verify(source, destination, nbytes)
                write({"kind": "transfer", "evidence": "MEASURED_PHYSICAL", "source": source,
                       "destination": destination, "bytes": nbytes, "size_labels": labels, "samples": timings,
                       "scope": "Contiguous preallocated buffers, CUDA P2P or pinned host DMA, no pack/unpack"})
    if not args.no_interference:
        for stage, pairs in CONCURRENT_ROUTES.items():
            stream = copies.streams[stage]
            compute = callbacks[stage]
            run_overlap(copies, stage, compute)
            for source, destination in pairs:
                for nbytes in (16 * MIB, 256 * MIB):
                    samples = []
                    count = GIB // nbytes
                    for rep in range(args.repeats):
                        if rep % 2 == 0:
                            before = run_overlap(copies, stage, compute)
                            isolated = copies.measure(source, destination, nbytes, count, stream=stream)
                            with_copy = run_overlap(copies, stage, compute, (source, destination), nbytes)
                        else:
                            with_copy = run_overlap(copies, stage, compute, (source, destination), nbytes)
                            isolated = copies.measure(source, destination, nbytes, count, stream=stream)
                            before = run_overlap(copies, stage, compute)
                        samples.append({"isolated_compute": before, "isolated_copy": isolated, "with_copy": with_copy})
                    copies.verify(source, destination, nbytes)
                    write({"kind": "interference", "evidence": "MEASURED_PHYSICAL", "active_stage": stage,
                           "compute_request": compute_case, "source": source, "destination": destination,
                           "bytes_per_copy": nbytes, "samples": samples,
                           "scope": "CUDA-event enclosures on the same GPU, not individual kernel intervals"})
    models.arena.release(pages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--request-id", action="append")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-interference", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if torch.cuda.device_count() != 3 or any(torch.cuda.get_device_name(i) != "NVIDIA H200" for i in range(3)):
        parser.error("Run with a lease selecting three H200 GPUs in E,P,D order")
    with args.out.open("x") as file, torch.inference_mode():
        def write(row):
            file.write(json.dumps(row, allow_nan=False) + "\n")
            file.flush()

        write({"kind": "config", "model": str(args.model.resolve()), "manifest": str(args.manifest.resolve()),
               "model_config_sha256": file_hash(args.model / "config.json"),
               "model_weight_index_sha256": file_hash(args.model / "model.safetensors.index.json"),
               "manifest_sha256": file_hash(args.manifest), "source_sha256": file_hash(Path(__file__)),
               "request_ids": args.request_id or DEFAULT_CASES, "repeats": args.repeats,
               "gpu_uuids": os.environ.get("CUDA_VISIBLE_DEVICES"), "cpu_affinity": sorted(os.sched_getaffinity(0)),
               "software": {name: version(name) for name in ("torch", "transformers", "vllm", "cuda-bindings")},
               "boundary": "Isolated HF Qwen SDPA encode/prefill, eager original paged FA3 decode, no online claim"})
        benchmark(args, write)
        write({"kind": "complete"})


if __name__ == "__main__":
    main()
