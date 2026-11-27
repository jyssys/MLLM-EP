"""Isolate native local and physically split BF16 FA3 at matched decode shapes."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

import torch

from .download_models import CHECKPOINTS
from .peer_attention import AssistedAttention, KVShard, PeerAccess, _bytes, _capture, fa3



def pack(sequences, device, layout):
    lengths = tuple(k.shape[0] for k, _ in sequences)
    if layout == "packed":
        keys, values = (torch.cat(rows).contiguous().to(device) for rows in zip(*sequences, strict=True))
        return KVShard.packed(keys, values, lengths)
    page_size = 128
    pages = tuple((length + page_size - 1)//page_size for length in lengths)
    size = sum(pages)
    reference = sequences[0][0]
    source = reference.device
    shape = (size, page_size, *reference.shape[1:])
    keys = torch.empty(shape, dtype=reference.dtype, device=source)
    values = torch.empty_like(keys)
    indices = []
    offset = 0
    for (k, v), count in zip(sequences, pages, strict=True):
        indices.append(list(range(offset, offset + count)) + [0] * (max(pages) - count))
        keys[offset:offset + count].flatten(0, 1)[:len(k)].copy_(k)
        values[offset:offset + count].flatten(0, 1)[:len(v)].copy_(v)
        offset += count
    table = torch.tensor(indices, dtype=torch.int32, device=device)
    return KVShard.paged(keys.to(device), values.to(device), lengths, table)


def reference_data(model_path, manifest, ids, device):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
    from ..mllm_specific_disaggregation_discovery.track_a import collect_prompt

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
    ).eval().to(device)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    rows = {row["request_id"]: row for row in json.loads(manifest.read_text())["requests"]}
    text = model.model.language_model
    cfg = text.config
    heads, kv_heads = cfg.num_attention_heads, cfg.num_key_value_heads
    dim = cfg.hidden_size//heads
    base_queries, layers, masks = [], [], []
    for ident in ids:
        prefill, visual, rope = collect_prompt(model, processor, rows[ident], device)
        state = text.embed_tokens(prefill.logits[0, 0].argmax().view(1, 1))
        length = prefill.past_key_values.get_seq_length()
        pos = torch.full((3,1,1), length + rope, device=device, dtype=torch.long)
        cs, sn = text.rotary_emb(state, pos)
        layer = text.layers[0]
        normalized = layer.input_layernorm(state)
        sa = layer.self_attn
        q = sa.q_proj(normalized).view(1,heads,1,dim)
        k = sa.k_proj(normalized).view(1,kv_heads,1,dim)
        v = sa.v_proj(normalized).view(1,kv_heads,dim)
        q, k = apply_multimodal_rotary_pos_emb(q,k,cs,sn,cfg.rope_parameters["mrope_section"])
        layer0 = prefill.past_key_values.layers[0]
        keys = torch.cat((layer0.keys[0].transpose(0,1),k[:, :, 0])).contiguous()
        values = torch.cat((layer0.values[0].transpose(0,1),v)).contiguous()
        mask = torch.zeros(len(keys), dtype=torch.bool, device=device)
        mask[visual.to(device)] = True
        layers.append((keys, values))
        masks.append(mask)
        base_queries.append(q[:, :, 0].contiguous())
        del prefill
    return torch.cat(base_queries), layers, masks


def fixture_data(path, device):
    captured = torch.load(path, map_location="cpu", weights_only=True)
    provenance = captured["provenance"]
    model, revision, _ = CHECKPOINTS["internvl3_8b"]
    if (provenance["checkpoint"] != model or provenance["revision"] != revision
            or provenance["layer_index"] != 0 or provenance["capture_point"] != "native_post_rope_first_decode"
            or not isinstance(provenance["request_id"], str)):
        raise ValueError("Unexpected InternVL Q/K/V fixture provenance")
    q, k, v, visual = (captured[name] for name in ("q", "k", "v", "visual"))
    if (q.shape != (1, 28, 128) or k.ndim != 3 or k.shape[1:] != (4, 128)
            or k.shape != v.shape or any(tensor.dtype != torch.bfloat16 for tensor in (q, k, v))
            or visual.dtype != torch.bool or visual.shape != (len(k),) or not visual.any() or visual[-1]):
        raise ValueError("Fixture requires BF16 InternVL B1 and its original visual prefix mask")
    with path.open("rb") as stream:
        receipt = provenance | {"fixture_sha256": hashlib.file_digest(stream, "sha256").hexdigest()}
    return (q.to(device), [(k.to(device), v.to(device))], [visual.to(device)]), receipt


def generated_data(batch, context, shape, device, rng):
    heads, kv_heads, dim = shape
    query = torch.randn((batch,heads,dim), device=device, dtype=torch.bfloat16, generator=rng)
    tensors = [torch.randn((batch,context+1,kv_heads,dim), device=device, dtype=torch.bfloat16,
                           generator=rng) for _ in range(2)]
    return query, list(zip(*tensors, strict=True)), None


def partition(data, ratio, masks, placement):
    local, remote = [], []
    for i, (k,v) in enumerate(data):
        if placement == "visual_first":
            mask = masks[i]
            if ratio < 1:
                indices = mask.nonzero().flatten()
                keep = int(len(indices) * ratio)
                mask = torch.zeros_like(mask)
                mask[indices[:keep]] = True
            local.append((k[~mask],v[~mask]))
            remote.append((k[mask],v[mask]))
        else:
            # Partition only the frozen prompt. The final one-query write remains
            # on D at every ratio, including all of the frozen prompt offloaded.
            rows = int(masks[i].sum()) if placement == "equal_count_prefix" else len(k) - 1
            before = int(rows * ratio)
            local.append((k[before:],v[before:]))
            remote.append((k[:before],v[:before]))
    return local, remote


def timed(fn, device, loops):
    with torch.cuda.device(device):
        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        wall = time.perf_counter_ns()
        begin.record()
        for _ in range(loops):
            fn()
        end.record()
        end.synchronize()
        return {"cuda_us": begin.elapsed_time(end)*1000/loops,
                "host_us": (time.perf_counter_ns()-wall)/1000/loops}


def link_counters(decode, helper):
    import pynvml

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if max(decode, helper) >= len(visible) or not all(code.startswith("GPU-") for code in visible):
        return None
    try:
        pynvml.nvmlInit()
        try:
            readings = {}
            fields = {"tx": pynvml.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX,
                      "rx": pynvml.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX}
            for role, index in (("D",decode),("E",helper)):
                handle = pynvml.nvmlDeviceGetHandleByUUID(visible[index])
                for field, code in fields.items():
                    data = pynvml.nvmlDeviceGetFieldValues(handle,[(code,link) for link in range(18)])
                    if len(data) != 18 or any(x.nvmlReturn != pynvml.NVML_SUCCESS for x in data):
                        return None
                    readings[f"{role}_{field}_bytes"] = sum(int(x.value.ullVal)*1024 for x in data)
            return readings
        finally:
            pynvml.nvmlShutdown()
    except (AttributeError,OSError,pynvml.NVMLError):
        return None


def verify_link(fn, args, path):
    baseline = link_counters(args.decode_device,args.helper_device)
    if baseline is None:
        return {"path":path,"status":"NOT_MEASURED"}
    timed(fn,args.decode_device,args.link_repeats)
    finish = link_counters(args.decode_device,args.helper_device)
    delta = None if finish is None else {key:finish[key]-previous for key,previous in baseline.items()}
    if delta is None or min(delta.values()) < 0:
        return {"path":path,"status":"NOT_MEASURED"}
    return {"path":path,"status":"MEASURED_GPU_NVLINK_DATA","steps":args.link_repeats,"delta":delta}


def compare(reference, estimate):
    x = reference.float().flatten()
    y = estimate.float().flatten()
    score = dict(
        relative_l2=float((x-y).norm()/x.norm().clamp_min(1e-20)),
        max_abs=float((x-y).abs().max()),
        cosine=float(torch.nn.functional.cosine_similarity(x,y,dim=0)),
        bitwise=bool(torch.equal(reference,estimate)),
    )
    if not all(math.isfinite(score[key]) for key in ("relative_l2","max_abs","cosine")):
        raise RuntimeError("Non-finite attention result")
    return score


class Events:
    def __init__(self):
        self.events = {}

    def __call__(self, name):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.events[name] = event

    def intervals(self):
        times = {
            "query_copy": ("remote_begin","query_transferred"),
            "remote_attention": ("query_transferred","remote_attention_done"),
            "result_copy": ("remote_attention_done","remote_result_transferred"),
            "local_attention": ("local_begin","local_attention_done"),
            "wait_for_remote": ("local_attention_done","merge_begin"),
            "merge": ("merge_begin","merge_done"),
        }
        return {name:self.events[a].elapsed_time(self.events[b])*1000 for name,(a,b) in times.items()}


def _local_capture(op, query, full):
    stream = torch.cuda.Stream(device=op.decode_home)
    captured, result = _capture(op.decode_device, stream, lambda: op.local(query, full))
    return captured, result


def run_case(geometry, args, peer, rng):
    query, data, masks = geometry
    producer = query.device
    d = torch.device("cuda",args.decode_device)
    e = torch.device("cuda",args.helper_device)
    kv_heads = data[0][0].shape[1]
    query = query.to(d).contiguous()
    scale = query.shape[-1] ** -0.5
    q_starts = torch.arange(len(data)+1, dtype=torch.int32, device=d)
    placement_start = time.perf_counter_ns()
    full = pack(data,d,args.layout)
    torch.cuda.synchronize(d)
    full_placement = (time.perf_counter_ns() - placement_start) / 1e6
    full_ref, full_lse = fa3(query, full, q_starts, scale, lse=True)
    splits = args.local_splits
    local_graph = {}
    raw = []
    for n in splits:
        op = AssistedAttention(tuple(query.shape),scale,d,e,peer=peer,num_splits=n)
        graph, _ = _local_capture(op,query,full)
        graph.replay()
        raw.append({"num_splits":n,"samples":[timed(graph.replay,d,args.calls) for _ in range(args.repeats)]})
        local_graph[n] = (graph,op)
    best = min(raw,key=lambda item:statistics.median(p["cuda_us"] for p in item["samples"]))
    local_fn = local_graph[best["num_splits"]][0].replay
    rows = []
    modes = [(ratio,"generic_prefix") for ratio in args.ratios]
    if masks:
        for placement in ("visual_first","equal_count_prefix"):
            modes.extend((ratio,placement) for ratio in args.ratios if ratio)
    random.Random(args.seed + len(data)).shuffle(modes)
    for ratio, placement in modes:
        if ratio == 0.:
            continue
        start = time.perf_counter_ns()
        d_data, e_data = partition(data,ratio,masks,placement)
        local, remote = pack(d_data,d,args.layout),pack(e_data,e,args.layout)
        torch.cuda.synchronize(d)
        torch.cuda.synchronize(e)
        initial = (time.perf_counter_ns()-start)/1e6
        op = AssistedAttention(tuple(query.shape),scale,d,e,peer=peer)
        result = op.split(query,local,remote)
        torch.cuda.synchronize(d)
        correctness = compare(full_ref,result)
        if remote.max_length:
            _, local_lse = op.local_part(query,local)
            merged_lse = torch.logaddexp(local_lse,op.remote_lse)
            correctness["lse_max_abs"] = float((full_lse - merged_lse).abs().max())
        if correctness["relative_l2"] > .008 or correctness["cosine"] < .9999:
            raise ValueError(f"Split result differs from local FA3: {correctness}")
        if correctness.get("lse_max_abs", 0.) > 2e-4:
            raise ValueError(f"Split normalization differs from local FA3: {correctness}")
        diag = []
        for _ in range(5):
            marker = Events()
            op.split(query,local,remote,record=marker)
            torch.cuda.synchronize(d)
            diag.append(marker.intervals())
        plans = {"local_best_graph": local_fn,
                 "local_eager": lambda: op.local(query,full),
                 "split_eager": lambda: op.split(query,local,remote)}
        graph, graph_error, graph_correctness, changed_correctness = None, None, None, None
        if not args.no_split_graph:
            try:
                graph = op.graph(query,local,remote,transport=args.graph_transport)
                graph_out = graph()
                torch.cuda.synchronize(d)
                graph_correctness = compare(full_ref,graph_out)
                if graph_correctness["relative_l2"] > .008:
                    raise ValueError(f"Graph composition differs from local FA3: {graph_correctness}")
                # Catch graph captures that accidentally hold an old query or a
                # stale output from warmup by changing the query at the same address.
                original = query.clone()
                query.mul_(-1)
                witness = op.local(query,full)
                transformed = graph()
                torch.cuda.synchronize(d)
                changed_correctness = compare(witness,transformed)
                query.copy_(original)
                if changed_correctness["relative_l2"] > .008:
                    raise ValueError(f"Graph did not follow updated decode Q: {changed_correctness}")
                plans["split_graph"] = graph
            except (RuntimeError, ValueError) as error:
                if args.require_split_graph:
                    raise
                graph_error = str(error)
                if graph is not None:
                    graph.close()
                    graph = None
        timings = defaultdict(list)
        for repetition in range(args.repeats):
            order = list(plans)
            rng.shuffle(order)
            for name in order:
                timings[name].append(timed(plans[name],d,args.calls))
        link = None
        if args.link_repeats:
            chosen = "split_graph" if graph is not None else "split_eager"
            link = verify_link(plans[chosen],args,chosen)
        if graph is not None:
            graph.close()
        rows.append({
            "requested_remote_fraction": ratio, "placement": placement,
            "remote_lengths": remote.lengths, "physical_local_lengths": local.lengths,
            "partition_setup_ms": initial, "source_device": str(producer),
            "destination_bytes": {"decode": _bytes(local.keys)+_bytes(local.values),
                                  "helper": _bytes(remote.keys)+_bytes(remote.values)},
            "correctness": correctness, "graph_correctness": graph_correctness,
            "graph_changed_query_correctness": changed_correctness, "graph_error": graph_error,
            "recurrent_link_bytes": op.link_bytes, "timings": timings, "instrumented_eager_us": diag,
            "nvlink_data":link,
        })
    return {"q_heads":query.shape[1],"kv_heads":kv_heads,"head_dim":query.shape[2],"context":full.lengths,
            "shape_type":"real_first_decoder_layer" if masks else "controlled_random_bf16",
            "full_placement_ms":full_placement,
            "full_physical_kv_bytes":_bytes(full.keys)+_bytes(full.values),
            "local_tuning":raw,"chosen_num_splits":best["num_splits"],"outcomes":rows}


def geometry(text):
    values = tuple(map(int,text.split(":")))
    if len(values) != 3 or min(values) < 1 or values[0] % values[1]:
        raise argparse.ArgumentTypeError("Expected query_heads:kv_heads:head_dim with complete GQA groups")
    return values


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--decode-device",type=int,default=0)
    parser.add_argument("--helper-device",type=int,default=1)
    parser.add_argument("--producer-device",type=int)
    parser.add_argument("--layout",choices=("paged","packed"),default="paged")
    parser.add_argument("--contexts",type=int,nargs="+",default=[4096,8192,16384,32768,65536])
    parser.add_argument("--batch",type=int,nargs="+",default=[1,4,8,16,32])
    parser.add_argument("--geometry",type=geometry,default=(28,4,128))
    parser.add_argument("--ratios",type=float,nargs="+",default=[.25,.5,.75,1.])
    parser.add_argument("--local-splits",type=int,nargs="+",default=[0,1,2,4,8])
    parser.add_argument("--repeats",type=int,default=5)
    parser.add_argument("--calls",type=int,default=24)
    parser.add_argument("--link-repeats",type=int,default=0)
    parser.add_argument("--seed",type=int,default=20261128)
    parser.add_argument("--no-split-graph",action="store_true")
    parser.add_argument("--require-split-graph",action="store_true")
    parser.add_argument("--graph-transport",choices=("memcpy","peer_kernel"),default="memcpy")
    parser.add_argument("--model",type=Path)
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--id",action="append",default=[])
    parser.add_argument("--fixture", type=Path, help="Native InternVL first-decode Q/K/V from internvl_native")
    args=parser.parse_args()
    if any(not 0. <= ratio <= 1. for ratio in args.ratios):
        parser.error("Partition fractions must lie in [0, 1]")
    if args.model and (not args.manifest or not args.id):
        parser.error("--model comes with --manifest and one or more --id selections")
    if args.fixture and (args.model or args.manifest or args.id):
        parser.error("Use either native InternVL fixture data or model/manifest selectors")
    if args.no_split_graph and args.require_split_graph:
        parser.error("Choose only one graph policy")
    if min(args.contexts + args.batch + [args.repeats, args.calls]) < 1:
        parser.error("Use positive shape extents and timing counts")
    with torch.cuda.device(args.decode_device):
        peer=PeerAccess(args.decode_device,args.helper_device)
        producer=torch.device("cuda",args.producer_device if args.producer_device is not None else args.decode_device)
        rng=torch.Generator(device=producer).manual_seed(args.seed)
        orders=random.Random(args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as stream:
            config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
            stream.write(json.dumps({"type":"config",**config})+"\n")
            stream.flush()
            if args.fixture:
                spec, provenance = fixture_data(args.fixture, producer)
                result = run_case(spec, args, peer, orders)
                stream.write(json.dumps({"type": "case", **result, "fixture_provenance": provenance}) + "\n")
                stream.flush()
            elif args.model:
                shapes=[reference_data(args.model,args.manifest,args.id,producer)]
                for spec in shapes:
                    result=run_case(spec,args,peer,orders)
                    stream.write(json.dumps({"type":"case",**result})+"\n")
                    stream.flush()
            else:
                for context in args.contexts:
                    for batch in args.batch:
                        spec=generated_data(batch,context,args.geometry,producer,rng)
                        result=run_case(spec,args,peer,orders)
                        stream.write(json.dumps({"type":"case",**result})+"\n")
                        stream.flush()
                        del spec
                        print(f"{args.layout=} {batch=} {context=} complete",flush=True)
            stream.write(json.dumps({"type":"complete"})+"\n")


if __name__=="__main__":
    main()
