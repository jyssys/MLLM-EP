"""Measure native Qwen encoder and physical E-assisted attention on shared E streams."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
from statistics import median
import threading
import time

from PIL import Image
from safetensors import safe_open
import torch
from transformers import AutoProcessor, Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel

from .operator_bench import compare, generated_data, pack, partition, reference_data, timed
from .peer_attention import AssistedAttention, _capture


ENCODE_IDS = ("lmms-lab__DocVQA__45142","TIGER-Lab__Mantis-Eval__wc_22",
              "yubo2333__MMLongBench-Doc__658")
RAGGED_IDS = ("lmms-lab__DocVQA__56399","yubo2333__MMLongBench-Doc__658",
              "lmms-lab__MP-DocVQA__52541","lmms-lab__MP-DocVQA__6771")


def vision_only(snapshot, device):
    config = Qwen2_5_VLConfig.from_pretrained(snapshot,local_files_only=True).vision_config
    config._attn_implementation = "sdpa"
    model = Qwen2_5_VisionTransformerPretrainedModel(config)
    weight_map = json.loads((snapshot/"model.safetensors.index.json").read_text())["weight_map"]
    shards = defaultdict(list)
    for name,path in weight_map.items():
        if name.startswith("visual."):
            shards[path].append(name)
    if not shards:
        raise ValueError("Pinned checkpoint has no separately indexed visual state")
    lookup = {}
    for name,keys in shards.items():
        with safe_open(snapshot/name,framework="pt",device="cpu") as shard:
            for key in keys:
                weight = shard.get_tensor(key)
                if weight.dtype != torch.bfloat16:
                    raise ValueError(f"Non-BF16 checkpoint weight: {key}")
                lookup[key.removeprefix("visual.")] = weight
    model.load_state_dict(lookup,strict=True,assign=True)
    model.to(device).eval()
    return model,sum(tensor.numel()*tensor.element_size() for tensor in lookup.values()),len(lookup)


def make_encoder_inputs(processor, dataset, ids, device):
    rows = {row["request_id"]:row for row in json.loads(dataset.read_text())["requests"]}
    for name in ids:
        row = rows[name]
        with_images = []
        try:
            for src in row["images"]:
                with Image.open(src["path"]) as image:
                    with_images.append(image.convert("RGB"))
            item = processor(text=[row["prompt"]],images=with_images,return_tensors="pt")
        finally:
            for image in with_images:
                image.close()
        if item["image_grid_thw"].tolist() != row["grid_thw"]:
            raise ValueError(f"Original image grid changed: {name}")
        if int((item["input_ids"] == processor.image_token_id).sum()) != row["visual_tokens"]:
            raise ValueError(f"Original visual token count changed: {name}")
        prepared = (item["pixel_values"].to(device,dtype=torch.bfloat16),item["image_grid_thw"].to(device))
        yield row,prepared


class Bench:
    def __init__(self,d,e,geometry,ratio,encoder_priority):
        self.d,self.e = d,e
        query,data,masks = geometry
        self.query = query.to(d).contiguous()
        self.contexts = [len(keys) for keys,_ in data]
        self.presentation = "real_qwen_first_layer" if masks is not None else "controlled_random_bf16"
        self.op = AssistedAttention(tuple(query.shape),decode_device=d,helper_device=e)
        self.full = pack(data,d,"paged")
        local,remote = partition(data,ratio,masks,"generic_prefix")
        self.local = pack(local,d,"paged")
        self.remote = pack(remote,e,"paged")
        torch.cuda.synchronize(d)
        torch.cuda.synchronize(e)
        self.d_stream = torch.cuda.Stream(device=d)
        self.e_stream = torch.cuda.Stream(device=e,priority=encoder_priority)
        tuning = []
        for splits in (0,1,2,4,8):
            self.op.num_splits = splits
            candidate,result = _capture(d,self.d_stream,lambda: self.op.local(self.query,self.full))
            for _ in range(3):
                candidate.replay()
            timings = [timed(candidate.replay,d,24)["cuda_us"] for _ in range(3)]
            tuning.append((median(timings),splits,candidate,result))
        best = min(tuning,key=lambda item:item[0])
        self.local_graph,self.local_out = best[2:]
        self.local_tuning = [{"num_splits":s,"cuda_us":t} for t,s,_,_ in tuning]
        self.local_splits = best[1]
        self.op.num_splits = 0
        self.split = self.op.graph(self.query,self.local,self.remote)
        partitioned = self.split()
        self.local_graph.replay()
        torch.cuda.synchronize(d)
        self.correctness = compare(self.local_out,partitioned)
        if self.correctness["relative_l2"] > .008:
            raise ValueError("Prepared shape did not preserve physical attention")
        self.trace = None
        self.trace_error = None
        try:
            self.trace = self.op.graph(self.query,self.local,self.remote,trace_helper=True)
            self.trace()
            torch.cuda.synchronize(d)
            if not all(event.query() for event in self.trace.helper_events):
                raise RuntimeError("Helper timing nodes failed to run")
        except (ValueError,RuntimeError) as error:
            self.trace_error = str(error)
            if self.trace is not None:
                self.trace.close()
                self.trace = None

    def pair(self, encoder, args, mode, calls):
        include_encode = mode not in {"isolated_local","isolated_assist"}
        if not include_encode:
            path = self.local_graph.replay if mode == "isolated_local" else self.split
            return {"mode":mode,"D":timed(path,self.d,calls),"encoder_gpu_ms":None}
        if mode == "encoder_only":
            with torch.cuda.device(self.e), torch.cuda.stream(self.e_stream):
                first,last = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                first.record()
                encoded = encoder(args)
                last.record()
                last.synchronize()
            return {"mode":mode,"D":None,"encoder_gpu_ms":first.elapsed_time(last),"visual_rows":len(encoded)}
        trace = mode == "observed_assist"
        if trace and self.trace is None:
            raise ValueError("No qualified helper event graph for this run")
        first,last,t0 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        t0.record(torch.cuda.current_stream(self.e))
        self.e_stream.wait_event(t0)
        start = threading.Barrier(2,timeout=60)
        submitted = threading.Event()
        errors = []
        encoded_rows = []

        def run_encoder():
            try:
                with torch.inference_mode(),torch.cuda.device(self.e),torch.cuda.stream(self.e_stream):
                    start.wait()
                    first.record()
                    encoded_rows.append(len(encoder(args)))
                    last.record()
            except BaseException as error:
                errors.append(error)
            finally:
                submitted.set()

        thread = threading.Thread(target=run_encoder,daemon=True)
        thread.start()
        with torch.cuda.device(self.d),torch.cuda.stream(self.d_stream):
            self.d_stream.wait_event(t0)
            prior,begin,end = (torch.cuda.Event(enable_timing=True) for _ in range(3))
            start.wait()
            wait_start = time.perf_counter_ns()
            prior.record()
            if mode == "encoder_first":
                if not submitted.wait(60) or errors:
                    raise RuntimeError("Encoder failed to submit")
                self.d_stream.wait_event(last)
            d_wall = time.perf_counter_ns()
            begin.record()
            path = self.local_graph.replay if mode == "local_with_encode" else self.split
            for index in range(calls):
                if trace and index == calls // 2:
                    self.trace()
                else:
                    path()
            end.record()
            end.synchronize()
            d_elapsed = (time.perf_counter_ns()-d_wall)/1000/calls
        thread.join(120)
        if thread.is_alive() or errors:
            raise RuntimeError("Encoder launch failed or did not return") from (errors[0] if errors else None)
        last.synchronize()
        row = {"mode":mode,"D":{"cuda_us":begin.elapsed_time(end)*1000/calls,"host_us":d_elapsed},
               "encoder_gpu_ms":first.elapsed_time(last),"visual_rows":encoded_rows[0],
               "D_queue_event_ms":prior.elapsed_time(begin),
               "D_host_submit_wait_ms":(d_wall-wait_start)/1e6,"calls":calls}
        if trace:
            helper_start,helper_end = (t0.elapsed_time(event) for event in self.trace.helper_events)
            encoder_start,encoder_end = t0.elapsed_time(first),t0.elapsed_time(last)
            row["physical_E_event_overlap"] = {
                "observed_D_invocation":calls//2,
                "E_encode_ms":[encoder_start,encoder_end],"E_assist_ms":[helper_start,helper_end],
                "intersect_ms":max(0.,min(encoder_end,helper_end)-max(encoder_start,helper_start))
            }
        return row

    def close(self):
        torch.cuda.synchronize(self.d)
        self.split.close()
        if self.trace:
            self.trace.close()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--decode-device",type=int,default=2)
    parser.add_argument("--helper-device",type=int,default=0)
    parser.add_argument("--producer-device",type=int,default=1)
    parser.add_argument("--encode-id",action="append")
    parser.add_argument("--kv-id",action="append")
    parser.add_argument("--contexts",type=int,nargs="+",default=[32768])
    parser.add_argument("--batch",type=int,default=8)
    parser.add_argument("--ratio",type=float,default=.5)
    parser.add_argument("--ragged-ratio",type=float,default=.25)
    parser.add_argument("--repeats",type=int,default=4)
    parser.add_argument("--calls",type=int,default=128)
    parser.add_argument("--encoder-first",action="store_true")
    parser.add_argument("--encoder-priority",type=int,default=-1)
    parser.add_argument("--vision-backend",choices=("sdpa","packed-fa3"),default="sdpa")
    args = parser.parse_args()
    if (not 0 < args.ratio <= 1 or not 0 < args.ragged_ratio <= 1
            or min(args.batch,args.repeats,args.calls,*args.contexts) < 1):
        parser.error("Select positive workload lengths and a nonzero helper fraction")
    d,e,p = (torch.device("cuda",ordinal) for ordinal in
             (args.decode_device,args.helper_device,args.producer_device))
    with torch.cuda.device(d):
        processor = AutoProcessor.from_pretrained(args.model,local_files_only=True)
        model, weight_bytes, tensors = vision_only(args.model,e)
        if args.vision_backend == "packed-fa3":
            from .model import install_packed_vision
            install_packed_vision(model)
        data = list(make_encoder_inputs(processor,args.manifest,args.encode_id or ENCODE_IDS,e))
        if not data:
            raise ValueError("No frozen image inputs available for encoding")
        query_generator = torch.Generator(device=p).manual_seed(3091)
        chooser = random.Random(3091)
        geometries = [(generated_data(args.batch,context,(28,4,128),p,query_generator),args.ratio)
                      for context in args.contexts]
        geometries.append((reference_data(args.model,args.manifest,args.kv_id or RAGGED_IDS,p),args.ragged_ratio))
        with args.manifest.open("rb") as source:
            manifest_sha256 = hashlib.file_digest(source,"sha256").hexdigest()
        with args.out.open("x") as raw:
            raw.write(json.dumps({"type":"config","vision_weights_bytes":weight_bytes,"vision_tensors":tensors,
                                  "decode":str(d),"encoder_helper":str(e),"producer":str(p),
                                  "vision_backend":args.vision_backend,"encoder_priority":args.encoder_priority,
                                  "bf16":True,"attention_calls_per_pair":args.calls,
                                  "model_revision":args.model.name,"manifest_sha256":manifest_sha256,
                                  "seed":3091,"encoder_requests":args.encode_id or ENCODE_IDS,
                                  "real_kv_requests":args.kv_id or RAGGED_IDS,
                                  "gpu_visibility":os.environ.get("CUDA_VISIBLE_DEVICES")})+"\n")
            raw.flush()
            for shape,ratio in geometries:
                bench = Bench(d,e,shape,ratio,args.encoder_priority)

                def encode(item):
                    return model(item[0],grid_thw=item[1]).pooler_output

                try:
                    for row, source in data:
                        # Warm the exact original encode geometry on this E path.
                        encode(source)
                        torch.cuda.synchronize(e)
                        arms = ["encoder_only","isolated_local","isolated_assist","local_with_encode",
                                "assist_with_encode"]
                        if args.encoder_first:
                            arms.append("encoder_first")
                        if bench.trace:
                            arms.append("observed_assist")
                        times = []
                        for replicate in range(args.repeats):
                            priority = arms.copy()
                            chooser.shuffle(priority)
                            for mode in priority:
                                torch.cuda.reset_peak_memory_stats(e)
                                result = bench.pair(encode,source,mode,args.calls)
                                times.append({"replicate":replicate,**result,
                                              "encoder_peak_allocated_bytes":torch.cuda.max_memory_allocated(e),
                                              "encoder_peak_reserved_bytes":torch.cuda.max_memory_reserved(e)})
                        raw.write(json.dumps({"type":"case","request_id":row["request_id"],
                                             "image_count":len(row["images"]),"visual_rows":row["visual_tokens"],
                                             "shape_source":bench.presentation,"kv_rows":bench.contexts,
                                             "helper_fraction":ratio,"correctness":bench.correctness,
                                             "link_bytes_per_call":bench.op.link_bytes,
                                             "helper_event_failure":bench.trace_error,
                                             "local_tuning":bench.local_tuning,"measurements":times})+"\n")
                        raw.flush()
                finally:
                    bench.close()
            raw.write(json.dumps({"type":"complete"})+"\n")


if __name__ == "__main__":
    main()
