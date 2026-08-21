"""Capture live visual routes and upstream encoder summaries on one-request DP waves."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
import socket
import time
import traceback
from pathlib import Path
from typing import Any

MODEL = "/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0)); return int(sock.getsockname()[1])


def _base_suite() -> list[dict[str, Any]]:
    ski = Path("/home/esjung/anaconda3/lib/python3.14/site-packages/skimage/data")
    mode = Path("/home/esjung/MLLM-EP/external/MODE/assets")
    tui = Path("/home/esjung/MLLM-EP/external/lmms-eval/docs/images")
    ds = Path("/home/esjung/anaconda3/lib/python3.14/site-packages/datashader/examples/assets/images")
    names: list[tuple[str, str, Path]] = []
    for name in ("astronaut.png", "camera.png", "chelsea.png", "coffee.png", "coins.png", "horse.png", "hubble_deep_field.jpg", "moon.png", "motorcycle_left.png", "motorcycle_right.png", "rocket.jpg"):
        names.append(("natural", Path(name).stem, ski / name))
    for name in ("brick.png", "cell.png", "clock_motion.png", "color.png", "grass.png", "gravel.png", "ihc.png", "microaneurysms.png", "phantom.png", "retina.jpg"):
        names.append(("fine_grained", Path(name).stem, ski / name))
    for name in ("chessboard_GRAY.png", "chessboard_RGB.png", "logo.png", "page.png", "text.png"):
        names.append(("chart_document", Path(name).stem, ski / name))
    for name in ("bit-allocate.png", "card_3.png", "fast_gptq.png", "method.png"):
        names.append(("chart_document", f"mode_{Path(name).stem}", mode / name))
    for name in ("tui-log-streaming.png", "tui-main.png", "tui-model-selection.png"):
        names.append(("chart_document", Path(name).stem, tui / name))
    for name in ("airport_connections.png", "chesapeake_farout.png", "chesbay_detail.png", "dashboard.png", "ds_hv_bokeh.png", "houston_district29.png", "landsat.png", "nyc_buildings.png", "nyc_pickups_vs_dropoffs.jpg", "nyc_races.jpg", "nyc_taxi_100k.png", "pcap.png", "pipeline.png", "sym_attractors.jpg", "uk_researchers.png"):
        names.append(("chart_document", f"ds_{Path(name).stem}", ds / name))
    assert len(names) == 48 and len({_sha(path) for _, _, path in names}) == 48
    prompts = {
        "natural": "Describe the important objects and their spatial arrangement briefly.",
        "fine_grained": "Describe the visible fine-grained texture or structure briefly.",
        "chart_document": "Summarize the visible chart, diagram, or interface briefly.",
    }
    return [{"category": cat, "sample_id": sid, "image_paths": [str(path)], "question": prompts[cat], "variant": "canonical", "source_ids": [_sha(path)]} for cat, sid, path in names]


def suite() -> list[dict[str, Any]]:
    rows = _base_suite()
    controlled = rows[:8]
    for row in controlled:
        for edge in (448, 896):
            rows.append({**row, "sample_id": f"{row['sample_id']}_res{edge}", "variant": "resolution", "resize_long_edge": edge})
        for index, question in enumerate(("What objects are present in this image?", "Give a concise summary of the scene.")):
            rows.append({**row, "sample_id": f"{row['sample_id']}_prompt{index}", "variant": "prompt", "question": question})
    assert len(rows) == 80
    return rows


def _prepare(processor: Any, row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    from PIL import Image
    images = [Image.open(path).convert("RGB") for path in row["image_paths"]]
    if row.get("resize_long_edge"):
        edge = int(row["resize_long_edge"]); resized = []
        for image in images:
            scale = edge / max(image.size)
            resized.append(image.resize((max(28, round(image.width * scale)), max(28, round(image.height * scale)))))
        images = resized
    content = [{"type": "image", "image": image} for image in images] + [{"type": "text", "text": row["question"]}]
    prompt = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    begin = time.perf_counter_ns(); processed = processor(text=[prompt], images=images, return_tensors="pt"); processor_ms = (time.perf_counter_ns()-begin)/1e6
    ids = processed["input_ids"][0].tolist(); grids = processed["image_grid_thw"].tolist()
    image_id = int(processor.tokenizer.convert_tokens_to_ids(processor.image_token)); merge = int(processor.image_processor.merge_size)
    spans=[]; cursor=0
    while cursor < len(ids):
        if ids[cursor] != image_id: cursor += 1; continue
        end=cursor+1
        while end < len(ids) and ids[end] == image_id: end += 1
        spans.append([cursor,end]); cursor=end
    meta_images=[]
    for image,path,grid,span in zip(images,row["image_paths"],grids,spans,strict=True):
        t,h,w=map(int,grid); expected=t*h*w//(merge*merge); assert span[1]-span[0] == expected
        meta_images.append({"path":path,"sha256":_sha(Path(path)),"input_size":list(image.size),"original_size":list(Image.open(path).size),"image_grid_thw":[t,h,w],"post_merge_grid_hw":[h//merge,w//merge],"vision_tokens":expected,"token_span":span})
    return {"prompt":prompt,"multi_modal_data":{"image":images[0] if len(images)==1 else images}}, {**row,"processor_ms":processor_ms,"prompt_tokens":len(ids),"vision_tokens":sum(x["vision_tokens"] for x in meta_images),"image_token_id":image_id,"images":meta_images}


def _generate(llm: Any, prompts: list[dict[str, Any]], sampling: Any, barrier: Any, wave: int) -> list[Any]:
    from vllm.outputs import RequestOutput
    from vllm.v1.engine import EngineCoreRequestType
    if prompts:
        barrier.wait(timeout=900); llm._add_completion_requests(prompts,sampling,use_tqdm=False); outputs=llm._run_engine(RequestOutput,use_tqdm=False)
    else:
        llm.llm_engine.engine_core._send_input(EngineCoreRequestType.START_DP_WAVE,(wave,-1)); barrier.wait(timeout=900); outputs=[]
    barrier.wait(timeout=900); return outputs


def _run_rank(rank: int, port: int, args: argparse.Namespace, barrier: Any, schedule: list[dict[str, Any]]) -> None:
    out=args.output_dir/f"driver.dp{rank}{'.resume' if args.resume else ''}.json"
    try:
        os.environ.update({"VLLM_DP_RANK":str(rank),"VLLM_DP_RANK_LOCAL":str(rank),"VLLM_DP_SIZE":"2","VLLM_DP_MASTER_IP":"127.0.0.1","VLLM_DP_MASTER_PORT":str(port),"FLASHVEP_PREROUTER_CONTROL":str((args.output_dir/"control.json").resolve()),"FLASHVEP_PREROUTER_RAW":str((args.output_dir/"raw").resolve()),"FLASHVEP_DEEPEP_PROOF_DIR":str((args.output_dir/"backend_proof").resolve()),"FLASHVEP_CONFIGURED_ALL2ALL_BACKEND":"deepep_high_throughput","FLASHVEP_CONFIGURED_DBO":"false"})
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
        processor=AutoProcessor.from_pretrained(args.model_path,trust_remote_code=True)
        prepared={row["sample_id"]:_prepare(processor,row)[0] for row in suite()}
        llm=LLM(model=args.model_path,dtype="bfloat16",tensor_parallel_size=2,enable_expert_parallel=True,expert_placement_strategy="linear",all2all_backend="deepep_high_throughput",enable_dbo=False,enable_return_routed_experts=True,enable_ep_weight_filter=True,trust_remote_code=True,gpu_memory_utilization=.90,kv_cache_memory_bytes=1<<30,max_model_len=4096,max_num_batched_tokens=8192,max_num_seqs=2,limit_mm_per_prompt={"image":2},skip_mm_profiling=True,enable_prefix_caching=False,enable_flashinfer_autotune=False,enforce_eager=True)
        sampling=SamplingParams(max_tokens=1,temperature=0.0); records=[]
        for entry in schedule:
            if rank==0:
                tmp=args.output_dir/"control.tmp.json"; _json(tmp,entry); tmp.replace(args.output_dir/"control.json")
            barrier.wait(timeout=900); prompt=[copy.deepcopy(prepared[entry["request_id"]])] if rank==entry["source_dp_rank"] else []
            begin=time.perf_counter_ns(); outputs=_generate(llm,prompt,sampling,barrier,int(entry["wave"])); wall=(time.perf_counter_ns()-begin)/1e6
            records.append({**entry,"driver_dp_rank":rank,"wall_ms":wall,"output_tokens":[int(t) for o in outputs for t in o.outputs[0].token_ids]})
        _json(out,{"ok":True,"records":records})
    except BaseException:
        _json(out,{"ok":False,"traceback":traceback.format_exc()}); raise


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",type=Path,required=True); parser.add_argument("--model-path",default=MODEL); parser.add_argument("--repeats",type=int,default=3); parser.add_argument("--resume",action="store_true"); args=parser.parse_args()
    if args.resume:
        schedule=json.loads((args.output_dir/"schedule.json").read_text())
        schedule=[entry for entry in schedule if not (args.output_dir/"raw"/"routes"/f"wave{entry['wave']}_dp{entry['source_dp_rank']}_tp0_layer0.npy").exists()]
        if not schedule: print("nothing to resume"); return
    else:
        args.output_dir.mkdir(parents=True,exist_ok=False)
        from transformers import AutoProcessor
        processor=AutoProcessor.from_pretrained(args.model_path,trust_remote_code=True); prepared=[_prepare(processor,row) for row in suite()]
        metadata=[item[1] for item in prepared]; _json(args.output_dir/"workload_manifest.json",{"model":args.model_path,"configuration":{"dtype":"BF16","tp":2,"dp":2,"ep":4,"pp":1,"all2all":"deepep_high_throughput","physical_gpus":[4,5,6,7]},"unique_source_images":len({x for row in metadata for x in row["source_ids"]}),"requests":metadata})
        schedule=[]
        for repeat in range(args.repeats):
            for row in metadata:
                schedule.append({"wave":len(schedule),"request_id":row["sample_id"],"repeat":repeat,"source_dp_rank":len(schedule)%2,"prompt_tokens":row["prompt_tokens"],"vision_tokens":row["vision_tokens"],"capture":True})
        _json(args.output_dir/"schedule.json",schedule)
    # Worker warmup happens before the first DP wave; never let it inherit a
    # stale real wave identity from a previous/resumed process.
    _json(args.output_dir/"control.json",{"wave":-1,"capture":False,"request_id":"engine_warmup","repeat":-1})
    context=mp.get_context("spawn"); barrier=context.Barrier(2); port=_port(); processes=[context.Process(target=_run_rank,args=(rank,port,args,barrier,schedule)) for rank in range(2)]
    for process in processes: process.start()
    for process in processes: process.join()
    if [p.exitcode for p in processes] != [0,0]: raise RuntimeError(f"capture failed: {[p.exitcode for p in processes]}")
    print(args.output_dir)


if __name__ == "__main__": main()
