"""Run a fixed request under isolated, controlled, or serving-like context."""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import socket
import time
import traceback
from pathlib import Path
from typing import Any

from poc_flashvep.live_prefill_execution_regime.run_live import _generate, _requests


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2)+"\n")


def _port() -> int:
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1",0)); return int(sock.getsockname()[1])


def _run(rank: int, port: int, args: argparse.Namespace, barrier: Any, schedule: list[dict[str,Any]]) -> None:
    output=args.output_dir/f"driver.dp_rank{rank}.json"
    try:
        os.environ.update({
            "VLLM_DP_RANK":str(rank),"VLLM_DP_RANK_LOCAL":str(rank),"VLLM_DP_SIZE":"2",
            "VLLM_DP_MASTER_IP":"127.0.0.1","VLLM_DP_MASTER_PORT":str(port),
            "FLASHVEP_RUNTIME_CONTROL":str((args.output_dir/"control.json").resolve()),
            "FLASHVEP_RUNTIME_RAW_DIR":str((args.output_dir/"raw").resolve()),
            "FLASHVEP_RUNTIME_MODE":args.context,
            "FLASHVEP_RUNTIME_ISOLATED_OUTPUT":str((args.output_dir/"isolated.json").resolve()),
            "FLASHVEP_CONFIGURED_ALL2ALL_BACKEND":"deepep_high_throughput","FLASHVEP_CONFIGURED_DBO":"false",
        })
        from vllm import LLM,SamplingParams
        requests=_requests(args.source,args.model_path); request=requests[args.request_id]
        llm=LLM(model=args.model_path,dtype="bfloat16",tensor_parallel_size=2,enable_expert_parallel=True,
                expert_placement_strategy="linear",all2all_backend="deepep_high_throughput",enable_dbo=False,
                enable_return_routed_experts=False,enable_ep_weight_filter=True,trust_remote_code=True,
                gpu_memory_utilization=.90,kv_cache_memory_bytes=1<<30,max_model_len=4096,
                max_num_batched_tokens=16384,max_num_seqs=4,skip_mm_profiling=True,
                enable_prefix_caching=False,enable_flashinfer_autotune=False,enforce_eager=True)
        sampling=SamplingParams(max_tokens=1,temperature=0.0); records=[]
        for entry in schedule:
            if rank==0:
                temporary=args.output_dir/"control.tmp.json"; _write(temporary,entry); temporary.replace(args.output_dir/"control.json")
            barrier.wait(timeout=900)
            if args.context in ("controlled","isolated"):
                prompts=[copy.deepcopy(request)] if rank==args.source_dp_rank else []
            else:
                prompts=[copy.deepcopy(request),copy.deepcopy(request)]
            start=time.perf_counter_ns(); outputs=_generate(llm,prompts,sampling,barrier,int(entry["wave"])); wall=(time.perf_counter_ns()-start)/1e6
            records.append({**entry,"driver_dp_rank":rank,"wall_ms":wall,"outputs":len(outputs)})
        flush={**schedule[-1],"wave":len(schedule),"flush":True,"instrument":False,"measured":False,"iteration":0}
        if rank==0:
            temporary=args.output_dir/"control.tmp.json"; _write(temporary,flush); temporary.replace(args.output_dir/"control.json")
        barrier.wait(timeout=900)
        prompts=[copy.deepcopy(request)] if rank==args.source_dp_rank else []
        _generate(llm,prompts,sampling,barrier,int(flush["wave"]))
        _write(output,{"ok":True,"records":records})
    except BaseException:
        _write(output,{"ok":False,"traceback":traceback.format_exc()}); raise


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--source",type=Path,required=True); parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--model-path",required=True); parser.add_argument("--context",choices=("isolated","controlled","serving"),required=True)
    parser.add_argument("--request-id",default="text_18_tui_main"); parser.add_argument("--source-dp-rank",type=int,default=0)
    parser.add_argument("--target-layer",type=int,default=45); parser.add_argument("--target-rank",type=int,default=0)
    parser.add_argument("--warmups",type=int,default=5); parser.add_argument("--iterations",type=int,default=30)
    args=parser.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=False)
    count=1 if args.context=="isolated" else args.warmups+args.iterations
    schedule=[]
    for index in range(count):
        schedule.append({"wave":index,"context":args.context,"request_id":args.request_id,"iteration":index-args.warmups,
                         "measured":args.context=="isolated" or index>=args.warmups,"instrument":args.context=="isolated" or index>=args.warmups,
                         "target_layer":args.target_layer,"target_rank":args.target_rank})
    _write(args.output_dir/"schedule.json",schedule)
    context=mp.get_context("spawn"); barrier=context.Barrier(2); port=_port()
    processes=[context.Process(target=_run,args=(rank,port,args,barrier,schedule)) for rank in range(2)]
    for process in processes: process.start()
    for process in processes: process.join()
    codes=[process.exitcode for process in processes]
    if codes != [0,0]: raise RuntimeError(f"context run failed: {codes}")


if __name__=="__main__": main()
