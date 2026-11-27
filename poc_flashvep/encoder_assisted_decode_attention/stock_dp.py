"""Optimized co-located three-replica control using real vLLM HTTP/SSE servers."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import sys
import time
import urllib.request

from ..mllm_epd_h200_prefill_heavy_batch_matched import runtime as rt
from ..mllm_specific_disaggregation_discovery.native_epd import prompt_messages


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--arrival", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--requests", type=int, required=True)
    p.add_argument("--asset-repo", type=Path, default=Path("../MLLM-EP-mllm-epd"))
    p.add_argument("--wall-seconds", type=int, default=1400)
    args=p.parse_args()
    arrival=rt.read(args.arrival)
    if args.requests < 1 or arrival["serial"] or len(arrival["trace"]) < args.requests:
        p.error("Use the frozen open-loop trace and a valid length")
    manifest=rt.read(args.manifest)
    index={row["request_id"]:row for row in manifest["requests"]}
    rows=[index[item["request_id"]] for item in arrival["trace"][:args.requests]]
    ports=(18853,18854,18855)
    for port in ports:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", port))
    run=args.out.resolve()
    run.mkdir()
    for directory in ("state","raw","logs","analysis","configs"):
        (run/directory).mkdir()
    model_id="Qwen/Qwen2.5-VL-7B-Instruct"
    runtime=Path(sys.executable).resolve().parent
    asset_repo = args.asset_repo.resolve(strict=True)
    rt.git_check(asset_repo)
    overlay = rt.prepare_overlay(asset_repo, run, research=Path.cwd())
    deadline=time.time()+args.wall_seconds
    nvidia_headers = runtime.parent / "lib/python3.14/site-packages/nvidia"
    include_flags = " ".join(f"-I{nvidia_headers / name / 'include'}"
                             for name in ("curand", "cublas", "cusparse", "cusolver"))
    env=os.environ | {
        "PATH":str(runtime)+os.pathsep+os.environ["PATH"],
        "CUDA_HOME":str(runtime.parent/"targets/x86_64-linux"),
        "FLASHINFER_NVCC":str(runtime/"nvcc"), "CC":str(runtime/"x86_64-conda-linux-gnu-cc"),
        "CXX":str(runtime/"x86_64-conda-linux-gnu-c++"), "VLLM_USE_FLASHINFER_SAMPLER":"0",
        "NVCC_CCBIN":str(runtime/"x86_64-conda-linux-gnu-c++"),
        "FLASHINFER_EXTRA_CFLAGS":include_flags,"FLASHINFER_EXTRA_CUDAFLAGS":include_flags,
        "FLASHINFER_WORKSPACE_BASE":str(run.parent/"flashinfer_workspace"),
        "OMP_NUM_THREADS":"3","MKL_NUM_THREADS":"3","TOKENIZERS_PARALLELISM":"false",
        "HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","MLLM_EPD_SESSION_ROOT":str(run),
        "MLLM_EPD_PINNED_REPO":str(asset_repo),"MLLM_EPD_SESSION_PYTHON":sys.executable,
        "MLLM_EPD_SESSION_DEADLINE":str(deadline),
        "MLLM_EPD_RUN_NONCE":secrets.token_hex(12),
    }
    devices=env["CUDA_VISIBLE_DEVICES"].split(",")
    if len(devices)!=3:
        p.error("Acquire a verified three-GPU lease with UUID visibility first")
    os.environ.update({name:value for name,value in env.items() if name.startswith("MLLM_EPD_")})
    sys.modules["poc_flashvep"].__path__.insert(0, str(overlay / "poc_flashvep"))
    client=rt.module("client")
    client.messages=prompt_messages
    metrics=rt.module("metrics")
    group=rt.OwnedProcesses(run,env,Path.cwd(),deadline)
    model=str(args.model.resolve())
    common_media=os.path.commonpath([str(Path(entry["path"]).parent) for row in rows for entry in row["images"]])
    required_max=max(row["image_count"] for row in rows)
    processes=[]
    urls=[]
    rt.write(run/"configs/hardware.json",{"evidence":"MEASURED_ONLINE","replicas":devices,
                                          "model_config_sha256":rt.sha(args.model/"config.json"),
                                          "original_arrival_SHA256":rt.sha(args.arrival)})
    def cancel(*_):
        raise TimeoutError("DP3 lease ended")
    signal.signal(signal.SIGTERM,cancel)
    signal.signal(signal.SIGINT,cancel)
    try:
        for ordinal,(uuid,port) in enumerate(zip(devices,ports,strict=True)):
            command=[str(runtime/"vllm"),"serve",model,"--served-model-name",model_id,
                     "--host","127.0.0.1","--port",str(port),"--dtype","bfloat16",
                     "--max-model-len","32768","--max-num-seqs","64","--max-num-batched-tokens","8192",
                     "--enable-chunked-prefill","--no-enable-prefix-caching",
                     "--generation-config","vllm","--seed","2026112401",
                     "--mm-processor-cache-gb","0","--renderer-num-workers","2","--block-size","128",
                     "--gpu-memory-utilization","0.9","--limit-mm-per-prompt",
                     json.dumps({"image":required_max,"video":0}),
                     "--allowed-local-media-path",common_media]
            process=group.start(f"colocated{ordinal}",command,CUDA_VISIBLE_DEVICES=uuid)
            url=f"http://127.0.0.1:{port}"
            processes.append(process)
            urls.append(url)
        for process,url in zip(processes,urls,strict=True):
            while time.time()<deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Colocated model exited before HTTP readiness at {url}")
                try:
                    with urllib.request.urlopen(url+"/v1/models",timeout=2) as response:
                        data=json.loads(response.read())["data"]
                    if data and data[0]["id"]==model_id:
                        break
                except OSError:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError(f"Colocated model did not start at {url}")
        # One actual warm request per replica. The measured cohort starts after all completions.
        warm=asyncio.run(client.execute(rows[:3],urls,"warmup",1.,arrival["seed"],16,serial=True))
        if len(warm)!=3 or any(item["status"]!="MEASURED_ONLINE" or len(item["token_ids"])!=16 for item in warm):
            raise RuntimeError("All co-located replicas must complete their fixed-output warmup")
        asyncio.run(client.execute(rows,urls,"stock_dp3",arrival["QPS"],arrival["seed"],None,
                                   unique_media_occurrence=True,defer_raw=True,
                                   metadata={"native":"three optimized co-located vLLM replicas"}))
        generated=rt.read(run/"configs/stock_dp3_arrival.json")["trace"]
        keys=("request_id","offset_seconds")
        if any(tuple(a[k] for k in keys)!=tuple(b[k] for k in keys)
               for a,b in zip(generated,arrival["trace"][:args.requests],strict=True)):
            raise RuntimeError("Native DP client produced a different arrival trace")
        metrics.summarize("stock_dp3")
    finally:
        group.close()


if __name__=="__main__":
    main()
