"""Fresh EP4 paper-baseline comparison with direct per-request token timestamps.

Reuses the project's validated two-DP-process engine setup, not old experimental
hooks. Same engine interleaves randomized policies; no runtime scheduler changes.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import socket
import time
from datetime import datetime,timezone
from pathlib import Path
from gpu_scope import allowed_devices


def record_token_timing(rec, output, received):
    """Snapshot V1 core timestamps before RequestStateStats is mutated again.

    Both clocks are Linux host monotonic clocks, not GPU absolute timestamps.
    Preserve frontend receipt time separately: synchronous frontend submission
    can delay polling while engine-core execution continues in another process.
    """
    metrics=output.metrics
    assert metrics is not None and metrics.last_token_ts>0, "Core timing required"
    ready=float(metrics.last_token_ts)
    assert rec["submitted_monotonic"]<=ready<=received, (rec,ready,received)
    count=len(output.outputs[0].token_ids)
    old=len(rec["token_timestamps"])
    assert count>=old
    rec["token_timestamps"].extend([ready]*(count-old))
    if old==0 and count:
        rec["token_timestamps"][0]=float(metrics.first_token_ts)
    if count-old>1:
        rec["coalesced_token_observations"]=rec.get("coalesced_token_observations",0)+1
    rec["frontend_token_timestamps"].extend([received]*(count-old))
    rec["engine_metrics"]={k:getattr(metrics,k,None) for k in
        ["arrival_time","first_token_latency","first_token_ts","last_token_ts","queued_ts","scheduled_ts"]}
    if output.finished:
        rec["frontend_completed_monotonic"]=received
        rec["frontend_e2e_ms"]=1000*(received-rec["submitted_monotonic"])
        rec["frontend_poll_delay_ms"]=1000*(received-ready)
    return ready


def worker(rank,args,barrier):
    import faulthandler
    stack_log=open(Path(args.out)/f"stack_dp{rank}.log","w")
    faulthandler.dump_traceback_later(90,repeat=True,file=stack_log)
    os.environ.update(VLLM_DP_RANK=str(rank),VLLM_DP_RANK_LOCAL=str(rank),VLLM_DP_SIZE="2",
                      VLLM_DP_MASTER_IP="127.0.0.1",VLLM_DP_MASTER_PORT=str(args.port))
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM,SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from successor_hook import install
    install()
    proc=AutoProcessor.from_pretrained(args.model)
    rows=[json.loads(s) for s in Path(args.data).read_text().splitlines()]
    rows=[r for r in rows if r["dataset"]==args.dataset and r["split"]==args.split]
    random.Random(args.seed).shuffle(rows)
    rows=rows[:args.requests]
    prompts=[]
    for row in rows:
        images=[Image.open(p).convert("RGB") for p in row.get("images",[])]
        content=[{"type":"image","image":im} for im in images]
        question=row["question"]+"\nAnswer the question using a single word or phrase."
        content.append({"type":"text","text":question})
        prompt=proc.apply_chat_template([{"role":"user","content":content}],tokenize=False,add_generation_prompt=True)
        item={"prompt":prompt}
        if images:
            item["multi_modal_data"]={"image":images[0] if len(images)==1 else images}
        prompts.append(item)
    llm=LLM(model=args.model,dtype="bfloat16",tensor_parallel_size=2,
            enable_expert_parallel=True,expert_placement_strategy="linear",
            all2all_backend="deepep_high_throughput",enable_dbo=False,enable_eplb=False,
            enable_ep_weight_filter=True,enable_return_routed_experts=False,
            kv_cache_memory_bytes=2*1024**3,max_model_len=8192,
            max_num_batched_tokens=args.max_batched_tokens,max_num_seqs=32,
            skip_mm_profiling=True,enable_prefix_caching=False,moe_backend="triton",
            enforce_eager=True,enable_flashinfer_autotune=False,
            mm_processor_kwargs={"max_pixels":args.max_pixels},
            limit_mm_per_prompt={"image":4},disable_log_stats=False)
    engine=llm.llm_engine
    pc=engine.vllm_config.parallel_config
    assert pc.data_parallel_size==2 and pc.tensor_parallel_size==2 and pc.enable_expert_parallel
    out=Path(args.out)
    (out/f"engine_config_dp{rank}.json").write_text(json.dumps({
        "tp":2,"dp":2,"ep":4,"pp":1,"dtype":"bfloat16","backend":pc.all2all_backend,
        "dbo":pc.enable_dbo,"eplb":pc.enable_eplb,"sequence_parallel":pc.use_sequence_parallel_moe,
        "visible_devices":os.environ["CUDA_VISIBLE_DEVICES"]},indent=2))
    policies=json.loads(Path(args.policies).read_text())
    counter=0
    seen_mm_hashes=set()
    def run(indices,policy,cohort,warmup=False):
        nonlocal counter
        if rank==0:
            Path(args.policy_file).write_text(json.dumps({**policy,"cohort":cohort}))
        barrier.wait(600)
        starts,records={},{}
        params=SamplingParams(temperature=0,max_tokens=args.output_tokens,
                              ignore_eos=args.fixed_output,output_kind=RequestOutputKind.CUMULATIVE)
        for idx in indices:
            request_key=f"p{policy['name']}__c{cohort}__r{rank}_c{counter}_{rows[idx]['id']}"
            item=dict(prompts[idx])
            if args.image_cache=="cold" and rows[idx].get("images"):
                # vLLM0.20 inputs.llm.MultiModalUUIDDict explicitly controls both
                # processor and encoder caches. Identical pixels, unique request
                # identities prevent policy-order-dependent encoder cache hits.
                item["multi_modal_uuids"]={"image":[f"{request_key}_image{i}"
                    for i in range(len(rows[idx]["images"]))]}
            start=time.monotonic()
            # Local vLLM0.20 deprecated InputPreprocessor._process_text silently
            # omits mm_uuids. Use the SAME renderer path as LLM.generate instead.
            # Rendering is still inside the measured request interval.
            rendered=llm._preprocess_cmpl_one(item)
            hashes=rendered.get("mm_hashes",{})
            flat_hashes=[h for hs in hashes.values() for h in hs]
            if args.image_cache=="cold" and rows[idx].get("images"):
                assert flat_hashes and not seen_mm_hashes.intersection(flat_hashes), (request_key,hashes)
                seen_mm_hashes.update(flat_hashes)
            rid=engine.add_request(request_key,rendered,params)
            starts[request_key]=start
            records[request_key]={"run":Path(args.out).name,"sample_seed":args.seed,
                          "order_seed":args.order_seed,"request_id":request_key,"internal_request_id":rid,"dataset_request":rows[idx]["id"],
                          "image_id":rows[idx]["image_id"],"answers":rows[idx]["answers"],
                          "policy":policy["name"],"cohort":cohort,"dp_rank":rank,
                          "image_cache_contract":args.image_cache,
                          "mm_hashes":hashes,"render_path":"LLM._preprocess_cmpl_one_RENDERER",
                          "submitted_monotonic":start,"token_timestamps":[],
                          "frontend_token_timestamps":[],"warmup":warmup,
                          "timing_contract":"HOST_SUBMISSION_TO_ENGINE_TOKEN_READY; CLIENT_RECEIPT_SEPARATE"}
            counter+=1
        # Multiprocess DP cores autonomously execute dummy participation. Consume
        # our actual requests, not an extra blocking get_output after local EOS.
        while engine.get_num_unfinished_requests()>0:
            outputs=engine.step()
            now=time.monotonic()
            for output in outputs:
                if output.request_id not in records or not output.outputs:
                    continue
                rec=records[output.request_id]
                tokens=output.outputs[0].token_ids
                ready=record_token_timing(rec,output,now)
                if output.finished:
                    rec.update(completed_monotonic=ready,e2e_ms=(ready-starts[output.request_id])*1000,
                               prompt_tokens=len(output.prompt_token_ids or []),output_tokens=list(tokens),
                               prediction=output.outputs[0].text)
                    times=rec["token_timestamps"]
                    rec["ttft_ms"]=(times[0]-starts[output.request_id])*1000 if times else None
                    rec["itl_observations_exact"]=rec.get("coalesced_token_observations",0)==0
                    rec["itls_ms"]=[1000*(b-a) for a,b in zip(times,times[1:])] if rec["itl_observations_exact"] else []
                    rec["tpot_ms"]=1000*(times[-1]-times[0])/(len(times)-1) if len(times)>1 else None
        barrier.wait(600)
        assert all("completed_monotonic" in r for r in records.values()),records
        with (out/f"requests_dp{rank}.jsonl").open("a") as f:
            for r in records.values():
                f.write(json.dumps(r)+"\n")
        print(json.dumps({"dp_rank":rank,"cohort":cohort,"policy":policy["name"],
                          "requests":len(records),"mean_e2e_ms":sum(r["e2e_ms"] for r in records.values())/len(records)}),flush=True)
    barrier.wait(600)
    local_n=args.batch_per_dp
    for j in range(args.warmups):
        for policy in policies:
            run(list(range(rank*local_n,(rank+1)*local_n)),policy,f"warmup{j}",True)
    for repeat in range(args.repetitions):
        order=list(policies)
        random.Random(args.order_seed+repeat).shuffle(order)
        for offset in range(0,len(rows),2*local_n):
            indices=list(range(offset+rank*local_n,min(offset+(rank+1)*local_n,len(rows))))
            if not indices:
                indices=[rank%len(rows)]
            for policy in order:
                run(indices,policy,f"rep{repeat}_offset{offset}")
    barrier.wait(600)
    # Explicit close terminates only engines owned by this worker.
    engine.engine_core.shutdown()
    faulthandler.cancel_dump_traceback_later()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True)
    ap.add_argument("--data",required=True)
    ap.add_argument("--policies",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--dataset",default="gqa")
    ap.add_argument("--split",default="heldout")
    ap.add_argument("--requests",type=int,default=16)
    ap.add_argument("--batch-per-dp",type=int,default=2)
    ap.add_argument("--output-tokens",type=int,default=32)
    ap.add_argument("--fixed-output",action="store_true")
    ap.add_argument("--max-pixels",type=int,default=1605632)
    ap.add_argument("--max-batched-tokens",type=int,default=8192)
    ap.add_argument("--warmups",type=int,default=2)
    ap.add_argument("--repetitions",type=int,default=3)
    ap.add_argument("--seed",type=int,default=7317)
    ap.add_argument("--order-seed",type=int,help="Independent run-order seed; does not change request selection")
    ap.add_argument("--instrument",action="store_true")
    ap.add_argument("--routes",action="store_true")
    ap.add_argument("--scheduler-context",action="store_true",
                    help="Separate low-overhead CPU scheduler-context diagnostic; not clean CUDA timing")
    ap.add_argument("--image-cache",choices=["cold","shared"],default="cold",
                    help="Cold uses unique image UUIDs per request, not altered pixels")
    args=ap.parse_args()
    if args.order_seed is None:args.order_seed=args.seed
    devices = allowed_devices()
    out=Path(args.out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    args.out=str(out)
    args.policy_file=str(out/"active_policy.json")
    Path(args.policy_file).write_text(json.dumps({"name":"vanilla","method":"vanilla"}))
    from transformers import AutoProcessor,AutoConfig
    proc=AutoProcessor.from_pretrained(args.model)
    cfg=AutoConfig.from_pretrained(args.model)
    token_path=out/"token_ids.json"
    token_path.write_text(json.dumps({"special_ids":proc.tokenizer.all_special_ids,
                                     "image_id":cfg.image_token_id,"video_id":cfg.video_token_id}))
    hook=Path(__file__).resolve().parent/"ep_hooks"
    source_paths=[Path(__file__),hook/'successor_hook.py',Path(args.policies)]
    (out/'source_hashes.json').write_text(json.dumps({str(p.resolve()):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},indent=2))
    os.environ.update(SUCCESSOR_EP_ENABLE="1",SUCCESSOR_EP_OUT=str(out),
                      SUCCESSOR_EP_POLICY=args.policy_file,SUCCESSOR_EP_TOKEN_IDS=str(token_path),
                      SUCCESSOR_EP_CATALOG=str(Path(args.policies).resolve()),
                      SUCCESSOR_EP_TIMING=str(int(args.instrument)),SUCCESSOR_EP_ROUTES=str(int(args.routes)),
                      SUCCESSOR_EP_SCHEDULER_CONTEXT=str(int(args.scheduler_context)),
                      PYTHONPATH=str(hook)+os.pathsep+os.environ.get("PYTHONPATH",""))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1",0))
        args.port=sock.getsockname()[1]
    ctx=mp.get_context("spawn")
    barrier=ctx.Barrier(2)
    started=datetime.now(timezone.utc).isoformat()
    start=time.monotonic()
    processes=[ctx.Process(target=worker,args=(rank,args,barrier)) for rank in range(2)]
    for p in processes:p.start()
    for p in processes:p.join(7200)
    for p in processes:
        if p.is_alive():p.terminate();p.join(10)
    result={"exit_codes":[p.exitcode for p in processes],"started_utc":started,
            "finished_utc":datetime.now(timezone.utc).isoformat(),"elapsed_seconds":time.monotonic()-start,
            "arguments":vars(args),"physical_gpus":devices}
    (out/"completed.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    raise SystemExit(0 if all(p.exitcode==0 for p in processes) else 1)


if __name__=="__main__":main()
