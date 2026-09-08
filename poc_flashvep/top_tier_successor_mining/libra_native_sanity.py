"""Run the supplied SGLang implementation itself with bounded fresh inputs.

Only the harness changes: local cached text, four permitted GPUs, randomized
is_ori switch, explicit output comparison. A reduced-layer dummy-weight run is
functional sanity ONLY and never paper performance or quality evidence.
"""
import argparse
import hashlib
import json
import os
import random
import time
from datetime import datetime,timezone
from pathlib import Path
from gpu_scope import allowed_devices, physical_gpu
import multiprocessing as mp


def worker(rank,args,server_args,port_args):
    import torch
    import torch.distributed as dist
    from sglang import bench_one_batch_test as bench
    from sglang.srt.distributed import destroy_distributed_environment
    if args.vl_inputs:
        from libra_vl_bridge import install_weight_name_bridge,install_captured_vl_context
        install_weight_name_bridge()
    torch.set_num_threads(2)
    torch.manual_seed(7317)
    started=datetime.now(timezone.utc).isoformat()
    model_runner,tokenizer=bench.load_model(server_args,port_args,rank)
    oracle=None
    if args.prediction_oracle or args.frozen_route_oracle:
        from libra_prediction_oracle import install_prediction_oracle
        oracle=install_prediction_oracle(model_runner.model,rank)
    if args.vl_inputs:
        sources=[torch.load(args.vl_inputs/f"{args.vl_group}_source{r}.pt",map_location="cpu",weights_only=True) for r in range(4)]
        local=sources[rank]
        assert all(s["M"]==args.tokens for s in sources)
        context=install_captured_vl_context(model_runner.model)
        for key in ("inputs_embeds","vision_mask","cos","sin"):
            context[key]=local[key].to("cuda")
        context["deepstack"]=[t.to("cuda") for t in local["deepstack"]]
        hf_reference=local["reference_logits"].to("cuda").float()
        flat=None
    else:
        rows=[json.loads(x) for x in args.data.read_text().splitlines()]
        flat=[]
        for row in rows:
            flat.extend(tokenizer.encode(row.get("text",row.get("question",""))))
            if len(flat)>=4*args.tokens*args.cases:break
        assert len(flat)>=4*args.tokens*args.cases
    records=[]
    debug=None
    if args.hidden_diagnostic:
        from libra_vl_diagnostics import install_hidden_diagnostic
        debug=install_hidden_diagnostic(model_runner.model.model.layers,native=True,rank=rank,local_m=args.tokens)
    for case in range(args.cases):
        if oracle is not None:
            oracle["logits"].clear()
        packed=([s["input_ids"].tolist() for s in sources] if args.vl_inputs else
                [flat[(case*4+r)*args.tokens:(case*4+r+1)*args.tokens] for r in range(4)])
        reference=None
        rng=random.Random(7317+case)
        for rep in range(-args.warmup,args.reps):
            order=["vanilla","libra"] + (["libra_oracle"] if oracle is not None else [])
            if args.frozen_route_oracle:order += ["libra_frozen","libra_frozen_oracle"]
            rng.shuffle(order)
            if reference is None:
                order=["vanilla","libra"] + (["libra_oracle"] if oracle is not None else [])
                if args.frozen_route_oracle:order += ["libra_frozen","libra_frozen_oracle"]
            for policy in order:
                if debug is not None:
                    debug["active"]=policy=="vanilla" and not debug["records"]
                if oracle is not None:
                    oracle["capture"]=policy=="vanilla" and not oracle["logits"]
                    oracle["enabled"]=policy in ("libra_oracle","libra_frozen_oracle")
                    oracle["freeze_current"]=policy in ("libra_frozen","libra_frozen_oracle")
                    oracle["calls"]=0
                    oracle["actual_logits"].clear()
                for layer in model_runner.model.model.layers:
                    layer.is_ori=policy=="vanilla"
                model_runner.req_to_token_pool.clear()
                model_runner.token_to_kv_pool_allocator.clear()
                reqs=bench.prepare_eval_inputs_for_latency_test(packed,1,args.tokens,rank)
                torch.cuda.synchronize();dist.barrier()
                if getattr(args,'profiled',False):
                    torch.cuda.nvtx.range_push(f"successor_native|rank={rank}|case={case}|rep={rep}|policy={policy}")
                t0=time.perf_counter()
                next_ids,logits,batch=bench.extend(reqs,model_runner)
                torch.cuda.synchronize()
                wall=(time.perf_counter()-t0)*1000
                if getattr(args,'profiled',False):
                    torch.cuda.nvtx.range_pop()
                if debug is not None and debug["active"]:
                    debug["active"]=False
                    torch.save({k:v.cpu() for k,v in debug["records"].items()},args.out/f"hidden_rank{rank}.pt")
                if policy=="vanilla":reference=logits.detach().float().clone()
                error=logits.float()-reference
                native_logprob=logits.float().log_softmax(-1)
                reference_logprob=reference.log_softmax(-1)
                record={"rank":rank,"case":case,"repeat":rep,"policy":policy,
                        "prefill_ms":wall,"token_ids":next_ids.tolist(),
                        "relative_logit_l2":float(error.norm()/reference.norm().clamp_min(1e-9)),
                        "logit_max_abs":float(error.abs().max()),
                        "logit_cosine":float(torch.nn.functional.cosine_similarity(logits.float().flatten(),reference.flatten(),dim=0)),
                        "logit_kl_vanilla_to_policy":float((reference_logprob.exp()*(reference_logprob-native_logprob)).sum(-1).mean()),
                        "greedy_first_equal":bool((logits.argmax(-1)==reference.argmax(-1)).all()),
                        "scope":("DUMMY_REDUCED_LAYER_FUNCTIONAL_ONLY" if args.dummy else
                                 "REAL_WEIGHTS_REDUCED_LAYER_FUNCTIONAL_ONLY" if args.layers<48 else
                                 "VL_POST_VISION_PREFILL_BRIDGE_NOT_FULL_E2E" if args.vl_inputs else
                                 "NATIVE_SUPPLEMENT_PREFILL_HARNESS"),
                        "concurrent_other_workload":args.concurrent_control,
                        "profiled_diagnostic":getattr(args,'profiled',False),
                        "hidden_diagnostic":args.hidden_diagnostic,
                        "fixed_current_routes_diagnostic":policy in ("libra_frozen","libra_frozen_oracle"),
                        "oracle_kind":"VANILLA_TRACE_LOOKAHEAD_KEEP_PREDICTOR_COMPUTE" if oracle is not None else None,
                        "oracle_prediction_calls":oracle["calls"] if oracle is not None else 0}
                if policy in ("libra_oracle","libra_frozen_oracle"):
                    assert oracle["calls"]==args.layers-1, record
                    agreement=[]
                    for index,actual in oracle["actual_logits"].items():
                        saved=oracle["logits"][index]
                        if len(saved)!=len(actual):saved=saved.chunk(4,dim=0)[rank]
                        assert saved.shape==actual.shape,(index,saved.shape,actual.shape)
                        predicted_ids=saved.topk(8,dim=-1).indices
                        actual_ids=actual.topk(8,dim=-1).indices
                        agreement.append(float((predicted_ids[:,:,None]==actual_ids[:,None,:]).any(-1).float().mean()))
                    assert len(agreement)==args.layers,len(agreement)
                    record["oracle_realized_topk_recall_mean"]=sum(agreement)/len(agreement)
                    record["oracle_realized_topk_recall_min"]=min(agreement)
                    if policy=="libra_frozen_oracle":
                        assert min(agreement)==1.0,record
                if args.vl_inputs:
                    hf_logprob=hf_reference.log_softmax(-1)
                    record.update(dataset_request=local["request"]["id"],
                        hf_reference_greedy_equal=bool((logits.argmax(-1)==hf_reference.argmax()).all()),
                        hf_reference_cosine=float(torch.nn.functional.cosine_similarity(logits.float().flatten(),hf_reference,dim=0)),
                        hf_reference_max_abs=float((logits.float().flatten()-hf_reference).abs().max()),
                        hf_reference_relative_l2=float((logits.float().flatten()-hf_reference).norm()/hf_reference.norm()))
                    record['hf_reference_kl_to_policy']=float((hf_logprob.exp()*(hf_logprob-native_logprob.flatten())).sum())
                records.append(record)
                with (args.out/f"rank{rank}.jsonl").open("a") as f:f.write(json.dumps(record)+"\n")
                print(json.dumps(record),flush=True)
    (args.out/f"completed_{rank}.json").write_text(json.dumps({"started_utc":started,
        "finished_utc":datetime.now(timezone.utc).isoformat(),"physical_gpu":physical_gpu(rank),
        "records":len(records),"dummy":args.dummy,"layers":args.layers,
        "torch":torch.__version__},indent=2))
    destroy_distributed_environment()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True)
    ap.add_argument("--data",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--tokens",type=int,default=128)
    ap.add_argument("--cases",type=int,default=2)
    ap.add_argument("--warmup",type=int,default=2)
    ap.add_argument("--reps",type=int,default=3)
    ap.add_argument("--dummy",action="store_true")
    ap.add_argument("--layers",type=int,default=48)
    ap.add_argument("--vl-inputs",type=Path)
    ap.add_argument("--vl-group",default="gqa_0")
    ap.add_argument("--concurrent-control",action="store_true")
    ap.add_argument("--prediction-oracle",action="store_true",
                    help="Diagnostic vanilla-trace future knowledge; not an online policy")
    ap.add_argument("--hidden-diagnostic",action="store_true")
    ap.add_argument("--profiled",action="store_true",
                    help="NVTX-attributed profiling diagnostic; never clean speed evidence")
    ap.add_argument("--frozen-route-oracle",action="store_true",
                    help="Separate controlled replay pair fixes current routes, not model-correctness evidence")
    args=ap.parse_args()
    allowed_devices()
    if args.vl_inputs:
        manifest=json.loads((args.vl_inputs/"workload_manifest.json").read_text())
        args.tokens=next(g["M_per_source"] for g in manifest["groups"] if g["id"]==args.vl_group)
    os.environ["EXPERT_DUP_FACTOR_N"]="8"
    os.environ["EXPERT_DUP_FACTOR_L"]="4"
    os.environ["SEQ_LENS_SUM"]=str(args.tokens)
    from sglang.srt.server_args import ServerArgs,PortArgs
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    kwargs=dict(model_path=args.model,tp_size=4,dp_size=4,ep_size=4,enable_dp_attention=True,
                enable_ep_moe=True,disable_cuda_graph=True,disable_radix_cache=True,
                chunked_prefill_size=-1,attention_backend="fa3",dtype="bfloat16",
                max_total_tokens=max(512,args.tokens*2),mem_fraction_static=.95)
    if args.dummy:
        args.layers=4
        kwargs["load_format"]="dummy"
    if args.layers!=48:
        assert 4<=args.layers<48
        kwargs["json_model_override_args"]=json.dumps({"num_hidden_layers":args.layers})
    server_args=ServerArgs(**kwargs)
    _set_envs_and_config(server_args)
    port_args=PortArgs.init_new(server_args)
    args.out.mkdir(parents=True,exist_ok=True)
    task=Path(__file__).resolve().parent
    source_paths=[Path(__file__),task/'libra_prediction_oracle.py',task/'libra_vl_bridge.py']
    (args.out/'source_hashes.json').write_text(json.dumps({str(p.resolve()):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},indent=2))
    (args.out/"arguments.json").write_text(json.dumps({**{k:str(v) for k,v in vars(args).items()},
        "server_args":kwargs,"source":"USER_SUPPLIED_LIBRA_DIFF","backend":"AUTHOR_ALLGATHER_NOT_DEEPEP"},indent=2))
    ctx=mp.get_context("spawn")
    workers=[ctx.Process(target=worker,args=(r,args,server_args,port_args)) for r in range(4)]
    try:
        for p in workers:p.start()
        while any(p.is_alive() for p in workers):
            for p in workers:
                p.join(.2)
                if p.exitcode not in (None,0):raise RuntimeError(f"Own worker {p.pid} failed: {p.exitcode}")
    finally:
        for p in workers:
            if p.is_alive():p.terminate()
        for p in workers:p.join()


if __name__=="__main__":main()
