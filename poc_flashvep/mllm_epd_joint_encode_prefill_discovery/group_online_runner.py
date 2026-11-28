"""Open-loop paired controls for a selected, physically validated Encoder group."""

import asyncio
import json
from pathlib import Path
import sys

import aiohttp

from ..mllm_epd_h200_prefill_heavy_batch_matched import capacity
from ..mllm_epd_prefill_heavy_modality_discovery import runner as native


POLICIES = {f"group_{name}": name for name in ("late_p", "early_p", "guarded_d")}


def routes(group, plan, native_routes):
    found = native_routes(group, {**plan, "cases": [case for case in plan["cases"] if case.get("route") not in POLICIES]})
    ec = Path("/dev/shm") / f"epd_modality_{group.env['MLLM_EPD_RUN_NONCE']}"
    for index, name in enumerate(sorted({case.get("route") for case in plan["cases"]} & POLICIES.keys())):
        port = 18854 + index
        tag = f"proxy_{name}"
        proxy = group.start(tag, [sys.executable,"-m",f"{__package__}.group_online_proxy",
            "--policy",POLICIES[name],"--port",str(port),"--ec-path",str(ec),
            "--trace",str(group.run/"raw"/f"proxy_{name}.jsonl")])
        group.ready(proxy,port,tag)
        found[name] = f"http://127.0.0.1:{port}"
    return found


async def survey(group, plan, manifests, routes_by_name, args):
    run = group.run
    client, metrics, owner = (native.rt.module(name) for name in ("client","metrics","ipc_ec_control"))
    client.messages = native.prompt_messages
    warm = next(iter(manifests.values()))[0]
    await client.execute([warm],[routes_by_name["native"]],"warmup",1,native.SEED,16,
                         serial=True,unique_media_occurrence=True)
    native.rt.require(native.rt.read(run/"state/warmup_complete.json")["passed"],"Warmup failed")
    owner.ownership_empty("warmup","cuda_ipc")
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30),trust_env=False) as session:
        for case in plan["cases"]:
            route = case.get("route","native")
            mode = plan["online_groups"]["mode"] if route in POLICIES else "stock"
            setting, = await capacity.json_reply(capacity.post_rpc(session,native.PORTS["encoder"],
                                                          "group_settings",{"mode":mode}),"Set online Encoder group")
            if setting["mode"] != mode:
                raise RuntimeError("Invalid online Encoder group setting")
            rows, tag = native.roster(case,manifests),case["tag"]
            qps, output = case["qps"],case.get("output")
            cohort = {"tag":tag,"group":tag,"source":case.get("source","natural"),
                      "cohort_id":case["cohort_id"],"qps":qps,"output":output,"count":len(rows),"route":route,
                      "observer":"clean","cache_gib":args.kv_gib,"arrival":"poisson","mode":mode,"ec_policy":"inline",
                      "manifest_sha256":native.rt.sha(case["manifest"]),"p_budget":args.p_budget,"d_budget":args.d_budget}
            with (run/"configs/cohorts.jsonl").open("a") as stream:
                stream.write(json.dumps(cohort)+"\n")
            if case.get("prime_count",0):
                prime = tag+"_prime"
                await client.execute(rows[:case["prime_count"]],[routes_by_name[route]],prime,qps,native.SEED,output,
                                     defer_raw=True,unique_media_occurrence=True)
                native.rt.require(native.rt.read(run/"state"/f"{prime}_complete.json")["passed"],"Case warmup failed")
                owner.ownership_empty(prime,"cuda_ipc")
            before = await asyncio.to_thread(native.metrics_snapshot)
            with native.sample_scheduler(run/"raw"/f"{tag}_scheduler.jsonl",args.queue_period), \
                    native.sample_gpm(run/"raw"/f"{tag}_gpm.json",native.GPU_MAP,args.gpm_period):
                result = await client.execute(rows,[routes_by_name[route]],tag,qps,native.SEED,output,
                                              defer_raw=True,unique_media_occurrence=True,metadata=cohort)
            after = await asyncio.to_thread(native.metrics_snapshot)
            native.rt.write(run/"raw"/f"{tag}_prometheus.json",{"before":before,"after":after})
            summary = metrics.summarize(tag)
            print("GROUP_ONLINE",tag,len(result),summary.get("gate_attainment"),summary.get("TTFT_p95_ms"),flush=True)
            owner.ownership_empty(tag,"cuda_ipc")
