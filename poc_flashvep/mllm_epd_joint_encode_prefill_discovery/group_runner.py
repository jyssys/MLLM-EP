"""Run the CUDA-ready Encoder sweep using the previous exact same-request continuation."""

import aiohttp
import json
import os
from pathlib import Path

from ..mllm_epd_h200_prefill_heavy_batch_matched import capacity
from ..mllm_epd_modality_aware_idle_execution_discovery.readiness_followup_runner import ARMS, run_arm
from ..mllm_epd_prefill_heavy_modality_discovery import runner as native


async def survey(group, plan, manifests, routes, args, native_survey):
    config = plan["groupwise"]
    candidates = json.loads(Path(config["cases"]).read_bytes())["cases"]
    chosen = {item["case_id"]: item for item in candidates}
    fixtures = [chosen[key] for key in config["fixtures"]]
    by_id = {r["request_id"]: r for rows in manifests.values() for r in rows}
    if any(by_id[f["request_id"]] != f["request"] for f in fixtures):
        raise ValueError("Fixture differs from the original byte-validated input roster")
    client = native.rt.module("client")
    client.messages = native.prompt_messages
    owner = native.rt.module("ipc_ec_control")
    ec = Path("/dev/shm") / f"epd_modality_{group.env['MLLM_EPD_RUN_NONCE']}"
    if config.get("stock_online"):
        await native_survey(group, plan, manifests, routes, args)
    else:
        await client.execute([fixtures[0]["request"]], [routes["native"]], "groups_warmup", 1, native.SEED, 16,
                             serial=True, unique_media_occurrence=True)
        native.rt.require(native.rt.read(group.run/"state/groups_warmup_complete.json")["passed"], "Warmup failed")
        owner.ownership_empty("groups_warmup", "cuda_ipc")

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180), trust_env=False) as session:
        async def mode_set(mode, signature=False):
            reply = await capacity.json_reply(capacity.post_rpc(session, native.PORTS["encoder"], "group_settings",
                                               {"mode": mode, "signature": signature}), "Encoder group settings")
            actual, = reply
            if actual["mode"] != mode or actual["signature"] is not signature:
                raise RuntimeError("E did not accept requested grouping and observation")

        async def measure(filename, mode, fixture, tag, arm, repeat, signature=False):
            value = await run_arm(session, client, ec, fixture, tag, arm, 16,
                                  require_cuda=mode != "stock", signatures=signature)
            value.update(group_mode=mode, repeat=repeat)
            with (group.run/"raw"/filename).open("a") as output:
                output.write(json.dumps(value) + "\n")
            print("GROUP_RESULT", mode, tag, arm, value["TTFT_ms"], flush=True)

        for mi, mode in enumerate(config["modes"]):
            await mode_set(mode)
            for index, fixture in enumerate(fixtures):
                for arm in ("late_p", "late_d"):
                    await measure("group_primes.jsonl", mode, fixture, f"group_prime_{mi}_{index}_{arm}", arm, -1)
                for repeat in range(config.get("repetitions", 2)):
                    shift = (mi + index + repeat) % len(ARMS)
                    for arm in ARMS[shift:] + ARMS[:shift]:
                        await measure("group_trials.jsonl", mode, fixture,
                                      f"group_trial_{mi}_{index}_{repeat}_{arm}", arm, repeat)
            owner.ownership_empty(f"groups_{mode}", "cuda_ipc")

        for mode in config.get("signature_modes", []):
            await mode_set(mode, signature=True)
            await measure("group_signatures.jsonl", mode, fixtures[0], f"group_signature_{mode}",
                          "late_p", 0, signature=True)
        if trace := config.get("trace"):
            mode, fixture = trace["mode"], chosen[trace["fixture"]]
            await mode_set(mode)
            tag = "group_kernel_overlap"
            async with native.profile(tag, True, False, 0, False) as receipts:
                for arm in ("early_p", "early_d"):
                    await measure("group_profiled.jsonl",mode,fixture,f"{tag}_{arm}",arm,0)
            native.rt.write(group.run/"raw"/f"{tag}_profile_receipts.json",receipts)
            for arm in ("early_p", "early_d"):
                await measure("group_profile_aa.jsonl",mode,fixture,f"{tag}_aa_{arm}",arm,0)
        await mode_set("stock")
        owner.ownership_empty("groups_final", "cuda_ipc")


def main():
    from . import group_online_runner

    launch_models, native_survey, native_routes = native.launch_models, native.survey, native.launch_routes
    native.APPROVED.update(group_online_runner.POLICIES)

    def launch(group, ec, devices, media, args):
        parent = group.start

        def start(role, command, **env):
            if role in native.PORTS:
                command = list(command)
                command[2] = f"{__package__}.group_server"
                if role == "encoder":
                    env["MLLM_EPD_LATE_METADATA"] = "1"
                    option = command.index("--worker-extension-cls") + 1
                    command[option] = f"{__package__}.group_encoder.GroupedEncoderWorker"
                else:
                    scheduler = ("poc_flashvep.mllm_epd_modality_aware_idle_execution_discovery."
                                 "readiness_scheduler.ReadinessScheduler")
                    if role == "decode" and (quantum := os.environ.get("MLLM_JOINT_D_QUANTUM_ROWS")):
                        env["MLLM_JOINT_D_QUANTUM_ROWS"] = quantum
                        scheduler = f"{__package__}.guarded_scheduler.GuardedScheduler"
                    command += ["--scheduler-cls",scheduler]
            return parent(role, command, **env)

        group.start = start
        try:
            launch_models(group, ec, devices, media, args)
        finally:
            group.start = parent

    async def run(group, plan, manifests, routes, args):
        if "online_groups" in plan:
            await group_online_runner.survey(group,plan,manifests,routes,args)
        else:
            await survey(group,plan,manifests,routes,args,native_survey)

    def routes(group, plan):
        if "online_groups" in plan:
            return group_online_runner.routes(group,plan,native_routes)
        return native_routes(group,plan)

    native.launch_models, native.survey, native.launch_routes = launch, run, routes
    native.main()


if __name__ == "__main__":
    main()
