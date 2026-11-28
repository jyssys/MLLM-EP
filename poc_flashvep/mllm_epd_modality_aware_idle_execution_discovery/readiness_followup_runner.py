"""Matched live early/late, P/D reciprocal controls on a native H200 lease."""

import asyncio
import hashlib
import json
import os
from pathlib import Path
import time

import aiohttp

from ..mllm_epd_h200_prefill_heavy_batch_matched import capacity
from ..mllm_epd_prefill_heavy_modality_discovery import runner as native
from .readiness_execution import FRONTIER_KEY, contract_path, published


ARMS = ("late_p", "early_p", "late_d", "early_d")


async def poll_publications(ec, metadata, key, fixture, role, ready):
    path = contract_path(metadata, key)
    manifest_path = path.with_name("manifest.json")
    async with asyncio.timeout(45):
        while not manifest_path.is_file():
            await asyncio.sleep(.001)
        manifest = json.loads(manifest_path.read_bytes())
        images = manifest["image_hashes"]
        cut, prefix = fixture["cut_tokens"], fixture["prefix_images"]
        if (manifest["key"] != key or manifest["prompt_tokens"] != fixture["total_tokens"]
                or manifest["image_spans"][prefix-1][1] + 1 != cut):
            raise ValueError("Live E manifest differs from the selected original cut")
        value = {"key": key, "role": role, "cut": cut, "prefix_images": prefix,
                 "prompt_tokens": manifest["prompt_tokens"], "images": images}
        with path.open("x") as stream:
            json.dump(value, stream)
        stamps = []
        all_not_yet_published_ns = None
        for image in images:
            while True:
                probe = time.monotonic_ns()
                if published(ec, image):
                    break
                all_not_yet_published_ns = probe
                await asyncio.sleep(.001)
            stamps.append(time.monotonic_ns())
            if len(stamps) == prefix:
                ready.set()
    return {"prefix_published_ns": stamps[prefix-1], "all_published_ns": stamps[-1],
            "all_not_yet_published_ns": all_not_yet_published_ns,
            "published_items": len(stamps)}


async def run_arm(session, client, ec, fixture, tag, arm, output):
    row = fixture["request"]
    role = "prefill" if arm.endswith("_p") else "decode"
    headers = capacity.headers(tag, "E:full")
    key = headers["x-campaign-metadata-key"]
    metadata = ec / "processed_prompts"
    ep, pp, dp = (native.PORTS[r] for r in ("encoder", "prefill", "decode"))
    url = lambda port: f"http://127.0.0.1:{port}/v1/chat/completions"
    ready = asyncio.Event()
    start = time.monotonic_ns()

    async def produce():
        result = await capacity.stage_request(session, ep, row, tag, "E:full", client, output_count=output)
        return round(result["finish"] * 1e9)

    e_task = asyncio.create_task(produce())
    publication = asyncio.create_task(poll_publications(ec, metadata, key, fixture, role, ready))
    await_prefix = asyncio.create_task(ready.wait())
    try:
        done, _ = await asyncio.wait([await_prefix, publication, e_task], return_when=asyncio.FIRST_COMPLETED)
        for task in done & {publication, e_task}:
            task.result()
        async with asyncio.timeout(50):
            await await_prefix
        if arm.startswith("late"):
            await e_task
        intent = {FRONTIER_KEY: key}
        submit_ns = time.monotonic_ns()
        p_done_ns = None
        p_first_token = None
        if role == "prefill":
            body = client.payload(row, tag, output, stream=False)
            body.update(max_tokens=1, kv_transfer_params=dict(capacity.PULL_PARAMS) | intent)
            result = await capacity.json_reply(session.post(url(pp), json=body,
                         headers=capacity.headers(tag, "P")), "early causal P")
            p_done_ns = time.monotonic_ns()
            p_first_token = (result["choices"][0].get("token_ids") or [None])[0]
            params = dict(result.get("kv_transfer_params") or {})
            params.pop(FRONTIER_KEY, None)
            if (result["usage"]["prompt_tokens"] != fixture["total_tokens"]
                    or params.get("remote_num_tokens") != fixture["total_tokens"]
                    or not params.get("do_remote_prefill") or not params.get("remote_block_ids")):
                raise ValueError("Original P did not return a full physical KV handoff")
            await publication
        else:
            params = intent
        body = client.payload(row, tag, output)
        body["kv_transfer_params"] = params
        d_submit_ns = time.monotonic_ns()
        response = await client.one(capacity.DecodeSession(session, tag),
                    f"http://127.0.0.1:{dp}", row, body, start / 1e9, tag, 0,
                    evidence_sink=capacity.MemorySink())
        e_ack_ns, observations = await asyncio.gather(e_task, publication)
        if response.get("status") != "MEASURED_ONLINE" or response.get("output_tokens") != output:
            raise RuntimeError({k: response.get(k) for k in ("status", "HTTP_status", "output_tokens", "error")})
        ids = response.get("prompt_token_ids")
        if not ids or len(ids) != fixture["total_tokens"]:
            raise ValueError("D did not return the complete original prompt identity")
        prompt_digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        await capacity.json_reply(capacity.post_rpc(session, ep, "ec_ipc/release", {"key": key}), "E release")
        return observations | {"tag": tag, "key": key, "arm": arm, "fixture": fixture["case_id"],
                "cut": fixture["cut_tokens"], "prefix_images": fixture["prefix_images"],
                "prompt_tokens": fixture["total_tokens"], "prompt_token_ids_SHA256": prompt_digest,
                "start_ns": start, "e_ack_ns": e_ack_ns, "recipient_submit_ns": submit_ns,
                "p_done_ns": p_done_ns, "d_submit_ns": d_submit_ns, "output_token_ids": response["token_ids"],
                "p_first_token_id": p_first_token, "TTFT_ms": response["TTFT_ms"],
                "TPOT_ms": response["TPOT_ms"], "ITL_ms": response["ITL_ms"]}
    finally:
        remaining = (await_prefix, publication, e_task)
        for task in remaining:
            if not task.done():
                task.cancel()
        await asyncio.gather(*remaining, return_exceptions=True)


async def survey(group, plan, manifests, routes, args):
    source = Path(plan["readiness"]["cases"])
    fixtures = json.loads(source.read_bytes())["cases"]
    client = native.rt.module("client")
    client.messages = native.prompt_messages
    by_id = {r["request_id"]: r for rows in manifests.values() for r in rows}
    for fixture in fixtures:
        if by_id[fixture["request_id"]] != fixture["request"]:
            raise ValueError("Fixture differs from hashed original source row")
    await client.execute([fixtures[0]["request"]], [routes["native"]], "ready_warmup", 1, native.SEED, 16,
                         serial=True, unique_media_occurrence=True)
    native.rt.require(native.rt.read(group.run / "state/ready_warmup_complete.json")["passed"], "Warmup failed")
    owner = native.rt.module("ipc_ec_control")
    owner.ownership_empty("ready_warmup", "cuda_ipc")
    ec = Path("/dev/shm") / f"epd_modality_{group.env['MLLM_EPD_RUN_NONCE']}"
    pending_output = group.run / "raw" / "ready_trials.jsonl"
    if plan["readiness"].get("prime"):
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=160), trust_env=False) as session:
            for index, fixture in enumerate(fixtures):
                for arm in ("late_p", "late_d"):
                    tag = f"ready_prime_{index}_{arm}"
                    witness = await run_arm(session, client, ec, fixture, tag, arm, 16)
                    with (group.run / "raw" / "ready_prime.jsonl").open("a") as handle:
                        handle.write(json.dumps(witness) + "\n")
        owner.ownership_empty("ready_prime", "cuda_ipc")
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=160), trust_env=False) as session:
        for repeat in range(plan["readiness"].get("repetitions", 3)):
            for index, fixture in enumerate(fixtures):
                shift = (repeat + index) % len(ARMS)
                for arm in ARMS[shift:] + ARMS[:shift]:
                    tag = f"ready_{repeat}_{index}_{arm}"
                    try:
                        witness = await run_arm(session, client, ec, fixture, tag, arm, 16)
                    except Exception as exc:
                        witness = {"tag": tag, "fixture": fixture["case_id"], "arm": arm,
                                   "error_type": type(exc).__name__, "detail": str(exc)[:500]}
                        pending = exc
                    else:
                        pending = None
                    with pending_output.open("a") as handle:
                        handle.write(json.dumps(witness | {"repeat": repeat}) + "\n")
                    print("READY_RESULT", tag, witness.get("TTFT_ms"), witness.get("error_type"), flush=True)
                    if pending is not None:
                        raise pending
        owner.ownership_empty("ready_trials", "cuda_ipc")
    if args.image_ready:
        await asyncio.sleep(.3)
        encoder_info = await asyncio.to_thread(capacity.rpc_get, native.PORTS["encoder"], "worker_info")
        native.rt.write(group.run / "state" / "image_ready_followup_complete.json", encoder_info)


def main():
    launch_models = native.launch_models

    def launch(group, ec, devices, media, args):
        start_worker = group.start

        def start(role, command, **env):
            if role in native.PORTS:
                command = list(command)
                command[2] = f"{__package__}.readiness_execution"
                if role != "encoder":
                    command.extend(["--scheduler-cls", f"{__package__}.readiness_scheduler.ReadinessScheduler"])
                else:
                    env["MLLM_EPD_LATE_METADATA"] = "1"
            return start_worker(role, command, **env)

        group.start = start
        try:
            launch_models(group, ec, devices, media, args)
        finally:
            group.start = start_worker

    native.launch_models = launch
    native.survey = survey
    native.main()


if __name__ == "__main__":
    main()
