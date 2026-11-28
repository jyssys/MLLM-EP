"""Run legal D-local prefix and real P→D suffix continuations under the Native EPD lease."""

import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import aiohttp

from ..mllm_epd_h200_prefill_heavy_batch_matched import capacity, runtime as rt
from ..mllm_epd_h200_modality_aware_dynamic_split.calibration import counter_delta, get, parse_nixl_metrics
from ..mllm_epd_prefill_heavy_modality_discovery import runner as native
from .joint_contract import JOINT_KEY, token_digest

NIXL_MODULE = "poc_flashvep.mllm_epd_prefill_heavy_token_split.token_nixl"
E, P, D = (native.PORTS[role] for role in ("encoder", "prefill", "decode"))


def contract(fixture, mode):
    return {JOINT_KEY: {"mode": mode, "cut": fixture["cut_tokens"], "images": fixture["prefix_images"],
                        "prefix_sha256": fixture["prefix_token_ids_SHA256_int64_le"],
                        "full_prompt_tokens": fixture["total_tokens"]}}


class RawDecode(capacity.DecodeSession):
    def post(self, url, **kwargs):
        # Keep the normal request identity without requesting an E render handoff.
        kwargs["headers"] = {"x-request-id": self.name + ":D"}
        return self.session.post(url, **kwargs)


def body(client, fixture, tag, *, cropped, output_tokens, stream=False, suffix_only=False):
    row = fixture["request"]
    if cropped or suffix_only:
        if "content_order" in row:
            raise ValueError("Frozen physical fixture expects images followed by the original question")
        split = fixture["prefix_images"]
        images = row["images"][split:] if suffix_only else row["images"][:split]
        row = row | {"images": images}
    return client.payload(row, tag, output_tokens, stream=stream)


def validate_prefix(result, fixture):
    ids = result.get("prompt_token_ids")
    if (result["usage"]["prompt_tokens"] != fixture["cut_tokens"] or not ids
            or len(ids) != fixture["cut_tokens"]
            or token_digest(ids) != fixture["prefix_token_ids_SHA256_int64_le"]):
        raise ValueError("Actual partial LM did not process the original frozen causal prefix")


async def stage(session, port, value, tag, role):
    headers = capacity.headers(tag, role) if role != "direct" else {"x-request-id": tag + "_local"}
    return await capacity.json_reply(session.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                                                 json=value, headers=headers), role)


async def timed(awaitable):
    start = time.monotonic_ns()
    value = await awaitable
    end = time.monotonic_ns()
    return value, {"begin_ns": start, "end_ns": end, "host_ms": (end - start) / 1e6}


async def release(session, tag):
    return await capacity.json_reply(capacity.post_rpc(session, E, "ec_ipc/release", {
        "key": capacity.headers(tag, "E:full")["x-campaign-metadata-key"],
    }), "E release")


async def observe(session, port, endpoint, params):
    response = await capacity.json_reply(capacity.post_rpc(session, port, endpoint, params), endpoint)
    result, = response
    return result


async def prefix_arm(session, client, fixture, tag, arm):
    if arm == "vision_only":
        value, work = await timed(observe(session, D, "joint_vision", {"fixture": fixture["case_id"]}))
        return {"vision": value, "measurement": work}
    request = body(client, fixture, tag, cropped=True, output_tokens=1)
    encoder = None
    start = time.monotonic_ns()
    if arm == "prefill_only":
        _, encoder = await timed(stage(session, E, request, tag, "E:full"))
    request["kv_transfer_params"] = contract(fixture, "local_prefix" if arm == "joint_prefix" else "remote_prefix")
    raw, quantum = await timed(stage(session, D, request, tag, "direct" if arm == "joint_prefix" else "D"))
    validate_prefix(raw, fixture)
    retire = None
    if arm == "prefill_only":
        _, retire = await timed(release(session, tag))
    output = raw["choices"][0].get("token_ids", [])
    return {"native_encoder": encoder, "measurement": quantum,
            "ack_release": retire, "internal_sample": output,
            "total_measured_ms": (time.monotonic_ns() - start) / 1e6}


async def suffix_arm(session, client, fixture, tag, arm):
    from ..mllm_epd_prefill_heavy_token_split.token_split import PLAN_KEY

    # The role-local route deliberately qualifies measured live D-side raw preprocessing.
    compact = arm == "joint_suffix_compact"
    joint = compact or arm == "joint_suffix"
    whole = arm == "native_epd"
    row = fixture["request"]
    request = body(client, fixture, tag, cropped=joint, output_tokens=16)
    request["max_tokens"] = 1
    nixl_before = parse_nixl_metrics(await get(session, D, "/metrics", as_text=True))
    start = time.monotonic_ns()
    encoded, e_stage = await timed(stage(session, E, request, tag, "E:full"))
    request["kv_transfer_params"] = dict(capacity.PULL_PARAMS)
    if not whole:
        request["kv_transfer_params"].update(contract(fixture, "remote_prefix"))
    prefetched, p_stage = await timed(stage(session, P, request, tag, "P"))
    remote = prefetched.get("kv_transfer_params") or {}
    expected = fixture["total_tokens"] if whole else fixture["cut_tokens"]
    if remote.get("remote_num_tokens") != expected or not remote.get("remote_block_ids"):
        raise ValueError("The original NIXL exporter did not retain the intended physical P prefix")
    if not whole:
        validate_prefix(prefetched, fixture)
        remote[PLAN_KEY] = {"prefix_tokens": fixture["cut_tokens"], "full_prompt_tokens": fixture["total_tokens"]}
    is_local = joint
    if is_local:
        remote.update(contract(fixture, "local_suffix_compact" if compact else "local_suffix"))
        if compact:
            owner = capacity.headers(tag, "E:full")["x-campaign-metadata-key"]
            remote[JOINT_KEY].update(metadata_key=owner, full_sha256=fixture["full_token_sha256"])
    final = body(client, fixture, tag + "_D_owned" if compact else tag, cropped=False,
                 output_tokens=16, stream=True, suffix_only=compact)
    final["kv_transfer_params"] = remote
    endpoint = f"http://127.0.0.1:{D}"
    if is_local:
        api = RawDecode(session, tag)
    else:
        api = capacity.DecodeSession(session, tag)
    result, d_stage = await timed(client.one(api, endpoint, row, final, start / 1e9, tag, 0,
                                             evidence_sink=capacity.MemorySink()))
    if (result.get("status") != "MEASURED_ONLINE" or result.get("output_tokens") != 16
            or token_digest(result.get("prompt_token_ids", [])) != fixture["full_token_sha256"]):
        raise RuntimeError("Fixed16 suffix/full EPD path failed or changed the original prompt: "
                           + str(result.get("error", result.get("usage"))))
    nixl_after = parse_nixl_metrics(await get(session, D, "/metrics", as_text=True))
    for _ in range(4):
        if nixl_after.get("transfers", 0) > nixl_before.get("transfers", 0):
            break
        await asyncio.sleep(0.05)
        nixl_after = parse_nixl_metrics(await get(session, D, "/metrics", as_text=True))
    nixl = counter_delta(nixl_before, nixl_after)
    if not nixl or not nixl.get("transfers"):
        nixl = None
    else:
        for name in ("transfer_ms", "post_ms"):
            if name in nixl:
                nixl[name] *= 1e3
    _, release_phase = await timed(release(session, tag))
    blocks = remote["remote_block_ids"]
    block_count = sum(len(group) for group in blocks) if isinstance(blocks[0], (list, tuple)) else len(blocks)
    full_bytes = block_count * capacity.KV_BLOCK_TOKENS * capacity.KV_BYTES_PER_TOKEN
    visual = row["visual_tokens"]
    suffix_visual = sum(img["visual_tokens"] for img in row["images"][fixture["prefix_images"]:])
    return {"native_encoder": e_stage, "prefill": p_stage, "measurement": d_stage,
            "ack_release": release_phase, "prefix_KV_tokens": expected, "padded_KV_bytes": full_bytes, "nixl": nixl,
            "E_actual_prompt_tokens": encoded["usage"]["prompt_tokens"],
            "E_encoded_images": fixture["prefix_images"] if joint else row["image_count"],
            "D_local_encoded_images": row["image_count"] - fixture["prefix_images"] if joint else 0,
            "D_local_visual_tokens": suffix_visual if joint else 0,
            "vision_IPC_logical_bytes": (visual - suffix_visual if joint else visual) * capacity.VISION_BYTES_PER_TOKEN,
            "output_token_ids": result["token_ids"], "TTFT_ms": result["TTFT_ms"],
            "TPOT_ms": result["TPOT_ms"], "ITL_ms": result["ITL_ms"], "E2E_ms": result["E2E_ms"]}


class WatchedDecode(capacity.DecodeSession):
    def __init__(self, session, name):
        super().__init__(session, name)
        self.first = asyncio.Event()

    @asynccontextmanager
    async def post(self, url, **kwargs):
        from ..mllm_epd_prefill_heavy_token_split.idle_admission import FirstToken

        parser = FirstToken()
        async with super().post(url, **kwargs) as response:
            async def chunks():
                async for chunk in response.content.iter_any():
                    if not self.first.is_set() and parser.feed(chunk):
                        self.first.set()
                    yield chunk
            yield SimpleNamespace(status=response.status, text=response.text, content=SimpleNamespace(iter_any=chunks))


async def interference(session, client, native_row, fixture, tag, mode):
    """Run actual decode with a pre-exported KV and record joint-start or active rejection."""
    await capacity.stage_request(session, E, native_row, tag, "E:full", client, output_count=128)
    export = await capacity.stage_request(session, P, native_row, tag, "P", client, export=True, output_count=128)
    decode_request = client.payload(native_row, tag, 128)
    decode_request["kv_transfer_params"] = export["kv_transfer_params"]
    watched = WatchedDecode(session, tag)
    prefix = None
    prefix_task = None
    if mode == "idle":
        prefix = await prefix_arm(session, client, fixture, tag + "_prefix", "joint_prefix")
    if mode == "upcoming":
        prefix_task = asyncio.create_task(prefix_arm(session, client, fixture, tag + "_prefix", "joint_prefix"))
        # The actual comparison uses timestamps for the later NIXL submission.
        await asyncio.sleep(0.020)
    submitted = time.monotonic()
    decoding = asyncio.create_task(client.one(
        watched, f"http://127.0.0.1:{D}", native_row, decode_request, submitted, tag, 0,
        evidence_sink=capacity.MemorySink(),
    ))
    rejected = None
    if mode == "active":
        active_observed = asyncio.create_task(watched.first.wait())
        done, _ = await asyncio.wait((decoding, active_observed), return_when=asyncio.FIRST_COMPLETED)
        if active_observed not in done:
            active_observed.cancel()
            raise RuntimeError("Unable to witness the active decode policy")
        if decoding.done():
            raise RuntimeError("Active decode was already retired at the decision point")
        rejected = time.monotonic_ns()
    result, *extra = await capacity.collect_all(decoding, *([prefix_task] if prefix_task else []))
    if extra:
        prefix, = extra
    if result["status"] != "MEASURED_ONLINE" or result["output_tokens"] != 128:
        raise RuntimeError(f"Native interference decode did not finish Fixed128: {result.get('error')}")
    await release(session, tag)
    return {"mode": mode, "tag": tag, "case_id": fixture["case_id"], "scope": "Physical intervention",
            "native_D_dispatch": result["dispatch"], "native_D_first_token": result["token_times"][0],
            "native_D_finish": result["finish"], "native_D_TTFT_ms_from_submit": result["TTFT_ms"],
            "native_D_TPOT_ms": result["TPOT_ms"], "native_D_ITL_ms": result["ITL_ms"],
            "native_D_explicit_tokens": result["token_ids"], "joint_prefix": prefix,
            "active_rejection_ns": rejected}


async def compare_correctness(session, client, fixture, tag, raw):
    from safetensors.torch import load_file
    import numpy as np
    import torch

    variants = {}
    for arm in ("prefill_only", "joint_prefix"):
        name = tag + "_" + arm
        await observe(session, D, "llm_probe/begin", {"name": name, "max_logits": 1, "include_decode_inputs": True})
        try:
            await prefix_arm(session, client, fixture, name, arm)
        finally:
            await observe(session, D, "llm_probe/end", {})
        manifest = json.loads((raw / "llm_reference" / name / "manifest.json").read_bytes())
        variants[arm] = (manifest, load_file(manifest["tensor_file"]))
    control, candidate = (variants[name][1] for name in ("prefill_only", "joint_prefix"))
    cut = fixture["cut_tokens"]
    with np.load(fixture["request"]["token_metadata_file"]) as original:
        positions = torch.from_numpy(original["positions"].reshape(3, -1)[:, :cut]).to(dtype=torch.int64)
    position_match = {name: torch.equal(t["full_prompt_mrope"].reshape(3, -1).to(dtype=torch.int64), positions)
                      for name, (_, t) in variants.items()}
    layer_exact = [torch.equal(control[f"prompt_KV_{i}"], candidate[f"prompt_KV_{i}"]) for i in range(28)]
    return {"scope": "Separate single-request correctness arm with CPU tensor captures. Not timed.",
            "position_match": position_match,
            "inputs_embeds_equal": torch.equal(control["inputs_embeds_0"], candidate["inputs_embeds_0"]),
            "layer_KV_exact": layer_exact,
            "top32_values_indices_equal": variants["prefill_only"][0]["logits"] == variants["joint_prefix"][0]["logits"]}


async def suffix_correctness(session, client, fixture, teacher, output):
    from safetensors import safe_open

    from ..mllm_epd_prefill_heavy_modality_discovery.logit_pair import tensor_metrics

    names = {arm: f"joint_teacher_{fixture['case_id']}_{arm}" for arm in
             ("native_epd", "native_repeat", "generic_token_suffix", "joint_suffix", "joint_suffix_compact")}
    probe = f"joint_teacher_{fixture['case_id']}"
    receipt = await observe(session, D, "greedy_probe/begin", {
        "name": probe, "ids": list(names.values()),
        "targets": {tag: {"steps": len(teacher) + 1, "prefix": teacher} for tag in names.values()},
    })
    try:
        for arm, tag in names.items():
            trial = await suffix_arm(session, client, fixture, tag, "native_epd" if arm == "native_repeat" else arm)
            if trial["output_token_ids"][:-1] != teacher:
                raise ValueError("Physical suffix diagnostic did not receive the shared teacher prefix")
    finally:
        await observe(session, D, "greedy_probe/end", {})
    path = Path(receipt["folder"]) / "manifest.json"
    manifest = rt.read(path)
    witness = Path(manifest["receipt"]["path"])
    if rt.sha(witness) != manifest["receipt"]["sha256"]:
        raise ValueError("Joint full-vocabulary logit tensor digest mismatch")
    result = {}
    with safe_open(witness, framework="pt", device="cpu") as tensors:
        for arm, tag in names.items():
            steps = []
            for step in range(len(teacher) + 1):
                control_id = f"{names['native_epd']}.{step}"
                candidate_id = f"{tag}.{step}"
                control = manifest["samples"][control_id]
                candidate = manifest["samples"][candidate_id]
                same = ("input_token_id", "mrope_position_ids")
                if any(control[key] != candidate[key] for key in same):
                    raise ValueError(f"Suffix correctness input or MRoPE differed at token {step}")
                steps.append({
                    "step": step + 1, "input_token_id": candidate["input_token_id"],
                    "mrope_position_ids": candidate["mrope_position_ids"],
                    "native_greedy": control["greedy_id"], "greedy_id": candidate["greedy_id"],
                    "native_margin": control["top1_top2_margin"], "margin": candidate["top1_top2_margin"],
                    "logits": tensor_metrics(tensors.get_tensor(control_id + ".logits"),
                                             tensors.get_tensor(candidate_id + ".logits")),
                })
            result[arm] = steps
    rt.write(output, {"evidence": "CORRECTNESS_ONLY", "fixture": fixture["case_id"],
                      "scope": "Shared actual Native teacher prefix, original logits before forced sampling.",
                      "cases": result, "manifest_SHA256": rt.sha(path)})


async def survey(group, plan, manifests, routes, args):
    import numpy as np

    run = group.run
    config = plan["joint"]
    fixtures = rt.read(Path(config["cases"]))["cases"]
    rows = {row["request_id"]: row for records in manifests.values() for row in records}
    if config.get("only"):
        fixtures = [fixture for fixture in fixtures if fixture["case_id"] in config["only"]]
    for fixture in fixtures:
        if rows[fixture["request_id"]] != fixture["request"]:
            raise ValueError("Joint input contract diverged from validated original request")
        with np.load(fixture["request"]["token_metadata_file"]) as tokens:
            fixture["full_token_sha256"] = token_digest(tokens["input_ids"].reshape(-1).tolist())
    client = rt.module("client")
    client.messages = native.prompt_messages
    await client.execute([fixtures[0]["request"]], [routes["native"]], "joint_warmup", 1, native.SEED, 16,
                         serial=True, unique_media_occurrence=True)
    rt.require(rt.read(run / "state/joint_warmup_complete.json")["passed"], "Native warmup failed")
    owner = rt.module("ipc_ec_control")
    owner.ownership_empty("joint_warmup", "cuda_ipc")
    log = run / "raw/joint_trials.jsonl"

    def record(value):
        with log.open("a") as stream:
            stream.write(json.dumps(value) + "\n")
        print("JOINT", value["tag"], value.get("TTFT_ms", value.get("measurement", {}).get("host_ms")),
              value.get("error"), flush=True)

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=140), trust_env=False) as session:
        teachers = {}
        for index, fixture in enumerate(fixtures):
            prefix = fixture["test"] == "prefix"
            if prefix:
                all_arms = ("vision_only", "prefill_only", "joint_prefix")
            else:
                all_arms = ("native_epd", "generic_token_suffix", "joint_suffix")
                if config.get("include_compact_suffix"):
                    all_arms += ("joint_suffix_compact",)
            if config.get("arms"):
                all_arms = tuple(arm for arm in all_arms if arm in config["arms"])
            invoke = prefix_arm if prefix else suffix_arm
            if not all_arms:
                continue
            for repeat in range(-1, config.get("repetitions", 5)):
                shift = (repeat + index) % len(all_arms)
                for arm in all_arms[shift:] + all_arms[:shift]:
                    tag = f"joint_{index}_{repeat+1}_{arm}"
                    common = {"tag": tag, "arm": arm, "fixture": fixture["case_id"], "repeat": repeat,
                              "cut_tokens": fixture["cut_tokens"], "prefix_images": fixture["prefix_images"],
                              "total_images": fixture["total_images"], "total_tokens": fixture["total_tokens"],
                              "warmup": repeat < 0, "test": fixture["test"]}
                    try:
                        result = await invoke(session, client, fixture, tag, arm)
                    except Exception as exc:
                        record(common | {"error": str(exc)[:500], "error_type": type(exc).__name__})
                        raise
                    record(common | result)
                    if not prefix and repeat >= 0 and arm == "native_epd":
                        teachers[fixture["case_id"]] = result["output_token_ids"][:-1]
        for index, fixture in enumerate(fixtures):
            if fixture["test"] == "prefix" and fixture["case_id"] in config.get("correctness", ()):
                result = await compare_correctness(session, client, fixture, f"joint_reference_{index}", run / "raw")
                rt.write(run / "analysis" / f"joint_correctness_{fixture['case_id']}.json", result)
        for fixture in fixtures:
            if fixture["test"] != "prefix" or fixture["case_id"] not in config.get("diagnostics", ()):
                continue
            for arm in ("prefill_only", "joint_prefix"):
                tag = f"joint_diag_{fixture['case_id']}_{arm}"
                await observe(session, D, "joint_profile", {"tag": tag})
                try:
                    result = await prefix_arm(session, client, fixture, tag, arm)
                finally:
                    timeline = await observe(session, D, "joint_profile", {"tag": None})
                rt.write(run / "analysis" / f"{tag}.json", {"request": result, "diagnostic": timeline})
        for fixture in fixtures:
            if fixture["case_id"] in config.get("suffix_correctness", ()):
                await suffix_correctness(session, client, fixture, teachers[fixture["case_id"]],
                                         run / "analysis" / f"joint_teacher_{fixture['case_id']}.json")
        for check in config.get("interference", ()):
            fixture = next(case for case in fixtures if case["case_id"] == check["fixture"])
            native_row = rows[check["native_request"]]
            for repeat in range(check["repetitions"]):
                for mode in ("control", "idle", "upcoming", "active"):
                    tag = f"joint_interference_{fixture['case_id']}_{repeat}_{mode}"
                    result = await interference(session, client, native_row, fixture, tag, mode)
                    with (run / "raw" / "joint_interference.jsonl").open("a") as stream:
                        stream.write(json.dumps(result) + "\n")
                    print("JOINT_INTERFERENCE", tag, result["native_D_TTFT_ms_from_submit"], flush=True)
    owner.ownership_empty("joint_trials", "cuda_ipc")


def main():
    launch = native.launch_models

    def launch_joint(group, ec, devices, media, args):
        original = group.start
        plan = rt.read(args.plan)
        cases = Path(plan["joint"]["cases"]).resolve(strict=True)

        def start(role, command, **env):
            if role in ("prefill", "decode"):
                command = list(command)
                command[2] = f"{__package__}.joint_server"
                command[command.index("--worker-extension-cls") + 1] = f"{__package__}.joint_worker.JointWorker"
                index = command.index("--kv-transfer-config") + 1
                cfg = json.loads(command[index])
                cfg.update(kv_connector="ExactTokenNixlConnector", kv_connector_module_path=NIXL_MODULE)
                command[index] = json.dumps(cfg)
                if role == "decode":
                    index = command.index("--ec-transfer-config") + 1
                    cfg = json.loads(command[index])
                    cfg.update(ec_connector="JointEncoderConnector", ec_connector_module_path=f"{__package__}.joint_connector")
                    command[index] = json.dumps(cfg)
                    env["MLLM_JOINT_CASES"] = str(cases)
            return original(role, command, **env)

        group.start = start
        try:
            launch(group, ec, devices, media, args)
        finally:
            group.start = original

    native.launch_models = launch_joint
    native.survey = survey
    native.main()


if __name__ == "__main__":
    main()
