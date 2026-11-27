"""Online Qwen BF16 EPD controls for exact GPU and host memory tiers."""

import argparse
import asyncio
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request

from ..mllm_epd_h200_prefill_heavy_batch_matched import runtime as rt
from ..mllm_epd_h200_prefill_heavy_batch_matched.resources import burner_status, experiment_lease
from ..mllm_epd_h200_wide_decode_batch_token_split.host import snapshot
from ..mllm_epd_h200_wide_decode_batch_token_split.online import configure, fingerprint_source

PACKAGE = "poc_flashvep.mllm_epd_memory_asymmetry_discovery"
GIB = 1024**3
ARRIVAL_SEED = 2026112402
METRIC_NAME = re.compile(r"^(vllm:[^ {]+)(?:\{[^}]*\})? ([^ ]+)(?: |$)")


def configure_kv(command, role, args):
    if role in args.apc_roles:
        command[command.index("--no-enable-prefix-caching")] = "--enable-prefix-caching"
        command += ["--prefix-caching-hash-algo", "sha256"]
    if role != "prefill" or args.tier == "none":
        return
    index = command.index("--kv-transfer-config") + 1
    native = json.loads(command[index])
    extra = {"cpu_bytes_to_use": int(args.tier_gib * GIB), "eviction_policy": args.eviction,
             "store_threshold": args.store_threshold}
    if args.tier == "peer":
        extra.update(spec_name="PeerOffloadingSpec", spec_module_path=f"{PACKAGE}.peer_kv",
                     tier_device=1, tier_guard_bytes=int(args.tier_guard_gib * GIB))
    command[index] = json.dumps({
        "kv_connector": "MultiConnector", "kv_role": "kv_both",
        "kv_connector_extra_config": {"connectors": [native, {
            "kv_connector": "OffloadingConnector", "kv_role": "kv_both", "kv_connector_extra_config": extra,
        }]},
    })


def launch(group, ec, devices, media, args):
    servers = rt.module("servers")
    model_ready = []
    for role, gpu, port, side_port in rt.ROLES:
        encoder = role == "encoder"
        matched = args.stable or args.matched_renderer
        command = servers.command(
            role, port, ec, side_port, observe=True, encoder_native_ops=encoder,
            compiled_encoder=encoder, encoder_sync_scheduling=encoder,
            kv_cache_gib=min(42, args.p_kv_gib) if role == "prefill" else args.d_kv_gib if role == "decode" else 42,
            mm_cache_gb=(.25 if matched else 0) if encoder else 1,
            renderer_workers=2 if encoder and not matched else 1,
            ec_transport="cuda_ipc", fixed_reduction=False, deterministic_reductions=False,
        )
        command[command.index("--allowed-local-media-path") + 1] = str(media)
        if role == "prefill" and args.p_kv_gib > 42:
            command[command.index("--kv-cache-memory-bytes") + 1] = str(int(args.p_kv_gib * GIB))
        configure_kv(command, role, args)
        command.append("--enable-prompt-tokens-details")
        if args.stable:
            from .stable_ec import configure_connector

            options = {"cache_device": args.encoder_cache_device} if args.encoder_cache_device != "gpu" else {}
            configure_connector(command, role, cap_bytes=int(args.ec_gib * GIB),
                                reuse_encoder=args.encoder_cache, **options)
        label = f"native_{role}"
        exposed = devices[gpu]
        if role == "prefill" and args.tier == "peer":
            exposed += f",{devices[4]}"
        compiler_root = (group.run.parent / "epd_runtime_cache" if args.shared_compiler_cache
                         else group.run / "runtime_cache")
        process = group.start(
            label, command, CUDA_VISIBLE_DEVICES=exposed, VLLM_NIXL_SIDE_CHANNEL_PORT=str(side_port),
            MLLM_EPD_ROLE=role, MLLM_EPD_TOKEN_SPLIT="0", VLLM_CACHE_ROOT=str(compiler_root / role),
            MLLM_EPD_EC_PATH=str(ec), MLLM_EPD_METADATA_PATH=str(ec / "processed_prompts"),
        )
        model_ready.append((role, gpu, port, label, process))
    for role, _, port, label, process in model_ready:
        group.ready(process, port, label, role=role)
    owners = {role: process.pid for role, _, _, _, process in model_ready}
    allowed = {devices[gpu]: [owners[role]] for role, gpu, _, _, _ in model_ready}
    if args.tier == "peer":
        allowed[devices[4]].append(owners["prefill"])
    occupancy = rt.nvidia("--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory")
    for gpu_uuid, processes in allowed.items():
        pids = [int(row[1]) for row in occupancy if row[0] == gpu_uuid]
        rt.require(pids and all(any(rt.pid_owned(pid, owner, group.env["MLLM_EPD_RUN_NONCE"])
                                  for owner in processes) for pid in pids), f"Unowned worker on {gpu_uuid}")
    rt.write(group.run / "state/gpu_owners.json", occupancy)
    if not args.stable:
        rt.module("native_common_control").set_common_ipc_policy()
        return rt.launch_proxy(group, "epd", "epd", 18847, decode_cap="free")

    rpc = rt.module("llm_probe_control").rpc
    for port in (18841, 18842):
        rpc(port, "ec_ipc/ack_policy", {"mode": "file"})
    command = [sys.executable, "-m", f"poc_flashvep.{rt.PACKAGE}.proxy", "--mode", "epd", "--name", "epd",
               "--encoder", "http://127.0.0.1:18840", "--prefill", "http://127.0.0.1:18841",
               "--decoders", "http://127.0.0.1:18842", "--port", "18847", "--processor-handoff", "grid_only",
               "--release-encoder-ipc"]
    proxy = group.start("epd", command)
    group.ready(proxy, 18847, "epd")
    return "http://127.0.0.1:18847"


def metrics_snapshot():
    result = {"monotonic": time.monotonic(), "roles": {}}
    for role, _, port, _ in rt.ROLES:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=3) as response:
            body = response.read().decode()
        result["roles"][role] = [line for line in body.splitlines() if line.startswith("vllm:")]
    return result


def numeric_metrics(snapshot_data):
    numeric = {}
    for role, lines in snapshot_data["roles"].items():
        values = {}
        for line in lines:
            parsed = METRIC_NAME.match(line)
            if parsed:
                name, value = parsed.groups()
                try:
                    number = float(value)
                    values[name] = values.get(name, 0.0) + number
                    if direction := re.search(r'transfer_type="([A-Za-z_]+)"', line):
                        directional = f"{name}:{direction.group(1)}"
                        values[directional] = values.get(directional, 0.0) + number
                except ValueError:
                    continue
        numeric[role] = values
    return numeric


@contextmanager
def sample_metrics(outfile, seconds):
    stopped = threading.Event()

    def sample():
        with outfile.open("x") as stream:
            while not stopped.is_set():
                try:
                    observation = numeric_metrics(metrics_snapshot())
                    wanted = ("kv_cache_usage", "num_preemptions", "num_requests", "prefix_cache", "offload")
                    observation = {role: {k: v for k, v in counters.items() if any(p in k for p in wanted)}
                                   for role, counters in observation.items()}
                    entry = {"monotonic": time.monotonic(), "values": observation}
                except (OSError, ValueError) as error:
                    entry = {"monotonic": time.monotonic(), "error": str(error)}
                stream.write(json.dumps(entry) + "\n")
                stream.flush()
                stopped.wait(seconds)

    if seconds == 0:
        yield
        return
    thread = threading.Thread(target=sample)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()


def checked_manifest(file, media):
    data = rt.read(file)
    rows = data["requests"]
    rt.require(rows and len({row["request_id"] for row in rows}) == len(rows), "Missing/duplicate roster IDs")
    verified = {}
    for row in rows:
        rt.require(len(row["images"]) == row["image_count"] > 0, "Expected non-empty original image turns")
        for item in row["images"]:
            path = Path(item["path"]).resolve(strict=True)
            rt.require(path.is_relative_to(media), f"Media outside allowed workspace: {path}")
            if path not in verified:
                verified[path] = rt.sha(path)
            rt.require(item.get("SHA256", verified[path]) == verified[path], f"Image hash changed: {path}")
            item["SHA256"] = verified[path]
    return rows


def measurements(run, rows, warm, url, args):
    client = rt.module("client")
    metrics = rt.module("metrics")
    native_payload = client.payload
    cache_scope = os.environ["MLLM_EPD_RUN_NONCE"]
    if args.stable:
        from ..mllm_specific_disaggregation_discovery.native_epd import prompt_messages

        def stable_payload(row, nonce, output_count=None, stream=True):
            result = native_payload(row, nonce, output_count, stream)
            history = prompt_messages(row, nonce)
            indices = ([part["image_index"] for part in row["content_order"] if part["type"] == "image"]
                       if "content_order" in row else range(len(row["images"])))
            image_messages = [part for part in history[0]["content"] if part["type"] == "image_url"]
            ordered_image_hash = hashlib.sha256(
                "".join(row["images"][index]["SHA256"] for index in indices).encode()
            ).hexdigest()
            for part, image_index in zip(image_messages, indices, strict=True):
                identity = row["images"][image_index]["SHA256"]
                if args.identity == "document":
                    identity = f"{ordered_image_hash}:{image_index}:{identity}"
                key = f"{cache_scope}:{rt.MODEL_REVISION}:default-image-v1:{identity}"
                part["uuid"] = hashlib.sha256(key.encode()).hexdigest()
            result.update(messages=history, cache_salt=cache_scope)
            return result

        client.payload = stable_payload

    def cohort(label, roster, qps, output, serial):
        before = metrics_snapshot()
        if args.stable:
            before["encoder_cache"] = rt.http_json(18840, "campaign/ec_ipc/state", timeout=15)
        with sample_metrics(run / "raw" / f"{label}_scheduler.jsonl", args.sample_period if not serial else 0):
            response = asyncio.run(client.execute(
                roster, [url], label, qps, ARRIVAL_SEED, output, serial=serial, defer_raw=True,
                unique_media_occurrence=not args.stable,
                metadata={"stable_scope": bool(args.stable), "P_APC": "prefill" in args.apc_roles,
                          "D_APC": "decode" in args.apc_roles, "native_encoder_reuse": args.encoder_cache,
                          "prefix_tier": args.tier, "legacy_top_level_client_cache_fields": "Not authoritative"},
            ))
        after = metrics_snapshot()
        if args.stable:
            after["encoder_cache"] = rt.http_json(18840, "campaign/ec_ipc/state", timeout=15)
        rt.write(run / "raw" / f"{label}_prometheus.json", {"before": before, "after": after})
        summary = metrics.summarize(label)
        print("ONLINE_SUMMARY", label, json.dumps(summary), flush=True)
        rt.require(len(response) == len(roster) and summary["primary_execution_valid"], f"Incomplete cohort: {label}")

    try:
        cohort("runtime_warmup", warm[:1], 1, 16, True)
        if args.kv_probe:
            from .kv_correctness import execute

            execute(run, rows, cohort, metrics_snapshot, numeric_metrics)
            return
        if args.cache_warm_count:
            cohort("cache_warmup", rows[:args.cache_warm_count], args.qps, 16, False)
        for index in range(args.cycles):
            cohort(f"measured_{index + 1}", rows, args.qps, args.output, False)
    finally:
        client.payload = native_payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-repo", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tier", choices=("none", "cpu", "peer"), default="none")
    parser.add_argument("--tier-gib", type=float, default=8)
    parser.add_argument("--tier-guard-gib", type=float, default=32)
    parser.add_argument("--eviction", choices=("lru", "arc"), default="lru")
    parser.add_argument("--store-threshold", type=int, default=0)
    parser.add_argument("--p-kv-gib", type=float, default=42)
    parser.add_argument("--d-kv-gib", type=float, default=42)
    parser.add_argument("--stable", action="store_true")
    parser.add_argument("--encoder-cache", action="store_true")
    parser.add_argument("--encoder-cache-device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--matched-renderer", action="store_true")
    parser.add_argument("--apc-roles", nargs="*", choices=("prefill", "decode"), default=[])
    parser.add_argument("--ec-gib", type=float, default=12)
    parser.add_argument("--identity", choices=("document", "image"), default="document")
    parser.add_argument("--qps", type=float, default=2.5)
    parser.add_argument("--output", type=int, choices=(16, 64, 128, 256))
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--kv-probe", action="store_true")
    parser.add_argument("--cache-warm-count", type=int, default=0)
    parser.add_argument("--sample-period", type=float, default=0)
    parser.add_argument("--shared-compiler-cache", action="store_true",
                        help="Reuse immutable compiler artifacts across independent engine starts")
    parser.add_argument("--wall-seconds", type=int, default=1200)
    args = parser.parse_args()
    if (args.qps <= 0 or args.cycles <= 0 or args.cache_warm_count < 0 or args.ec_gib <= 0
            or not 0 < args.p_kv_gib <= 90 or not 0 < args.d_kv_gib <= 42
            or args.tier != "none" and ("prefill" not in args.apc_roles or args.tier_gib <= 0)
            or (args.apc_roles or args.encoder_cache) and not args.stable
            or args.encoder_cache_device == "cpu" and not args.encoder_cache
            or args.kv_probe and args.tier != "peer"):
        parser.error("Prefix tiers require P APC and verified stable input identity")
    base = Path(__file__).resolve().parents[2]
    media = base.parent
    asset, receipt, manifest, run = map(Path.resolve, (args.asset_repo, args.receipt, args.manifest, args.run_dir))
    rt.require(run.is_relative_to(media) and not run.is_relative_to(base), "Use a fresh external run directory")
    rows = checked_manifest(manifest, media)
    rt.git_check(asset)
    model, old_manifests = rt.verify_inputs(asset, receipt)
    warm = checked_manifest(old_manifests[rt.CALIBRATION], media)
    inventory = {int(gpu): (uuid, name) for gpu, uuid, name in rt.nvidia("--query-gpu=index,uuid,name")}
    devices = {gpu: inventory[gpu][0] for _, gpu, _, _ in rt.ROLES}
    rt.require(all(inventory[gpu][1] == "NVIDIA H200" for gpu in devices), "Expected the registered H200 group")
    deadline = time.time() + args.wall_seconds
    run.mkdir(mode=0o700)
    for directory in ("state", "logs", "raw", "analysis", "data", "configs", "references"):
        (run / directory).mkdir()
    shutil.copyfile(manifest, run / "data/frozen_manifest.json")
    rt.write(run / "state/model_download_complete.json", {"directory": str(model), "revision": rt.MODEL_REVISION})
    source, settings = configure(base, asset, run, deadline, freeze_auxiliary=True)
    rt.write(run / "state/environment.json", {
        "runtime": rt.runtime_inventory(), "HEAD": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=base, text=True).strip(),
        "manifest_SHA256": rt.sha(manifest), "worker_source": fingerprint_source(source),
        "campaign_source": {p.name: rt.sha(p) for p in (source / Path(*PACKAGE.split("."))).glob("*.py")},
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "authorized_uuid_by_gpu": devices,
    })
    ec = Path("/dev/shm") / f"epd_cache_{settings['MLLM_EPD_RUN_NONCE']}"
    group = rt.OwnedProcesses(run, os.environ.copy(), source, deadline)

    def deadline_handler(*_):
        signal.setitimer(signal.ITIMER_REAL, 0)
        raise TimeoutError("Native online cache experiment cancelled or reached its deadline")

    signal.signal(signal.SIGALRM, deadline_handler)
    signal.signal(signal.SIGTERM, deadline_handler)
    signal.setitimer(signal.ITIMER_REAL, max(.1, deadline - time.time()))
    try:
        rt.write(run / "state/burners_before.json", burner_status())
        rt.write(run / "state/host_before.json", snapshot(topology=True))
        with experiment_lease(devices, run / "state") as stopped:
            gpu_inventory, occupancy = rt.verify_devices(devices)
            rt.write(run / "state/lease.json", {"stopped_burners": stopped, "gpu_inventory": gpu_inventory,
                                                 "other_gpu_processes": occupancy})
            rt.verify_ports()
            ec.mkdir(mode=0o700)
            try:
                url = launch(group, ec, devices, media, args)
                measurements(run, rows, warm, url, args)
                rt.write(run / "state/workers_after_drain.json", {
                    role: rt.http_json(port, "campaign/worker_info", timeout=30)[0]
                    for role, _, port, _ in rt.ROLES
                })
            finally:
                group.close()
                shutil.rmtree(ec)
                rt.verify_devices(devices)
    except BaseException as error:
        rt.write(run / "state/failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    rt.write(run / "state/complete.json", {"status": "COMPLETE", "time": time.time()})


if __name__ == "__main__":
    main()
