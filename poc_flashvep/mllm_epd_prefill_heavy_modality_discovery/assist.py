"""Co-resident LM Prefill producer for a matched three-GPU EPD intervention."""

import json
import socket

from ..mllm_epd_h200_prefill_heavy_batch_matched import runtime as rt
from ..mllm_epd_h200_prefill_heavy_batch_matched.trace import rpc


HELPER_PORT = 18843
HELPER_SIDE_PORT = 19843
HELPER_LABEL = "encoder_prefill_helper"
HELPER_KV_GIB = 8


def launch_helper(group, ec, encoder_uuid, media, args):
    """Start an additional Prefill process on the original encoder GPU."""
    encoders = [process for label, process in group.processes if label == "encoder"]
    rt.require(len(encoders) == 1 and encoders[0].poll() is None,
               "Expected one live owned encoder before staging the Prefill helper")
    encoder = encoders[0]
    nonce = group.env["MLLM_EPD_RUN_NONCE"]
    rt.require(group.env.get("MLLM_EPD_EC_TRANSPORT") == "cuda_ipc",
               "The helper requires the original encoder's CUDA IPC transport")
    apps_query = "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory"

    def verify_gpu_owners(*processes):
        rows = [row for row in rt.nvidia(apps_query) if row[0] == encoder_uuid]
        owners = {process.pid: [] for process in processes}
        for row in rows:
            pid = int(row[1])
            matching = [session for session in owners if rt.pid_owned(pid, session, nonce)]
            rt.require(len(matching) == 1, f"Unexpected GPU process {pid} on the encoder device")
            owners[matching[0]].append(pid)
        rt.require(all(owners.values()), "An expected worker does not own memory on the encoder device")
        return {"compute_apps": rows, "sessions_to_gpu_pids": owners}

    before = verify_gpu_owners(encoder)
    for port in (HELPER_PORT, HELPER_SIDE_PORT):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", port))

    servers = rt.module("servers")
    command = servers.command(
        "prefill", HELPER_PORT, ec, HELPER_SIDE_PORT, observe=True,
        kv_cache_gib=HELPER_KV_GIB, gpu_memory_utilization=.35, mm_cache_gb=0,
        ec_transport="cuda_ipc",
    )
    values = {
        "--allowed-local-media-path": media,
        "--max-num-batched-tokens": args.p_budget,
        "--max-num-seqs": args.max_seqs,
        "--worker-extension-cls": f"{__package__}.observer.DiscoveryWorker",
    }
    for flag, value in values.items():
        command[command.index(flag) + 1] = str(value)
    command.extend([
        "--attention-backend", "FLASH_ATTN",
        "--attention-config", json.dumps({"flash_attn_version": 3}),
        "--enable-prompt-tokens-details",
    ])
    helper = group.start(
        HELPER_LABEL, command, CUDA_VISIBLE_DEVICES=encoder_uuid,
        VLLM_NIXL_SIDE_CHANNEL_PORT=str(HELPER_SIDE_PORT), MLLM_EPD_ROLE="prefill",
        MLLM_EPD_TOKEN_SPLIT="0", VLLM_CACHE_ROOT=str(args.cache_root.resolve() / HELPER_LABEL),
        MLLM_EPD_EC_PATH=str(ec), MLLM_EPD_METADATA_PATH=str(ec / "processed_prompts"),
    )
    group.ready(helper, HELPER_PORT, HELPER_LABEL, role="prefill")
    after = verify_gpu_owners(encoder, helper)
    ack, = rpc(HELPER_PORT, "ec_ipc/ack_policy", {"mode": "file"})
    rt.require(ack["actual"] == "file" and rt.pid_owned(ack["pid"], helper.pid, nonce),
               "File acknowledgement was not set on the owned Prefill helper")
    rt.write(group.run / "state" / "encoder_prefill_helper_ownership.json", {
        "encoder_gpu_uuid": encoder_uuid, "helper_port": HELPER_PORT,
        "helper_side_port": HELPER_SIDE_PORT, "helper_kv_gib": HELPER_KV_GIB,
        "before": before, "after": after, "helper_ack_policy": ack,
    })
    return helper
