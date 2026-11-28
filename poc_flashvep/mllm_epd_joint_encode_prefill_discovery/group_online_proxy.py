"""Opt-in open-loop proxy for grouped CUDA-ready Prefill recipient comparisons."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import re
import time

import aiohttp
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..mllm_epd_h200_prefill_heavy_batch_matched.capacity import PULL_PARAMS, headers, json_reply, post_rpc
from ..mllm_epd_h200_prefill_heavy_batch_matched.runtime import ROLES
from ..mllm_epd_modality_aware_idle_execution_discovery.readiness_execution import (
    FRONTIER_KEY,
    contract_path,
)
from ..mllm_epd_prefill_heavy_token_split.idle_admission import FirstToken

POLICIES = ("late_p", "early_p", "guarded_d")
PORTS = {role: port for role, _, port, _ in ROLES}
TARGET = "/v1/chat/completions"


def image_count(payload):
    images = 0
    for message in payload["messages"]:
        content = message.get("content")
        if isinstance(content, str):
            continue
        for item in content or ():
            if item["type"] == "image_url":
                images += 1
            elif item["type"] != "text":
                raise ValueError("Only original image and text input is supported")
    if not images:
        raise ValueError("Grouped readiness requires original images")
    return images


def prefix_contract(manifest, key, count, prefix_images):
    images, spans = manifest["image_hashes"], manifest["image_spans"]
    tokens = manifest["prompt_tokens"]
    if (manifest["key"] != key or manifest["mode"] != "grid_only"
            or manifest["images"] != count or len(images) != count or len(spans) != count
            or len(set(images)) != count or any(not re.fullmatch(r"[A-Za-z0-9_-]+", image) for image in images)):
        raise ValueError("Producer render metadata differs from the original image request")
    previous = 0
    for begin, end in spans:
        if not previous <= begin < end < tokens:
            raise ValueError("Producer image spans do not follow the original prompt order")
        previous = end
    if count <= prefix_images:
        return None
    cut = spans[prefix_images - 1][1] + 1
    if not cut < tokens or cut > spans[prefix_images][0]:
        raise ValueError("No causal cut after the original prefix's closing vision delimiter")
    return {"key": key, "cut": cut, "prefix_images": prefix_images, "prompt_tokens": tokens,
            "images": images, "require_cuda": True}


def choose_recipient(policy, contract, p_outstanding, d_outstanding):
    if contract is None:
        return "prefill", "late_p", "SHORT_INPUT"
    if policy == "late_p":
        return "prefill", "late_p", "LATE_CONTROL"
    if policy == "guarded_d" and p_outstanding >= 2 and not d_outstanding:
        return "decode", "early_d", "P_BACKLOG_AND_D_HTTP_EMPTY"
    reason = "EARLY_CONTROL" if policy == "early_p" else "D_HTTP_BUSY_OR_P_BELOW_THRESHOLD"
    return "prefill", "early_p", reason


async def manifest_ready(path, producer):
    while not path.is_file():
        if producer.done():
            producer.result()
            raise ValueError("E completed without publishing the original render manifest")
        await asyncio.sleep(.002)
    return json.loads(path.read_bytes())


async def cuda_ready(ec, images, producer):
    last_ready_ns = 0
    for image in images:
        folder = ec / image
        if folder.is_symlink():
            raise ValueError("CUDA-ready publication cannot use a linked image directory")
        marker = folder / "cuda_group_ready.json"
        # The grouped producer writes this marker after observing its group event complete.
        while not marker.is_file():
            if producer.done():
                producer.result()
            await asyncio.sleep(.002)
        value = json.loads(marker.read_bytes())
        if (value["identifier"] != image or not (folder / "cuda_ipc_inline_prefill.json").is_file()
                or not (folder / "cuda_ipc_ready.json").is_file()
                or (folder / "cuda_ipc_released.json").exists()):
            raise ValueError("CUDA-ready marker is missing its live original IPC export")
        last_ready_ns = max(last_ready_ns, value["ready_ns"])
    return last_ready_ns, time.monotonic_ns()


def validate_p_handoff(reply, prompt_tokens):
    params = dict(reply.get("kv_transfer_params") or {})
    if (reply["usage"]["prompt_tokens"] != prompt_tokens or params.get("remote_num_tokens") != prompt_tokens
            or not params.get("do_remote_prefill") or not params.get("remote_block_ids")):
        raise ValueError("P did not return all original prompt KV pages")
    params.pop(FRONTIER_KEY, None)
    return params


def application(policy, ec_path, trace_path, *, ports=None, prefix_images=8, ready_timeout=180):
    if policy not in POLICIES or prefix_images < 1 or ready_timeout <= 0:
        raise ValueError("Invalid grouped readiness policy")
    ports = PORTS if ports is None else ports
    ec = Path(ec_path)
    trace_path = Path(trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    p_outstanding, d_outstanding, requests = set(), set(), set()
    session = None
    failed = None

    @asynccontextmanager
    async def lifespan(app):
        nonlocal session
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2400, sock_read=1800),
                                         connector=aiohttp.TCPConnector(limit=0), trust_env=False) as session:
            yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"name": f"proxy_group_{policy}", "ready": failed is None, "policy": policy}

    def post(role, payload, request_id, stage):
        return session.post(f"http://127.0.0.1:{ports[role]}{TARGET}", json=payload,
                            headers=headers(request_id, stage))

    @app.post(TARGET)
    async def completion(request: Request):
        nonlocal failed
        if failed is not None:
            return JSONResponse({"error": "Previous grouped request requires ownership recovery"}, 503)
        request_id = request.headers.get("x-request-id")
        if not request_id:
            return JSONResponse({"error": "Expected a unique native request identifier"}, 400)
        payload = await request.json()
        if request_id in requests:
            return JSONResponse({"error": "Request identifier is already in flight"}, 400)
        if not payload.get("stream") or not payload.get("return_token_ids"):
            return JSONResponse({"error": "Expected the streaming token-ID client contract"}, 400)
        try:
            count = image_count(payload)
        except (KeyError, TypeError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, 400)

        key = headers(request_id, "E:full")["x-campaign-metadata-key"]
        path = contract_path(ec / "processed_prompts", key)
        stage_payload = dict(payload, stream=False, max_tokens=1)
        for stage_field in ("stream_options", "max_completion_tokens", "kv_transfer_params"):
            stage_payload.pop(stage_field, None)

        trace = {"runtime_request_id": request_id, "name": policy, "mode": "groupwise_ready",
                 "received": time.monotonic(), "status": "FAILED_EXECUTION", "key": key,
                 "image_count": count, "prefix_images": prefix_images}

        def finish():
            requests.discard(request_id)
            d_outstanding.discard(request_id)
            trace["finished"] = time.monotonic()
            with trace_path.open("a") as sink:
                sink.write(json.dumps(trace) + "\n")

        trace["E_submit"] = time.monotonic()
        requests.add(request_id)

        async def encode():
            reply = await json_reply(post("encoder",stage_payload,request_id,"E:full"),"Encoder")
            trace["E_complete"] = time.monotonic()
            return reply

        producer = asyncio.create_task(encode())
        response = None
        try:
            async with asyncio.timeout(ready_timeout):
                manifest = await manifest_ready(path.with_name("manifest.json"), producer)
                contract = prefix_contract(manifest, key, count, prefix_images)
                images = manifest["image_hashes"]
                trace.update(input_tokens=manifest["prompt_tokens"], fingerprint=manifest["fingerprint"],
                             first_visual_offset=manifest["image_spans"][0][0])
                late = policy == "late_p" or contract is None
                needed = images if late else images[:prefix_images]
                cuda_ns, ready_ns = await cuda_ready(ec, needed, producer)
                if late:
                    trace["all_images_cuda_ready_ns"] = cuda_ns
                    trace["all_images_observed_ready_ns"] = ready_ns
                    encoded = await producer
                    if encoded["usage"]["prompt_tokens"] != manifest["prompt_tokens"]:
                        raise ValueError("Encoder HTTP response differs from the rendering contract")
                else:
                    trace["prefix_cuda_ready_ns"] = cuda_ns
                    trace["prefix_observed_ready_ns"] = ready_ns

            if failed is not None:
                raise RuntimeError("Previous grouped request requires ownership recovery")
            p_count, d_count = len(p_outstanding), len(d_outstanding)
            role, arm, reason = choose_recipient(policy, contract, p_count, d_count)
            trace.update(deflected=role == "decode", arm=arm, reason=reason,
                         p_http_outstanding=p_count, d_http_outstanding=d_count, cut=None)
            intent = {}
            if contract is not None:
                contract.update(role=role, decode_protection=role == "decode")
                with path.open("x") as sink:
                    json.dump(contract, sink)
                trace["cut"] = contract["cut"]
                intent[FRONTIER_KEY] = key
            if role == "prefill":
                trace["P_submit"] = time.monotonic()
                p_outstanding.add(request_id)
                try:
                    p_payload = dict(stage_payload, kv_transfer_params=dict(PULL_PARAMS) | intent)
                    p_reply = await json_reply(post("prefill", p_payload, request_id, "P"), "Prefill")
                finally:
                    p_outstanding.discard(request_id)
                transfer = validate_p_handoff(p_reply, manifest["prompt_tokens"])
                trace["P_ready"] = time.monotonic()
                trace["P_KV_tokens"] = transfer["remote_num_tokens"]
            else:
                transfer = intent

            d_outstanding.add(request_id)
            trace["D_submit"] = time.monotonic()
            response = await post("decode", dict(payload, kv_transfer_params=transfer), request_id, "D")
            trace["D_headers"] = time.monotonic()
            if response.status != 200:
                raise RuntimeError(f"Decode {response.status}: {(await response.text())[:1000]}")

            async def stream():
                nonlocal failed
                first_token = FirstToken()
                done = False
                tail = b""
                try:
                    async for block in response.content.iter_any():
                        if first_token.feed(block):
                            trace["D_first_generated_token"] = time.monotonic()
                        done = done or b"\ndata: [DONE]\n" in tail + block
                        tail = (tail + block)[-24:]
                        yield block
                    if not (first_token.seen and done):
                        raise RuntimeError("Decode SSE stream ended without token IDs and a final marker")
                    trace["D_http_complete"] = time.monotonic()
                    encoded = await producer
                    if encoded["usage"]["prompt_tokens"] != manifest["prompt_tokens"]:
                        raise ValueError("Encoder HTTP response differs from the rendering contract")
                    if not late:
                        async with asyncio.timeout(ready_timeout):
                            actual_ns, observed_ns = await cuda_ready(ec, images, producer)
                        trace["all_images_cuda_ready_ns"] = actual_ns
                        trace["all_images_observed_ready_ns"] = observed_ns
                    trace["E_embedding_release"] = await json_reply(
                        post_rpc(session, ports["encoder"], "ec_ipc/release", {"key": key}), "E release")
                    trace["status"] = "MEASURED_ONLINE"
                except BaseException as exc:
                    failed = request_id
                    trace.update(error=type(exc).__name__, detail=str(exc)[:1000])
                    raise
                finally:
                    response.release()
                    if not producer.done():
                        producer.cancel()
                    finish()
                    await asyncio.gather(producer, return_exceptions=True)

            return StreamingResponse(stream(), media_type="text/event-stream")
        except BaseException as exc:
            failed = request_id
            trace.update(error=type(exc).__name__, detail=str(exc)[:1000])
            if response is not None:
                response.release()
            if not producer.done():
                producer.cancel()
            finish()
            await asyncio.gather(producer, return_exceptions=True)
            if isinstance(exc, asyncio.CancelledError):
                raise
            return JSONResponse({"error": str(exc)}, status_code=502)

    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, choices=POLICIES)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--ec-path", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--prefix-images", default=8, type=int)
    parser.add_argument("--ready-timeout", default=180, type=float)
    args = parser.parse_args()
    uvicorn.run(application(args.policy, args.ec_path, args.trace, prefix_images=args.prefix_images,
                            ready_timeout=args.ready_timeout), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
