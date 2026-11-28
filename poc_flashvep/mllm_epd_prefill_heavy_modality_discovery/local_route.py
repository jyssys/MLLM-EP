"""Whole-request D-local control on the existing CUDA IPC / NIXL path."""

import argparse
from contextlib import asynccontextmanager
import time

import aiohttp
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..mllm_epd_h200_prefill_heavy_batch_matched.capacity import PULL_PARAMS, headers
from ..mllm_epd_prefill_heavy_token_split.common import ROOT, append
from ..mllm_epd_prefill_heavy_token_split.idle_admission import FirstToken, IdleAdmission


KV_BYTES_PER_TOKEN = 57344
EMBEDDING_BYTES_PER_ROW = 7168


def application(policy, p_speed, d_speed, link_speed, typical_visual_ratio):
    session = None
    outstanding = {}
    admission = IdleAdmission()
    e_port, p_port, d_port = 18840, 18841, 18842
    target = "/v1/chat/completions"

    @asynccontextmanager
    async def lifespan(app):
        nonlocal session
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2400, sock_read=1800),
                                         connector=aiohttp.TCPConnector(limit=0), trust_env=False) as session:
            yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"name": policy, "ready": True}

    async def post(port, path, payload, request_headers=None):
        async with session.post(f"http://127.0.0.1:{port}{path}", json=payload,
                                headers=request_headers) as response:
            if response.status != 200:
                raise RuntimeError(f"{port} {response.status}: {(await response.text())[:1500]}")
            return await response.json()

    @app.post(target)
    async def completion(request: Request):
        payload = await request.json()
        request_id = request.headers["x-request-id"]
        trace = {"runtime_request_id": request_id, "name": policy, "mode": "whole_request",
                 "received": time.monotonic(), "status": "FAILED_EXECUTION"}
        image_parts = sum(piece.get("type") == "image_url"
                          for message in payload["messages"] for piece in message.get("content", []))
        partial = dict(payload, stream=False, max_tokens=1)
        partial.pop("stream_options", None)

        def stage_headers(stage):
            return headers(request_id, stage) if image_parts else {"x-request-id": request_id + ":" + stage}

        async def release():
            if image_parts:
                trace["E_embedding_release"] = await post(e_port, "/campaign/ec_ipc/release",
                    {"key": stage_headers("E")["x-campaign-metadata-key"]})

        admitted = False
        try:
            if image_parts:
                trace["E_submit"] = time.monotonic()
                encoded = await post(e_port, target, partial, stage_headers("E:full"))
                trace["E_complete"] = time.monotonic()
                usage = encoded["usage"]
                trace["E_items"] = [{"begin":trace["E_submit"], "end":trace["E_complete"],
                                     "image_count":image_parts, "usage":usage}]
                tokens = usage["prompt_tokens"]
                details = usage.get("prompt_tokens_details") or {}
                visual = (details.get("multimodal_tokens") or {}).get("image", 0)
            else:
                if policy != "text":
                    raise ValueError("Image input is required for this route")
                tokens = visual = 0

            quiet = not (admission.active or admission.pending or admission.prefill_owner)
            now = time.monotonic()
            queued_tokens = sum(item["tokens"] for item in outstanding.values())
            p_finish_prediction = min((item["expected_ready"] for item in outstanding.values()), default=now)
            p_load = len(outstanding)
            cost_save_s = (tokens + queued_tokens) / p_speed - tokens / d_speed
            projected_visual = tokens * typical_visual_ratio if policy == "cost_blind" else visual
            projected_embedding_s = projected_visual * EMBEDDING_BYTES_PER_ROW / link_speed
            kv_discount_s = tokens * KV_BYTES_PER_TOKEN / link_speed
            # The same embedding is consumed over the same fabric in both arms.
            # Include the embedding once in local duration and cancel it in the route delta.
            local_time_s = tokens / d_speed + projected_embedding_s
            horizon_s = p_finish_prediction - now
            probe = policy in ("probe_e", "probe_d") and int(request_id.rsplit("_", 1)[1]) % 4 == 0
            use_e_helper = probe and policy == "probe_e"
            if policy == "force":
                use_local = True
            elif policy == "probe_d":
                use_local = probe
            elif policy in ("naive", "naive_isolated"):
                use_local = quiet and p_load >= 2
            elif policy in ("cost", "cost_blind"):
                use_local = (quiet and p_load >= 1 and cost_save_s + kv_discount_s > .025
                             and horizon_s >= local_time_s - .020)
            else:
                use_local = False
            trace.update(deflected=use_local, encoder_assisted=use_e_helper,
                         visual_tokens=visual, input_tokens=tokens,
                         decision={"P_HTTP_outstanding": p_load, "D_quiet_HTTP": quiet,
                                   "P_outstanding_prompt_tokens": queued_tokens,
                                   "predicted_ttft_saving_s": cost_save_s + kv_discount_s,
                                   "estimated_visual_rows": projected_visual,
                                   "predicted_embedding_transfer_s": projected_embedding_s,
                                   "predicted_native_P_ready_s": horizon_s,
                                   "predicted_local_first_s":local_time_s,
                                   "scope":"Causal HTTP state and calibrated cost, no observed future slack."})
            if not use_local:
                role = "H" if use_e_helper else "P"
                trace[f"{role}_submit"] = time.monotonic()
                if not use_e_helper:
                    serial_end = max((item["expected_ready"] for item in outstanding.values()), default=now)
                    outstanding[request_id] = {"tokens":tokens, "expected_ready":max(now, serial_end) + tokens / p_speed}
                try:
                    prefetched = await post(18843 if use_e_helper else p_port, target,
                                            dict(partial, kv_transfer_params=dict(PULL_PARAMS)),
                                            stage_headers("P:e_assist" if use_e_helper else "P"))
                finally:
                    if not use_e_helper:
                        del outstanding[request_id]
                trace[f"{role}_ready"] = time.monotonic()
                params = prefetched.get("kv_transfer_params")
                if (not params or not params.get("do_remote_prefill") or not params.get("remote_block_ids")
                        or params.get("remote_num_tokens") != prefetched["usage"]["prompt_tokens"]):
                    raise RuntimeError("No physical native KV handoff")
                payload["kv_transfer_params"] = params
                trace[f"{role}_response"] = {"usage":prefetched.get("usage"), "kv_transfer_params":params}
            trace["D_admission"] = await admission.enter(request_id, use_local)
            admitted = True
            trace["D_submit"] = time.monotonic()
            reply = await session.post(f"http://127.0.0.1:{d_port}{target}", json=payload, headers=stage_headers("D"))
            trace["D_headers"] = time.monotonic()
            if reply.status != 200:
                error = await reply.text()
                reply.release()
                raise RuntimeError(f"D {reply.status}: {error[:1500]}")

            async def stream():
                detector = FirstToken()
                try:
                    async for block in reply.content.iter_any():
                        if detector.feed(block):
                            trace["D_first_generated_token"] = time.monotonic()
                            if not (use_local and policy == "naive_isolated"):
                                await admission.first_token(request_id)
                        yield block
                    if not detector.seen:
                        raise RuntimeError("No D output token arrived")
                    trace["D_http_complete"] = time.monotonic()
                    await release()
                    trace["status"] = "MEASURED_ONLINE"
                except BaseException as error:
                    trace["error"] = str(error)
                    raise
                finally:
                    reply.release()
                    await admission.leave(request_id, failed=trace["status"] != "MEASURED_ONLINE")
                    trace["finished"] = time.monotonic()
                    append(ROOT/"raw"/f"proxy_{policy}.jsonl", trace)

            return StreamingResponse(stream(), media_type="text/event-stream")
        except (ValueError, RuntimeError, aiohttp.ClientError, TimeoutError) as error:
            if admitted:
                await admission.leave(request_id, failed=True)
            trace.update(error=str(error), finished=time.monotonic())
            append(ROOT/"raw"/f"proxy_{policy}.jsonl", trace)
            return JSONResponse({"error":str(error)}, status_code=502)
    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", choices=("ondemand", "naive", "naive_isolated", "cost", "cost_blind",
                                           "force", "text", "probe_e", "probe_d"),
                        required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--p-budget", type=float, required=True, help="Calibrated P prompt tokens/s")
    parser.add_argument("--d-prefill-speed", type=float, required=True, help="Calibrated D prompt tokens/s")
    parser.add_argument("--link-speed", type=float, default=20 * 1024**3, help="Effective peer bytes/s")
    parser.add_argument("--typical-visual-ratio", type=float, default=.75,
                        help="Cohort embedding-density reference for the cost-only comparator")
    args = parser.parse_args()
    if not 0 <= args.typical_visual_ratio <= 1:
        parser.error("--typical-visual-ratio must be in [0, 1]")
    uvicorn.run(application(args.name, args.p_budget, args.d_prefill_speed, args.link_speed,
                            args.typical_visual_ratio),
                host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
