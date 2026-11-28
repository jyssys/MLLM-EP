"""Local startup hooks for the measured joint multimodal execution."""

import os
import time

from .joint_contract import apply_contract, plan_from_request


def main():
    from fastapi import Request
    from vllm.entrypoints.openai import api_server
    from vllm.v1.engine.input_processor import InputProcessor

    from ..mllm_epd_prefill_heavy_token_split import server_entry
    from ..mllm_epd_prefill_heavy_token_split.common import ROOT, append
    from . import joint_splice

    process = InputProcessor.process_inputs
    role = os.environ["MLLM_EPD_ROLE"]

    def process_inputs(instance, *args, **kwargs):
        result = process(instance, *args, **kwargs)
        if plan_from_request(result)[0] is None:
            return result
        result = apply_contract(result, role)
        append(ROOT / "raw" / f"joint_contract_{role}.jsonl", {
            "request_id": result.request_id, "monotonic_ns": time.monotonic_ns(),
            "role": role, "prompt_tokens": len(result.prompt_token_ids),
            "contract": plan_from_request(result)[0],
        })
        return result

    InputProcessor.process_inputs = process_inputs
    build = api_server.build_app

    def build_app(*args, **kwargs):
        app = build(*args, **kwargs)

        @app.post("/campaign/joint_vision")
        async def vision_probe(request: Request):
            if role != "decode":
                raise ValueError("This fixed vision probe runs only on the owned decoder")
            body = await request.json()
            return await app.state.engine_client.collective_rpc(
                "campaign_joint_vision", args=(body["fixture"],), timeout=120
            )

        @app.post("/campaign/joint_profile")
        async def joint_profile(request: Request):
            body = await request.json()
            return await app.state.engine_client.collective_rpc(
                "campaign_joint_timing", args=(body["tag"],), timeout=60
            )

        return app

    api_server.build_app = build_app
    joint_splice.install()
    server_entry.main()


if __name__ == "__main__":
    main()
