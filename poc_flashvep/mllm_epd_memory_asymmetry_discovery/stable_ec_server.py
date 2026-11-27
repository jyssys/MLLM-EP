"""Use distinct renderer, encoder-worker, and persistent consumer identities."""

import asyncio
from contextvars import ContextVar
import hashlib
import logging
import os

from fastapi import Request
from fastapi.responses import JSONResponse

from ..mllm_epd_prefill_heavy_token_split import processor_handoff, server_entry
from ..mllm_epd_prefill_heavy_token_split.host_timeline import APITimelineMiddleware
from .stable_ec import checked_hash, source_alias


ENCODER_REQUEST = ContextVar("stable_ec_encoder_request", default=None)
LOGGER = logging.getLogger(__name__)


def occurrence_request(request, owner):
    """Preserve original processed data even if the same image appears twice."""
    keys = []
    messages = []
    for message in request.messages:
        content = message.get("content")
        if not isinstance(content, list):
            messages.append(message)
            continue
        items = []
        for item in content:
            if item["type"] == "image_url":
                keys.append(checked_hash(item.get("uuid")))
                identity = f"processor:{owner}:{len(keys)}".encode()
                item = dict(item, uuid=hashlib.sha256(identity).hexdigest())
            items.append(item)
        messages.append(dict(message, content=items))
    if not keys:
        raise ValueError("Stable EC requires at least one image")
    return request.model_copy(update={"messages": messages}), keys


def with_image_keys(rendered, keys):
    conversation, prompts = rendered
    if len(prompts) != 1 or prompts[0]["type"] != "multimodal":
        raise ValueError("Stable EC requires a single original multimodal prompt")
    prompt = prompts[0]
    if set(prompt["mm_hashes"]) != {"image"} or len(prompt["mm_hashes"]["image"]) != len(keys):
        raise ValueError("Original processor image order does not match the request")
    if any(item is None for item in prompt["mm_kwargs"]["image"]):
        raise ValueError("A request-occurrence processor input unexpectedly omitted an original image")
    return conversation, [dict(prompt, mm_hashes={"image": keys})]


def install_renderer():
    from vllm.renderers.online_renderer import OnlineRenderer

    original_render = OnlineRenderer.render_chat
    original_save = processor_handoff.RenderStore.save

    async def render(renderer, request, *args, **kwargs):
        contract = processor_handoff.HANDOFF_CONTEXT.get()
        lease = ENCODER_REQUEST.get()
        if contract is None or lease is None:
            raise RuntimeError("Stable EC request lacks its EPD processor handoff")
        owner, _ = contract
        checked_hash(owner)
        if request.cache_salt != os.environ["MLLM_EPD_RUN_NONCE"] or request.mm_processor_kwargs:
            raise ValueError("Stable EC requires the current cache scope and frozen image processor settings")
        occurrence, keys = occurrence_request(request, owner)
        rendered = await original_render(renderer, occurrence, *args, **kwargs)
        result = with_image_keys(rendered, [source_alias(owner, key) for key in keys])
        lease.update(owner=owner, keys=keys)
        acquired = await lease["engine"].collective_rpc("campaign_stable_ec_acquire", timeout=60,
                                                         args=(owner, keys))
        if len(acquired) != 1 or acquired[0]["owner"] != owner:
            raise RuntimeError("Expected one exact producer to acquire the encoder lease")
        return result

    def save(store, owner, request, rendered, mode):
        lease = ENCODER_REQUEST.get()
        if lease is None or lease.get("owner") != owner:
            raise RuntimeError("Processed prompt must correspond to a live encoder lease")
        return original_save(store, owner, request, with_image_keys(rendered, lease["keys"]), mode)

    OnlineRenderer.render_chat = render
    processor_handoff.RenderStore.save = save


class StableECMiddleware(APITimelineMiddleware):
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] not in (
                "/campaign/ec_ipc/release", "/v1/chat/completions"):
            return await super().__call__(scope, receive, send)
        engine = scope["app"].state.engine_client
        if scope["path"] == "/campaign/ec_ipc/release" and scope["method"] == "POST":
            request = Request(scope, receive=receive)
            try:
                owner = checked_hash((await request.json())["key"])
            except (KeyError, TypeError, ValueError):
                response = JSONResponse({"error": "Malformed encoder request owner"}, status_code=400)
            else:
                result = await engine.collective_rpc("campaign_stable_ec_release", timeout=60, args=(owner,))
                response = JSONResponse(result)
            return await response(scope, receive, send)
        lease = {"engine": engine}
        token = ENCODER_REQUEST.set(lease)
        status = None

        async def capture_status(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            return await super().__call__(scope, receive, capture_status)
        finally:
            ENCODER_REQUEST.reset(token)
            if "owner" in lease and (status is None or status >= 400):
                try:
                    await asyncio.shield(engine.collective_rpc("campaign_stable_ec_release", timeout=60,
                                                               args=(lease["owner"],)))
                except Exception:
                    LOGGER.exception("Encoder stage could not retire an unsuccessful request lease")


if __name__ == "__main__":
    # vLLM resolves middleware by qualified name, whereas -m runs this file as
    # __main__. Install using the imported module so both see one ContextVar.
    from .stable_ec_server import install_renderer

    install_renderer()
    server_entry.main()
