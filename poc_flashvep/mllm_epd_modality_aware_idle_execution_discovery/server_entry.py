"""Add a fixed local endpoint to the pinned EPD server in an owned lease."""

from fastapi import Request
from vllm.entrypoints.openai import api_server

from ..mllm_epd_prefill_heavy_token_split import server_entry


def main():
    original = api_server.build_app

    def build(*args, **kwargs):
        app = original(*args, **kwargs)

        @app.post("/campaign/idle/vision")
        async def native_vision(request: Request):
            body = await request.json()
            return await app.state.engine_client.collective_rpc(
                "campaign_idle_vision", timeout=90, args=(body,))

        return app

    api_server.build_app = build
    server_entry.main()


if __name__ == "__main__":
    main()
