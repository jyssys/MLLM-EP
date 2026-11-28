"""Keep the existing readiness worker API and add a localhost Encoder settings RPC."""

from fastapi import Request
from vllm.entrypoints.openai import api_server

from ..mllm_epd_modality_aware_idle_execution_discovery import readiness_execution


def main():
    previous = api_server.build_app

    def build(*args, **kwargs):
        app = previous(*args, **kwargs)

        @app.post("/campaign/group_settings")
        async def group_settings(request: Request):
            body = await request.json()
            return await app.state.engine_client.collective_rpc(
                "campaign_group_settings", timeout=30, args=(body["mode"], body.get("signature", False)))

        return app

    api_server.build_app = build
    readiness_execution.main()


if __name__ == "__main__":
    main()
